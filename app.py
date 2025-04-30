import os
import streamlit as st
import numpy as np
import faiss
import pandas as pd
import time
from datetime import datetime
from gensim.models import Word2Vec
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import openai
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# --- Initial setup ---
st.set_page_config(page_title="🇱🇰 Legal Chatbot", layout="centered")

# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Go to", ["🔐 Login", "📝 Register"])

# --- Styling ---
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stButton>button {
            background-color: #00ffc3;
            color: black;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .stTextInput>div>div>input {
            background-color: #161b22;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Utility: Load documents ---
@st.cache_resource
def load_documents():
    with open("Sri Lanka Constitution-Sinhala.txt", "r", encoding="utf-8") as f:
        docs = f.readlines()
    return [doc.strip() for doc in docs if len(doc.strip()) > 10]

# --- Utility: Load Word2Vec + Build indices ---
@st.cache_resource
def labour_act_docs(docs):
    w2v = Word2Vec.load("word2vec_sinhala.model")
    vector_size = w2v.vector_size

    def embed(sentence):
        words = sentence.split()
        vectors = [w2v.wv[word] for word in words if word in w2v.wv]
        return np.mean(vectors, axis=0) if vectors else None

    valid_docs, valid_embeddings = [], []
    for doc in docs:
        vec = embed(doc)
        if vec is not None:
            valid_docs.append(doc)
            valid_embeddings.append(vec)

    embeddings = np.array(valid_embeddings, dtype="float32")
    faiss_index = faiss.IndexFlatL2(vector_size)
    faiss_index.add(embeddings)
    bm25_index = BM25Okapi([doc.split() for doc in valid_docs])

    return w2v, faiss_index, bm25_index, valid_docs

# --- Utility: Get sentence embedding ---
def get_sentence_embedding(sentence, w2v):
    words = sentence.split()
    vectors = [w2v.wv[word] for word in words if word in w2v.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v.vector_size)

# --- Hybrid Retrieval ---
def hybrid_retrieve(query, w2v, faiss_index, bm25_index, docs, top_k=5):
    query_vec = get_sentence_embedding(query, w2v).reshape(1, -1)
    _, faiss_ids = faiss_index.search(query_vec, top_k)
    faiss_hits = [docs[i] for i in faiss_ids[0] if i < len(docs)]
    bm25_hits = bm25_index.get_top_n(query.split(), docs, n=top_k)
    return list(set(faiss_hits + bm25_hits))

# --- GPT Response Generator ---
def generate_response(query, context):
    system_role = "You are a legal assistant answering questions based on the Sri Lankan Constitution."
    context_text = "\n".join(context) if context else "No relevant context found."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{query}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI API error: {e}"

# --- Logging ---
def log_user_query(username, query, response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "Username": username,
        "Timestamp": timestamp,
        "Query": query,
        "Response": response
    }
    if not os.path.exists("logs"):
        os.makedirs("logs")

    log_file = "logs/user_logs.csv"
    if not os.path.exists(log_file):
        pd.DataFrame([log_entry]).to_csv(log_file, index=False)
    else:
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
        df.to_csv(log_file, index=False)

def delete_user_history(username):
    log_file = "logs/user_logs.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = df[df["Username"] != username]
        df.to_csv(log_file, index=False)

# --- CONFIG LOAD ---
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"]
)

# ===============================
# 🔐 LOGIN PAGE
# ===============================
if page == "🔐 Login":

    name, authentication_status, username = authenticator.login(
        location="main",
        fields={
            "Form name": "Login",
            "Username": "Username",
            "Password": "Password",
            "Login": "Login"
        }
    )

    if authentication_status is False:
        st.error("❌ Incorrect username or password.")
    elif authentication_status is None:
        st.warning("🟡 Please enter your username and password.")
    elif authentication_status:
        authenticator.logout("🚪 Logout", "sidebar", key="logout_button")
        st.sidebar.success(f"👋 Welcome, {name}")

        st.title("🇱🇰 Sri Lankan Legal Chatbot")
        query = st.text_input("📜 Enter your legal question:")

        if query:
            with st.spinner("🤖 GPT-4 is typing..."):
                docs = load_documents()
                w2v, faiss_idx, bm25_idx, valid_docs = labour_act_docs(docs)

                if not w2v or not valid_docs:
                    st.error("❌ Model or documents failed to load.")
                else:
                    results = hybrid_retrieve(query, w2v, faiss_idx, bm25_idx, valid_docs)
                    response = generate_response(query, results)
                    log_user_query(username, query, response)

                    st.markdown(
                        f"""
                        <div style='
                            background-color: #1e1e1e;
                            padding: 1rem;
                            border-radius: 10px;
                            border-left: 4px solid #ffc400;
                            margin-top: 1rem;
                            font-size: 1rem;
                            color: #ffffff;
                        '>
                            <b>🧑 You:</b><br>{query}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    placeholder = st.empty()
                    animated_response = ""

                    for word in response.split():
                        animated_response += word + " "
                        placeholder.markdown(
                            f"""
                            <div style='
                                background-color: #161b22;
                                padding: 1rem;
                                border-radius: 10px;
                                border-left: 4px solid #00ffc3;
                                margin-top: 1rem;
                                font-size: 1rem;
                                color: #c9d1d9;
                            '>
                                <b>🤖 GPT-4:</b><br>{animated_response}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        time.sleep(0.05)

        # 🗑️ History deletion
        if st.button("🗑️ Delete My History"):
            delete_user_history(username)
            st.success("✅ Your chat history has been deleted.")

        # 📂 Show chat history
        if os.path.exists("logs/user_logs.csv"):
            df = pd.read_csv("logs/user_logs.csv")
            user_history = df[df["Username"] == username]

            if not user_history.empty:
                st.markdown("### 🗂️ Your Chat History")
                for i, row in user_history.tail(10).iterrows():
                    with st.expander(f"🕒 {row['Timestamp']} - {row['Query']}"):
                        st.markdown(f"**🧠 GPT:** {row['Response']}")
            else:
                st.info("ℹ️ You have no saved chat history yet.")

# ===============================
# 📝 REGISTER PAGE
# ===============================
elif page == "📝 Register":
    st.title("📝 Create a New Account")

    name = st.text_input("👤 Full Name")
    username_signup = st.text_input("🆔 Choose a Username")
    email = st.text_input("📧 Email Address")
    password = st.text_input("🔐 Password", type="password")
    confirm = st.text_input("🔁 Confirm Password", type="password")

    if st.button("Register"):
        if not all([name, username_signup, email, password, confirm]):
            st.warning("Please fill all fields.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            if username_signup in config["credentials"]["usernames"]:
                st.error("❌ Username already exists.")
            else:
                hashed_pw = stauth.Hasher([password]).generate()[0]
                config["credentials"]["usernames"][username_signup] = {
                    "email": email,
                    "name": name,
                    "password": hashed_pw
                }

                with open("config.yaml", "w") as file:
                    yaml.dump(config, file, default_flow_style=False)

                st.success("✅ Registration complete! Please return to the login page.")
