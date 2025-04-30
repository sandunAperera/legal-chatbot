import os
import streamlit as st
import numpy as np
import faiss
import pandas as pd
from datetime import datetime
from gensim.models import Word2Vec
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import openai
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# ---- Streamlit Page Config ----
st.set_page_config(page_title="Sri Lankan Legal Chatbot", page_icon="⚖️", layout="centered")

# -- Load environment variables --
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# -- Load config.yaml for login system --
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"]
)


# ---- Login ----
name, authentication_status, username = authenticator.login(
    location="main",
    fields={
        "Form name": "🔐 Login",
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

    authenticator.logout("🚪 Logout", "sidebar")
    st.sidebar.success(f"👋 Welcome, {name}")

    st.title("🇱🇰 Sri Lankan Constitution Legal Chatbot")
    st.markdown("Ask your legal question in **Sinhala** or **English**.")

    # ---- Load Constitution Text ----
    @st.cache_resource
    def load_documents():
        with open("Sri Lanka Constitution-Sinhala.txt", "r", encoding="utf-8") as f:
            docs = f.readlines()
        return [doc.strip() for doc in docs if len(doc.strip()) > 10]

    # ---- Load Word2Vec & Build FAISS/BM25 Indices ----
    @st.cache_resource
    def labour_act_docs(docs):
        w2v = Word2Vec.load("word2vec_sinhala.model")
        vector_size = w2v.vector_size

        def embed(sentence):
            words = sentence.split()
            vectors = [w2v.wv[word] for word in words if word in w2v.wv]
            if not vectors:
                return None
            return np.mean(vectors, axis=0)

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

    def get_sentence_embedding(sentence, w2v):
        words = sentence.split()
        vectors = [w2v.wv[word] for word in words if word in w2v.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(w2v.vector_size)

    def hybrid_retrieve(query, w2v, faiss_index, bm25_index, docs, top_k=5):
        query_vec = get_sentence_embedding(query, w2v).reshape(1, -1)
        _, faiss_ids = faiss_index.search(query_vec, top_k)
        faiss_hits = [docs[i] for i in faiss_ids[0] if i < len(docs)]
        bm25_hits = bm25_index.get_top_n(query.split(), docs, n=top_k)
        return list(set(faiss_hits + bm25_hits))

    def generate_response(query, context):
        system_role = "You are a legal assistant who answers questions based on the Sri Lankan Constitution using the provided context."
        context_text = "\n".join(context) if context else "No relevant context available."

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

    # ---- Logging Function ----
    def log_user_query(username, query, response):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "Username": username,
            "Timestamp": timestamp,
            "Query": query,
            "Response": response
        }
        log_file = "logs/user_logs.csv"

        if not os.path.exists("logs"):
            os.makedirs("logs")

        if not os.path.exists(log_file):
            df = pd.DataFrame([log_entry])
            df.to_csv(log_file, index=False)
        else:
            df = pd.read_csv(log_file)
            df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
            df.to_csv(log_file, index=False)
            

    # ---- Query Box ----
    query = st.text_input("📜 Enter your legal question:")

    if query:
        with st.spinner("💡 Thinking like a lawyer..."):
            docs = load_documents()
            w2v, faiss_idx, bm25_idx, valid_docs = labour_act_docs(docs)

            if not w2v or not valid_docs:
                st.error("❌ Model or document loading failed.")
            else:
                results = hybrid_retrieve(query, w2v, faiss_idx, bm25_idx, valid_docs)
                response = generate_response(query, results)

                log_user_query(username, query, response)
                

                st.markdown("### 📚 Relevant Sections:")
                for i, doc in enumerate(results):
                    st.markdown(f"<div style='padding: 0.5rem; background-color: #1e1e1e; border-left: 4px solid #00ffc3; margin-bottom: 1rem;'><b>{i+1}.</b> {doc}</div>", unsafe_allow_html=True)

                st.markdown("### 💬 Chatbot Answer:")
                st.success(response)
                
                # ---- Show Chat History for Logged-in User ----
                
if os.path.exists("logs/user_logs.csv"):
    df = pd.read_csv("logs/user_logs.csv")
    user_history = df[df["Username"] == username]

    if not user_history.empty:
        st.markdown("### 🗂️ Your Chat History")
        for i, row in user_history.tail(10).iterrows():  # last 10 entries
            with st.expander(f"🕒 {row['Timestamp']} - {row['Query']}"):
                st.markdown(f"**🧠 Answer:** {row['Response']}")
    else:
        st.info("ℹ️ You have no saved chat history yet.")
        
def delete_user_history(username):
    log_file = "logs/user_logs.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = df[df["Username"] != username]  # keep all but current user
        df.to_csv(log_file, index=False)
if st.button("🗑️ Delete My History"):
    delete_user_history(username)
    st.success("✅ Your chat history has been deleted.")


