import os
import streamlit as st
import numpy as np
import faiss
from gensim.models import Word2Vec
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import openai

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Style Customization ---
st.set_page_config(page_title="Legora | Legal Chatbot", page_icon="🤖", layout="centered")

# ---- HEADER SECTION ----
st.markdown("""
    <style>
    .main { background-color: #0f1117; color: #ffffff; }
    h1 { color: #00ffcc; text-align: center; font-size: 3em; }
    .block-container { padding-top: 2rem; }
    .stTextInput > label { font-weight: bold; font-size: 1.1em; color: #00ffd2; }
    .stMarkdown h3 { margin-top: 2rem; color: #ffffff; }
    .stSpinner { color: #00ffcc !important; }
    .stSuccess { background-color: #161b22 !important; border-left: 5px solid #00ffcc; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖| Sri Lankan Legal Chatbot")

st.markdown("""
Welcome to your AI legal assistant.  
Ask me a question based on the **Sri Lankan Constitution** — in Sinhala or English —  
and I’ll provide context-aware answers using real constitutional text and GPT-4.

🧠 Example:
- "ශ්‍රී ලංකාවේ ජනාධිපතිට ඇති බලතල මොනවාද?"
- "What are the rights of a detained person under Sri Lankan law?"
""")

# ---- Load Constitution Text ----
@st.cache_resource
def load_documents():
    try:
        with open("Sri Lanka Constitution-Sinhala.txt", "r", encoding="utf-8") as f:
            docs = f.readlines()
        return [doc.strip() for doc in docs if len(doc.strip()) > 10]
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return []

# ---- Load Embedding Models and Indices ----
@st.cache_resource
def labour_act_docs(docs):
    w2v = Word2Vec.load("word2vec_sinhala.model")
    vector_size = w2v.vector_size

    def embed(sentence):
        words = sentence.split()
        vectors = [w2v.wv[word] for word in words if word in w2v.wv]
        if not vectors:
            return None
        vec = np.mean(vectors, axis=0)
        return vec if vec.shape[0] == vector_size else None

    valid_docs, valid_embeddings = [], []
    for doc in docs:
        vec = embed(doc)
        if vec is not None:
            valid_docs.append(doc)
            valid_embeddings.append(vec)

    if not valid_embeddings:
        st.error("❌ No valid embeddings generated.")
        return None, None, None, []

    embeddings = np.array(valid_embeddings, dtype="float32")
    faiss_index = faiss.IndexFlatL2(vector_size)
    faiss_index.add(embeddings)
    bm25_index = BM25Okapi([doc.split() for doc in valid_docs])

    return w2v, faiss_index, bm25_index, valid_docs

# ---- Sentence Embedding for Query ----
def get_sentence_embedding(sentence, w2v):
    words = sentence.split()
    vectors = [w2v.wv[word] for word in words if word in w2v.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v.vector_size)

# ---- Hybrid Search ----
def hybrid_retrieve(query, w2v, faiss_index, bm25_index, docs, top_k=5):
    query_vec = get_sentence_embedding(query, w2v).reshape(1, -1)
    _, faiss_ids = faiss_index.search(query_vec, top_k)
    faiss_hits = [docs[i] for i in faiss_ids[0] if i < len(docs)]
    bm25_hits = bm25_index.get_top_n(query.split(), docs, n=top_k)
    return list(set(faiss_hits + bm25_hits))

# ---- GPT-4 Based Response ----
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

# ---- Query Section ----
query = st.text_input("📜 Enter your legal question here:")

if query:
    with st.spinner("💡 Thinking like a lawyer..."):
        docs = load_documents()
        w2v, faiss_idx, bm25_idx, valid_docs = labour_act_docs(docs)

        if not w2v or not valid_docs:
            st.error("❌ Model or document loading failed.")
        else:
            results = hybrid_retrieve(query, w2v, faiss_idx, bm25_idx, valid_docs)
            response = generate_response(query, results)

            st.markdown("### 📚 Top Relevant Sections Retrieved:")
            for i, doc in enumerate(results):
                st.markdown(f"<div style='padding: 0.5rem; background-color: #1e1e1e; border-left: 4px solid #00ffc3; margin-bottom: 1rem;'><b>{i+1}.</b> {doc}</div>", unsafe_allow_html=True)

            st.markdown("### 💬 AI-Powered Legal Response:")
            st.success(response)
