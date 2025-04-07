import os
import streamlit as st
import numpy as np
import faiss
from gensim.models import Word2Vec, FastText
from rank_bm25 import BM25Okapi
import openai
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load Sinhala constitution
@st.cache_resource
def load_documents():
    with open("Sri Lanka Constitution-Sinhala.txt", "r", encoding="utf-8") as f:
        docs = f.readlines()
    return docs

@st.cache_resource
def load_models_and_indices(docs):
    tokenized = [doc.split() for doc in docs]
    
    w2v = Word2Vec.load("word2vec_sinhala.model")
    ft = FastText.load("fasttext_sinhala.model")

    def embed(sentence):
        words = sentence.split()
        vectors = [
            w2v.wv[word] if word in w2v.wv else ft.wv[word]
            for word in words if word in w2v.wv or word in ft.wv
        ]
        return np.mean(vectors, axis=0) if vectors else np.zeros(w2v.vector_size)

    embeddings = np.array([embed(doc) for doc in docs], dtype="float32")

    faiss_index = faiss.IndexFlatL2(300)
    faiss_index.add(embeddings)

    bm25_index = BM25Okapi(tokenized)

    return w2v, ft, faiss_index, bm25_index, tokenized

def get_sentence_embedding(sentence, w2v, ft):
    words = sentence.split()
    vectors = [
        w2v.wv[word] if word in w2v.wv else ft.wv[word]
        for word in words if word in w2v.wv or word in ft.wv
    ]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v.vector_size)

def hybrid_retrieve(query, w2v, ft, faiss_index, bm25_index, docs, top_k=5):
    query_vec = get_sentence_embedding(query, w2v, ft).reshape(1, -1)
    _, faiss_ids = faiss_index.search(query_vec, top_k)
    faiss_hits = [docs[i] for i in faiss_ids[0] if i < len(docs)]
    bm25_hits = bm25_index.get_top_n(query.split(), docs, n=top_k)
    return list(set(faiss_hits + bm25_hits))

def generate_response(query, context):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a Sinhala legal expert. Use the context from the Sri Lankan Constitution."},
                {"role": "user", "content": f"Question: {query}\nContext: {context}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# --- Streamlit UI ---
st.title("ශ්‍රී ලංකා ව්‍යවස්ථා නීති සෙවීමේ සහායකය")

query = st.text_input("ඔබේ ප්‍රශ්නය ඇතුළත් කරන්න (සිංහලෙන්)", "")

if query:
    with st.spinner("ප්‍රශ්නය සෙවීම..."):
        docs = load_documents()
        w2v, ft, faiss_idx, bm25_idx, _ = load_models_and_indices(docs)
        retrieved = hybrid_retrieve(query, w2v, ft, faiss_idx, bm25_idx, docs)

        st.subheader("ප්‍රශ්නයට අදාල කොටස්")
        for i, d in enumerate(retrieved):
            st.markdown(f"**{i+1}.** {d.strip()}")

        st.subheader("GPT පිළිතුර (RAG සමඟ)")
        rag_response = generate_response(query, " ".join(retrieved[:2]))
        st.success(rag_response)

        st.subheader("GPT පිළිතුර (context නොමැතිව)")
        baseline_response = generate_response(query, "")
        st.info(baseline_response)