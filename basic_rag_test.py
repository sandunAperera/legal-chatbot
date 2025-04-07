from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "business-support-rag"

index = pc.Index(name=INDEX_NAME)
print("✅ Pinecone index connected!")

def get_embedding(text):
    """Generate an embedding for the given text using OpenAI."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def retrieve_from_pinecone(query, top_k=5):
    """Retrieves the most relevant chunks from Pinecone based on query embedding."""
    print("🔍 Retrieving relevant chunks from Pinecone...")

    query_embedding = get_embedding(query)

    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    results = [match["metadata"]["text"] for match in response["matches"]]
    return results

def generate_response(query, retrieved_chunks):
    """Generates a final response by augmenting the query with retrieved context."""
    context = "\n".join(retrieved_chunks)
    prompt = f"""
    You are an AI assistant that answers business-related questions.
    Use the following retrieved information to answer the query.
    
    Retrieved Context:
    {context}
    
    User Query:
    {query}
    
    Provide a detailed but concise response.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

def basic_rag_pipeline(query):
    """Executes the Basic RAG pipeline for a given query."""
    retrieved_chunks = retrieve_from_pinecone(query)
    
    if not retrieved_chunks:
        return "⚠️ No relevant information found in the database."

    response = generate_response(query, retrieved_chunks)
    return response

if __name__ == "__main__":
    user_query = "What is this business in short?"
    print("🟢 User Query:", user_query)
    
    rag_response = basic_rag_pipeline(user_query)
    print("\n💡 AI Response:", rag_response)
