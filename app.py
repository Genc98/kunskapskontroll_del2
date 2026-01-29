import streamlit as st
from google import genai
from google.genai import types
import pickle
import numpy as np

API_KEY = "----"
client = genai.Client(api_key=API_KEY)

st.title("SpaceBot - Ask me anything about space")

with open("rag_bot.pkl", "rb") as f:
    data = pickle.load(f)

chunks_clean = data["chunks_clean"]
embeddings = np.array(data["embeddings"])

def embed_query(query):
    resp = client.models.embed_content(
        model="text-embedding-004",
        contents=[query],
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        ) 
    return resp.embeddings[0].values

def cosine_similarity(vec1, vec2):
    return (np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2)))

def semantic_search(query, k=5):
    query_embedding = embed_query(query) 
    
    similarity_scores = []
    
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(query_embedding, chunk_embedding)
        similarity_scores.append((i, similarity_score))
        
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]
    
    return [chunks_clean[index] for index in top_indices]


question = st.text_input("Ask any question about space.")

if question:
    with st.spinner("Generating answer..."):
        context = "\n".join(semantic_search(question))
        system_prompt = """I'm going to ask you a question, and I want you to answer
            based only on the context I'm sending you, and no other information.
            If there's not enough information in the context to answer the question,
            say "I don't know". Don't try to guess.
            Keep your answer simple and break it down into nice paragraphs. """
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            contents=f"Question: {question}\nContext: {context}"
        )
        st.markdown(response.text)
