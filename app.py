import streamlit as st
from google import genai
from google.genai import types
import pickle
import numpy as np 

API_KEY = "---"
client = genai.Client(api_key=API_KEY)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            to bottom,
            #0b1026,
            #1e3a8a,
            #22c55e
        );
    }

    h1, h2, h3 {
        color: #ffffff;
        text-shadow: 0 0 12px rgba(34,197,94,0.7);
    }

    p, span, label, div {
        color: #f9fafb;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
question = st.text_input("Ask any question about space.")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
        
    with st.spinner("Generating answer..."):
            context = "\n".join(semantic_search(question))
            system_prompt = """
                I will ask you a question, and I want you to answer based on the given context and no other information. 
                If there is not enough information, respond exactly like this:
                "I don't know." 
                Do not try to guess or answer.
                Keep your answers simple and break them down to nice paragrahs.
                Do not write phrases like "Based on the context" or anything similiar """
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(system_instruction=system_prompt),
                contents=f"Question: {question}\nContext: {context}"
            )
            answer_text = response.text

            st.session_state.chat_history.append({"role": "assistant", "content": answer_text})

for msg in reversed(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
