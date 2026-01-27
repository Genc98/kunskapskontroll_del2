import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
import numpy as np

API_KEY = "------"
client = genai.Client(api_key=API_KEY)

st.title("SpaceBot - Ask me anything about space")

reader = PdfReader("pdf_file/solarsystem.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

chunks = []
n = 800
overlap = 150

for i in range(0, len(text), n - overlap):
    chunks.append(text[i:i + n])


chunks_clean = []
for chunk in chunks:
    chunk = chunk.replace("\n", " ").replace("- ", "").strip()
    if(
       len(chunk) > 400
       and chunk.count(".") >= 2
       and "www.nasa.gov" not in chunk
       and "OUR SOLAR SYSTEM" not in chunk.upper()
    ):
        chunks_clean.append(chunk)

def create_embeddings(text_list, model="text-embedding-004", task_type="SEMANTIC_SIMILARITY"):
    embeddings = []
    for i in range(0, len(text_list), 100):  
        resp = client.models.embed_content(
            model=model,
            contents=text_list[i:i+100],
            config=types.EmbedContentConfig(task_type=task_type)
        )
        embeddings += resp.embeddings  
    return embeddings

embeddings = create_embeddings(chunks_clean)

def cosine_similarity(vec1, vec2):
    return (np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2)))

def semantic_search(query, chunks_clean, embeddings, k=5):
    query_embedding = create_embeddings([query])[0].values  
    
    similarity_scores = []
    
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(query_embedding, chunk_embedding.values)
        similarity_scores.append((i, similarity_score))
        
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]
    
    return [chunks_clean[index] for index in top_indices]


question = st.text_input("Ask any question about space.")

if question:
    with st.spinner("Generating answer..."):
        context = "\n".join(semantic_search(question, chunks_clean, embeddings))
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
