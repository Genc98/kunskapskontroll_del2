import pickle
import numpy as np
from google import genai
from google.genai import types


class ragBot:
    def __init__(self, api_key, data_path="rag_bot.pkl"):
        self.client = genai.Client(api_key=api_key)

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.chunks_clean = data["chunks_clean"]
        self.embeddings = np.array(data["embeddings"])

    def create_embeddings(self, text_list):
        resp = self.client.models.embed_content(
            model="text-embedding-004",
            contents=text_list,
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY"
            )
        )
        return resp.embeddings
    
    def cosine_similarity(self, vec1, vec2):
         return np.dot(vec1, vec2) / (
             np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def semantic_search(self, query, k=5):
        query_emb = self.create_embeddings([query])[0].values

        scores = []
        for i, emb in enumerate(self.embeddings):
            score = self.cosine_similarity(query_emb, emb)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)  # fixad sortering
        top_idx = [i for i, _ in scores[:k]]  

        return [self.chunks_clean[i] for i in top_idx]

    def generate_prompt(self, query):
        context = "\n".join(self.semantic_search(query))

        system_prompt = """I'm going to ask you a question, and I want you to answer
        based only on the context I'm sending you, and no other information.
        If there's not enough information in the context to answer the question,
        say "I don't know". Don't try to guess.
        Keep your answer simple and break it down into nice paragraphs. """

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt
            ),
            contents=f"Question: {query}\nContext: {context}"
        )

        return response.text
