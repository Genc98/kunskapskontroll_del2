import streamlit as st
from rag_bot import ragBot

st.set_page_config(page_title="SpaceBot RAG")

API_KEY = "--"

@st.cache_resource
def load_bot():
    return ragBot(api_key=API_KEY)


spaceBot = load_bot()

st.title("SpaceBot - Ask anything about space")

question = st.text_input("Ask anything related to space")

if question:
    with st.spinner("Generating answer..."):
        answer = spaceBot.generate_prompt(question)
        st.markdown(answer)
