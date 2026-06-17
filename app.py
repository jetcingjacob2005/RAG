"""
app.py
======
Streamlit UI for the Wikipedia AI/ML RAG chatbot (Phase 3).

Run locally with:
    streamlit run app.py

Deploy on Streamlit Cloud by connecting this GitHub repo and setting
GEMINI_KEY / GROQ_KEY in the app's Secrets manager (see README.md).
"""

import streamlit as st
from rag_pipeline import ask_question

st.set_page_config(page_title="AI/ML RAG Chatbot", page_icon="🤖")

st.title("🤖 My RAG Chatbot")
st.write("Ask a question about the documents I've loaded!")
st.caption(
    "I know about: retrieval-augmented generation, large language models, "
    "transformers, word embeddings, CNNs, RNNs, attention, BERT, GPT-4, "
    "knowledge graphs, vector databases, and RLHF — sourced from Wikipedia."
)

question = st.text_input("Your question:")

if question:
    with st.spinner("Searching documents and thinking..."):
        try:
            result = ask_question(question)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.write("**Answer:**")
    st.write(result["answer"])

    if result.get("source_names"):
        st.caption("Sources: " + ", ".join(result["source_names"]))

    with st.expander("See the document chunks I used"):
        if result["sources"]:
            for i, chunk in enumerate(result["sources"], 1):
                st.caption(f"Chunk {i}: {chunk[:200]}...")
        else:
            st.write("No chunks were retrieved for this question.")

st.markdown("---")
st.caption(
    "Built with LangGraph, Gemini embeddings (gemini-embedding-001), "
    "ChromaDB, and Groq (Llama 3.3 70B)."
)
