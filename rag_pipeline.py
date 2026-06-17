"""
rag_pipeline.py
================
Core RAG pipeline for the Wikipedia AI/ML chatbot.

This module wraps the exact ingestion + LangGraph logic built and verified
in Phase 2 (Internship_Phase_2 notebook) into reusable functions that
app.py (the Streamlit UI) imports and calls.

Pipeline summary (matches Phase 2 exactly):
  - Source docs : 12 Wikipedia articles on AI/ML topics
  - Chunking    : RecursiveCharacterTextSplitter, 500 chars, 50 overlap
  - Embeddings  : Gemini "models/gemini-embedding-001" via the raw
                  google-generativeai SDK (NOT langchain's wrapper —
                  the langchain wrapper hits a v1beta 404 bug)
  - Vector store: a native ChromaDB PersistentClient collection
                  named "rag_documents" (NOT langchain's Chroma class)
  - LLM         : Groq "llama-3.3-70b-versatile" via the raw groq.Groq
                  client (NOT langchain_groq.ChatGroq)
  - Orchestration: LangGraph StateGraph with retrieve -> generate
                  (or error_handler) -> END

It expects a pre-built ChromaDB collection on disk at CHROMA_DIR,
created by ingest.py (Phase 2, Cells 3-5). Run ingest.py once before
running the Streamlit app if chroma_db/ does not already exist.
"""

import os
from typing import TypedDict, List, Optional

import chromadb
import google.generativeai as genai
from groq import Groq
from langgraph.graph import StateGraph, END

# ── Configuration (matches Phase 2 exactly) ─────────────────────────────────
CHROMA_DIR        = "./chroma_db"
COLLECTION_NAME   = "rag_documents"
EMBEDDING_MODEL   = "models/gemini-embedding-001"
LLM_MODEL         = "llama-3.3-70b-versatile"
RETRIEVAL_K       = 5


def _load_api_keys() -> None:
    """
    Load GEMINI_KEY and GROQ_KEY from whichever secrets source is available:
    - Streamlit Cloud / local Streamlit secrets (st.secrets)
    - Google Colab secrets (userdata)
    - Plain environment variables (fallback, e.g. for local testing)
    Configures the google-generativeai SDK and sets GROQ_API_KEY so the
    Groq client can pick it up.
    """
    gemini_key = None
    groq_key   = None

    # 1. Try Streamlit secrets (used on Streamlit Cloud)
    try:
        import streamlit as st
        gemini_key = st.secrets.get("GEMINI_KEY")
        groq_key   = st.secrets.get("GROQ_KEY")
    except Exception:
        pass

    # 2. Try Google Colab secrets (used when running inside Colab)
    if not gemini_key or not groq_key:
        try:
            from google.colab import userdata
            gemini_key = gemini_key or userdata.get("GEMINI_KEY")
            groq_key   = groq_key or userdata.get("GROQ_KEY")
        except Exception:
            pass

    # 3. Fallback to plain environment variables (local dev / CI)
    gemini_key = gemini_key or os.environ.get("GEMINI_KEY")
    groq_key   = groq_key or os.environ.get("GROQ_KEY")

    if not gemini_key or not groq_key:
        raise RuntimeError(
            "Missing API keys. Set GEMINI_KEY and GROQ_KEY in Streamlit "
            "secrets, Colab secrets, or environment variables."
        )

    os.environ["GOOGLE_API_KEY"] = gemini_key
    os.environ["GROQ_API_KEY"]   = groq_key
    genai.configure(api_key=gemini_key)


class GeminiEmbeddings:
    """
    Thin wrapper around the google-generativeai SDK's embed_content call.

    This bypasses langchain_google_genai's GoogleGenerativeAIEmbeddings,
    which (at the time this was built) called the v1beta endpoint and
    returned a 404 for gemini-embedding-001. Calling genai.embed_content
    directly works correctly.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document",
            )
            vectors.append(result["embedding"])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]


class RAGState(TypedDict):
    """State object passed between LangGraph nodes."""
    question:    str
    chunks:      List[str]
    sources:     List[str]
    answer:      str
    error:       Optional[str]


_pipeline    = None  # cached compiled LangGraph app
_embeddings  = None  # cached GeminiEmbeddings instance
_collection  = None  # cached ChromaDB collection
_groq_client = None  # cached Groq client


def _build_pipeline():
    """
    Build and compile the LangGraph pipeline exactly as in Phase 2 Cell 7:
    retrieve_node -> (generate_node | error_handler_node) -> END
    Returns the compiled app, ready to .invoke({"question": ...}).
    """
    global _embeddings, _collection, _groq_client

    _load_api_keys()

    _embeddings = GeminiEmbeddings(EMBEDDING_MODEL)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    _collection   = chroma_client.get_collection(COLLECTION_NAME)

    _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def retrieve_node(state: RAGState) -> RAGState:
        """Retrieve top-k relevant chunks from ChromaDB."""
        try:
            query_vec = _embeddings.embed_query(state["question"])
            results = _collection.query(
                query_embeddings=[query_vec],
                n_results=RETRIEVAL_K,
                include=["documents", "metadatas"],
            )
            docs  = results["documents"][0]
            metas = results["metadatas"][0]
            return {
                **state,
                "chunks":  docs,
                "sources": [m.get("source", "Unknown") for m in metas],
                "error":   None,
            }
        except Exception as e:
            return {**state, "chunks": [], "sources": [], "error": f"Retrieval error: {e}"}

    def generate_node(state: RAGState) -> RAGState:
        """Generate answer using Groq based on retrieved chunks."""
        if state.get("error"):
            return state

        if not state["chunks"]:
            return {**state, "answer": "No relevant chunks found.", "error": "No chunks retrieved."}

        context = "\n\n---\n\n".join(
            f"[Source: {src}]\n{chunk}"
            for src, chunk in zip(state["sources"], state["chunks"])
        )

        prompt = f"""You are a helpful AI assistant. Answer the question using ONLY the context provided below.
Cite the source for every fact you state. If the context does not contain enough information, say so clearly.

CONTEXT:
{context}

QUESTION: {state['question']}

ANSWER:"""

        try:
            response = _groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            return {**state, "answer": response.choices[0].message.content, "error": None}
        except Exception as e:
            return {**state, "answer": "", "error": f"Generation error: {e}"}

    def error_handler_node(state: RAGState) -> RAGState:
        """Return a graceful, user-friendly message when something fails."""
        return {
            **state,
            "answer": f"⚠️ Pipeline error: {state.get('error', 'Unknown error')}. "
                      "Please check your API keys and document store.",
        }

    def route_after_retrieval(state: RAGState) -> str:
        return "error_handler" if state.get("error") else "generate"

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("error_handler", error_handler_node)
    builder.set_entry_point("retrieve")
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {"generate": "generate", "error_handler": "error_handler"},
    )
    builder.add_edge("generate", END)
    builder.add_edge("error_handler", END)

    return builder.compile()


def get_pipeline():
    """Return a cached compiled LangGraph pipeline, building it on first call."""
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


def ask_question(question: str) -> dict:
    """
    Run a question through the full RAG pipeline.

    Returns:
        {
            "answer":  str        -> the generated answer
            "sources": List[str]  -> the retrieved chunks used as context
                                      (kept as "sources" for compatibility
                                      with the Phase 3 app.py spec template,
                                      which iterates result["sources"] as
                                      chunk text)
        }
    """
    app = get_pipeline()
    result = app.invoke({
        "question": question,
        "chunks":   [],
        "sources":  [],
        "answer":   "",
        "error":    None,
    })
    return {
        "answer":  result.get("answer", ""),
        "sources": result.get("chunks", []),   # chunk text, shown in the expander
        "source_names": list(set(result.get("sources", []))),  # article titles
    }
