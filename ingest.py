"""
ingest.py
=========
One-time ingestion script for the Wikipedia AI/ML RAG chatbot.

Run this ONCE to build the ChromaDB vector store:

    python ingest.py

It mirrors Phase 2, Cells 3-5 exactly:
  1. Fetch 12 Wikipedia AI/ML articles and save as .txt files
  2. Load and split them into 500-char chunks (50-char overlap)
  3. Embed each chunk with Gemini ("models/gemini-embedding-001") via the
     raw google-generativeai SDK, batched (20 chunks/batch) with retry on
     429 quota errors
  4. Store everything in a ChromaDB collection named "rag_documents",
     persisted to ./chroma_db

Once chroma_db/ exists, you do NOT need to run this again — the
Streamlit app (app.py) loads the persisted collection directly via
rag_pipeline.py.
"""

import os
import time

import chromadb
import google.generativeai as genai
import wikipediaapi
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration (matches Phase 2 exactly) ─────────────────────────────────
DOC_DIR           = "./docs"
CHROMA_DIR        = "./chroma_db"
COLLECTION_NAME   = "rag_documents"
EMBEDDING_MODEL   = "models/gemini-embedding-001"
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50
BATCH_SIZE        = 20

TOPICS = [
    "Retrieval-augmented generation",
    "Large language model",
    "Transformer (deep learning architecture)",
    "Word embedding",
    "Convolutional neural network",
    "Recurrent neural network",
    "Attention (machine learning)",
    "BERT (language model)",
    "GPT-4",
    "Knowledge graph",
    "Vector database",
    "Reinforcement learning from human feedback",
]


def load_api_key() -> None:
    """Load GEMINI_KEY from environment and configure the genai SDK."""
    gemini_key = os.environ.get("GEMINI_KEY")
    if not gemini_key:
        raise RuntimeError(
            "GEMINI_KEY environment variable not set.\n"
            "Run: export GEMINI_KEY=your_key_here   (Linux/Mac)\n"
            "Or:  set GEMINI_KEY=your_key_here       (Windows)"
        )
    os.environ["GOOGLE_API_KEY"] = gemini_key
    genai.configure(api_key=gemini_key)


def fetch_wikipedia_articles() -> list:
    """Fetch the 12 source Wikipedia articles and save each as a .txt file."""
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="RAGInternshipBot/1.0 (internship project)",
    )

    os.makedirs(DOC_DIR, exist_ok=True)

    fetched = []
    for topic in TOPICS:
        try:
            page = wiki.page(topic)
            if page.exists():
                safe_name = topic.replace("/", "_").replace(" ", "_")
                filepath  = os.path.join(DOC_DIR, f"{safe_name}.txt")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(page.text[:8000])  # cap at 8000 chars, same as Phase 2
                fetched.append(filepath)
                print(f"✅ Saved: {topic}")
            else:
                print(f"⚠️  Not found: {topic}")
        except Exception as e:
            print(f"❌ Error processing '{topic}': {e}")

    print(f"\n📚 Total documents fetched: {len(fetched)}")
    return fetched


def load_and_chunk_documents() -> list:
    """Load all .txt files from DOC_DIR and split them into chunks."""
    all_docs = []
    for filename in os.listdir(DOC_DIR):
        if filename.endswith(".txt"):
            try:
                filepath = os.path.join(DOC_DIR, filename)
                loader   = TextLoader(filepath, encoding="utf-8")
                docs     = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename.replace(".txt", "").replace("_", " ")
                all_docs.extend(docs)
                print(f"📄 Loaded: {filename}  ({len(docs[0].page_content)} chars)")
            except Exception as e:
                print(f"❌ Error loading '{filename}': {e}")

    print(f"\n✅ Total documents loaded: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks


class GeminiEmbeddings:
    """Same embedding wrapper used in rag_pipeline.py — kept in sync."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name

    def embed_documents(self, texts: list) -> list:
        vectors = []
        for text in texts:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document",
            )
            vectors.append(result["embedding"])
        return vectors

    def embed_query(self, text: str) -> list:
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]


def embed_and_store(chunks) -> None:
    """Embed chunks in batches and persist to a native ChromaDB collection."""
    embeddings = GeminiEmbeddings(EMBEDDING_MODEL)

    print("🔍 Testing embedding model …")
    test_vec = embeddings.embed_query("hello world")
    print(f"✅ Embedding works! Vector dimension: {len(test_vec)}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  🗑️  Cleared existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total  = len(chunks)
    doc_id = 0

    print(f"\n🚀 Starting ingestion — {total} chunks into ChromaDB …\n")

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(
            f"  Batch {i // BATCH_SIZE + 1}/{(total + BATCH_SIZE - 1) // BATCH_SIZE} "
            f"(chunks {i + 1}–{min(i + BATCH_SIZE, total)} of {total})",
            end=" ... ",
        )

        for attempt in range(5):
            try:
                texts     = [c.page_content for c in batch]
                metadatas = [c.metadata for c in batch]
                ids       = [f"doc_{doc_id + j}" for j in range(len(batch))]
                vectors   = embeddings.embed_documents(texts)

                collection.add(
                    documents=texts,
                    embeddings=vectors,
                    metadatas=metadatas,
                    ids=ids,
                )
                doc_id += len(batch)
                print("✅")
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    wait = 60 * (attempt + 1)
                    print(f"\n  ⚠️  Quota hit — waiting {wait}s (retry {attempt + 1}/5)", end=" ")
                    time.sleep(wait)
                else:
                    print(f"\n  ❌ Unexpected error: {e}")
                    raise

        time.sleep(1)  # polite pause between batches

    print(f"\n✅ ChromaDB built at   : {CHROMA_DIR}")
    print(f"✅ Total chunks stored : {collection.count()}")


def main():
    load_api_key()
    fetch_wikipedia_articles()
    chunks = load_and_chunk_documents()
    if not chunks:
        print("No chunks were created. Check that Wikipedia articles were fetched correctly.")
        return
    embed_and_store(chunks)


if __name__ == "__main__":
    main()
