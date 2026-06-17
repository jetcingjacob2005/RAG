# 🤖 AI/ML Wikipedia RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
core AI/ML concepts — retrieval-augmented generation, large language models,
transformers, embeddings, neural network architectures, BERT, GPT-4,
knowledge graphs, vector databases, and RLHF — by retrieving relevant
passages from Wikipedia and generating grounded answers with an LLM.

**Live demo:** _[add your deployed Streamlit Cloud URL here once deployed]_

---

## What this bot knows

The bot's knowledge comes from 12 Wikipedia articles, fetched and indexed
ahead of time:

- Retrieval-augmented generation
- Large language model
- Transformer (deep learning architecture)
- Word embedding
- Convolutional neural network
- Recurrent neural network
- Attention (machine learning)
- BERT (language model)
- GPT-4
- Knowledge graph
- Vector database
- Reinforcement learning from human feedback

It answers **only** from this source material. If something isn't covered
in these articles, the bot says so instead of guessing.

## How to use it

1. Open the live demo link above (or run it locally — see below)
2. Type a question about any of the AI/ML topics above into the text box
3. Read the generated answer (with source articles noted)
4. Click "See the document chunks I used" to view the exact passages the
   answer was based on

That's it — no login, no setup, just ask a question.

---

## How it works (for the curious)

```
12 Wikipedia articles
     │
     ▼
Chunking (500 chars, 50 overlap)
     │
     ▼
Gemini embeddings (gemini-embedding-001) ──────► ChromaDB (vector store)
                                                       │
User question                                         │
     │                                                 │
     ▼                                                 │
Gemini embeddings ─────────────────────────────────────┘
     │
     ▼
Retrieve top-5 relevant chunks  (LangGraph: retrieve_node)
     │
     ▼
Groq (Llama 3.3 70B) generates answer  (LangGraph: generate_node)
     │
     ▼
Answer + source chunks shown to user
```

The pipeline is orchestrated with **LangGraph** (with a dedicated
`error_handler` node so retrieval or generation failures degrade
gracefully instead of crashing), and embeddings are generated through the
raw `google-generativeai` SDK rather than LangChain's wrapper, which hit a
compatibility bug with the `gemini-embedding-001` model.

The system was evaluated on 20 hand-written test questions using a
RAGAS-style scoring approach — faithfulness, answer relevancy, and context
precision — with Groq itself acting as the judge LLM (the official `ragas`
library had a dependency conflict in this environment; see the notebook for
details).

**Evaluation results:**

| Metric | Score |
|---|---|
| Faithfulness | 0.975 |
| Answer Relevancy | 0.980 |
| Context Precision | 0.818 |
| **Overall Average** | **0.924** |

The weakest area was context precision on questions not well covered by the
source articles (e.g. "How is context precision measured in RAG
evaluation?" — a question about evaluation methodology itself, which none
of the 12 source articles address).

---

## Repository structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — the chatbot's front end |
| `rag_pipeline.py` | Core RAG logic (retrieval + generation) imported by `app.py` |
| `ingest.py` | One-time script to fetch the Wikipedia articles and build the ChromaDB vector store |
| `Internship_Phase_2_Cleaned.ipynb` | Cleaned, documented Colab notebook showing the full build process and evaluation |
| `requirements.txt` | Python dependencies |
| `.streamlit/secrets.toml.example` | Template showing what secrets to set (no real keys) |
| `docs/` | Fetched Wikipedia `.txt` files (not committed — see note below) |
| `chroma_db/` | Persisted vector store (not committed — rebuild with `ingest.py`) |

---

## Running it yourself

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API keys

You need two free API keys:
- **Gemini** — from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Groq** — from [console.groq.com](https://console.groq.com/keys)

**For local development**, copy the secrets template and fill in your keys:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml with your real keys
```

**Never commit real API keys.** `.streamlit/secrets.toml` is already in
`.gitignore`.

### 4. Build the vector database (first time only)

```bash
export GEMINI_KEY=your-gemini-key-here
python ingest.py
```
This fetches the 12 Wikipedia articles, chunks them, embeds every chunk,
and saves everything to `chroma_db/`. It only needs to be run once —
`chroma_db/` is reused on every subsequent app run.

### 5. Run the app
```bash
streamlit run app.py
```
Open the URL Streamlit prints in your terminal (usually `http://localhost:8501`).

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub (make sure `chroma_db/`, `docs/`, and any real
   secrets are **not** committed — check `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**, select this repo, branch, and `app.py` as the main file
4. Before deploying, go to **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_KEY = "your-real-gemini-key"
   GROQ_KEY   = "your-real-groq-key"
   ```
5. Click **Deploy** — you'll get a public URL like
   `https://<your-app-name>.streamlit.app`

> **Note:** `chroma_db/` must exist in the deployed environment for the
> app to answer questions. Either commit a small pre-built `chroma_db/`
> folder to the repo (it's only ~300 chunks, small enough to commit), or
> add a startup step that runs `ingest.py` once when the app first launches.

---

## Evaluation methodology

See `Internship_Phase_2_Cleaned.ipynb`, Section 8–9, for the full scorecard
and per-question breakdown. Each of the 20 test questions was run through
the live pipeline, then scored by Groq (acting as an evaluator) on a 0–1
scale for faithfulness, answer relevancy, and context precision.

---

## Built with

google-generativeai (Gemini embeddings) · ChromaDB · Groq (Llama 3.3 70B) ·
LangGraph · LangChain (text splitting + document loading) ·
wikipedia-api · Streamlit
