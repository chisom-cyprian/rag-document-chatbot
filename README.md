# Research Paper RAG Chat

A chatbot that answers questions about research papers and technical documents,
grounded entirely in the documents you upload — with page-level source citations
on every answer.

Built to understand RAG (Retrieval-Augmented Generation) from the ground up,
not just call a framework's one-liner.

## What it does

1. Upload a PDF (research paper, technical spec, etc.)
2. The paper is split into chunks, embedded, and stored in a local ChromaDB vector store
3. Ask a question in plain English
4. The most relevant chunks are retrieved and passed to an LLM, which answers
   **only from that context** — and the response shows exactly which document
   and page the answer came from

## Why RAG, and why this matters

A general-purpose LLM can't answer questions about a paper it's never seen,
and it will confidently hallucinate if you ask anyway. RAG fixes this by
retrieving the actual relevant text first, then asking the model to answer
using only that retrieved context. This is the same pattern behind most
production AI products that answer questions over private or specialized data.

## Architecture

```
PDF upload
   │
   ▼
PyPDFLoader (load_documents)      → one Document per page, with source + page metadata
   │
   ▼
RecursiveCharacterTextSplitter    → overlapping ~1000-char chunks
   │
   ▼
OpenAI Embeddings                 → text-embedding-3-small
   │
   ▼
ChromaDB (persisted to disk)      → vector store
   │
   ▼  (at query time)
similarity_search(question, k=4)  → top 4 most relevant chunks
   │
   ▼
Prompt: "Answer using only this context" → gpt-4o-mini
   │
   ▼
Answer + source citations  → FastAPI response → frontend
```

**Stack:** Python, FastAPI, LangChain, ChromaDB, OpenAI API

## Setup

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd rag-research-chat

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI API key
cp .env.example .env
# edit .env and paste in your real key

# 5. Run the server
uvicorn app.main:app --reload
```

Open **http://localhost:8000** in your browser. Upload a PDF, then ask it questions.

## Project structure

```
app/
├── core/
│   ├── config.py       # settings, API key loading
│   ├── ingest.py        # PDF → chunks → embeddings → ChromaDB
│   └── rag_chain.py     # retrieval + answer generation + citations
├── api/
│   └── routes.py        # /api/upload and /api/chat endpoints
└── main.py                # FastAPI app entrypoint
static/
└── index.html              # frontend (vanilla HTML/CSS/JS)
docs/                        # uploaded PDFs land here
chroma_db/                    # persisted vector store (gitignored)
```

## Known limitations

- Re-ingests **all** documents in `docs/` on every upload rather than
  incrementally adding just the new file — fine for a small personal
  document set, not efficient at scale
- No conversation memory — each question is independent, not a multi-turn
  conversation with follow-up context
- No chunk-level relevance scoring shown to the user (just which doc/page,
  not how confident the match was)
- Single ChromaDB collection shared across all documents — no per-user or
  per-project isolation

## Possible extensions

- Add conversation memory so follow-up questions ("what about section 3?") work
- Show similarity scores next to each source citation
- Support incremental ingestion instead of full re-ingestion on upload
- Swap OpenAI for a local model via Ollama for a fully offline version
- Add a "delete document" endpoint to remove a paper from the index
