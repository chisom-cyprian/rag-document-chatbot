"""
Ingestion pipeline.

Flow: PDF files in DOCS_DIR -> load text per page -> split into overlapping
chunks -> embed each chunk -> store in a persistent ChromaDB collection.

Run this file directly to (re)build the vector store from whatever PDFs
are in the docs/ folder:

    python -m app.core.ingest
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings


def load_documents(docs_dir: str = settings.DOCS_DIR):
    """
    Loads every PDF in docs_dir. PyPDFLoader returns one LangChain
    Document per PAGE, and automatically stamps metadata={'source': filename,
    'page': page_number} on each one. That metadata is what lets us cite
    sources later — we never have to track it manually.
    """
    documents = []
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in '{docs_dir}/'. Add some research papers and re-run.")
        return documents

    for filename in pdf_files:
        filepath = os.path.join(docs_dir, filename)
        loader = PyPDFLoader(filepath)
        pages = loader.load()  # one Document per page
        documents.extend(pages)
        print(f"Loaded {len(pages)} pages from {filename}")

    return documents


def chunk_documents(documents):
    """
    Splits documents into overlapping chunks.

    Why chunk at all: embedding models have a context limit, and retrieval
    quality drops if a chunk is too big (it dilutes the match) or too small
    (it loses context). ~1000 characters is a reasonable default for
    research papers.

    Why overlap: without overlap, a sentence that straddles a chunk boundary
    gets cut in half and loses meaning. 150 chars of overlap keeps continuity
    between chunks.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then sentences, then words — so it prefers cutting at natural boundaries
    rather than mid-word.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks):
    """
    Embeds each chunk and persists it to ChromaDB on disk.

    Chroma.from_documents handles three things at once:
    1. Calls the embedding model on every chunk's text
    2. Stores the resulting vectors alongside the chunk's text + metadata
    3. Persists everything to CHROMA_DIR so it survives a restart
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.CHROMA_DIR,
        collection_name="research_docs",
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at '{settings.CHROMA_DIR}/'")
    return vector_store


def ingest():
    """Full pipeline: load -> chunk -> embed -> store."""
    documents = load_documents()
    if not documents:
        return None
    chunks = chunk_documents(documents)
    return build_vector_store(chunks)


if __name__ == "__main__":
    ingest()
