"""
The RAG chain itself.

Flow: user question -> embed the question -> similarity search against
ChromaDB to get the K most relevant chunks -> stuff those chunks into a
prompt -> ask the chat model to answer USING ONLY that context -> return
the answer plus which chunks (source + page) it came from.

This is intentionally written as plain functions rather than LangChain's
LCEL chain syntax, so every step is visible and debuggable rather than
hidden behind a pipe operator.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings

SYSTEM_PROMPT = """You are a research assistant answering questions strictly \
based on the provided context from the user's documents.

Rules:
- Only use information found in the context below.
- If the context doesn't contain the answer, say so explicitly. Do not guess \
or use outside knowledge.
- Be precise and concise. Prefer direct answers over hedging.
- When relevant, mention which finding/section the information came from.

Context:
{context}
"""


def get_vector_store():
    """
    Re-opens the persisted Chroma collection. This does NOT re-embed
    anything — it just loads the existing vectors from disk so the API
    server can query them without re-running ingestion every time.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )
    return Chroma(
        persist_directory=settings.CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="research_docs",
    )


def retrieve_chunks(question: str, k: int = settings.RETRIEVAL_K):
    """
    Embeds the question and runs a similarity search.
    Returns LangChain Documents — each has .page_content (the text) and
    .metadata (source filename + page number, set automatically back in
    ingest.py by PyPDFLoader).
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search(question, k=k)
    return results


def format_context(chunks):
    """
    Joins retrieved chunks into a single context block, labeling each one
    with its source so the model CAN reference where information came from
    if asked, and so we can cross-check the model's claims against sources.
    """
    blocks = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", None)
        page_display = page + 1 if isinstance(page, int) else "?"
        blocks.append(f"[Chunk {i+1} | {source}, page {page_display}]\n{chunk.page_content}")
    return "\n\n".join(blocks)


def format_sources(chunks):
    """
    Builds a clean, deduplicated list of {source, page} for the API response,
    so the frontend can show 'Sources: paper.pdf p.4, paper.pdf p.7' under
    the answer.
    """
    seen = set()
    sources = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", None)
        page_display = page + 1 if isinstance(page, int) else "?"
        key = (source, page_display)
        if key not in seen:
            seen.add(key)
            sources.append({"source": source, "page": page_display})
    return sources


def answer_question(question: str) -> dict:
    """
    The full RAG call: retrieve -> build prompt -> generate -> return
    answer + sources together so the API layer has everything it needs.
    """
    chunks = retrieve_chunks(question)

    if not chunks:
        return {
            "answer": "No documents have been ingested yet, so I have no context to answer from.",
            "sources": [],
        }

    context = format_context(chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGoogleGenerativeAI(
        model=settings.CHAT_MODEL,
        temperature=0,  # deterministic, factual answers — not creative ones
        google_api_key=settings.GOOGLE_API_KEY,
    )

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "sources": format_sources(chunks),
    }
