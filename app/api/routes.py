"""
API routes. Two real endpoints:

POST /api/upload  - accepts a PDF, saves it to docs/, re-runs ingestion
POST /api/chat     - accepts a question, returns answer + sources

Kept thin on purpose: routes only handle HTTP concerns (request/response
shape, status codes). All actual logic lives in app/core/.
"""

import os
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.ingest import ingest
from app.core.rag_chain import answer_question

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Saves an uploaded PDF into docs/, then re-runs the full ingestion
    pipeline. Simple approach: re-ingest everything in docs/ on every
    upload. Fine for a portfolio project; for production you'd ingest
    only the new file and append to the existing collection instead.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(settings.DOCS_DIR, exist_ok=True)
    save_path = os.path.join(settings.DOCS_DIR, file.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    vector_store = ingest()
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Ingestion failed.")

    return {"message": f"'{file.filename}' uploaded and ingested successfully."}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answers a question using the currently ingested documents."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = answer_question(request.question)
    return ChatResponse(answer=result["answer"], sources=result["sources"])
