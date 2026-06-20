"""
FastAPI app entrypoint.

Run with:
    uvicorn app.main:app --reload

Then open http://localhost:8000 for the frontend.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router

app = FastAPI(title="Research Paper RAG Chat")

app.include_router(router, prefix="/api")

# Serve the frontend's static assets (script.js, styles if you split them out)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    """Serves the single-page frontend at the root URL."""
    return FileResponse("static/index.html")
