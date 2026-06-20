"""
Central config. Loads environment variables once so every other
module just imports `settings` instead of calling os.getenv() everywhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file in project root


class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    DOCS_DIR: str = os.getenv("DOCS_DIR", "docs")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "chroma_db")

    # Gemini's free-tier embedding + chat models
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    CHAT_MODEL: str = "gemini-2.5-flash"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    RETRIEVAL_K: int = 4


settings = Settings()

if not settings.GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Create a .env file (see .env.example) "
        "and add your key from https://aistudio.google.com/apikey"
    )