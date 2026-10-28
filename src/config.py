"""Configuration settings for Phase 1 vector storage and embeddings."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file if present
load_dotenv(override=True)

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
CORPUS_PATH = Path(os.getenv("CORPUS_FILE_PATH", BASE_DIR / "corpus.jsonl"))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", BASE_DIR / "data" / "chromadb"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "asterline_kb")

# Local embedding model name (Sentence Transformers)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# SQLite database for persistent user memories
SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", BASE_DIR / "data" / "memories.db"))
