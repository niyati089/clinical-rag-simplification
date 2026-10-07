"""
config.py — Central configuration for the RAG pipeline.
All tunable parameters live here. Import from this module everywhere else;
never scatter magic values throughout the codebase.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env so callers don't have to ─────────────────────────────────────
load_dotenv()

# ── Project paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
GUIDELINES_DIR = KNOWLEDGE_BASE_DIR / "guidelines"
TEXTBOOKS_DIR = KNOWLEDGE_BASE_DIR / "textbooks"
TRUSTED_SOURCES_DIR = KNOWLEDGE_BASE_DIR / "trusted_sources"
EVALUATION_SAMPLES_DIR = DATA_DIR / "evaluation_samples"

VECTOR_STORE_DIR = BASE_DIR / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_STORE_PATH = VECTOR_STORE_DIR / "metadata.json"

EVALUATION_RESULTS_DIR = BASE_DIR / "evaluation" / "results"

# ── Embedding ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION: int = 384  # matches all-MiniLM-L6-v2; update if model changes

# ── Chunking ────────────────────────────────────────────────────────────────
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "300"))
CHUNK_OVERLAP_SENTENCES: int = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "1"))

# ── Retrieval ───────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))

# ── Reranking ───────────────────────────────────────────────────────────────
USE_RERANKER: bool = os.getenv("USE_RERANKER", "false").lower() == "true"
RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "3"))

# ── LLM ────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")  # groq | openai | google | anthropic
# Groq models: qwen/qwen3.8-27b | openai/gpt-oss-120b | openai/gpt-oss-20b
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1500"))

# ── Reading levels ──────────────────────────────────────────────────────────
VALID_READING_LEVELS = ("Basic", "Intermediate", "Advanced")
DEFAULT_READING_LEVEL: str = os.getenv("DEFAULT_READING_LEVEL", "Basic")

# ── Safety ─────────────────────────────────────────────────────────────────
DISCLAIMER: str = (
    "This output is an AI-assisted simplification for educational purposes only. "
    "It is NOT a substitute for professional medical advice. Always consult a "
    "qualified healthcare provider for medical decisions."
)
