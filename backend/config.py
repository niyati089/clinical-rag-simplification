"""
backend/config.py — Backend-specific configuration.

All backend settings are read from environment variables (via .env).
A's and B's configs are loaded independently by their own modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

# ── Backend feature flags ────────────────────────────────────────────────────

MOCK_MODE: bool = os.getenv("MOCK_MODE", "false").lower() in ("1", "true", "yes")
ENABLE_AUTH: bool = os.getenv("ENABLE_AUTH", "false").lower() in ("1", "true", "yes")
ENABLE_CLAIM_VERIFICATION: bool = os.getenv(
    "ENABLE_CLAIM_VERIFICATION", "false"
).lower() in ("1", "true", "yes")

# ── File upload limits ───────────────────────────────────────────────────────

MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "image/png",
    "image/jpeg",
}

# ── CORS ─────────────────────────────────────────────────────────────────────

_cors_raw: str = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
)
CORS_ORIGINS: List[str] = [o.strip() for o in _cors_raw.split(",") if o.strip()]

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR: Path = Path(__file__).resolve().parent.parent
MOCK_DATA_DIR: Path = BASE_DIR / "data" / "mock_A_outputs"

# ── Job store ────────────────────────────────────────────────────────────────

# Maximum age in seconds before a completed/failed job is pruned from memory
JOB_TTL_SECONDS: int = int(os.getenv("JOB_TTL_SECONDS", "3600"))  # 1 hour

# ── Auth (only used when ENABLE_AUTH=true) ───────────────────────────────────

JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM: str = "HS256"
JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
AUTH_DB_PATH: Path = BASE_DIR / "backend" / "auth.db"
