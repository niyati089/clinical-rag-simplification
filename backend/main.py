"""
backend/main.py — FastAPI application entry point.

Start with:
    uvicorn backend.main:app --reload --port 8000

(Run from the project root: clinical-rag-simplification/)
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import CORS_ORIGINS, LOG_LEVEL, MOCK_MODE


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Lifespan: load heavy resources once at startup ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: warm RAGPipeline (FAISS load + embedding model) and LLMEngine.
    Shutdown: clean up thread-pool executor.
    """
    logger.info("=== Clinical Document Simplification API starting up ===")
    logger.info("MOCK_MODE = %s", MOCK_MODE)

    # Thread-pool for blocking pipeline work (1 worker per concurrent job)
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pipeline")
    app.state.executor = executor

    # In-memory doc store: { doc_id: { filename, text } }
    app.state.doc_store: dict[str, Any] = {}

    if not MOCK_MODE:
        # Pre-load RAGPipeline so the first request isn't slow
        try:
            from backend.pipeline import get_rag_pipeline
            get_rag_pipeline()
        except FileNotFoundError:
            logger.warning(
                "FAISS index not found at startup. "
                "Run `python ingest_knowledge_base.py` before processing documents."
            )
        except EnvironmentError as exc:
            logger.warning("LLM API key issue at startup: %s", exc)
        except Exception as exc:
            logger.warning("RAGPipeline warm-up failed (non-fatal): %s", exc)

        # Pre-load B's LLM engine
        try:
            from simplifier.simplify import _get_llm_engine
            _get_llm_engine()
            logger.info("Simplifier LLM engine warmed up.")
        except Exception as exc:
            logger.warning("Simplifier LLM engine warm-up failed: %s", exc)
    else:
        logger.info("MOCK_MODE: skipping model warm-up.")

    logger.info("=== Server ready ===")

    yield  # --- application runs here ---

    logger.info("Shutting down thread-pool executor…")
    executor.shutdown(wait=False)
    logger.info("=== Server shut down ===")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Clinical Document Simplification API",
    description=(
        "RAG-powered clinical document simplification. "
        "Upload a medical document, extract its text, and get a plain-language "
        "version with glossary, readability scores, and source references."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global error handlers ─────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: never return raw stack traces to the client."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "message": str(exc)},
    )


# ── Routers ───────────────────────────────────────────────────────────────────

from backend.routers.health import router as health_router
from backend.routers.documents import router as documents_router
from backend.routers.simplify import router as simplify_router

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(simplify_router)


# ── Root redirect ─────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse(
        content={
            "message": "Clinical Document Simplification API",
            "docs": "/docs",
            "health": "/api/health",
        }
    )
