"""
ingestion.py — Knowledge base ingestion pipeline.

Reads trusted medical reference documents from the knowledge_base/ folder,
cleans and chunks them, generates embeddings, and stores them in the
FAISS vector index with full metadata.

Supported formats:
  - .txt  (plain text)
  - .md   (Markdown, stripped to plain text)

Usage:
    from src.ingestion import ingest_knowledge_base
    ingest_knowledge_base()          # builds/updates the index

Or run directly:
    python ingest_knowledge_base.py
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

from src.config import (
    KNOWLEDGE_BASE_DIR,
    GUIDELINES_DIR,
    TEXTBOOKS_DIR,
    TRUSTED_SOURCES_DIR,
    EMBEDDING_DIMENSION,
)
from src.chunking import chunk_clinical_text, ClinicalChunk
from src.embeddings import embed_documents, get_embedding_dimension
from src.retrieval import FAISSStore

logger = logging.getLogger(__name__)

# ── Source category map ──────────────────────────────────────────────────────

_CATEGORY_PATHS = {
    "Clinical Guideline": GUIDELINES_DIR,
    "Medical Textbook": TEXTBOOKS_DIR,
    "Trusted Reference": TRUSTED_SOURCES_DIR,
}

_SUPPORTED_EXTENSIONS = {".txt", ".md"}


# ── Text cleaning for knowledge-base documents ───────────────────────────────

def _clean_kb_text(text: str) -> str:
    """Strip Markdown syntax and normalise whitespace."""
    # Remove markdown headings markers but keep the text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"[*_]{1,3}(.+?)[*_]{1,3}", r"\1", text)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Normalise whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Document discovery ───────────────────────────────────────────────────────

def _iter_documents(
    base_dir: Path = KNOWLEDGE_BASE_DIR,
) -> Iterator[Tuple[Path, str]]:
    """
    Yield (file_path, category_name) for every supported document
    found under base_dir.
    """
    if not base_dir.exists():
        logger.warning("Knowledge base directory not found: %s", base_dir)
        return

    for category_name, category_dir in _CATEGORY_PATHS.items():
        if not category_dir.exists():
            logger.debug("Category directory missing, skipping: %s", category_dir)
            continue
        for file_path in sorted(category_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in _SUPPORTED_EXTENSIONS:
                yield file_path, category_name


# ── Single-document processing ───────────────────────────────────────────────

def _process_document(
    file_path: Path,
    category: str,
) -> List[dict]:
    """
    Read, clean, chunk a document and return a list of metadata dicts.

    Each dict contains:
        chunk_id, text, source, document_name, page_number, section_title
    """
    logger.info("Processing: %s [%s]", file_path.name, category)
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    clean_text = _clean_kb_text(raw_text)

    if not clean_text.strip():
        logger.warning("Empty document after cleaning: %s — skipped.", file_path)
        return []

    chunks: List[ClinicalChunk] = chunk_clinical_text(clean_text)

    metadata_list = []
    for chunk in chunks:
        # Deterministic chunk_id based on source + content
        content_hash = hashlib.sha256(
            f"{file_path.name}::{chunk.text}".encode()
        ).hexdigest()[:12]
        chunk_id = f"kb_{content_hash}"

        metadata_list.append(
            {
                "chunk_id": chunk_id,
                "text": chunk.text,
                "source": category,
                "document_name": file_path.stem.replace("_", " ").title(),
                "page_number": None,          # TXT/MD files don't have pages
                "section_title": chunk.section_title or None,
            }
        )

    logger.debug("  → %d chunks created.", len(metadata_list))
    return metadata_list


# ── Main ingestion function ──────────────────────────────────────────────────

def ingest_knowledge_base(
    base_dir: Path = KNOWLEDGE_BASE_DIR,
    force_rebuild: bool = False,
) -> FAISSStore:
    """
    Ingest all trusted medical documents into the FAISS vector store.

    Parameters
    ----------
    base_dir : Path
        Root of the knowledge base folder structure.
    force_rebuild : bool
        If True, rebuild the index from scratch even if it already exists.

    Returns
    -------
    FAISSStore
        The populated (and saved) FAISS store instance.
    """
    store = FAISSStore()

    # Determine actual embedding dimension from model
    actual_dim = get_embedding_dimension()
    store.embedding_dim = actual_dim

    store.create_index()
    logger.info("Starting knowledge base ingestion from: %s", base_dir)

    # ── Collect all metadata records ───────────────────────────────────────
    all_metadata: List[dict] = []
    for file_path, category in _iter_documents(base_dir):
        doc_meta = _process_document(file_path, category)
        all_metadata.extend(doc_meta)

    if not all_metadata:
        logger.warning(
            "No documents found in knowledge base at %s. "
            "Add .txt or .md files to data/knowledge_base/ subdirectories.",
            base_dir,
        )
        store.save()
        return store

    # ── Generate embeddings in one batch ──────────────────────────────────
    texts = [m["text"] for m in all_metadata]
    logger.info("Generating embeddings for %d chunks...", len(texts))
    embeddings: np.ndarray = embed_documents(texts, show_progress=True)

    # ── Add to FAISS store ─────────────────────────────────────────────────
    store.add_chunks(embeddings, all_metadata)
    store.save()

    logger.info(
        "Ingestion complete. %d chunks indexed and saved to vector store.",
        store.total_vectors(),
    )
    return store
