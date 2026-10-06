"""
embeddings.py — Sentence Transformer embedding module.

Provides two public functions:
    embed_text(text)           → np.ndarray  (1-D vector)
    embed_documents(documents) → np.ndarray  (2-D matrix, one row per doc)

The embedding model is configured in src/config.py (EMBEDDING_MODEL).
The SentenceTransformer instance is cached as a module-level singleton so the
model is only loaded once per process.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


# ── Model singleton ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Load and cache the SentenceTransformer model.
    The model is loaded only once and reused across calls.
    """
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded successfully.")
    return model


# ── Public API ──────────────────────────────────────────────────────────────

def embed_text(text: str, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """
    Generate a normalised embedding vector for a single text string.

    Parameters
    ----------
    text : str
        The text to embed. Must be a non-empty string.
    model_name : str
        Sentence-Transformers model identifier. Defaults to EMBEDDING_MODEL
        from config.

    Returns
    -------
    np.ndarray
        1-D float32 array of shape (embedding_dim,).

    Raises
    ------
    ValueError
        If ``text`` is empty or whitespace-only.
    """
    if not text or not text.strip():
        raise ValueError("embed_text received an empty or whitespace-only string.")

    model = _get_model(model_name)
    vector: np.ndarray = model.encode(
        text.strip(),
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalised → cosine = dot product
    )
    return vector.astype(np.float32)


def embed_documents(
    documents: List[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Generate normalised embedding vectors for a list of texts.

    Parameters
    ----------
    documents : List[str]
        List of text strings to embed.
    model_name : str
        Sentence-Transformers model identifier.
    batch_size : int
        How many texts to encode per forward pass.
    show_progress : bool
        Whether to show a progress bar during encoding.

    Returns
    -------
    np.ndarray
        2-D float32 array of shape (len(documents), embedding_dim).

    Raises
    ------
    ValueError
        If ``documents`` is empty.
    """
    if not documents:
        raise ValueError("embed_documents received an empty document list.")

    # Strip and validate
    clean_docs = [d.strip() for d in documents]
    empty_indices = [i for i, d in enumerate(clean_docs) if not d]
    if empty_indices:
        raise ValueError(
            f"Documents at indices {empty_indices} are empty or whitespace-only."
        )

    model = _get_model(model_name)
    logger.info("Embedding %d documents...", len(clean_docs))

    matrix: np.ndarray = model.encode(
        clean_docs,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    logger.info("Embedding complete. Shape: %s", matrix.shape)
    return matrix.astype(np.float32)


def get_embedding_dimension(model_name: str = EMBEDDING_MODEL) -> int:
    """Return the output dimension of the configured embedding model."""
    model = _get_model(model_name)
    return model.get_sentence_embedding_dimension()
