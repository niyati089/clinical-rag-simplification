"""
reranking.py — Optional cross-encoder reranking stage.

When USE_RERANKER=True in config, the top-k FAISS results are passed through
a cross-encoder model to produce more accurate relevance scores.

The module degrades gracefully: if the cross-encoder library is not installed
or USE_RERANKER is False, reranking is skipped and the original ranked list
is returned unchanged.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional

from src.config import (
    USE_RERANKER,
    RERANKER_MODEL,
    RERANKER_TOP_K,
)
from src.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


# ── Cross-encoder singleton ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str = RERANKER_MODEL):
    """
    Load and cache the cross-encoder model.
    Import is deferred so that missing sentence-transformers[cross-encoder]
    only fails when reranking is actually requested.
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Cross-encoder reranking requires 'sentence-transformers'. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    logger.info("Loading cross-encoder model: %s", model_name)
    model = CrossEncoder(model_name)
    logger.info("Cross-encoder loaded.")
    return model


# ── Public API ───────────────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: List[RetrievedChunk],
    top_k: int = RERANKER_TOP_K,
    model_name: str = RERANKER_MODEL,
    enabled: bool = USE_RERANKER,
) -> List[RetrievedChunk]:
    """
    Optionally rerank a list of RetrievedChunks using a cross-encoder.

    Parameters
    ----------
    query : str
        The original clinical chunk text used as the reranking query.
    candidates : List[RetrievedChunk]
        Candidates returned by FAISS retrieval.
    top_k : int
        Number of chunks to keep after reranking.
    model_name : str
        Cross-encoder model identifier.
    enabled : bool
        If False, candidates are returned as-is (truncated to top_k).

    Returns
    -------
    List[RetrievedChunk]
        Reranked (or original) list of at most ``top_k`` chunks.
    """
    if not candidates:
        return []

    if not enabled:
        logger.debug("Reranking disabled — returning original FAISS ranking.")
        return candidates[:top_k]

    if not query.strip():
        logger.warning("Reranking query is empty — skipping reranking.")
        return candidates[:top_k]

    try:
        model = _get_cross_encoder(model_name)
        pairs = [(query, c.text) for c in candidates]
        raw_scores = model.predict(pairs)

        # Attach reranker scores and sort descending
        scored = sorted(
            zip(raw_scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        reranked = []
        for score, chunk in scored[:top_k]:
            chunk.score = float(score)   # overwrite FAISS score with CE score
            reranked.append(chunk)

        logger.info(
            "Reranked %d candidates → kept top %d.",
            len(candidates),
            len(reranked),
        )
        return reranked

    except Exception as exc:
        logger.error(
            "Reranking failed (%s). Falling back to original FAISS ranking.", exc
        )
        return candidates[:top_k]
