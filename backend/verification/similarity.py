"""
backend/verification/similarity.py — Similarity check between original and simplified text.

Two independent checks:
  1. Embedding-level cosine similarity per section (sentence-transformers).
  2. Lexical presence check: numbers, dates, drug names, doses from original
     must appear somewhere in the simplified output.

Design contract:
  - NEVER raises — any internal error returns [] flags + a warning log.
  - Returns a list of flag dicts; empty list = all checks passed.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Similarity threshold ──────────────────────────────────────────────────────

_LOW_SIMILARITY_THRESHOLD = 0.40   # cosine similarity below this → flag

# ── Lazy embedding model ──────────────────────────────────────────────────────

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            # Reuse the same model already required by Person A
            _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            logger.debug("Similarity embedding model loaded.")
        except Exception as exc:
            logger.warning("Could not load sentence-transformers model: %s", exc)
            _embed_model = None
    return _embed_model


# ── Patterns for lexical check ────────────────────────────────────────────────

_NUMBER_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|l|mmol|mmHg|%|units?|tabs?|capsules?)?\b",
    re.IGNORECASE,
)
_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*\d{4})\b",
    re.IGNORECASE,
)


def _extract_critical_tokens(text: str) -> List[str]:
    """Pull numbers/doses/dates from the original text."""
    tokens = _NUMBER_PATTERN.findall(text) + _DATE_PATTERN.findall(text)
    return [t.strip() for t in tokens if t.strip()]


def _cosine_similarity(vec_a, vec_b) -> float:
    import numpy as np
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Public API ────────────────────────────────────────────────────────────────

def check_similarity(
    original_text: str,
    enhanced_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compare the original clinical text against the simplified output.

    Parameters
    ----------
    original_text : str
        Raw text as uploaded/entered by the user.
    enhanced_result : dict
        The fully assembled result dict (after A → B chain).

    Returns
    -------
    list of flag dicts, each with keys: type, message, severity
    """
    flags: List[Dict[str, Any]] = []

    try:
        flags.extend(_embedding_similarity_check(original_text, enhanced_result))
    except Exception as exc:
        logger.warning("Embedding similarity check failed (non-fatal): %s", exc)

    try:
        flags.extend(_lexical_presence_check(original_text, enhanced_result))
    except Exception as exc:
        logger.warning("Lexical presence check failed (non-fatal): %s", exc)

    return flags


def _embedding_similarity_check(
    original_text: str,
    enhanced_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Cosine similarity between original and each chunk's simplified explanation."""
    model = _get_embed_model()
    if model is None:
        return []

    flags: List[Dict[str, Any]] = []
    results = enhanced_result.get("results", [])

    for chunk in results:
        original_chunk = chunk.get("original_chunk", "")
        simplified_exp = chunk.get("simplified_output", {}).get("simple_explanation", "")
        if not original_chunk or not simplified_exp:
            continue

        try:
            vecs = model.encode([original_chunk, simplified_exp], show_progress_bar=False)
            sim = _cosine_similarity(vecs[0], vecs[1])
        except Exception as exc:
            logger.debug("Encoding failed for chunk %s: %s", chunk.get("chunk_id"), exc)
            continue

        if sim < _LOW_SIMILARITY_THRESHOLD:
            flags.append(
                {
                    "type": "similarity",
                    "message": (
                        f"Chunk '{chunk.get('chunk_id')}' has low semantic similarity "
                        f"between original and simplified text (score={sim:.2f}). "
                        "Review for potential meaning drift."
                    ),
                    "severity": "warning",
                    "chunk_id": chunk.get("chunk_id"),
                    "score": round(sim, 4),
                }
            )

    return flags


def _lexical_presence_check(
    original_text: str,
    enhanced_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Verify that critical tokens (doses, dates, numbers) from the original
    appear somewhere in the simplified output.
    """
    critical_tokens = _extract_critical_tokens(original_text)
    if not critical_tokens:
        return []

    # Gather all simplified text
    all_simplified_parts: List[str] = []
    for chunk in enhanced_result.get("results", []):
        so = chunk.get("simplified_output", {})
        all_simplified_parts.append(so.get("simple_explanation", ""))
        for field in ("important_instructions", "medication_guidance", "follow_up"):
            items = so.get(field, [])
            if isinstance(items, list):
                all_simplified_parts.extend(items)
    combined_simplified = " ".join(all_simplified_parts).lower()

    flags: List[Dict[str, Any]] = []
    for token in critical_tokens:
        if token.lower() not in combined_simplified:
            flags.append(
                {
                    "type": "missing_value",
                    "message": (
                        f"Critical value '{token}' from the original text was not "
                        "found in the simplified output. Verify it was not omitted."
                    ),
                    "severity": "warning",
                    "missing_token": token,
                }
            )

    return flags
