"""
backend/verification/claims.py — LLM-as-judge claim verification.

For each simplified section, extracts declarative claims and asks the LLM
(using Person A's existing _call_llm adapter) whether each claim is:
  - Supported    — corroborated by the retrieved references
  - Unsupported  — cannot be verified from references
  - Inconsistent — contradicts the references

Activation: only runs when ENABLE_CLAIM_VERIFICATION=true in .env.
Design contract: NEVER raises — returns [] flags on any failure.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Verdict options ───────────────────────────────────────────────────────────
VERDICTS = {"Supported", "Unsupported", "Inconsistent"}


# ── Claim extraction (heuristic) ──────────────────────────────────────────────

def _extract_claims(text: str, max_claims: int = 3) -> List[str]:
    """
    Extract up to max_claims declarative sentences from simplified text.
    Declarative = ends with period, contains a verb, ≥ 8 words.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = []
    for sent in sentences:
        words = sent.split()
        if len(words) >= 8 and sent.endswith("."):
            claims.append(sent)
        if len(claims) >= max_claims:
            break
    return claims


# ── Prompt ────────────────────────────────────────────────────────────────────

_JUDGE_PROMPT_TEMPLATE = """\
You are a medical fact-checker. Given the retrieved reference excerpts below, \
evaluate whether the CLAIM is Supported, Unsupported, or Inconsistent with those references.

REFERENCES:
{references}

CLAIM:
{claim}

Respond with exactly one word: Supported, Unsupported, or Inconsistent.
Do not add any other text.
"""


def _build_judge_prompt(claim: str, references: List[Dict[str, Any]]) -> str:
    ref_texts = []
    for i, ref in enumerate(references[:5], 1):  # limit to top-5 refs
        snippet = ref.get("text", "")[:300]
        source = ref.get("source", "Unknown")
        ref_texts.append(f"[{i}] ({source}): {snippet}")
    refs_str = "\n".join(ref_texts) if ref_texts else "(no references provided)"
    return _JUDGE_PROMPT_TEMPLATE.format(references=refs_str, claim=claim)


# ── LLM call (reuses A's adapter) ────────────────────────────────────────────

def _call_judge(claim: str, references: List[Dict[str, Any]]) -> str:
    """Call the LLM and return a verdict string."""
    from src.llm import _call_llm  # Person A's adapter — already in scope
    prompt = _build_judge_prompt(claim, references)
    raw = _call_llm(prompt).strip()
    # Normalise
    for verdict in VERDICTS:
        if verdict.lower() in raw.lower():
            return verdict
    return "Unsupported"  # conservative default


# ── Public API ────────────────────────────────────────────────────────────────

def verify_claims(
    enhanced_result: Dict[str, Any],
    references: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """
    LLM-as-judge claim verification.

    Parameters
    ----------
    enhanced_result : dict
        The full result dict (after A → B chain).
    references : list, optional
        Aggregated references; if None, gathered from chunk results.

    Returns
    -------
    list of flag dicts with keys: type, claim, verdict, severity
    """
    from backend.config import ENABLE_CLAIM_VERIFICATION
    if not ENABLE_CLAIM_VERIFICATION:
        return []

    flags: List[Dict[str, Any]] = []

    try:
        # Collect references if not provided
        if references is None:
            references = []
            for chunk in enhanced_result.get("results", []):
                references.extend(chunk.get("retrieved_references", []))

        # Process each chunk's simplified explanation
        for chunk in enhanced_result.get("results", []):
            simplified_exp = chunk.get("simplified_output", {}).get(
                "simple_explanation", ""
            )
            if not simplified_exp:
                continue

            claims = _extract_claims(simplified_exp)
            chunk_refs = chunk.get("retrieved_references", references)

            for claim in claims:
                try:
                    verdict = _call_judge(claim, chunk_refs)
                except Exception as exc:
                    logger.warning(
                        "LLM judge call failed for claim '%s…': %s",
                        claim[:60],
                        exc,
                    )
                    continue

                if verdict in ("Unsupported", "Inconsistent"):
                    flags.append(
                        {
                            "type": "claim",
                            "claim": claim,
                            "verdict": verdict,
                            "chunk_id": chunk.get("chunk_id"),
                            "severity": (
                                "error" if verdict == "Inconsistent" else "warning"
                            ),
                            "message": (
                                f"Claim is '{verdict}' by retrieved references: \"{claim}\""
                            ),
                        }
                    )

    except Exception as exc:
        logger.warning("Claim verification failed (non-fatal): %s", exc)
        flags.append(
            {
                "type": "verification_warning",
                "message": f"Claim verification could not complete: {exc}",
                "severity": "info",
            }
        )

    return flags
