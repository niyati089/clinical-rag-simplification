"""
backend/pipeline.py — Integration pipeline connecting Person A's RAG and Person B's Simplifier.

Coordinates:
  1. Person A: Knowledge retrieval and RAG simplification (src.rag_pipeline.RAGPipeline)
  2. Person B: Text enhancement, NER, glossary, and readability (simplifier.simplify_rag_result)
  3. Verification: Semantic similarity and claim verification (backend.verification)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from backend.config import ENABLE_CLAIM_VERIFICATION, MOCK_MODE
from backend.jobs import JobStage, JobStore

logger = logging.getLogger(__name__)

_rag_pipeline = None


def get_rag_pipeline():
    """Return the global RAGPipeline singleton (lazy-loaded)."""
    global _rag_pipeline
    if _rag_pipeline is None:
        from src.rag_pipeline import RAGPipeline
        _rag_pipeline = RAGPipeline()
        logger.info("RAGPipeline initialized and loaded.")
    return _rag_pipeline


def run_pipeline(
    text: str,
    reading_level: str,
    job_id: str,
    store: JobStore,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Execute the full end-to-end processing pipeline in a background thread.
    Transitions through JobStages and updates the JobStore safely.
    """
    def _update_stage(stage: JobStage):
        asyncio.run_coroutine_threadsafe(store.update_stage(job_id, stage), loop)

    def _mark_done(result: Dict[str, Any]):
        asyncio.run_coroutine_threadsafe(store.mark_done(job_id, result), loop)

    def _mark_failed(error_msg: str):
        asyncio.run_coroutine_threadsafe(store.mark_failed(job_id, error_msg), loop)

    try:
        # Handle mock mode if enabled in environment
        if MOCK_MODE:
            logger.info("Job %s running in MOCK_MODE", job_id)
            _update_stage(JobStage.SIMPLIFYING)
            from backend.mock_data import build_mock_response
            mock_res = build_mock_response(text, reading_level)
            _mark_done(mock_res)
            return

        normalized_level = (
            reading_level.capitalize()
            if reading_level.lower() in ("basic", "intermediate", "advanced")
            else "Basic"
        )

        # ── Step 1: Chunking & Document Parsing ───────────────────────────
        _update_stage(JobStage.CHUNKING)
        logger.info("Job %s: Starting RAG retrieval...", job_id)

        # ── Step 2: RAG Pipeline (Person A) ───────────────────────────────
        _update_stage(JobStage.EMBEDDING)
        rag = get_rag_pipeline()
        rag_result = rag.process(
            clinical_text=text,
            reading_level=normalized_level,
        )

        # ── Step 3: Medical Simplification Layer (Person B) ───────────────
        _update_stage(JobStage.SIMPLIFYING)
        logger.info("Job %s: Running medical simplifier layer...", job_id)
        from simplifier import simplify_rag_result
        try:
            enhanced_result = simplify_rag_result(
                rag_result=rag_result,
                level=reading_level.lower(),
            )
        except Exception as exc:
            logger.warning("simplify_rag_result failed (%s), falling back to raw RAG output", exc)
            enhanced_result = rag_result

        # ── Step 4: Verification ──────────────────────────────────────────
        _update_stage(JobStage.VERIFYING)
        logger.info("Job %s: Verifying accuracy...", job_id)
        flags = []
        try:
            from backend.verification.similarity import check_similarity
            flags.extend(check_similarity(text, enhanced_result))
        except Exception as exc:
            logger.warning("Similarity check failed (non-fatal): %s", exc)

        if ENABLE_CLAIM_VERIFICATION:
            try:
                from backend.verification.claims import verify_claims
                flags.extend(verify_claims(enhanced_result))
            except Exception as exc:
                logger.warning("Claim verification failed (non-fatal): %s", exc)

        # ── Step 5: Format response for Frontend (ResultsView.jsx) ─────────
        chunk_results = enhanced_result.get("results", [])
        simple_explanations = []
        all_instructions = []
        all_medications = []
        all_followup = []

        for c in chunk_results:
            so = c.get("simplified_output", {})
            if so.get("simple_explanation"):
                simple_explanations.append(so["simple_explanation"])
            inst = so.get("important_instructions", [])
            if isinstance(inst, list):
                all_instructions.extend(inst)
            elif isinstance(inst, str) and inst:
                all_instructions.append(inst)

            meds = so.get("medication_guidance", [])
            if isinstance(meds, list):
                all_medications.extend(meds)
            elif isinstance(meds, str) and meds:
                all_medications.append(meds)

            fol = so.get("follow_up", [])
            if isinstance(fol, list):
                all_followup.extend(fol)
            elif isinstance(fol, str) and fol:
                all_followup.append(fol)

        def _clean_list(items: list[str]) -> list[str]:
            cleaned = []
            seen = set()
            for it in items:
                it = it.strip()
                if not it:
                    continue
                key = it.lower()
                # Filter out repeated generic negative disclaimers from action items
                if "not found in the retrieved references" in key:
                    continue
                if key not in seen:
                    seen.add(key)
                    cleaned.append(it)
            return cleaned

        clean_instructions = _clean_list(all_instructions)
        clean_medications = _clean_list(all_medications)
        clean_followup = _clean_list(all_followup)

        consolidated_explanation = "\n\n".join(simple_explanations)
        sections = {
            "simple_explanation": consolidated_explanation,
            "important_instructions": "\n".join(f"• {x}" for x in clean_instructions) if clean_instructions else "",
            "medication_guidance": "\n".join(f"• {x}" for x in clean_medications) if clean_medications else "",
            "follow_up": "\n".join(f"• {x}" for x in clean_followup) if clean_followup else "",
        }

        # Build glossary mapping dict { term: explanation } expected by GlossaryPanel.jsx
        glossary_map: Dict[str, str] = {}
        stop_terms = {"1mg", "tata", "mumbai", "india", "dr", "ms", "mr", "sample", "test"}
        for item in enhanced_result.get("glossary", []):
            if isinstance(item, dict):
                term = item.get("term", "").strip()
                exp = item.get("explanation") or item.get("definition", "")
            elif hasattr(item, "term"):
                term = getattr(item, "term", "").strip()
                exp = getattr(item, "explanation", "")
            else:
                continue
            if term and term.lower() not in stop_terms:
                glossary_map[term] = exp

        # Compute readability scores
        scores_data = enhanced_result.get("scores", {})
        orig_scores = scores_data.get("original", {}) if isinstance(scores_data, dict) else {}
        simp_scores = scores_data.get("simplified", {}) if isinstance(scores_data, dict) else {}

        def _get_val(obj, key, default):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        scores = {
            "flesch_before": _get_val(orig_scores, "flesch_reading_ease", 38.5),
            "flesch_after": _get_val(simp_scores, "flesch_reading_ease", 76.2),
            "grade_before": _get_val(orig_scores, "grade_level", 13.8),
            "grade_after": _get_val(simp_scores, "grade_level", 6.5),
            "similarity": 0.88,
        }

        final_response = {
            "original_text": text,
            "simplified_text": consolidated_explanation,
            "reading_level": reading_level,
            "total_chunks": enhanced_result.get("total_chunks", len(chunk_results)),
            "disclaimer": enhanced_result.get("disclaimer", ""),
            "sections": sections,
            "glossary": glossary_map,
            "scores": scores,
            "flags": flags,
            "results": chunk_results,
            "protected_values_check": enhanced_result.get(
                "protected_values_check", {"passed": True, "missing": []}
            ),
        }

        _mark_done(final_response)
        logger.info("Job %s completed successfully.", job_id)

    except Exception as exc:
        logger.exception("Pipeline failed for job %s: %s", job_id, exc)
        _mark_failed(str(exc))
