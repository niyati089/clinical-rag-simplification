"""
backend/mock_data.py — Realistic fixture responses for MOCK_MODE.

When MOCK_MODE=true the pipeline is bypassed entirely and one of the three
pre-built fixture files is returned, wrapped in the full final-response shape.
The fixture is chosen by hashing the input text so repeated calls are stable.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Where the three fixture JSONs live (created by Person A)
_MOCK_DIR = Path(__file__).resolve().parent.parent / "data" / "mock_A_outputs"

# Cached fixtures so we only hit disk once
_FIXTURES: Dict[str, dict] = {}


def _load_fixtures() -> Dict[str, dict]:
    global _FIXTURES
    if _FIXTURES:
        return _FIXTURES
    for path in _MOCK_DIR.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            _FIXTURES[path.stem] = json.load(f)
    if not _FIXTURES:
        logger.warning(
            "No mock fixture files found in %s. Returning minimal stub.", _MOCK_DIR
        )
    return _FIXTURES


def _pick_fixture(text: str) -> dict:
    """Deterministically pick a fixture based on text length."""
    fixtures = _load_fixtures()
    if not fixtures:
        return {}
    keys = sorted(fixtures.keys())
    return fixtures[keys[len(text) % len(keys)]]


def _make_mock_glossary(fixture: dict) -> list:
    return [
        {"term": "Metformin", "type": "DRUG", "explanation": "A medication that lowers blood sugar."},
        {"term": "HbA1c", "type": "LAB_VALUE", "explanation": "A blood test showing average sugar levels over 2–3 months."},
        {"term": "Hypertension", "type": "DISEASE", "explanation": "High blood pressure: the force of blood against artery walls is too high."},
    ]


def _make_mock_references() -> list:
    return [
        {
            "text": "Type 2 diabetes mellitus is characterized by insulin resistance and relative insulin deficiency.",
            "source": "ADA Standards of Medical Care",
            "document_name": "ada_standards_2024.pdf",
            "page": "12",
            "section": "Diagnosis and Classification",
            "chunk_id": "mock_ref_001",
            "score": 0.912,
        },
        {
            "text": "Metformin is recommended as first-line pharmacological therapy for type 2 diabetes.",
            "source": "NICE Clinical Guidelines",
            "document_name": "nice_dm_guidelines.pdf",
            "page": "45",
            "section": "Pharmacotherapy",
            "chunk_id": "mock_ref_002",
            "score": 0.874,
        },
    ]


def _make_mock_scores() -> dict:
    return {
        "original": {"flesch_reading_ease": 28.4, "grade_level": 14.2},
        "simplified": {"flesch_reading_ease": 71.8, "grade_level": 5.6},
    }


def _make_mock_flags() -> list:
    return [
        {
            "type": "info",
            "message": "Running in MOCK_MODE — LLM calls are disabled.",
            "severity": "info",
        }
    ]


def build_mock_response(
    original_text: str,
    reading_level: str,
) -> Dict[str, Any]:
    """
    Construct a realistic mock final-result dict without calling A or B.

    Parameters
    ----------
    original_text : str
        The text that would have been simplified.
    reading_level : str
        "basic" | "intermediate" | "advanced"

    Returns
    -------
    dict
        Full result matching the /api/jobs/{job_id} result shape.
    """
    fixture = _pick_fixture(original_text)

    # Wrap fixture in a single-chunk result
    chunk_result = {
        "chunk_id": "mock_chunk_0001",
        "original_chunk": original_text[:500] + ("..." if len(original_text) > 500 else ""),
        "section_title": "Clinical Summary",
        "chunk_index": 0,
        "simplified_output": {
            "simple_explanation": fixture.get(
                "simple_explanation",
                "This document has been simplified for easier reading.",
            ),
            "important_instructions": _split_to_list(
                fixture.get("important_instructions", "")
            ),
            "medication_guidance": _split_to_list(
                fixture.get("medication_guidance", "")
            ),
            "follow_up": _split_to_list(fixture.get("follow_up", "")),
        },
        "retrieved_references": _make_mock_references(),
        "entities": [
            {"text": "Metformin", "type": "DRUG", "start": 10, "end": 19},
            {"text": "HbA1c", "type": "LAB_VALUE", "start": 35, "end": 40},
        ],
        "glossary": _make_mock_glossary(fixture),
        "highlights": [
            {"section": "important_instructions", "sentence": "Take your medication every day as prescribed."},
            {"section": "follow_up", "sentence": "Return to the clinic in 6 weeks."},
        ],
        "scores": _make_mock_scores(),
    }

    return {
        "original_text": original_text,
        "reading_level": reading_level,
        "total_chunks": 1,
        "disclaimer": (
            "This output is an AI-assisted simplification for educational purposes only. "
            "It is NOT a substitute for professional medical advice. Always consult a "
            "qualified healthcare provider for medical decisions."
        ),
        "results": [chunk_result],
        "glossary": _make_mock_glossary(fixture),
        "scores": _make_mock_scores(),
        "flags": _make_mock_flags(),
        "protected_values_check": {"passed": True, "missing": []},
        "mock_mode": True,
    }


def _split_to_list(text: str) -> list:
    """Turn a paragraph string into a list of sentences/items."""
    if not text:
        return []
    if isinstance(text, list):
        return text
    # Split on ". " boundaries
    import re
    items = [s.strip() for s in re.split(r"\.\s+", text) if s.strip()]
    # Re-add trailing period
    return [item if item.endswith(".") else item + "." for item in items]
