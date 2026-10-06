#!/usr/bin/env python3
"""
evaluate_rag.py — RAG vs. Plain LLM evaluation script.

Compares:
  A. Plain LLM (no retrieval)
  B. RAG + LLM (retrieval-augmented)

For each of the 10 sample clinical documents in data/evaluation_samples/,
it runs both pipelines and saves:
  - Side-by-side JSON outputs
  - A structured comparison table (CSV + console display)
  - A manual evaluation template (Markdown)

Usage:
    python evaluation/evaluate_rag.py
    python evaluation/evaluate_rag.py --level Intermediate
    python evaluation/evaluate_rag.py --samples path/to/samples/

The evaluation criteria (see _CRITERIA below) require human medical review
for ground-truth accuracy. The script generates the data and the template;
it does NOT fabricate evaluation scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    EVALUATION_SAMPLES_DIR,
    EVALUATION_RESULTS_DIR,
    DEFAULT_READING_LEVEL,
    VALID_READING_LEVELS,
    DISCLAIMER,
)
from src.rag_pipeline import RAGPipeline
from src.llm import generate_plain_response
from src.chunking import chunk_clinical_text

logger = logging.getLogger(__name__)

# ── Evaluation criteria (for manual review template) ────────────────────────

_CRITERIA = [
    "factual_grounding",        # Are facts traceable to a source?
    "medical_accuracy",         # Is the medical content correct?
    "unsupported_claims",       # Count of invented/hallucinated statements
    "instruction_coverage",     # Are important instructions preserved?
    "relevance",                # Is the simplified output relevant?
    "explanation_quality",      # Is the explanation clear and useful?
]

_SCORE_SCALE = "0–5 (0=very poor, 5=excellent). For unsupported_claims: count (lower=better)."


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_samples(samples_dir: Path) -> List[Dict[str, str]]:
    """
    Load all .txt sample clinical documents from samples_dir.
    Returns list of {"name": filename_stem, "text": content}.
    """
    samples = []
    if not samples_dir.exists():
        raise FileNotFoundError(
            f"Evaluation samples directory not found: {samples_dir}\n"
            "Add .txt files to data/evaluation_samples/ and re-run."
        )
    for path in sorted(samples_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            samples.append({"name": path.stem, "text": text})
    return samples


def _run_plain_llm(text: str, reading_level: str) -> Dict[str, Any]:
    """Run plain LLM on the full text (chunked, then merge outputs)."""
    chunks = chunk_clinical_text(text)
    results = []
    for chunk in chunks:
        try:
            output = generate_plain_response(
                clinical_chunk=chunk.text,
                reading_level=reading_level,
            )
        except Exception as exc:
            output = {
                "simple_explanation": f"[Error: {exc}]",
                "important_instructions": [],
                "medication_guidance": [],
                "follow_up": [],
            }
        results.append({"chunk_id": chunk.chunk_id, "output": output})
    return {"chunks": results}


def _run_rag(text: str, reading_level: str, pipeline: RAGPipeline) -> Dict[str, Any]:
    """Run the full RAG pipeline on the text."""
    return pipeline.process(clinical_text=text, reading_level=reading_level)


def _count_references(rag_result: dict) -> int:
    total = 0
    for r in rag_result.get("results", []):
        total += len(r.get("retrieved_references", []))
    return total


def _generate_manual_template(
    doc_name: str,
    plain_output: dict,
    rag_output: dict,
) -> str:
    """
    Generate a Markdown manual evaluation template for a single document.
    A human medical reviewer fills this in.
    """
    lines = [
        f"# Manual Evaluation — {doc_name}",
        "",
        f"> Scoring scale: {_SCORE_SCALE}",
        "",
        "## Plain LLM Output",
        "```json",
        json.dumps(plain_output, indent=2)[:3000],  # truncate for readability
        "```",
        "",
        "## RAG + LLM Output",
        "```json",
        json.dumps(rag_output, indent=2)[:3000],
        "```",
        "",
        "## Evaluation Scores",
        "",
        "| Criterion | Plain LLM Score | RAG Score | Notes |",
        "|-----------|-----------------|-----------|-------|",
    ]
    for criterion in _CRITERIA:
        lines.append(f"| {criterion} | ___ | ___ | |")

    lines += [
        "",
        "## Reviewer Notes",
        "",
        "**Hallucinations / Unsupported Claims (Plain LLM):**",
        "",
        "_List any statements not grounded in fact:_",
        "",
        "**Hallucinations / Unsupported Claims (RAG):**",
        "",
        "_List any statements not grounded in retrieved references:_",
        "",
        "**Overall Assessment:**",
        "",
        "_Which output is more suitable for patient communication and why?_",
        "",
        f"---",
        f"*Reviewer:* ___________  *Date:* ___________",
        "",
        f"> {DISCLAIMER}",
    ]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG vs. Plain LLM on clinical sample documents."
    )
    parser.add_argument(
        "--level",
        choices=list(VALID_READING_LEVELS),
        default=DEFAULT_READING_LEVEL,
        help="Reading level for simplification (default: Basic).",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=EVALUATION_SAMPLES_DIR,
        help="Path to evaluation samples directory.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    reading_level: str = args.level
    samples_dir: Path = args.samples

    # ── Load samples ────────────────────────────────────────────────────────
    print(f"\nLoading evaluation samples from: {samples_dir}")
    samples = _load_samples(samples_dir)
    if not samples:
        print("No .txt sample files found. Add samples and re-run.")
        sys.exit(1)
    print(f"Found {len(samples)} sample(s).")

    # ── Output directory ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = EVALUATION_RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {run_dir}\n")

    # ── Initialise RAG pipeline once ────────────────────────────────────────
    pipeline = RAGPipeline()

    # ── Per-document evaluation ─────────────────────────────────────────────
    summary_rows: List[dict] = []

    for i, sample in enumerate(samples, start=1):
        doc_name = sample["name"]
        doc_text = sample["text"]
        print(f"[{i}/{len(samples)}] Processing: {doc_name}")

        # Plain LLM
        print("  → Running Plain LLM...")
        try:
            plain_output = _run_plain_llm(doc_text, reading_level)
            plain_status = "OK"
        except Exception as exc:
            logger.error("Plain LLM failed for %s: %s", doc_name, exc)
            plain_output = {"error": str(exc)}
            plain_status = f"ERROR: {exc}"

        # RAG
        print("  → Running RAG pipeline...")
        try:
            rag_output = _run_rag(doc_text, reading_level, pipeline)
            rag_status = "OK"
            num_refs = _count_references(rag_output)
        except Exception as exc:
            logger.error("RAG failed for %s: %s", doc_name, exc)
            rag_output = {"error": str(exc)}
            rag_status = f"ERROR: {exc}"
            num_refs = 0

        # ── Save raw outputs ─────────────────────────────────────────────
        doc_dir = run_dir / doc_name
        doc_dir.mkdir(exist_ok=True)

        (doc_dir / "plain_llm_output.json").write_text(
            json.dumps(plain_output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (doc_dir / "rag_output.json").write_text(
            json.dumps(rag_output, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ── Generate manual evaluation template ──────────────────────────
        template_md = _generate_manual_template(doc_name, plain_output, rag_output)
        (doc_dir / "manual_evaluation_template.md").write_text(
            template_md, encoding="utf-8"
        )

        # ── Summary row ───────────────────────────────────────────────────
        summary_rows.append(
            {
                "document": doc_name,
                "reading_level": reading_level,
                "plain_llm_status": plain_status,
                "rag_status": rag_status,
                "rag_references_retrieved": num_refs,
                "plain_llm_accuracy": "MANUAL_REVIEW_NEEDED",
                "rag_accuracy": "MANUAL_REVIEW_NEEDED",
                "plain_llm_unsupported_claims": "MANUAL_REVIEW_NEEDED",
                "rag_unsupported_claims": "MANUAL_REVIEW_NEEDED",
                "manual_template": str(doc_dir / "manual_evaluation_template.md"),
            }
        )
        print(f"  ✓ Done. References retrieved by RAG: {num_refs}\n")

    # ── Save summary CSV ────────────────────────────────────────────────────
    csv_path = run_dir / "evaluation_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    # ── Print summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(
        f"{'Document':<30} {'Plain LLM':>12} {'RAG':>6} {'RAG Refs':>10}"
    )
    print("-" * 70)
    for row in summary_rows:
        print(
            f"{row['document']:<30} "
            f"{row['plain_llm_status']:>12} "
            f"{row['rag_status']:>6} "
            f"{row['rag_references_retrieved']:>10}"
        )
    print("=" * 70)
    print(
        "\nNOTE: Medical accuracy and unsupported-claims scores require\n"
        "      human review. Fill in the generated Markdown templates:\n"
        f"      {run_dir}\n"
    )
    print(
        "The columns 'plain_llm_accuracy', 'rag_accuracy', etc. in the CSV\n"
        "are marked 'MANUAL_REVIEW_NEEDED' — do not fabricate these scores.\n"
        "A qualified reviewer should assess each template and fill them in."
    )
    print(f"\nSummary CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
