#!/usr/bin/env python3
"""
test_rag.py — Standalone interactive test for the RAG pipeline.

Usage:
    python test_rag.py

The script will:
  1. Ask you to paste raw clinical text (type END on a new line to finish).
  2. Ask for the reading level (Basic / Intermediate / Advanced).
  3. Run the complete RAG pipeline.
  4. Display the structured simplified output and retrieved references.

This works independently of Person C's frontend.
"""

import json
import logging
import sys
from pathlib import Path
from textwrap import fill, indent

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import VALID_READING_LEVELS, DISCLAIMER
from src.rag_pipeline import RAGPipeline

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,   # suppress INFO noise during interactive use
    format="%(levelname)s — %(message)s",
)

SEPARATOR = "=" * 70
THIN_SEP  = "-" * 70


def read_multiline(prompt: str) -> str:
    """Collect multi-line input until user types 'END' on its own line."""
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def choose_reading_level() -> str:
    """Prompt user to choose a reading level."""
    options = " / ".join(VALID_READING_LEVELS)
    num_map = {"1": "Basic", "2": "Intermediate", "3": "Advanced"}
    while True:
        choice = input(f"\nReading level [{options}] (default: Basic, or 1/2/3): ").strip()
        if not choice:
            return "Basic"
        if choice in num_map:
            return num_map[choice]
        for level in VALID_READING_LEVELS:
            if choice.lower() == level.lower():
                return level
        print(f"  Invalid choice. Please enter one of: {options} (or 1, 2, 3)")


def format_list(items: list, indent_str: str = "  ") -> str:
    if not items:
        return f"{indent_str}(none)"
    return "\n".join(f"{indent_str}• {item}" for item in items)


def display_result(result: dict) -> None:
    """Pretty-print the pipeline result to stdout."""
    reading_level = result.get("reading_level", "")
    total = result.get("total_chunks", 0)

    print()
    print(SEPARATOR)
    print(f"  RAG PIPELINE OUTPUT  |  Reading Level: {reading_level}  |  Chunks: {total}")
    print(SEPARATOR)
    print(f"\n{DISCLAIMER}\n")

    for chunk_result in result.get("results", []):
        idx = chunk_result["chunk_index"] + 1
        section = chunk_result.get("section_title", "")
        section_label = f" [{section}]" if section else ""

        print(SEPARATOR)
        print(f"  CHUNK {idx}{section_label}")
        print(SEPARATOR)

        print("\nOriginal Clinical Text:")
        print(indent(fill(chunk_result["original_chunk"], width=66), "  "))

        refs = chunk_result.get("retrieved_references", [])
        print(f"\nRetrieved References ({len(refs)} found):")
        if refs:
            for i, ref in enumerate(refs, start=1):
                src = ref.get("source", "Unknown")
                doc = ref.get("document_name", "")
                page = ref.get("page")
                score = ref.get("score", 0.0)
                page_str = f", Page {page}" if page else ""
                print(f"  {i}. {src} — {doc}{page_str}  [score: {score:.3f}]")
        else:
            print("  (No references retrieved — knowledge base may be empty)")

        simplified = chunk_result.get("simplified_output", {})

        print("\nSimple Explanation:")
        explanation = simplified.get("simple_explanation", "")
        print(indent(fill(explanation, width=66), "  "))

        instructions = simplified.get("important_instructions", [])
        print(f"\nImportant Instructions ({len(instructions)}):")
        print(format_list(instructions))

        meds = simplified.get("medication_guidance", [])
        print(f"\nMedication Guidance ({len(meds)}):")
        print(format_list(meds))

        followup = simplified.get("follow_up", [])
        print(f"\nFollow-up ({len(followup)}):")
        print(format_list(followup))

        print()

    print(SEPARATOR)
    print("  END OF REPORT")
    print(SEPARATOR)


def main() -> None:
    print(SEPARATOR)
    print("  Clinical RAG Pipeline — Interactive Test")
    print(SEPARATOR)
    print(
        "\nThis tool simplifies clinical text using a RAG pipeline.\n"
        "Ensure the knowledge base is ingested first:\n"
        "  python ingest_knowledge_base.py\n"
    )

    clinical_text = read_multiline(
        "Paste your clinical text below.\n"
        "When done, type  END  on a new line and press Enter:\n"
    )

    if not clinical_text.strip():
        print("No clinical text provided. Exiting.")
        sys.exit(1)

    reading_level = choose_reading_level()

    print(f"\nRunning RAG pipeline (reading_level={reading_level})...")
    print("This may take a moment on first run (model loading).\n")

    try:
        pipeline = RAGPipeline()
        result = pipeline.process(
            clinical_text=clinical_text,
            reading_level=reading_level,
        )
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        print("Run `python ingest_knowledge_base.py` to build the index.")
        sys.exit(1)
    except EnvironmentError as exc:
        print(f"\n[ERROR] {exc}")
        print("Set LLM_API_KEY in your .env file.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[UNEXPECTED ERROR] {exc}")
        raise

    display_result(result)

    # Optionally save raw JSON
    save = input("\nSave raw JSON output to file? [y/N]: ").strip().lower()
    if save == "y":
        out_path = Path("rag_output.json")
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
