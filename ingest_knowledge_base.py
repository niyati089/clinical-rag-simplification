#!/usr/bin/env python3
"""
ingest_knowledge_base.py — Build / update the FAISS vector index.

Place your trusted medical reference documents in:
    data/knowledge_base/guidelines/      ← Clinical guidelines (.txt / .md)
    data/knowledge_base/textbooks/       ← Medical textbooks  (.txt / .md)
    data/knowledge_base/trusted_sources/ ← Other trusted references

Then run:
    python ingest_knowledge_base.py

Options:
    --rebuild   Force a full rebuild even if an index already exists.
    --verbose   Show DEBUG-level log output.
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Ensure project root is on sys.path when run directly ────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion import ingest_knowledge_base
from src.config import KNOWLEDGE_BASE_DIR, VECTOR_STORE_DIR, FAISS_INDEX_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest trusted medical knowledge base into FAISS vector store."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a full rebuild of the vector index.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    args = parser.parse_args()

    # ── Logging setup ────────────────────────────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Pre-flight checks ────────────────────────────────────────────────────
    if not KNOWLEDGE_BASE_DIR.exists():
        logging.warning(
            "Knowledge base directory not found: %s\n"
            "Creating the directory structure. Add your documents and re-run.",
            KNOWLEDGE_BASE_DIR,
        )
        (KNOWLEDGE_BASE_DIR / "guidelines").mkdir(parents=True, exist_ok=True)
        (KNOWLEDGE_BASE_DIR / "textbooks").mkdir(parents=True, exist_ok=True)
        (KNOWLEDGE_BASE_DIR / "trusted_sources").mkdir(parents=True, exist_ok=True)
        print(
            "\nCreated empty knowledge base folders:\n"
            f"  {KNOWLEDGE_BASE_DIR / 'guidelines'}\n"
            f"  {KNOWLEDGE_BASE_DIR / 'textbooks'}\n"
            f"  {KNOWLEDGE_BASE_DIR / 'trusted_sources'}\n\n"
            "Add .txt or .md reference documents to those folders and re-run."
        )
        return

    if FAISS_INDEX_PATH.exists() and not args.rebuild:
        print(
            f"Vector index already exists at {FAISS_INDEX_PATH}.\n"
            "Pass --rebuild to force a full rebuild."
        )
        return

    # ── Run ingestion ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Clinical RAG — Knowledge Base Ingestion")
    print("=" * 60)
    print(f"  Source : {KNOWLEDGE_BASE_DIR}")
    print(f"  Output : {VECTOR_STORE_DIR}")
    print()

    store = ingest_knowledge_base(force_rebuild=args.rebuild)

    print()
    print("=" * 60)
    print(f"  Done! {store.total_vectors()} chunks indexed.")
    print(f"  Index  : {FAISS_INDEX_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
