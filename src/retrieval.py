"""
retrieval.py — FAISS-backed vector store for medical knowledge retrieval.

Responsibilities:
  - Create / save / load a FAISS index.
  - Add chunk embeddings with associated metadata.
  - Search the index and return ranked results with text + metadata.

The metadata store (chunk text + source info) is persisted as a JSON file
alongside the FAISS binary index so both are always in sync.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from src.config import (
    FAISS_INDEX_PATH,
    METADATA_STORE_PATH,
    EMBEDDING_DIMENSION,
    TOP_K,
)

logger = logging.getLogger(__name__)


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text: str
    source: str
    document_name: str
    page_number: Optional[str]
    section_title: Optional[str]
    chunk_id: str
    score: float           # cosine similarity (higher = more relevant)
    faiss_index: int = -1  # internal FAISS row index

    def to_dict(self) -> dict:
        return asdict(self)


# ── FAISSStore ──────────────────────────────────────────────────────────────

class FAISSStore:
    """
    Manages a FAISS IndexFlatIP (inner-product = cosine on L2-normalised vectors)
    together with a parallel JSON metadata store.
    """

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_STORE_PATH,
        embedding_dim: int = EMBEDDING_DIMENSION,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_dim = embedding_dim

        self._index: Optional[faiss.Index] = None
        self._metadata: List[dict] = []   # parallel to FAISS rows

    # ── Index lifecycle ────────────────────────────────────────────────────

    def create_index(self) -> None:
        """Create a brand-new empty FAISS index (Inner Product on L2-normed vecs)."""
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._metadata = []
        logger.info("Created new FAISS IndexFlatIP (dim=%d).", self.embedding_dim)

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        if self._index is None:
            raise RuntimeError("No index to save. Call create_index() first.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

        logger.info(
            "Saved FAISS index (%d vectors) → %s",
            self._index.ntotal,
            self.index_path,
        )

    def load(self) -> None:
        """Load an existing FAISS index and its metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run `python ingest_knowledge_base.py` first."
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata store not found at {self.metadata_path}."
            )

        self._index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        logger.info(
            "Loaded FAISS index (%d vectors) from %s",
            self._index.ntotal,
            self.index_path,
        )

    def is_loaded(self) -> bool:
        return self._index is not None

    def ensure_loaded(self) -> None:
        """Load the index if it hasn't been loaded yet."""
        if not self.is_loaded():
            self.load()

    # ── Adding vectors ─────────────────────────────────────────────────────

    def add_chunks(
        self,
        embeddings: np.ndarray,
        metadata_list: List[dict],
    ) -> None:
        """
        Add a batch of L2-normalised embeddings with their metadata.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n, embedding_dim), float32, L2-normalised.
        metadata_list : List[dict]
            One metadata dict per embedding. Required keys:
            chunk_id, text, source, document_name.
            Optional: page_number, section_title.
        """
        if self._index is None:
            raise RuntimeError("Index not initialised. Call create_index() first.")
        if len(embeddings) != len(metadata_list):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and metadata_list "
                f"({len(metadata_list)}) must have the same length."
            )

        # Ensure float32
        vecs = np.array(embeddings, dtype=np.float32)
        self._index.add(vecs)
        self._metadata.extend(metadata_list)
        logger.debug("Added %d vectors. Total: %d", len(vecs), self._index.ntotal)

    # ── Searching ──────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            1-D float32 L2-normalised embedding of the query.
        top_k : int
            Number of results to return.

        Returns
        -------
        List[RetrievedChunk]
            Ordered by descending cosine similarity (score).
        """
        self.ensure_loaded()

        if self._index.ntotal == 0:
            logger.warning("FAISS index is empty — no results returned.")
            return []

        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)

        t0 = time.perf_counter()
        scores, indices = self._index.search(query_vec, k)
        elapsed = time.perf_counter() - t0
        logger.debug("FAISS search took %.3f s for top-%d.", elapsed, k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS padding for insufficient results
            meta = self._metadata[idx]
            results.append(
                RetrievedChunk(
                    text=meta.get("text", ""),
                    source=meta.get("source", "Unknown"),
                    document_name=meta.get("document_name", ""),
                    page_number=meta.get("page_number"),
                    section_title=meta.get("section_title"),
                    chunk_id=meta.get("chunk_id", f"row_{idx}"),
                    score=float(score),
                    faiss_index=int(idx),
                )
            )

        return results

    # ── Utility ────────────────────────────────────────────────────────────

    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0


# ── Module-level singleton (lazy-loaded) ────────────────────────────────────

_store: Optional[FAISSStore] = None


def get_store() -> FAISSStore:
    """Return the module-level FAISSStore singleton (lazy-loaded)."""
    global _store
    if _store is None:
        _store = FAISSStore()
    return _store


def retrieve(
    query_embedding: np.ndarray,
    top_k: int = TOP_K,
    store: Optional[FAISSStore] = None,
) -> List[RetrievedChunk]:
    """
    Convenience wrapper: retrieve top-k chunks for a query embedding.

    Parameters
    ----------
    query_embedding : np.ndarray
        1-D float32 embedding of the clinical chunk text.
    top_k : int
        Number of references to retrieve.
    store : FAISSStore, optional
        Use a custom store instance instead of the default singleton.

    Returns
    -------
    List[RetrievedChunk]
    """
    active_store = store or get_store()
    active_store.ensure_loaded()
    return active_store.search(query_embedding, top_k=top_k)
