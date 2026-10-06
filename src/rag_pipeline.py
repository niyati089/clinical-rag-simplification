"""
rag_pipeline.py — Complete RAG pipeline (Person A's main deliverable).

Exposes the RAGPipeline class which Person C can import into the FastAPI backend:

    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    result = pipeline.process(
        clinical_text="...",
        reading_level="Basic"
    )

The returned result is fully JSON-serialisable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import (
    TOP_K,
    USE_RERANKER,
    RERANKER_TOP_K,
    VALID_READING_LEVELS,
    DEFAULT_READING_LEVEL,
    DISCLAIMER,
)
from src.chunking import chunk_clinical_text, ClinicalChunk
from src.embeddings import embed_text
from src.retrieval import FAISSStore, RetrievedChunk, retrieve
from src.reranking import rerank
from src.llm import generate_response

logger = logging.getLogger(__name__)


# ── Result data models ───────────────────────────────────────────────────────

@dataclass
class ChunkResult:
    chunk_id: str
    original_chunk: str
    section_title: str
    chunk_index: int
    simplified_output: Dict[str, Any]
    retrieved_references: List[Dict[str, Any]]

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "original_chunk": self.original_chunk,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "simplified_output": self.simplified_output,
            "retrieved_references": self.retrieved_references,
        }


@dataclass
class PipelineResult:
    results: List[ChunkResult]
    reading_level: str
    total_chunks: int
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "reading_level": self.reading_level,
            "total_chunks": self.total_chunks,
            "disclaimer": self.disclaimer,
            "results": [r.to_dict() for r in self.results],
        }


# ── RAGPipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Complete clinical RAG pipeline.

    Parameters
    ----------
    top_k : int
        Number of FAISS results to retrieve per clinical chunk.
    use_reranker : bool
        Whether to apply cross-encoder reranking.
    reranker_top_k : int
        Number of results to keep after reranking.
    store : FAISSStore, optional
        Inject a custom FAISS store (useful for testing).

    Usage (Person C integration)
    ----------------------------
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    result = pipeline.process(
        clinical_text=raw_text,
        reading_level="Basic"
    )
    # result is a dict — JSON-serialisable
    """

    def __init__(
        self,
        top_k: int = TOP_K,
        use_reranker: bool = USE_RERANKER,
        reranker_top_k: int = RERANKER_TOP_K,
        store: Optional[FAISSStore] = None,
    ) -> None:
        self.top_k = top_k
        self.use_reranker = use_reranker
        self.reranker_top_k = reranker_top_k
        self._store = store  # lazy-load if None

    # ── Store management ───────────────────────────────────────────────────

    def _get_store(self) -> FAISSStore:
        if self._store is None:
            self._store = FAISSStore()
        self._store.ensure_loaded()
        return self._store

    # ── Internal steps ─────────────────────────────────────────────────────

    def _chunk(self, clinical_text: str) -> List[ClinicalChunk]:
        """Step 1: Semantic chunking."""
        chunks = chunk_clinical_text(clinical_text)
        logger.info("Chunking complete → %d chunks.", len(chunks))
        return chunks

    def _retrieve_for_chunk(
        self,
        chunk: ClinicalChunk,
        store: FAISSStore,
    ) -> List[RetrievedChunk]:
        """Steps 2–4: Embed → FAISS search → optional reranking."""
        embedding = embed_text(chunk.text)
        candidates = store.search(embedding, top_k=self.top_k)

        if not candidates:
            logger.warning(
                "No references retrieved for chunk %s. "
                "The knowledge base may be empty or the index not built.",
                chunk.chunk_id,
            )

        final = rerank(
            query=chunk.text,
            candidates=candidates,
            top_k=self.reranker_top_k,
            enabled=self.use_reranker,
        )
        return final

    def _generate(
        self,
        chunk: ClinicalChunk,
        references: List[RetrievedChunk],
        reading_level: str,
    ) -> Dict[str, Any]:
        """Step 5: Prompt construction + LLM call."""
        return generate_response(
            clinical_chunk=chunk.text,
            retrieved_context=references,
            reading_level=reading_level,
        )

    # ── Public process() method ────────────────────────────────────────────

    def process(
        self,
        clinical_text: str,
        reading_level: str = DEFAULT_READING_LEVEL,
    ) -> dict:
        """
        Run the full RAG pipeline on a clinical document.

        Parameters
        ----------
        clinical_text : str
            Raw clinical/medical text to simplify.
        reading_level : str
            "Basic" | "Intermediate" | "Advanced"

        Returns
        -------
        dict
            JSON-serialisable result matching the agreed schema:
            {
              "reading_level": "...",
              "total_chunks": N,
              "disclaimer": "...",
              "results": [
                {
                  "chunk_id": "...",
                  "original_chunk": "...",
                  "section_title": "...",
                  "chunk_index": N,
                  "simplified_output": {
                    "simple_explanation": "...",
                    "important_instructions": [...],
                    "medication_guidance": [...],
                    "follow_up": [...]
                  },
                  "retrieved_references": [
                    {"text": "...", "source": "...", "page": "...", "score": 0.0, ...}
                  ]
                }
              ]
            }

        Raises
        ------
        ValueError
            If clinical_text is empty or reading_level is invalid.
        FileNotFoundError
            If the FAISS index has not been built yet.
        EnvironmentError
            If LLM_API_KEY is not set.
        """
        # ── Input validation ────────────────────────────────────────────────
        if not clinical_text or not clinical_text.strip():
            raise ValueError(
                "clinical_text is empty. Please provide a non-empty clinical document."
            )
        if reading_level not in VALID_READING_LEVELS:
            raise ValueError(
                f"Invalid reading_level '{reading_level}'. "
                f"Valid options: {VALID_READING_LEVELS}"
            )

        logger.info(
            "Starting RAG pipeline | reading_level=%s", reading_level
        )

        # ── Load vector store (fails fast if not built) ─────────────────────
        store = self._get_store()
        if store.total_vectors() == 0:
            logger.warning(
                "FAISS index is empty. Responses will lack factual grounding. "
                "Run `python ingest_knowledge_base.py` to populate the index."
            )

        # ── Step 1: Semantic chunking ────────────────────────────────────────
        chunks = self._chunk(clinical_text)

        # ── Steps 2–5: Per-chunk retrieval + LLM generation ─────────────────
        chunk_results: List[ChunkResult] = []

        for chunk in chunks:
            logger.debug(
                "Processing chunk %s (%d chars)",
                chunk.chunk_id,
                len(chunk.text),
            )

            # Retrieve references
            try:
                references = self._retrieve_for_chunk(chunk, store)
            except Exception as exc:
                logger.error(
                    "Retrieval failed for chunk %s: %s", chunk.chunk_id, exc
                )
                references = []

            # Generate simplified output
            try:
                simplified = self._generate(chunk, references, reading_level)
            except Exception as exc:
                logger.error(
                    "LLM generation failed for chunk %s: %s", chunk.chunk_id, exc
                )
                simplified = {
                    "simple_explanation": (
                        f"[Error: Could not generate simplified output. {exc}]"
                    ),
                    "important_instructions": [],
                    "medication_guidance": [],
                    "follow_up": [],
                }

            # Serialise references
            ref_dicts = []
            for ref in references:
                ref_dicts.append(
                    {
                        "text": ref.text,
                        "source": ref.source,
                        "document_name": ref.document_name,
                        "page": ref.page_number,
                        "section": ref.section_title,
                        "chunk_id": ref.chunk_id,
                        "score": round(ref.score, 4),
                    }
                )

            chunk_results.append(
                ChunkResult(
                    chunk_id=chunk.chunk_id,
                    original_chunk=chunk.text,
                    section_title=chunk.section_title,
                    chunk_index=chunk.chunk_index,
                    simplified_output=simplified,
                    retrieved_references=ref_dicts,
                )
            )

        pipeline_result = PipelineResult(
            results=chunk_results,
            reading_level=reading_level,
            total_chunks=len(chunk_results),
        )

        logger.info(
            "Pipeline complete. %d chunks processed.", len(chunk_results)
        )
        return pipeline_result.to_dict()
