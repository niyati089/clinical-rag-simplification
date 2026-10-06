"""
tests/test_retrieval.py — Unit tests for the FAISS retrieval module.
"""

import json
import numpy as np
import pytest
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import FAISSStore, RetrievedChunk


def _make_store_with_data(tmp_path: Path) -> FAISSStore:
    """Helper: create a FAISSStore with 5 dummy vectors."""
    dim = 384
    store = FAISSStore(
        index_path=tmp_path / "test_index.bin",
        metadata_path=tmp_path / "test_meta.json",
        embedding_dim=dim,
    )
    store.create_index()

    # Create 5 distinct random unit vectors
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((5, dim)).astype(np.float32)
    # L2-normalise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    metadata = [
        {
            "chunk_id": f"kb_{i:04d}",
            "text": f"Medical reference text number {i}.",
            "source": "Clinical Guideline",
            "document_name": f"Document {i}",
            "page_number": str(i * 10),
            "section_title": f"Section {i}",
        }
        for i in range(5)
    ]
    store.add_chunks(vecs, metadata)
    return store


class TestFAISSStoreCreate:
    def test_create_index(self, tmp_path):
        store = FAISSStore(
            index_path=tmp_path / "idx.bin",
            metadata_path=tmp_path / "meta.json",
        )
        store.create_index()
        assert store.total_vectors() == 0

    def test_add_chunks_increases_count(self, tmp_path):
        store = _make_store_with_data(tmp_path)
        assert store.total_vectors() == 5

    def test_mismatched_embeddings_and_metadata_raises(self, tmp_path):
        store = FAISSStore(
            index_path=tmp_path / "idx.bin",
            metadata_path=tmp_path / "meta.json",
        )
        store.create_index()
        vecs = np.random.randn(3, 384).astype(np.float32)
        metadata = [{"chunk_id": "1", "text": "a", "source": "s", "document_name": "d"}]
        with pytest.raises(ValueError):
            store.add_chunks(vecs, metadata)


class TestFAISSStoreSaveLoad:
    def test_save_and_load(self, tmp_path):
        store = _make_store_with_data(tmp_path)
        store.save()

        store2 = FAISSStore(
            index_path=tmp_path / "test_index.bin",
            metadata_path=tmp_path / "test_meta.json",
            embedding_dim=384,
        )
        store2.load()
        assert store2.total_vectors() == 5

    def test_load_missing_index_raises(self, tmp_path):
        store = FAISSStore(
            index_path=tmp_path / "nonexistent.bin",
            metadata_path=tmp_path / "meta.json",
        )
        with pytest.raises(FileNotFoundError):
            store.load()


class TestFAISSStoreSearch:
    def test_search_returns_list(self, tmp_path):
        store = _make_store_with_data(tmp_path)
        query = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query)
        results = store.search(query, top_k=3)
        assert isinstance(results, list)

    def test_search_returns_correct_count(self, tmp_path):
        store = _make_store_with_data(tmp_path)
        query = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query)
        results = store.search(query, top_k=3)
        assert len(results) == 3

    def test_search_returns_retrieved_chunk_instances(self, tmp_path):
        store = _make_store_with_data(tmp_path)
        query = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query)
        results = store.search(query, top_k=2)
        for r in results:
            assert isinstance(r, RetrievedChunk)

    def test_search_result_has_required_fields(self, tmp_path):
        store = _make_store_with_data(tmp_path)
        query = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query)
        results = store.search(query, top_k=1)
        r = results[0]
        assert r.text
        assert r.source
        assert r.chunk_id
        assert isinstance(r.score, float)

    def test_search_empty_index_returns_empty(self, tmp_path):
        store = FAISSStore(
            index_path=tmp_path / "idx.bin",
            metadata_path=tmp_path / "meta.json",
        )
        store.create_index()
        query = np.random.randn(384).astype(np.float32)
        results = store.search(query, top_k=5)
        assert results == []

    def test_top_k_capped_at_total_vectors(self, tmp_path):
        store = _make_store_with_data(tmp_path)
        query = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query)
        results = store.search(query, top_k=100)
        assert len(results) == 5  # only 5 vectors in store
