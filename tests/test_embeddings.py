"""
tests/test_embeddings.py — Unit tests for the embeddings module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import embed_text, embed_documents, get_embedding_dimension


class TestEmbedText:
    def test_returns_numpy_array(self):
        vec = embed_text("Aspirin is used for pain relief.")
        assert isinstance(vec, np.ndarray)

    def test_correct_dtype(self):
        vec = embed_text("Test sentence.")
        assert vec.dtype == np.float32

    def test_1d_shape(self):
        vec = embed_text("Test sentence.")
        assert vec.ndim == 1

    def test_dimension_matches_model(self):
        dim = get_embedding_dimension()
        vec = embed_text("Test sentence.")
        assert vec.shape[0] == dim

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            embed_text("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            embed_text("   ")

    def test_similar_texts_have_high_cosine_similarity(self):
        v1 = embed_text("The patient has diabetes.")
        v2 = embed_text("The patient suffers from diabetes mellitus.")
        # Cosine similarity = dot product (vectors are L2-normalised)
        similarity = float(np.dot(v1, v2))
        assert similarity > 0.7, f"Expected high similarity, got {similarity:.3f}"

    def test_dissimilar_texts_have_lower_similarity(self):
        v1 = embed_text("The patient has diabetes.")
        v2 = embed_text("The stock market rose by 3% today.")
        similarity = float(np.dot(v1, v2))
        assert similarity < 0.7


class TestEmbedDocuments:
    DOCS = [
        "Aspirin reduces the risk of blood clots.",
        "Metformin is used for type 2 diabetes.",
        "Lithium is used for bipolar disorder.",
    ]

    def test_returns_2d_array(self):
        matrix = embed_documents(self.DOCS)
        assert matrix.ndim == 2

    def test_correct_shape(self):
        matrix = embed_documents(self.DOCS)
        dim = get_embedding_dimension()
        assert matrix.shape == (len(self.DOCS), dim)

    def test_correct_dtype(self):
        matrix = embed_documents(self.DOCS)
        assert matrix.dtype == np.float32

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            embed_documents([])

    def test_list_with_empty_string_raises(self):
        with pytest.raises(ValueError):
            embed_documents(["Valid text", "", "Another valid"])
