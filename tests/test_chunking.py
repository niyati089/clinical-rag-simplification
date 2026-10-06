"""
tests/test_chunking.py — Unit tests for the semantic chunking module.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import (
    chunk_clinical_text,
    ClinicalChunk,
    _clean_text,
    _is_heading,
    _approx_tokens,
)


class TestCleanText:
    def test_normalises_line_endings(self):
        result = _clean_text("line1\r\nline2\rline3")
        assert "\r" not in result

    def test_collapses_excessive_newlines(self):
        result = _clean_text("para1\n\n\n\n\npara2")
        assert "\n\n\n" not in result

    def test_strips_trailing_whitespace(self):
        result = _clean_text("  hello world  ")
        assert result == "hello world"

    def test_empty_string(self):
        assert _clean_text("") == ""


class TestIsHeading:
    def test_allcaps_heading(self):
        assert _is_heading("DIAGNOSIS:") is True
        assert _is_heading("MEDICATIONS AT DISCHARGE") is True

    def test_title_case_with_colon(self):
        assert _is_heading("Chief Complaint:") is True

    def test_normal_sentence_not_heading(self):
        assert _is_heading("The patient should take aspirin daily.") is False

    def test_empty_line_not_heading(self):
        assert _is_heading("") is False
        assert _is_heading("   ") is False


class TestApproxTokens:
    def test_returns_positive_int(self):
        assert _approx_tokens("hello") > 0

    def test_proportional_to_length(self):
        short = _approx_tokens("hi")
        long = _approx_tokens("This is a much longer sentence with many more words and characters.")
        assert long > short


class TestChunkClinicalText:
    SAMPLE_TEXT = """DIAGNOSIS:
The patient presents with acute myocardial infarction.

MEDICATIONS:
Aspirin 75 mg once daily.
Ticagrelor 90 mg twice daily.

FOLLOW-UP:
Return to cardiology clinic in 4 weeks.
"""

    def test_returns_list_of_chunks(self):
        chunks = chunk_clinical_text(self.SAMPLE_TEXT)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_each_chunk_is_clinical_chunk(self):
        chunks = chunk_clinical_text(self.SAMPLE_TEXT)
        for chunk in chunks:
            assert isinstance(chunk, ClinicalChunk)

    def test_chunk_has_required_fields(self):
        chunks = chunk_clinical_text(self.SAMPLE_TEXT)
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.text
            assert isinstance(chunk.chunk_index, int)

    def test_chunk_ids_are_unique(self):
        chunks = chunk_clinical_text(self.SAMPLE_TEXT)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_empty_text_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            chunk_clinical_text("")

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError):
            chunk_clinical_text("   \n   ")

    def test_to_dict_serialisable(self):
        chunks = chunk_clinical_text(self.SAMPLE_TEXT)
        import json
        for chunk in chunks:
            d = chunk.to_dict()
            # Should not raise
            json.dumps(d)

    def test_large_paragraph_is_split(self):
        """A very long paragraph should be split into multiple chunks."""
        long_para = "This is sentence number {}. " * 50
        long_para = long_para.format(*range(50))
        chunks = chunk_clinical_text(long_para, max_tokens=100)
        assert len(chunks) > 1

    def test_short_text_returns_single_chunk(self):
        short = "Take aspirin 75 mg once daily."
        chunks = chunk_clinical_text(short)
        assert len(chunks) == 1
        assert short in chunks[0].text
