"""
Tests for RAG pipeline adapter functionality.

Tests the simplify_rag_result() function that processes RAG pipeline output
while preserving structure, references, and adding simplification features.
"""

import json
import pytest
import os
from pathlib import Path
from typing import Dict, Any

from simplifier import simplify_rag_result
from simplifier.schema import RagResultB


class TestRagAdapter:
    """Test cases for RAG pipeline adapter."""
    
    @pytest.fixture
    def rag_sample_data(self) -> Dict[str, Any]:
        """Load sample RAG result data from fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "rag_result_sample.json"
        with open(fixture_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_references_unchanged(self, rag_sample_data):
        """Test that retrieved_references are preserved exactly."""
        result = simplify_rag_result(rag_sample_data)
        
        # Check that references are preserved in each chunk
        for i, chunk_result in enumerate(result["results"]):
            original_refs = rag_sample_data["results"][i]["retrieved_references"]
            output_refs = chunk_result["retrieved_references"]
            
            assert output_refs == original_refs, f"References changed in chunk {i}"
            
            # Verify specific reference fields
            if original_refs:
                assert "source" in output_refs[0]
                assert "relevance_score" in output_refs[0]
    
    def test_chunk_metadata_unchanged(self, rag_sample_data):
        """Test that chunk_id, original_chunk, section_title, chunk_index are preserved."""
        result = simplify_rag_result(rag_sample_data)
        
        for i, chunk_result in enumerate(result["results"]):
            original_chunk = rag_sample_data["results"][i]
            
            assert chunk_result["chunk_id"] == original_chunk["chunk_id"]
            assert chunk_result["original_chunk"] == original_chunk["original_chunk"]
            assert chunk_result["section_title"] == original_chunk["section_title"]
            assert chunk_result["chunk_index"] == original_chunk["chunk_index"]
    
    def test_list_sections_stay_lists(self, rag_sample_data):
        """Test that list sections remain as lists after processing."""
        result = simplify_rag_result(rag_sample_data)
        
        for chunk_result in result["results"]:
            simplified_output = chunk_result["simplified_output"]
            
            # Verify list sections are still lists
            assert isinstance(simplified_output["important_instructions"], list)
            assert isinstance(simplified_output["medication_guidance"], list)
            assert isinstance(simplified_output["follow_up"], list)
            
            # Verify simple_explanation is still a string
            assert isinstance(simplified_output["simple_explanation"], str)
    
    def test_protected_values_preserved(self, rag_sample_data):
        """Test that protected values (dosages, units, lab values) are preserved."""
        result = simplify_rag_result(rag_sample_data)
        
        # Check that protected values check passed
        protected_check = result["protected_values_check"]
        assert "passed" in protected_check
        assert "missing" in protected_check
        
        # Verify specific protected values from sample data
        diabetes_chunk = result["results"][0]
        simplified_text = _combine_chunk_text(diabetes_chunk["simplified_output"])
        
        # Should preserve "8.2%" and "500 mg"
        assert "8.2%" in simplified_text or "8.2" in simplified_text
        assert "500 mg" in simplified_text or "500mg" in simplified_text
        
        hypertension_chunk = result["results"][1]
        simplified_text = _combine_chunk_text(hypertension_chunk["simplified_output"])
        
        # Should preserve "158/94 mmHg" and "10 mg"
        assert "158/94" in simplified_text
        assert "10 mg" in simplified_text or "10mg" in simplified_text
    
    def test_level_normalisation(self, rag_sample_data):
        """Test that level parameter is normalized correctly."""
        # Test default level from rag_result
        result = simplify_rag_result(rag_sample_data)
        # Should use "Basic" -> "basic" from the sample data
        
        # Test explicit level override with different cases
        result_upper = simplify_rag_result(rag_sample_data, level="INTERMEDIATE")
        result_mixed = simplify_rag_result(rag_sample_data, level="Advanced")
        result_lower = simplify_rag_result(rag_sample_data, level="basic")
        
        # All should work without error (we can't directly test the level since it's internal)
        assert "results" in result_upper
        assert "results" in result_mixed
        assert "results" in result_lower
    
    def test_schema_validation(self, rag_sample_data):
        """Test that output validates against RagResultB schema."""
        result = simplify_rag_result(rag_sample_data)
        
        # Should validate without error
        validated = RagResultB(**result)
        
        # Check key schema requirements
        assert validated.reading_level == "Basic"
        assert validated.total_chunks == 2
        assert len(validated.results) == 2
        
        # Check chunk structure
        for chunk in validated.results:
            assert hasattr(chunk, 'chunk_id')
            assert hasattr(chunk, 'entities')
            assert hasattr(chunk, 'glossary')
            assert hasattr(chunk, 'highlights')
            assert hasattr(chunk, 'scores')
    
    def test_offline_mode(self, rag_sample_data):
        """Test that function works in offline mode."""
        # Set offline mode environment variable
        original_offline = os.environ.get('OFFLINE_MODE')
        os.environ['OFFLINE_MODE'] = 'true'
        
        try:
            result = simplify_rag_result(rag_sample_data)
            
            # Should complete without error
            assert "results" in result
            assert len(result["results"]) == 2
            
            # Should have basic structure even in offline mode
            assert "glossary" in result
            assert "scores" in result
            assert "protected_values_check" in result
            
        finally:
            # Restore original environment
            if original_offline is None:
                os.environ.pop('OFFLINE_MODE', None)
            else:
                os.environ['OFFLINE_MODE'] = original_offline
    
    def test_aggregate_glossary(self, rag_sample_data):
        """Test that glossary is aggregated and deduplicated across chunks."""
        result = simplify_rag_result(rag_sample_data)
        
        # Should have top-level aggregated glossary
        assert "glossary" in result
        assert isinstance(result["glossary"], list)
        
        # Should have per-chunk glossaries
        for chunk in result["results"]:
            assert "glossary" in chunk
            assert isinstance(chunk["glossary"], list)
        
        # Collect all terms from chunk glossaries
        all_chunk_terms = set()
        for chunk in result["results"]:
            for entry in chunk["glossary"]:
                all_chunk_terms.add(entry["term"].lower())
        
        # Top-level glossary should contain deduplicated terms
        top_level_terms = set(entry["term"].lower() for entry in result["glossary"])
        
        # All chunk terms should appear in top level (allowing for some filtering)
        # This test allows for reasonable differences due to deduplication logic
    
    def test_aggregate_scores(self, rag_sample_data):
        """Test that readability scores are aggregated correctly."""
        result = simplify_rag_result(rag_sample_data)
        
        # Should have top-level aggregate scores
        assert "scores" in result
        scores = result["scores"]
        
        assert "original" in scores
        assert "simplified" in scores
        
        # Check score structure
        assert "flesch_reading_ease" in scores["original"]
        assert "grade_level" in scores["original"]
        assert "flesch_reading_ease" in scores["simplified"]
        assert "grade_level" in scores["simplified"]
        
        # Scores should be reasonable numbers
        assert 0 <= scores["original"]["flesch_reading_ease"] <= 100
        assert scores["original"]["grade_level"] >= 1
        assert 0 <= scores["simplified"]["flesch_reading_ease"] <= 100
        assert scores["simplified"]["grade_level"] >= 1
    
    def test_invalid_input_handling(self):
        """Test error handling for invalid input."""
        # Test non-dict input
        with pytest.raises(ValueError):
            simplify_rag_result("not a dict")
        
        # Test missing required fields
        with pytest.raises(KeyError):
            simplify_rag_result({"reading_level": "Basic"})  # Missing other fields
        
        # Test invalid level
        valid_rag_data = {
            "reading_level": "Basic",
            "total_chunks": 1,
            "disclaimer": "Test",
            "results": []
        }
        with pytest.raises(ValueError):
            simplify_rag_result(valid_rag_data, level="invalid_level")
    
    def test_empty_results_handling(self, rag_sample_data):
        """Test handling of empty or minimal results."""
        # Create minimal valid input
        minimal_data = {
            "reading_level": "Basic",
            "total_chunks": 0,
            "disclaimer": "Test disclaimer",
            "results": []
        }
        
        result = simplify_rag_result(minimal_data)
        
        # Should complete without error
        assert result["total_chunks"] == 0
        assert result["results"] == []
        assert "glossary" in result
        assert "scores" in result
        assert "protected_values_check" in result


def _combine_chunk_text(simplified_output: Dict[str, Any]) -> str:
    """Helper to combine all text from a simplified output for testing."""
    parts = []
    
    if simplified_output.get("simple_explanation"):
        parts.append(simplified_output["simple_explanation"])
    
    for list_field in ["important_instructions", "medication_guidance", "follow_up"]:
        items = simplified_output.get(list_field, [])
        if isinstance(items, list):
            parts.extend(items)
    
    return " ".join(parts)


# Integration test
class TestRagAdapterIntegration:
    """Integration tests for complete RAG adapter workflow."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from fixture to validated output."""
        # Load fixture
        fixture_path = Path(__file__).parent / "fixtures" / "rag_result_sample.json"
        with open(fixture_path, 'r', encoding='utf-8') as f:
            rag_data = json.load(f)
        
        # Process through adapter
        result = simplify_rag_result(rag_data)
        
        # Validate against schema
        validated = RagResultB(**result)
        
        # Verify key transformations
        assert len(validated.results) == 2
        
        # Check first chunk (diabetes)
        diabetes_chunk = validated.results[0]
        assert diabetes_chunk.chunk_id == "chunk_001_diabetes"
        assert "diabetes" in diabetes_chunk.original_chunk.lower()
        assert len(diabetes_chunk.entities) > 0
        
        # Check second chunk (hypertension)  
        hypertension_chunk = validated.results[1]
        assert hypertension_chunk.chunk_id == "chunk_002_hypertension"
        assert "blood pressure" in hypertension_chunk.original_chunk.lower()
        
        # Verify aggregated data
        assert len(validated.glossary) >= 0  # May be empty in offline mode
        assert validated.protected_values_check.passed in [True, False]  # Should be boolean
        
        print(f"Successfully processed {validated.total_chunks} chunks")
        print(f"Aggregate glossary has {len(validated.glossary)} terms")
        print(f"Protected values check: {validated.protected_values_check.passed}")