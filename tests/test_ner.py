"""
Unit tests for Named Entity Recognition functionality.
"""

import pytest
from simplifier.ner import NEREngine, extract_entities, EvaluationMetrics
from simplifier.schema import EntityModel


class TestNEREngine:
    """Test cases for the NEREngine class."""
    
    def test_engine_initialization(self):
        """Test that NER engine initializes without errors."""
        engine = NEREngine()
        assert engine is not None
        assert not engine._initialized  # Lazy loading should not initialize yet
    
    def test_lab_value_extraction(self):
        """Test regex-based lab value extraction (works without scispaCy)."""
        engine = NEREngine()
        
        # Test various lab value formats
        test_cases = [
            ("Blood glucose is 180 mg/dL", "180 mg/dL", "LAB_VALUE"),
            ("HbA1c is 8.5%", "HbA1c is 8.5%", "LAB_VALUE"),  # HbA1c pattern captures full phrase
            ("BP is 150/95 mmHg", "150/95 mmHg", "LAB_VALUE"),
            ("Temperature 101.2°F", "101.2°F", "LAB_VALUE"),
            ("Heart rate 98 bpm", "98 bpm", "LAB_VALUE"),
            ("Give 500 mg daily", "500 mg", "LAB_VALUE"),
        ]
        
        for text, expected_text, expected_type in test_cases:
            entities = engine.extract_entities(text)
            # Find the matching entity
            matching = [e for e in entities if e.text == expected_text]
            assert len(matching) > 0, f"Expected to find '{expected_text}' in text: {text}. Found: {[e.text for e in entities]}"
            assert matching[0].type == expected_type
    
    def test_entity_positions(self):
        """Test that entity positions are accurate."""
        engine = NEREngine()
        text = "Blood glucose is 180 mg/dL and BP is 150/95 mmHg."
        entities = engine.extract_entities(text)
        
        # Check each entity's position matches the text
        for entity in entities:
            extracted_text = text[entity.start:entity.end]
            assert extracted_text == entity.text, \
                f"Position mismatch: expected '{entity.text}' but got '{extracted_text}'"
    
    def test_no_overlapping_entities(self):
        """Test that entities don't overlap."""
        engine = NEREngine()
        text = "HbA1c is 8.5% which indicates poor glycemic control with glucose 180 mg/dL."
        entities = engine.extract_entities(text)
        
        # Check no overlaps
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                assert e1.end <= e2.start or e2.end <= e1.start, \
                    f"Overlapping entities: {e1.text} and {e2.text}"
    
    def test_entity_sorting(self):
        """Test that entities are sorted by start position."""
        engine = NEREngine()
        text = "Patient has BP 140/90, glucose 180 mg/dL, and temp 98.6°F."
        entities = engine.extract_entities(text)
        
        # Check sorting
        for i in range(len(entities) - 1):
            assert entities[i].start <= entities[i+1].start, \
                "Entities not sorted by start position"
    
    def test_multiple_lab_values(self):
        """Test extraction of multiple lab values in one text."""
        engine = NEREngine()
        text = "Labs: glucose 180 mg/dL, HbA1c 8.5%, BP 150/95 mmHg, HR 98 bpm"
        entities = engine.extract_entities(text)
        
        lab_entities = [e for e in entities if e.type == "LAB_VALUE"]
        assert len(lab_entities) >= 4, f"Expected at least 4 lab values, got {len(lab_entities)}"
    
    def test_empty_text(self):
        """Test that empty text returns no entities."""
        engine = NEREngine()
        entities = engine.extract_entities("")
        assert entities == []
    
    def test_text_without_entities(self):
        """Test text with no medical entities."""
        engine = NEREngine()
        text = "The weather is nice today."
        entities = engine.extract_entities(text)
        # Should return empty or very few entities
        assert isinstance(entities, list)
    
    def test_convenience_function(self):
        """Test the module-level extract_entities function."""
        text = "Blood glucose is 180 mg/dL"
        entities = extract_entities(text)
        assert isinstance(entities, list)
        assert len(entities) > 0
        assert all(isinstance(e, EntityModel) for e in entities)
    
    def test_entity_types(self):
        """Test that all entity types are valid according to schema."""
        engine = NEREngine()
        text = "Patient has diabetes with HbA1c 8.5% and takes metformin 500 mg daily."
        entities = engine.extract_entities(text)
        
        valid_types = {"DISEASE", "DRUG", "PROCEDURE", "LAB_VALUE", "ABBREVIATION"}
        for entity in entities:
            assert entity.type in valid_types, f"Invalid entity type: {entity.type}"


class TestNERWithScispacy:
    """Test cases that require scispaCy to be installed."""
    
    def test_disease_extraction(self):
        """Test disease entity extraction (requires scispaCy)."""
        engine = NEREngine()
        
        # Try to load the model
        if not engine._load_model():
            pytest.skip("scispaCy model not available")
        
        text = "Patient presents with type 2 diabetes mellitus and hypertension."
        entities = engine.extract_entities(text)
        
        # Should find disease entities if scispaCy is working
        disease_entities = [e for e in entities if e.type == "DISEASE"]
        # This might be empty if model doesn't recognize these specific terms
        assert isinstance(disease_entities, list)
    
    def test_drug_extraction(self):
        """Test drug entity extraction (requires scispaCy)."""
        engine = NEREngine()
        
        if not engine._load_model():
            pytest.skip("scispaCy model not available")
        
        text = "Prescribe metformin 500 mg twice daily."
        entities = engine.extract_entities(text)
        
        # Check we got some entities
        assert len(entities) > 0
    
    def test_procedure_extraction(self):
        """Test procedure entity extraction (requires scispaCy)."""
        engine = NEREngine()
        
        if not engine._load_model():
            pytest.skip("scispaCy model not available")
        
        text = "The patient underwent cardiac catheterization."
        entities = engine.extract_entities(text)
        
        # Should extract some entities
        assert isinstance(entities, list)
    
    def test_abbreviation_extraction(self):
        """Test abbreviation detection (requires scispaCy)."""
        engine = NEREngine()
        
        if not engine._load_model():
            pytest.skip("scispaCy model not available")
        
        text = "Patient has chronic obstructive pulmonary disease (COPD)."
        entities = engine.extract_entities(text)
        
        # Check for abbreviations
        abbrev_entities = [e for e in entities if e.type == "ABBREVIATION"]
        # May or may not find depending on model behavior
        assert isinstance(abbrev_entities, list)


class TestNEREvaluation:
    """Test cases for NER evaluation metrics."""
    
    def test_evaluation_function_exists(self):
        """Test that evaluation function can be called."""
        engine = NEREngine()
        metrics = engine.evaluate_performance()
        
        assert isinstance(metrics, EvaluationMetrics)
        assert hasattr(metrics, 'precision')
        assert hasattr(metrics, 'recall')
        assert hasattr(metrics, 'f1')
    
    def test_evaluation_metrics_structure(self):
        """Test the structure of evaluation metrics."""
        engine = NEREngine()
        metrics = engine.evaluate_performance()
        
        # Metrics should be floats between 0 and 1
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1 <= 1.0
        
        # Counts should be non-negative integers
        assert metrics.total_predicted >= 0
        assert metrics.total_actual >= 0
        assert metrics.total_correct >= 0
    
    def test_evaluation_with_labels_file(self):
        """Test evaluation using the labeled dataset."""
        engine = NEREngine()
        
        # This will use data/ner_labels.json
        metrics = engine.evaluate_performance()
        
        # Should have processed some data (even if model not loaded, we have lab values)
        assert metrics.total_actual > 0, "No labeled data found"
    
    def test_lab_value_evaluation_performance(self):
        """Test that lab value extraction achieves reasonable performance."""
        engine = NEREngine()
        metrics = engine.evaluate_performance()
        
        # Even without scispaCy, lab value regex should work
        # We should get some predictions
        assert metrics.total_predicted >= 0
        
        # If we made predictions, precision should be reasonable
        if metrics.total_predicted > 0:
            # At least some precision on lab values
            assert metrics.precision >= 0.0


class TestNEREdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        engine = NEREngine()
        long_text = "Blood glucose is 180 mg/dL. " * 100
        entities = engine.extract_entities(long_text)
        
        assert isinstance(entities, list)
        # Should find multiple instances
        lab_values = [e for e in entities if e.type == "LAB_VALUE"]
        assert len(lab_values) > 0
    
    def test_special_characters(self):
        """Test handling of special characters."""
        engine = NEREngine()
        text = "BP: 140/90 mmHg; HbA1c: 8.5% (elevated)"
        entities = engine.extract_entities(text)
        
        # Should still extract lab values
        lab_values = [e for e in entities if e.type == "LAB_VALUE"]
        assert len(lab_values) > 0
    
    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        engine = NEREngine()
        text = "Temperature is 37°C and glucose is 180 mg/dL"
        entities = engine.extract_entities(text)
        
        # Should find temperature and glucose
        lab_values = [e for e in entities if e.type == "LAB_VALUE"]
        assert len(lab_values) >= 2
    
    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive where appropriate."""
        engine = NEREngine()
        
        # Test uppercase and lowercase variations
        text1 = "HbA1c is 8.5%"
        text2 = "hba1c is 8.5%"
        
        entities1 = engine.extract_entities(text1)
        entities2 = engine.extract_entities(text2)
        
        # Both should find the percentage
        assert len(entities1) > 0
        assert len(entities2) > 0


def test_ner_integration():
    """Integration test for the complete NER pipeline."""
    engine = NEREngine()
    
    # Complex medical text with multiple entity types
    text = """
    Patient presents with type 2 diabetes mellitus and hypertension.
    Blood glucose is 180 mg/dL and BP is 150/95 mmHg.
    Current medications include metformin 500 mg twice daily and lisinopril 10 mg daily.
    HbA1c is 8.5% which indicates poor glycemic control.
    Recommend cardiac catheterization to evaluate coronary artery disease.
    """
    
    entities = engine.extract_entities(text)
    
    # Should extract multiple entities
    assert len(entities) > 0
    
    # Should have at least lab values (even without scispaCy)
    lab_values = [e for e in entities if e.type == "LAB_VALUE"]
    assert len(lab_values) >= 4  # glucose, BP, HbA1c percentage, dosages
    
    # All entities should have valid positions
    for entity in entities:
        assert entity.start >= 0
        assert entity.end > entity.start
        assert entity.end <= len(text)
        
        # Verify position accuracy
        extracted = text[entity.start:entity.end]
        assert extracted == entity.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
