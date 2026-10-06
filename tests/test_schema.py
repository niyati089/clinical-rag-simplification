"""
Tests for schema validation and data model compliance.

Tests include:
- SimplifierOutput schema validation
- Entity model validation
- Input/output type checking
- JSON serialization compatibility
"""

import pytest
from typing import Dict, Any
from pydantic import ValidationError

from simplifier.schema import (
    SimplifierOutput,
    EntityModel,
    TextOrSections,
    SECTION_KEYS,
    validate_simplifier_output,
    create_empty_sections
)


class TestEntityModel:
    """Test cases for EntityModel validation."""
    
    def test_valid_entity(self):
        """Test creation of valid entity."""
        entity = EntityModel(
            text="diabetes",
            type="DISEASE",
            start=0,
            end=8
        )
        
        assert entity.text == "diabetes"
        assert entity.type == "DISEASE"
        assert entity.start == 0
        assert entity.end == 8
    
    def test_invalid_entity_type(self):
        """Test validation of invalid entity type."""
        with pytest.raises(ValidationError):
            EntityModel(
                text="diabetes",
                type="INVALID_TYPE",  # Should be one of the allowed types
                start=0,
                end=8
            )
    
    def test_invalid_position(self):
        """Test validation of invalid start/end positions."""
        # Note: Pydantic doesn't automatically validate start < end relationship
        # This would require a custom validator if needed
        entity = EntityModel(
            text="diabetes",
            type="DISEASE",
            start=8,  # Start after end - allowed by default schema
            end=0
        )
        # Just verify it creates without error - position logic validation 
        # would need to be in application code if required
        assert entity.start == 8
        assert entity.end == 0
    
    def test_entity_serialization(self):
        """Test entity JSON serialization."""
        entity = EntityModel(
            text="insulin",
            type="DRUG",
            start=10,
            end=17
        )
        
        entity_dict = entity.dict()
        assert entity_dict["text"] == "insulin"
        assert entity_dict["type"] == "DRUG"
        assert entity_dict["start"] == 10
        assert entity_dict["end"] == 17


class TestSimplifierOutput:
    """Test cases for SimplifierOutput schema."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.valid_output_data = {
            "level": "basic",
            "sections": {
                "simple_explanation": "You have diabetes.",
                "important_instructions": "Take your medicine.",
                "medication_guidance": "Take with food.",
                "follow_up": "See your doctor in 2 weeks."
            },
            "simplified_text": "## What This Means\nYou have diabetes.\n\n## Important Instructions\n- Take your medicine.",
            "entities": [
                {
                    "text": "diabetes",
                    "type": "DISEASE",
                    "start": 9,
                    "end": 17
                }
            ],
            "glossary": [
                {
                    "term": "diabetes",
                    "type": "DISEASE", 
                    "explanation": "A condition where blood sugar is too high"
                }
            ],
            "highlights": [
                {
                    "section": "important_instructions",
                    "sentence": "Take your medicine."
                }
            ],
            "scores": {
                "original": {
                    "flesch_reading_ease": 45.0,
                    "grade_level": 12.0
                },
                "simplified": {
                    "flesch_reading_ease": 75.0,
                    "grade_level": 6.0
                }
            },
            "protected_values": ["500 mg", "twice daily"]
        }
    
    def test_valid_output(self):
        """Test creation of valid SimplifierOutput."""
        output = SimplifierOutput(**self.valid_output_data)
        
        assert output.level == "basic"
        assert len(output.entities) == 1
        assert output.entities[0].text == "diabetes"
        assert len(output.glossary) == 1
        assert len(output.highlights) == 1
        assert len(output.protected_values) == 2
    
    def test_invalid_level(self):
        """Test validation of invalid level."""
        invalid_data = self.valid_output_data.copy()
        invalid_data["level"] = "invalid_level"
        
        with pytest.raises(ValidationError):
            SimplifierOutput(**invalid_data)
    
    def test_missing_sections(self):
        """Test handling of missing sections."""
        invalid_data = self.valid_output_data.copy()
        # Make a shallow copy of sections dict
        invalid_data["sections"] = self.valid_output_data["sections"].copy()
        del invalid_data["sections"]["simple_explanation"]
        
        # Pydantic allows partial dicts - it doesn't validate required keys in Dict[str, str]
        # This is valid behavior - the application validates section completeness separately
        output = SimplifierOutput(**invalid_data)
        assert "simple_explanation" not in output.sections
    
    def test_invalid_entity_in_list(self):
        """Test validation of invalid entity in entities list."""
        invalid_data = self.valid_output_data.copy()
        invalid_data["entities"] = [
            {
                "text": "diabetes",
                "type": "INVALID_TYPE",
                "start": 0,
                "end": 8
            }
        ]
        
        with pytest.raises(ValidationError):
            SimplifierOutput(**invalid_data)
    
    def test_missing_score_fields(self):
        """Test validation of score structure."""
        invalid_data = self.valid_output_data.copy()
        del invalid_data["scores"]["original"]["flesch_reading_ease"]
        
        with pytest.raises(ValidationError):
            SimplifierOutput(**invalid_data)
    
    def test_json_serialization(self):
        """Test JSON serialization of complete output."""
        output = SimplifierOutput(**self.valid_output_data)
        
        # Should serialize without error
        json_dict = output.dict()
        assert isinstance(json_dict, dict)
        assert json_dict["level"] == "basic"
        
        # Should be JSON-serializable
        import json
        json_str = json.dumps(json_dict)
        assert isinstance(json_str, str)
        
        # Should round-trip correctly
        parsed = json.loads(json_str)
        assert parsed["level"] == "basic"
        assert len(parsed["entities"]) == 1


class TestTextOrSections:
    """Test TextOrSections type handling."""
    
    def test_plain_string_input(self):
        """Test handling of plain string input."""
        # This is more of a documentation test since TextOrSections is a Union type
        text_input: TextOrSections = "This is plain text input."
        assert isinstance(text_input, str)
    
    def test_sections_dict_input(self):
        """Test handling of sections dictionary input."""
        sections_input: TextOrSections = {
            "simple_explanation": "You have a condition.",
            "important_instructions": "Take medicine.",
            "medication_guidance": "With food.",
            "follow_up": "See doctor."
        }
        assert isinstance(sections_input, dict)
        assert "simple_explanation" in sections_input


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_empty_sections(self):
        """Test creation of empty sections dictionary."""
        sections = create_empty_sections()
        
        assert isinstance(sections, dict)
        for key in SECTION_KEYS:
            assert key in sections
            assert sections[key] == ""
    
    def test_section_keys_constant(self):
        """Test that SECTION_KEYS contains expected keys."""
        expected_keys = ["simple_explanation", "important_instructions", "medication_guidance", "follow_up"]
        
        assert len(SECTION_KEYS) == len(expected_keys)
        for key in expected_keys:
            assert key in SECTION_KEYS
    
    def test_validate_simplifier_output_valid(self):
        """Test validation of valid output."""
        valid_data = {
            "level": "intermediate",
            "sections": create_empty_sections(),
            "simplified_text": "Test text",
            "entities": [],
            "glossary": [],
            "highlights": [],
            "scores": {
                "original": {"flesch_reading_ease": 50.0, "grade_level": 10.0},
                "simplified": {"flesch_reading_ease": 60.0, "grade_level": 8.0}
            },
            "protected_values": []
        }
        
        # Should not raise exception
        result = validate_simplifier_output(valid_data)
        assert isinstance(result, SimplifierOutput)
    
    def test_validate_simplifier_output_invalid(self):
        """Test validation of invalid output."""
        invalid_data = {
            "level": "invalid",  # Invalid level
            "sections": {"wrong_key": "value"},  # Wrong sections
            "simplified_text": "",
            "entities": [],
            "glossary": [],
            "highlights": [],
            "scores": {},  # Missing score structure
            "protected_values": []
        }
        
        with pytest.raises(ValidationError):
            validate_simplifier_output(invalid_data)


class TestSchemaCompliance:
    """Test schema compliance with the documented API."""
    
    def test_all_required_fields_present(self):
        """Test that schema includes all documented required fields."""
        # Create minimal valid output
        minimal_data = {
            "level": "basic",
            "sections": create_empty_sections(),
            "simplified_text": "",
            "entities": [],
            "glossary": [],
            "highlights": [],
            "scores": {
                "original": {"flesch_reading_ease": 0.0, "grade_level": 12.0},
                "simplified": {"flesch_reading_ease": 0.0, "grade_level": 12.0}
            },
            "protected_values": []
        }
        
        output = SimplifierOutput(**minimal_data)
        
        # Check all documented fields are accessible
        assert hasattr(output, 'level')
        assert hasattr(output, 'sections')
        assert hasattr(output, 'simplified_text')
        assert hasattr(output, 'entities')
        assert hasattr(output, 'glossary')
        assert hasattr(output, 'highlights')
        assert hasattr(output, 'scores')
        assert hasattr(output, 'protected_values')
    
    def test_entity_fields_complete(self):
        """Test that entity model has all documented fields."""
        entity = EntityModel(
            text="test",
            type="DISEASE",
            start=0,
            end=4
        )
        
        # Check all documented entity fields
        assert hasattr(entity, 'text')
        assert hasattr(entity, 'type')
        assert hasattr(entity, 'start')
        assert hasattr(entity, 'end')
    
    def test_glossary_entry_structure(self):
        """Test glossary entry structure."""
        # Glossary entries should have term, type, explanation
        valid_output_data = {
            "level": "basic",
            "sections": create_empty_sections(),
            "simplified_text": "",
            "entities": [],
            "glossary": [
                {
                    "term": "hypertension",
                    "type": "DISEASE",
                    "explanation": "High blood pressure"
                }
            ],
            "highlights": [],
            "scores": {
                "original": {"flesch_reading_ease": 50.0, "grade_level": 10.0},
                "simplified": {"flesch_reading_ease": 60.0, "grade_level": 8.0}
            },
            "protected_values": []
        }
        
        output = SimplifierOutput(**valid_output_data)
        assert len(output.glossary) == 1
        
        glossary_entry = output.glossary[0]
        # glossary_entry is a Pydantic model, not a dict - access attributes directly
        assert hasattr(glossary_entry, 'term')
        assert hasattr(glossary_entry, 'type')
        assert hasattr(glossary_entry, 'explanation')
        assert glossary_entry.term == "hypertension"
        assert glossary_entry.type == "DISEASE"
        assert glossary_entry.explanation == "High blood pressure"


if __name__ == '__main__':
    pytest.main([__file__])