"""
Comprehensive tests for the text masking and number safety system.

Tests all critical functionality for preserving medical values during text
processing, including edge cases and validation.
"""

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from simplifier.masking import (
    MaskingEngine,
    mask_text,
    restore_text,
    validate_protected_values,
    get_protected_values
)


class TestMaskingEngine:
    """Test the MaskingEngine class functionality."""
    
    def setup_method(self):
        """Set up test instance."""
        self.engine = MaskingEngine()
    
    def test_basic_dosage_masking(self):
        """Test masking of basic dosage patterns."""
        text = "Take 500 mg twice daily with food."
        masked_text, placeholder_map = self.engine.mask_text(text)
        
        # Should mask both the dosage and frequency
        assert "500 mg" not in masked_text
        assert "twice daily" not in masked_text
        assert "__PROTECTED_001__" in masked_text
        assert "__PROTECTED_002__" in masked_text
        
        # Verify placeholder map
        assert len(placeholder_map) == 2
        assert "500 mg" in placeholder_map.values()
        assert "twice daily" in placeholder_map.values()
        
        # Test restoration
        restored_text = self.engine.restore_text(masked_text, placeholder_map)
        assert restored_text == text
    
    def test_complex_dosages(self):
        """Test various dosage formats."""
        test_cases = [
            "2.5 mg tablet",
            "10-20 mg daily", 
            "1/2 tablet",
            "500 mcg",
            "1000 units",
            "5 ml",
            "2 capsules",
            "3 drops"
        ]
        
        for dosage in test_cases:
            text = f"Patient should take {dosage} as prescribed."
            masked_text, placeholder_map = self.engine.mask_text(text)
            
            assert dosage not in masked_text, f"Failed to mask: {dosage}"
            assert len(placeholder_map) >= 1
            
            restored_text = self.engine.restore_text(masked_text, placeholder_map)
            assert restored_text == text, f"Failed to restore: {dosage}"
    
    def test_frequency_patterns(self):
        """Test frequency pattern recognition."""
        frequencies = [
            "once daily",
            "twice daily", 
            "three times daily",
            "every 8 hours",
            "every 12 hours",
            "bid",
            "tid", 
            "qid",
            "q6h",
            "q12h",
            "prn",
            "as needed",
            "2 times per day",
            "once weekly"
        ]
        
        for freq in frequencies:
            text = f"Take medication {freq}."
            masked_text, placeholder_map = self.engine.mask_text(text)
            
            assert freq not in masked_text, f"Failed to mask frequency: {freq}"
            assert len(placeholder_map) >= 1
            
            restored_text = self.engine.restore_text(masked_text, placeholder_map)
            assert restored_text == text, f"Failed to restore frequency: {freq}"
    
    def test_lab_value_patterns(self):
        """Test lab value pattern recognition."""
        lab_values = [
            "HbA1c 7.2%",
            "A1C 6.8%", 
            "BP 120/80",
            "blood pressure 130/85 mmHg",
            "glucose 150 mg/dL",
            "cholesterol 200 mg/dL",
            "LDL 120 mg/dL",
            "HDL 45 mg/dL",
            "creatinine 1.2 mg/dL",
            "TSH 2.5 mIU/L",
            "hemoglobin 12.5 g/dL"
        ]
        
        for lab_value in lab_values:
            text = f"Your {lab_value} indicates normal range."
            masked_text, placeholder_map = self.engine.mask_text(text)
            
            assert lab_value not in masked_text, f"Failed to mask lab value: {lab_value}"
            assert len(placeholder_map) >= 1
            
            restored_text = self.engine.restore_text(masked_text, placeholder_map)
            assert restored_text == text, f"Failed to restore lab value: {lab_value}"
    
    def test_unit_patterns(self):
        """Test standalone unit recognition."""
        text = "The measurement was in mg/dL units with mmHg pressure readings."
        masked_text, placeholder_map = self.engine.mask_text(text)
        
        assert "mg/dL" not in masked_text
        assert "mmHg" not in masked_text
        assert len(placeholder_map) >= 2
        
        restored_text = self.engine.restore_text(masked_text, placeholder_map)
        assert restored_text == text
    
    def test_numeric_ranges(self):
        """Test numeric range patterns."""
        ranges = [
            "120-140 mg",
            "2.5-5.0 units", 
            "100-150 mg/dL",
            "10-20%"
        ]
        
        for range_val in ranges:
            text = f"Target range is {range_val}."
            masked_text, placeholder_map = self.engine.mask_text(text)
            
            assert range_val not in masked_text, f"Failed to mask range: {range_val}"
            
            restored_text = self.engine.restore_text(masked_text, placeholder_map)
            assert restored_text == text
    
    def test_overlapping_patterns(self):
        """Test handling of overlapping patterns (should keep longest match)."""
        text = "Take 500 mg tablets twice daily."
        masked_text, placeholder_map = self.engine.mask_text(text)
        
        # Should mask "500 mg" and "twice daily" but not duplicate "mg"
        protected_values = self.engine.get_protected_values(text)
        
        # Verify no overlapping values
        for i, val1 in enumerate(protected_values):
            for j, val2 in enumerate(protected_values):
                if i != j:
                    assert not (val1 in val2 or val2 in val1), \
                        f"Overlapping values detected: '{val1}' and '{val2}'"
    
    def test_special_characters(self):
        """Test handling of special characters in medical values."""
        text = "Blood pressure is 120/80 mmHg, glucose 150.5 mg/dL."
        masked_text, placeholder_map = self.engine.mask_text(text)
        
        protected_values = self.engine.get_protected_values(text)
        
        # Should include values with special characters
        assert any("120/80" in val for val in protected_values)
        assert any("150.5" in val for val in protected_values)
        
        restored_text = self.engine.restore_text(masked_text, placeholder_map)
        assert restored_text == text
    
    def test_case_insensitive_matching(self):
        """Test case insensitive pattern matching."""
        text = "Take 500 MG TWICE DAILY and check HBA1C 7.2%."
        masked_text, placeholder_map = self.engine.mask_text(text)
        
        assert "MG" not in masked_text
        assert "TWICE DAILY" not in masked_text
        assert "HBA1C" not in masked_text
        
        restored_text = self.engine.restore_text(masked_text, placeholder_map)
        assert restored_text == text
    
    def test_empty_text(self):
        """Test handling of empty or whitespace text."""
        empty_texts = ["", " ", "\n", "\t"]
        
        for empty_text in empty_texts:
            masked_text, placeholder_map = self.engine.mask_text(empty_text)
            assert masked_text == empty_text
            assert placeholder_map == {}
    
    def test_text_without_protected_values(self):
        """Test text with no medical values."""
        text = "This is just regular text with no medical information."
        masked_text, placeholder_map = self.engine.mask_text(text)
        
        assert masked_text == text
        assert placeholder_map == {}
    
    def test_validation_success(self):
        """Test successful validation of protected values."""
        original_text = "Take 500 mg twice daily."
        
        # Process through masking/restoration
        masked_text, placeholder_map = self.engine.mask_text(original_text)
        restored_text = self.engine.restore_text(masked_text, placeholder_map)
        
        # Should not raise any exception
        self.engine.validate_protected_values(original_text, restored_text)
    
    def test_validation_failure_missing_value(self):
        """Test validation failure when values are missing."""
        original_text = "Take 500 mg twice daily."
        modified_text = "Take medication as prescribed."  # Values removed
        
        with pytest.raises(ValueError) as excinfo:
            self.engine.validate_protected_values(original_text, modified_text)
        
        assert "Missing values" in str(excinfo.value)
        assert "500 mg" in str(excinfo.value)
        assert "twice daily" in str(excinfo.value)
    
    def test_validation_failure_added_value(self):
        """Test validation failure when values are added."""
        original_text = "Take medication as prescribed."
        modified_text = "Take 500 mg twice daily."  # Values added
        
        with pytest.raises(ValueError) as excinfo:
            self.engine.validate_protected_values(original_text, modified_text)
        
        assert "Added values" in str(excinfo.value)
        assert "500 mg" in str(excinfo.value)
        assert "twice daily" in str(excinfo.value)
    
    def test_validation_failure_changed_value(self):
        """Test validation failure when values are changed."""
        original_text = "Take 500 mg twice daily."
        modified_text = "Take 250 mg once daily."  # Values changed
        
        with pytest.raises(ValueError) as excinfo:
            self.engine.validate_protected_values(original_text, modified_text)
        
        error_str = str(excinfo.value)
        # Should have both missing and added values
        assert "Missing values" in error_str or "Added values" in error_str
    
    def test_get_protected_values(self):
        """Test extraction of protected values list."""
        text = "Take 500 mg twice daily, check HbA1c 7.2%."
        protected_values = self.engine.get_protected_values(text)
        
        assert len(protected_values) == 3
        assert "500 mg" in protected_values
        assert "twice daily" in protected_values
        assert "HbA1c 7.2%" in protected_values
    
    def test_complex_medical_text(self):
        """Test complex medical text with multiple value types."""
        text = (
            "Patient should take metformin 500 mg twice daily with meals. "
            "Check blood pressure weekly - target is below 130/80 mmHg. "
            "Latest HbA1c was 7.2% which is improved from 8.5%. "
            "Adjust insulin from 10 units to 15 units if glucose exceeds 180 mg/dL. "
            "Follow up in 3 months or as needed."
        )
        
        masked_text, placeholder_map = self.engine.mask_text(text)
        
        # Verify multiple values are masked
        protected_values = self.engine.get_protected_values(text)
        assert len(protected_values) >= 6  # Multiple dosages, lab values, etc.
        
        # Verify no protected values remain in masked text
        for value in protected_values:
            assert value not in masked_text, f"Value not masked: {value}"
        
        # Verify complete restoration
        restored_text = self.engine.restore_text(masked_text, placeholder_map)
        assert restored_text == text
        
        # Verify validation passes
        self.engine.validate_protected_values(text, restored_text)


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_mask_text_function(self):
        """Test mask_text convenience function."""
        text = "Take 500 mg daily."
        masked_text, placeholder_map = mask_text(text)
        
        assert "500 mg" not in masked_text
        assert len(placeholder_map) >= 1
    
    def test_restore_text_function(self):
        """Test restore_text convenience function."""
        text = "Take 500 mg daily."
        masked_text, placeholder_map = mask_text(text)
        restored_text = restore_text(masked_text, placeholder_map)
        
        assert restored_text == text
    
    def test_validate_protected_values_function(self):
        """Test validate_protected_values convenience function."""
        original = "Take 500 mg daily."
        processed = "Take 500 mg daily."
        
        # Should not raise exception
        validate_protected_values(original, processed)
    
    def test_get_protected_values_function(self):
        """Test get_protected_values convenience function."""
        text = "Take 500 mg twice daily."
        values = get_protected_values(text)
        
        assert "500 mg" in values
        assert "twice daily" in values


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        # Create long text with scattered protected values
        base_text = "Patient information. " * 100
        medical_text = f"{base_text} Take 500 mg daily. {base_text} HbA1c 7.2%. {base_text}"
        
        engine = MaskingEngine()
        masked_text, placeholder_map = engine.mask_text(medical_text)
        restored_text = engine.restore_text(masked_text, placeholder_map)
        
        assert restored_text == medical_text
        engine.validate_protected_values(medical_text, restored_text)
    
    def test_unicode_characters(self):
        """Test handling of unicode characters near protected values."""
        text = "Dosage: 500 mg → twice daily (μg/mL)"
        engine = MaskingEngine()
        
        masked_text, placeholder_map = engine.mask_text(text)
        restored_text = engine.restore_text(masked_text, placeholder_map)
        
        assert restored_text == text
    
    def test_multiple_spaces_and_formatting(self):
        """Test handling of irregular spacing and formatting."""
        text = "Take   500  mg    twice   daily  ."
        engine = MaskingEngine()
        
        protected_values = engine.get_protected_values(text)
        # Should still recognize values despite irregular spacing
        assert any("500" in val for val in protected_values)
        assert any("twice" in val for val in protected_values)
    
    def test_boundary_punctuation(self):
        """Test values at sentence boundaries with punctuation."""
        text = "Medication: 500mg. Frequency: twice daily!"
        engine = MaskingEngine()
        
        masked_text, placeholder_map = engine.mask_text(text)
        restored_text = engine.restore_text(masked_text, placeholder_map)
        
        assert restored_text == text
        engine.validate_protected_values(text, restored_text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])