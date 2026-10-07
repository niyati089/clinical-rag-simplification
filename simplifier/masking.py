"""
Text masking system for preserving critical medical values during text processing.

This module implements safety-critical functionality to ensure that dosages,
lab values, units, and frequencies are preserved exactly during text
simplification. All protected values are masked with placeholders during
processing and restored afterward with validation.
"""

import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass


@dataclass
class ProtectedValue:
    """A medical value that must be preserved exactly."""
    text: str
    start: int
    end: int
    value_type: str


class MaskingEngine:
    """
    Engine for masking and restoring protected medical values in text.
    
    This class handles the critical safety requirement of preserving all
    medical values exactly as they appear in the original text.
    """
    
    def __init__(self):
        self.placeholder_prefix = "__PROTECTED_"
        self.placeholder_suffix = "__"
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for different types of medical values."""
        
        # Dosage patterns: "500 mg", "2.5 tablets", "10-20 mg", "1/2 tablet"
        self.dosage_pattern = re.compile(
            r'\b(?:\d+(?:[./]\d+)?(?:-\d+(?:[./]\d+)?)?)\s*(?:mg|g|ml|l|mcg|µg|units?|iu|tablet[s]?|capsule[s]?|drop[s]?|pill[s]?)\b',
            re.IGNORECASE
        )
        
        # Frequency patterns: "twice daily", "every 8 hours", "bid", "q6h"
        self.frequency_pattern = re.compile(
            r'\b(?:'
            r'(?:once\s+daily|twice\s+daily|three\s+times?\s+daily|four\s+times?\s+daily)|'
            r'(?:once|twice|three times?|four times?)\s+(?:daily|weekly|monthly)|'
            r'(?:every\s+\d+\s+(?:hour[s]?|day[s]?|week[s]?))|'
            r'(?:bid|tid|qid|qd|q\d+h|prn|as needed)|'
            r'(?:\d+\s*times?\s*(?:per\s+)?(?:day|week|month))|'
            r'(?:daily|weekly|monthly)'
            r')\b',
            re.IGNORECASE
        )
        
        # Lab value patterns: "HbA1c 7.2%", "BP 120/80 mmHg", "glucose 150 mg/dL"
        # Also captures thyroid-specific: "Free T3: 4.2 pmol/L", "TSH: 2.5 mIU/L"
        self.lab_value_pattern = re.compile(
            r'(?:'
            r'(?:HbA1c|HBA1C|A1C)\s+\d+(?:\.\d+)?%|'
            r'(?:BP|blood\s+pressure)\s+\d+/\d+(?:\s*mmHg)?|'
            r'\d+/\d+\s*mmHg|'  # Standalone BP readings like "120/80 mmHg"
            r'(?:glucose|sugar)\s+\d+(?:\.\d+)?(?:\s*mg/dL|\s*mmol/L)?|'
            r'(?:cholesterol|LDL|HDL)\s+\d+(?:\.\d+)?(?:\s*mg/dL|\s*mmol/L)?|'
            r'(?:creatinine)\s+\d+(?:\.\d+)?(?:\s*mg/dL|\s*µmol/L)?|'
            r'(?:Free\s+T3|Free\s+T4|FT3|FT4|fT3|fT4)\s*[:\-]?\s*\d+(?:\.\d+)?(?:\s*pmol/L|\s*pg/mL|\s*nmol/L)?|'
            r'(?:TSH)\s*[:\-]?\s*\d+(?:\.\d+)?(?:\s*mIU/L|\s*µIU/mL|\s*mlU/L)?|'
            r'(?:hemoglobin|Hgb|Hb)\s+\d+(?:\.\d+)?(?:\s*g/dL)?'
            r')',
            re.IGNORECASE
        )
        
        # Measurement units that might appear standalone: "mg/dL", "mmHg", "mEq/L"
        # Simple pattern without complex lookbehinds
        self.unit_pattern = re.compile(
            r'\b(?:'
            r'mg/dL|mmol/L|µmol/L|mEq/L|mmHg|mIU/L|g/dL|'
            r'mg|mcg|µg|ml|units?|iu'
            r')(?=\s|$|\.|,)',
            re.IGNORECASE
        )
        
        # Numeric ranges and values with units: "120-140 mg", "2.5-5.0 units", "180 mg/dL"
        # Also captures standalone blood pressure readings and percentage ranges
        self.numeric_range_pattern = re.compile(
            r'(?:'
            r'\d+/\d+(?:\s*mmHg)?|'  # Blood pressure readings like "120/80" or "120/80 mmHg"
            r'\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s*(?:'
            r'mg/dL|mmol/L|µmol/L|mEq/L|mmHg|mIU/L|g/dL|'
            r'mg|g|ml|l|mcg|µg|units?|iu|%'
            r')'
            r')',
            re.IGNORECASE
        )
    
    def _extract_protected_values(self, text: str) -> List[ProtectedValue]:
        """Extract all protected values from text with their positions."""
        values = []
        
        # Find all matches for each pattern type - ORDER MATTERS (longest matches first)
        patterns = [
            (self.lab_value_pattern, "LAB_VALUE"),  # Check lab values first (longer matches)
            (self.numeric_range_pattern, "NUMERIC_RANGE"),
            (self.dosage_pattern, "DOSAGE"),
            (self.frequency_pattern, "FREQUENCY"),
            (self.unit_pattern, "UNIT")  # Check units last (shortest matches)
        ]
        
        for pattern, value_type in patterns:
            for match in pattern.finditer(text):
                values.append(ProtectedValue(
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    value_type=value_type
                ))
        
        # Sort by position and remove overlaps (keep the longest match)
        values.sort(key=lambda x: (x.start, -len(x.text)))
        filtered_values = []
        last_end = -1
        
        for value in values:
            if value.start >= last_end:
                filtered_values.append(value)
                last_end = value.end
        
        return filtered_values
    
    def mask_text(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Mask protected values in text with placeholders.
        
        Args:
            text: Input text containing medical values
            
        Returns:
            Tuple of (masked_text, placeholder_map) where placeholder_map
            maps placeholders back to original values
        """
        protected_values = self._extract_protected_values(text)
        
        if not protected_values:
            return text, {}
        
        # Sort by position in reverse order for replacement
        protected_values.sort(key=lambda x: x.start, reverse=True)
        
        masked_text = text
        placeholder_map = {}
        
        for i, value in enumerate(protected_values):
            placeholder = f"{self.placeholder_prefix}{i+1:03d}{self.placeholder_suffix}"
            placeholder_map[placeholder] = value.text
            
            # Replace the value with placeholder
            masked_text = (
                masked_text[:value.start] +
                placeholder +
                masked_text[value.end:]
            )
        
        return masked_text, placeholder_map
    
    def restore_text(self, masked_text: str, placeholder_map: Dict[str, str]) -> str:
        """
        Restore original protected values from masked text.
        
        Args:
            masked_text: Text with placeholders
            placeholder_map: Mapping from placeholders to original values
            
        Returns:
            Text with original protected values restored
        """
        restored_text = masked_text
        
        for placeholder, original_value in placeholder_map.items():
            restored_text = restored_text.replace(placeholder, original_value)
        
        return restored_text
    
    def get_protected_values(self, text: str) -> List[str]:
        """
        Extract list of protected values from text.
        
        Args:
            text: Input text
            
        Returns:
            List of protected value strings
        """
        protected_values = self._extract_protected_values(text)
        return [value.text for value in protected_values]
    
    def validate_protected_values(self, original_text: str, restored_text: str) -> None:
        """
        Validate that protected values are preserved exactly.
        
        Args:
            original_text: Original input text
            restored_text: Text after processing and restoration
            
        Raises:
            ValueError: If protected values don't match exactly
        """
        original_values = set(self.get_protected_values(original_text))
        restored_values = set(self.get_protected_values(restored_text))
        
        if original_values != restored_values:
            missing = original_values - restored_values
            added = restored_values - original_values
            
            error_msg = "Protected value validation failed!\n"
            if missing:
                error_msg += f"Missing values: {sorted(missing)}\n"
            if added:
                error_msg += f"Added values: {sorted(added)}\n"
            
            raise ValueError(error_msg.strip())


# Global instance for easy access
masking_engine = MaskingEngine()


def mask_text(text: str) -> Tuple[str, Dict[str, str]]:
    """Convenience function for masking text."""
    return masking_engine.mask_text(text)


def restore_text(masked_text: str, placeholder_map: Dict[str, str]) -> str:
    """Convenience function for restoring text."""
    return masking_engine.restore_text(masked_text, placeholder_map)


def validate_protected_values(original_text: str, restored_text: str) -> None:
    """Convenience function for validation."""
    masking_engine.validate_protected_values(original_text, restored_text)


def get_protected_values(text: str) -> List[str]:
    """Convenience function to get protected values."""
    return masking_engine.get_protected_values(text)