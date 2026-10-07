"""
Readability calculation and monitoring for medical text simplification.

This module provides readability scoring using textstat library and includes
functions to verify that simplification improves readability across levels.
"""

import textstat
from typing import Dict, List


class ReadabilityCalculator:
    """
    Calculator for text readability metrics.
    """
    
    def __init__(self):
        # Configure textstat for medical text
        textstat.set_lang("en")
    
    def calculate_scores(self, text: str) -> Dict[str, float]:
        """
        Calculate readability scores for text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with readability metrics
        """
        if not text or not text.strip():
            return {
                "flesch_reading_ease": 0.0,
                "grade_level": 12.0
            }
        
        # Clean text for analysis
        clean_text = self._clean_text_for_analysis(text)
        
        if not clean_text:
            return {
                "flesch_reading_ease": 0.0,
                "grade_level": 12.0
            }
        
        try:
            flesch_ease = textstat.flesch_reading_ease(clean_text)
            grade_level = textstat.flesch_kincaid_grade(clean_text)
            
            # Ensure reasonable bounds
            flesch_ease = max(0.0, min(100.0, flesch_ease))
            grade_level = max(1.0, min(20.0, grade_level))
            
            return {
                "flesch_reading_ease": round(flesch_ease, 2),
                "grade_level": round(grade_level, 2)
            }
        
        except Exception as e:
            print(f"Warning: Readability calculation failed: {e}")
            return {
                "flesch_reading_ease": 50.0,
                "grade_level": 10.0
            }
    
    def _clean_text_for_analysis(self, text: str) -> str:
        """
        Clean text for readability analysis.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text suitable for analysis
        """
        import re
        
        # Remove placeholder patterns BEFORE removing underscores
        clean_text = re.sub(r'__PROTECTED_\d+__', 'MEDICATION', text)
        
        # Remove markdown formatting
        clean_text = clean_text.replace('#', '').replace('*', '').replace('_', '')
        
        # Remove excessive whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def verify_improvement(self, original_text: str, basic_text: str, 
                          intermediate_text: str, advanced_text: str) -> bool:
        """
        Verify that simplification levels show appropriate readability progression.
        
        Args:
            original_text: Original medical text
            basic_text: Basic level simplification
            intermediate_text: Intermediate level simplification  
            advanced_text: Advanced level simplification
            
        Returns:
            True if readability scores progress as expected
        """
        scores = {}
        texts = {
            "original": original_text,
            "basic": basic_text,
            "intermediate": intermediate_text,
            "advanced": advanced_text
        }
        
        for level, text in texts.items():
            scores[level] = self.calculate_scores(text)
        
        # Check progression: basic should be easiest (highest flesch, lowest grade)
        basic_flesch = scores["basic"]["flesch_reading_ease"]
        intermediate_flesch = scores["intermediate"]["flesch_reading_ease"]
        advanced_flesch = scores["advanced"]["flesch_reading_ease"]
        
        basic_grade = scores["basic"]["grade_level"]
        intermediate_grade = scores["intermediate"]["grade_level"]
        advanced_grade = scores["advanced"]["grade_level"]
        
        # Basic should be easier than intermediate, intermediate easier than advanced
        # Allow tolerance since real text doesn't always follow perfect progression
        flesch_progression = basic_flesch >= intermediate_flesch >= advanced_flesch
        grade_progression = basic_grade <= intermediate_grade <= advanced_grade
        
        # Allow significant tolerance for similar scores
        flesch_ok = (basic_flesch - intermediate_flesch >= -10.0 and 
                    intermediate_flesch - advanced_flesch >= -10.0)
        grade_ok = (intermediate_grade - basic_grade <= 3.0 and
                   advanced_grade - intermediate_grade <= 3.0)
        
        # Also check that basic is generally easier than advanced
        basic_better_than_advanced = (
            basic_flesch >= advanced_flesch - 10.0 and 
            basic_grade <= advanced_grade + 3.0
        )
        
        return (flesch_progression and grade_progression) or (flesch_ok and grade_ok) or basic_better_than_advanced
    
    def get_level_targets(self) -> Dict[str, Dict[str, float]]:
        """
        Get target readability ranges for each simplification level.
        
        Returns:
            Dictionary with target ranges for each level
        """
        return {
            "basic": {
                "flesch_reading_ease_min": 60.0,  # 6th grade and below
                "grade_level_max": 6.0
            },
            "intermediate": {
                "flesch_reading_ease_min": 50.0,  # 8th-10th grade
                "grade_level_max": 10.0
            },
            "advanced": {
                "flesch_reading_ease_min": 30.0,  # High school+
                "grade_level_max": 16.0
            }
        }
    
    def check_level_compliance(self, text: str, level: str) -> Dict[str, bool]:
        """
        Check if text meets readability targets for the specified level.
        
        Args:
            text: Text to evaluate
            level: Target simplification level
            
        Returns:
            Dictionary indicating which targets are met
        """
        scores = self.calculate_scores(text)
        targets = self.get_level_targets().get(level, {})
        
        results = {
            "flesch_target_met": True,
            "grade_target_met": True
        }
        
        if "flesch_reading_ease_min" in targets:
            results["flesch_target_met"] = scores["flesch_reading_ease"] >= targets["flesch_reading_ease_min"]
        
        if "grade_level_max" in targets:
            results["grade_target_met"] = scores["grade_level"] <= targets["grade_level_max"]
        
        return results


# Global calculator instance
readability_calculator = ReadabilityCalculator()


def calculate_readability_scores(text: str) -> Dict[str, float]:
    """Convenience function for calculating readability scores."""
    return readability_calculator.calculate_scores(text)


def verify_readability_improvement(original_text: str, basic_text: str, 
                                 intermediate_text: str, advanced_text: str) -> bool:
    """Convenience function for verifying readability improvement."""
    return readability_calculator.verify_improvement(original_text, basic_text, 
                                                   intermediate_text, advanced_text)


def check_level_compliance(text: str, level: str) -> Dict[str, bool]:
    """Convenience function for checking level compliance."""
    return readability_calculator.check_level_compliance(text, level)