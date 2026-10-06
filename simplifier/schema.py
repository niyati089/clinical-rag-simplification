"""
Pydantic models for the medical text simplification API schema.

This module defines the complete JSON-serializable output schema for the
simplify() function, ensuring type safety and validation for downstream
FastAPI integration.
"""

from typing import List, Dict, Union, Literal
from pydantic import BaseModel, Field


class EntityModel(BaseModel):
    """Named entity recognized in the medical text."""
    text: str = Field(description="The entity text as it appears in the document")
    type: Literal["DISEASE", "DRUG", "PROCEDURE", "LAB_VALUE", "ABBREVIATION"] = Field(
        description="Entity type classification"
    )
    start: int = Field(description="Character start position in the simplified text")
    end: int = Field(description="Character end position in the simplified text")


class GlossaryEntry(BaseModel):
    """Glossary definition for a medical term."""
    term: str = Field(description="The medical term being defined")
    type: str = Field(description="Entity type (DISEASE, DRUG, etc.)")
    explanation: str = Field(description="Plain-language explanation of the term")


class Highlight(BaseModel):
    """Important sentence that should be visually highlighted."""
    section: str = Field(description="Section containing the sentence")
    sentence: str = Field(description="Full sentence text to highlight")


class ReadabilityScore(BaseModel):
    """Readability metrics for text."""
    flesch_reading_ease: float = Field(description="Flesch Reading Ease score (0-100, higher = easier)")
    grade_level: float = Field(description="Flesch-Kincaid grade level")


class ReadabilityScores(BaseModel):
    """Readability comparison between original and simplified text."""
    original: ReadabilityScore = Field(description="Metrics for the original text")
    simplified: ReadabilityScore = Field(description="Metrics for the simplified text")


class SimplifierOutput(BaseModel):
    """Complete output schema for the text simplification API."""
    level: Literal["basic", "intermediate", "advanced"] = Field(
        description="Simplification level applied"
    )
    sections: Dict[str, str] = Field(
        description="Text organized by sections (simple_explanation, important_instructions, medication_guidance, follow_up)"
    )
    simplified_text: str = Field(
        description="Complete simplified text in markdown format"
    )
    entities: List[EntityModel] = Field(
        description="Named entities extracted from the text"
    )
    glossary: List[GlossaryEntry] = Field(
        description="Alphabetically sorted glossary of terms with explanations"
    )
    highlights: List[Highlight] = Field(
        description="Important sentences to visually emphasize"
    )
    scores: ReadabilityScores = Field(
        description="Readability metrics comparison"
    )
    protected_values: List[str] = Field(
        description="Dosages, units, frequencies, and lab values preserved exactly from input"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            float: lambda v: round(v, 2)  # Round floats to 2 decimal places in JSON
        }


# Type alias for the input parameter
TextOrSections = Union[str, Dict[str, str]]

# Standard section keys for structured input
SECTION_KEYS = [
    "simple_explanation",
    "important_instructions", 
    "medication_guidance",
    "follow_up"
]


def create_empty_sections() -> Dict[str, str]:
    """
    Create an empty sections dictionary with all required keys.
    
    Returns:
        Dictionary with all section keys initialized to empty strings
    """
    return {key: "" for key in SECTION_KEYS}


def validate_simplifier_output(output_dict: dict) -> SimplifierOutput:
    """
    Validate and parse a dictionary into a SimplifierOutput model.
    
    Args:
        output_dict: Dictionary to validate
        
    Returns:
        Validated SimplifierOutput instance
        
    Raises:
        ValidationError: If the dictionary doesn't match the schema
    """
    return SimplifierOutput(**output_dict)