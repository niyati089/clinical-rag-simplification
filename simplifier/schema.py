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


# RAG Pipeline Integration Models
# These models support the simplify_rag_result() function

class ReferenceItem(BaseModel):
    """Retrieved reference item from RAG pipeline."""
    # This model matches whatever structure Person A's RAG pipeline produces
    # We keep it flexible to accept any dict structure
    pass

    class Config:
        extra = "allow"  # Allow any additional fields from RAG pipeline


class ProtectedValuesCheck(BaseModel):
    """Result of protected values validation across all chunks."""
    passed: bool = Field(description="True if all protected values are preserved")
    missing: List[str] = Field(
        default_factory=list,
        description="List of protected values that were lost during simplification"
    )


class ChunkScores(BaseModel):
    """Readability scores for original and simplified text in a chunk."""
    original: ReadabilityScore = Field(description="Metrics for the original chunk text")
    simplified: ReadabilityScore = Field(description="Metrics for the simplified output text")


class SimplifiedOutputB(BaseModel):
    """Simplified output sections for a single chunk (matches RAG pipeline format)."""
    simple_explanation: str = Field(description="Main content explanation")
    important_instructions: List[str] = Field(
        default_factory=list, 
        description="Critical instructions as list items"
    )
    medication_guidance: List[str] = Field(
        default_factory=list,
        description="Medication-related information as list items"
    )
    follow_up: List[str] = Field(
        default_factory=list,
        description="Follow-up care instructions as list items"
    )


class ChunkResultB(BaseModel):
    """Single chunk result for RAG pipeline integration."""
    # Original fields from RAG pipeline - preserved exactly
    chunk_id: str = Field(description="Unique identifier for this chunk")
    original_chunk: str = Field(description="Original text content")
    section_title: str = Field(description="Section title from source document")
    chunk_index: int = Field(description="Index of this chunk in the document")
    simplified_output: SimplifiedOutputB = Field(description="Processed simplified content")
    retrieved_references: List[Dict] = Field(
        default_factory=list,
        description="References retrieved by RAG pipeline (preserved as-is)"
    )
    
    # New fields added by simplify_rag_result
    entities: List[EntityModel] = Field(description="Entities extracted from this chunk")
    glossary: List[GlossaryEntry] = Field(description="Terms defined for this chunk")
    highlights: List[Highlight] = Field(description="Important sentences to emphasize")
    scores: ChunkScores = Field(description="Before/after readability comparison")


class RagResultB(BaseModel):
    """Complete RAG result with simplification enhancements."""
    # Original top-level fields from RAG pipeline - preserved exactly
    reading_level: str = Field(description="Original reading level from RAG pipeline")
    total_chunks: int = Field(description="Total number of chunks processed")
    disclaimer: str = Field(description="Disclaimer text from RAG pipeline")
    results: List[ChunkResultB] = Field(description="Enhanced chunk results")
    
    # New top-level fields added by simplify_rag_result
    glossary: List[GlossaryEntry] = Field(
        description="Combined and deduplicated glossary across all chunks"
    )
    scores: ReadabilityScores = Field(description="Aggregate readability metrics")
    protected_values_check: ProtectedValuesCheck = Field(
        description="Validation that protected values are preserved"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            float: lambda v: round(v, 2)  # Round floats to 2 decimal places in JSON
        }