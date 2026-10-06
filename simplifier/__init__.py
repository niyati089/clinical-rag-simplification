"""
Medical Text Simplification Package

A Python package for simplifying medical text across different reading levels
while preserving critical information like dosages, lab values, and important
instructions. Designed for integration with RAG pipelines and FastAPI backends.

Main API:
    simplify(text_or_sections, level) -> dict

Example:
    >>> from simplifier import simplify
    >>> result = simplify("Your HbA1c is 8.2%. Take 500mg metformin twice daily.", level="basic")
    >>> print(result['simplified_text'])
"""

from .simplify import simplify
from .schema import SimplifierOutput, EntityModel, GlossaryEntry, Highlight, ReadabilityScores
from .masking import mask_text, restore_text, validate_protected_values, get_protected_values
from .ner import NEREngine, extract_entities
from .glossary import generate_glossary, get_simple_replacement
from .readability import calculate_readability_scores, verify_readability_improvement, check_level_compliance
from .layout import format_markdown_output, identify_highlights

__version__ = "0.1.0"
__all__ = [
    "simplify", 
    "SimplifierOutput", 
    "EntityModel", 
    "GlossaryEntry", 
    "Highlight", 
    "ReadabilityScores",
    "mask_text",
    "restore_text", 
    "validate_protected_values",
    "get_protected_values",
    "NEREngine",
    "extract_entities",
    "generate_glossary",
    "get_simple_replacement",
    "calculate_readability_scores",
    "verify_readability_improvement", 
    "check_level_compliance",
    "format_markdown_output",
    "identify_highlights"
]