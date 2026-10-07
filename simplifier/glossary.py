"""
Glossary generation for medical text simplification.

This module provides functionality to automatically generate glossaries from
extracted entities, with deduplication and alphabetical sorting.
"""

import json
import os
from typing import List, Dict, Set, Optional
from .schema import EntityModel, GlossaryEntry


def load_term_map(term_map_file: Optional[str] = None) -> Dict[str, str]:
    """
    Load the curated term mapping dictionary.
    
    Args:
        term_map_file: Path to term mapping JSON file. Defaults to data/term_map.json
        
    Returns:
        Dictionary mapping medical terms to plain-language explanations
    """
    if term_map_file is None:
        term_map_file = "data/term_map.json"
    
    # Make path relative to project root
    if not os.path.isabs(term_map_file):
        project_root = os.path.dirname(os.path.dirname(__file__))
        term_map_file = os.path.join(project_root, term_map_file)
    
    try:
        with open(term_map_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return empty dict if file doesn't exist
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in term map: {e}")
        return {}


def get_simple_replacement(term: str, term_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Get a simple replacement for a medical term from the curated dictionary.
    
    Args:
        term: Medical term to look up
        term_map: Optional term mapping dict (will load from file if not provided)
        
    Returns:
        Plain-language replacement if found, None otherwise
    """
    if term_map is None:
        term_map = load_term_map()
    
    # Try exact match first (case-insensitive)
    term_lower = term.lower().strip()
    for map_term, replacement in term_map.items():
        if map_term.lower() == term_lower:
            return replacement
    
    # Try partial matches for abbreviations or common variations
    for map_term, replacement in term_map.items():
        map_term_lower = map_term.lower()
        # Check if the term is contained within a mapped term or vice versa
        if (term_lower in map_term_lower and len(term_lower) >= 3) or \
           (map_term_lower in term_lower and len(map_term_lower) >= 3):
            return replacement
    
    return None


def generate_glossary(entities: List[EntityModel], llm_explanations: Optional[Dict[str, str]] = None, 
                     term_map: Optional[Dict[str, str]] = None) -> List[GlossaryEntry]:
    """
    Generate an alphabetically sorted glossary from extracted entities.
    
    Args:
        entities: List of entities extracted from the text
        llm_explanations: Optional dictionary of LLM-generated explanations
        term_map: Optional term mapping dict (will load from file if not provided)
        
    Returns:
        List of GlossaryEntry objects, deduplicated and alphabetically sorted
    """
    if term_map is None:
        term_map = load_term_map()
    
    if llm_explanations is None:
        llm_explanations = {}
    
    # Collect unique terms
    unique_terms: Dict[str, EntityModel] = {}  # normalized_term -> EntityModel
    
    for entity in entities:
        # Normalize the term (lowercase for deduplication)
        term_normalized = entity.text.lower().strip()
        
        # Only add if we haven't seen this exact term before, or if this one is longer
        if term_normalized not in unique_terms or len(entity.text) > len(unique_terms[term_normalized].text):
            unique_terms[term_normalized] = entity
    
    # Generate glossary entries
    glossary_entries = []
    
    for term_normalized, entity in unique_terms.items():
        original_term = entity.text.strip()
        
        # Get explanation from multiple sources (priority order: term_map -> llm_explanations)
        explanation = get_simple_replacement(original_term, term_map)
        
        if explanation is None:
            # Try LLM explanations (case-insensitive lookup)
            for llm_term, llm_explanation in llm_explanations.items():
                if llm_term.lower().strip() == term_normalized:
                    explanation = llm_explanation
                    break
        
        # If still no explanation found, skip this term (don't create generic explanations)
        if explanation is None:
            continue
        
        glossary_entries.append(GlossaryEntry(
            term=original_term,
            type=entity.type,
            explanation=explanation
        ))
    
    # Sort alphabetically by term (case-insensitive)
    glossary_entries.sort(key=lambda entry: entry.term.lower())
    
    return glossary_entries