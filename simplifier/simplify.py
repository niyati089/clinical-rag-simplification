"""
Main text simplification function and orchestration logic.

This module provides the primary simplify() function that coordinates all
simplification steps: entity recognition, number masking, term simplification,
readability optimization, and output formatting.
"""

import os
import json
import hashlib
from typing import Dict, Literal, Union, Optional
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, use existing environment variables
    pass
from .schema import SimplifierOutput, TextOrSections, SECTION_KEYS, EntityModel
from .masking import mask_text, restore_text, get_protected_values, validate_protected_values
from .ner import extract_entities
from .glossary import generate_glossary, get_simple_replacement
from .readability import calculate_readability_scores, ReadabilityCalculator, readability_calculator
from .layout import format_markdown_output, identify_highlights


class LLMEngine:
    """
    LLM integration with caching and offline fallback for text simplification.
    Supports both Anthropic Claude and Google Gemini APIs.
    """
    
    def __init__(self, cache_dir: str = None, offline_mode: bool = False):
        """Initialize LLM engine with optional cache directory and offline mode."""
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "data" / "llm_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine which API to use based on environment variables
        self.api_provider = self._detect_api_provider()
        self.offline_mode = offline_mode or (self.api_provider == "offline")
        
        # Set model based on provider
        if self.api_provider == "anthropic":
            self.model = os.environ.get("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        elif self.api_provider == "gemini":
            self.model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        else:
            self.model = "offline"
        
        # Initialize client based on provider
        self.client = None
        if not self.offline_mode:
            self.client = self._initialize_client()
    
    def _detect_api_provider(self) -> str:
        """Detect which API provider to use based on environment variables."""
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        else:
            return "offline"
    
    def _initialize_client(self):
        """Initialize the appropriate API client."""
        try:
            if self.api_provider == "anthropic":
                import anthropic
                return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            elif self.api_provider == "gemini":
                import google.genai as genai
                client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
                return client
            else:
                return None
        except ImportError as e:
            print(f"Warning: Required package not installed for {self.api_provider}: {e}")
            self.offline_mode = True
            return None
        except Exception as e:
            print(f"Warning: Failed to initialize {self.api_provider} client: {e}")
            self.offline_mode = True
            return None
    
    def _get_cache_key(self, text: str, level: str, task: str) -> str:
        """Generate cache key for LLM request."""
        content = f"{task}:{level}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load result from cache if available."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("result")
        except Exception as e:
            print(f"Warning: Failed to load from cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, result: str) -> None:
        """Save result to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({"result": result}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save to cache: {e}")
    
    def simplify_text(self, masked_text: str, level: str) -> str:
        """
        Simplify masked text using LLM or fallback.
        
        Args:
            masked_text: Text with protected values masked
            level: Simplification level (basic, intermediate, advanced)
            
        Returns:
            Simplified text with same masking
        """
        cache_key = self._get_cache_key(masked_text, level, "simplify")
        
        # Try cache first
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Generate result
        if self.offline_mode:
            result = self._rule_based_simplification(masked_text, level)
        else:
            try:
                result = self._llm_simplification(masked_text, level)
            except Exception as e:
                print(f"Warning: LLM call failed: {e}, using rule-based fallback")
                result = self._rule_based_simplification(masked_text, level)
        
        # Cache the result
        self._save_to_cache(cache_key, result)
        return result
    
    def explain_term(self, term: str) -> str:
        """
        Generate explanation for an unknown medical term.
        
        Args:
            term: Medical term to explain
            
        Returns:
            Plain-language explanation
        """
        cache_key = self._get_cache_key(term, "", "explain")
        
        # Try cache first
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Generate explanation
        if self.offline_mode:
            result = self._rule_based_explanation(term)
        else:
            try:
                result = self._llm_explanation(term)
            except Exception as e:
                print(f"Warning: LLM call failed for term '{term}': {e}")
                result = self._rule_based_explanation(term)
        
        # Cache the result
        self._save_to_cache(cache_key, result)
        return result
    
    def _llm_simplification(self, masked_text: str, level: str) -> str:
        """Use LLM to simplify text."""
        level_configs = {
            "basic": {
                "target_grade": "6th grade or below",
                "sentence_length": "under 15 words",
                "terminology": "replace all medical terms with simple words",
                "explanations": "explain difficult concepts immediately in the same sentence"
            },
            "intermediate": {
                "target_grade": "8th to 10th grade",
                "sentence_length": "moderate length, up to 20 words",
                "terminology": "keep some medical terms but add simple explanations in brackets",
                "explanations": "provide brief explanations for key terms"
            },
            "advanced": {
                "target_grade": "high school and above", 
                "sentence_length": "can be longer for complex concepts",
                "terminology": "keep medical terminology",
                "explanations": "minimal explanation, preserve professional language"
            }
        }
        
        config = level_configs.get(level, level_configs["basic"])
        
        prompt = f"""Simplify this medical text for {config['target_grade']} reading level.

Requirements:
- Target grade level: {config['target_grade']}
- Sentence length: {config['sentence_length']}
- Terminology: {config['terminology']}
- Explanations: {config['explanations']}
- CRITICAL: Do not change any text that looks like __PROTECTED_XXX__ - these are placeholders that must remain exactly as written
- Keep the same overall structure and meaning
- Make it clear and easy to understand

Text to simplify:
{masked_text}

Simplified text:"""
        
        if self.api_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        
        elif self.api_provider == "gemini":
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    'temperature': 0.1,
                    'max_output_tokens': 1000,
                }
            )
            return response.text.strip()
        
        else:
            # Fallback to rule-based
            return self._rule_based_simplification(masked_text, level)
    
    def _llm_explanation(self, term: str) -> str:
        """Use LLM to explain a medical term."""
        prompt = f"""Provide a simple, clear explanation of this medical term in plain language that a patient could understand. Keep it to 1-2 sentences.

Medical term: {term}

Explanation:"""
        
        if self.api_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        
        elif self.api_provider == "gemini":
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    'temperature': 0.1,
                    'max_output_tokens': 200,
                }
            )
            return response.text.strip()
        
        else:
            # Fallback to rule-based
            return self._rule_based_explanation(term)
    
    def _rule_based_simplification(self, masked_text: str, level: str) -> str:
        """Fallback rule-based simplification."""
        from .glossary import get_simple_replacement
        
        text = masked_text
        
        if level == "basic":
            # Replace known medical terms with simple versions
            # Use a more sophisticated approach to avoid multiple replacements
            from .glossary import load_term_map
            term_map = load_term_map()
            
            # Sort terms by length (longest first) to avoid partial replacements
            terms_by_length = sorted(term_map.keys(), key=len, reverse=True)
            
            for medical_term in terms_by_length:
                simple_term = term_map[medical_term]
                # Case-insensitive replacement, preserving word boundaries
                import re
                pattern = r'\b' + re.escape(medical_term) + r'\b'
                text = re.sub(pattern, simple_term, text, flags=re.IGNORECASE)
        
        elif level == "intermediate":
            # Add bracket explanations for known terms
            from .glossary import load_term_map
            term_map = load_term_map()
            
            # Sort terms by length (longest first) to avoid partial replacements
            terms_by_length = sorted(term_map.keys(), key=len, reverse=True)
            
            for medical_term in terms_by_length:
                simple_term = term_map[medical_term]
                # Add explanations in brackets, but only if not already explained
                import re
                pattern = r'\b' + re.escape(medical_term) + r'\b(?!\s*\([^)]+\))'
                replacement = f"{medical_term} ({simple_term})"
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Advanced level keeps original text unchanged
        return text
    
    def _rule_based_explanation(self, term: str) -> str:
        """Fallback rule-based explanation."""
        # Simple fallback - just return a generic explanation
        return f"a medical term that your healthcare provider can explain in detail"


# Global LLM engine instance - will be initialized on first use
llm_engine = None


def _get_llm_engine():
    """Get or create the global LLM engine instance."""
    global llm_engine
    if llm_engine is None:
        offline = os.environ.get('OFFLINE_MODE', '').lower() in ('1', 'true', 'yes')
        llm_engine = LLMEngine(offline_mode=offline)
    return llm_engine


def simplify(
    text_or_sections: TextOrSections,
    level: Literal["basic", "intermediate", "advanced"] = "basic"
) -> Dict:
    """
    Simplify medical text for different reading levels while preserving critical information.
    
    This is the main API function that processes medical text through multiple
    stages: entity recognition, number protection, terminology simplification,
    readability optimization, and structured output generation.
    
    Args:
        text_or_sections: Either a plain string or a dict with keys:
            - simple_explanation: Main content explanation
            - important_instructions: Critical instructions for the patient
            - medication_guidance: Medication-related information
            - follow_up: Follow-up care instructions
        level: Target simplification level:
            - "basic": 6th grade reading level, all terms simplified inline
            - "intermediate": 8th-10th grade, key terms annotated in brackets
            - "advanced": Original terminology preserved, glossary definitions only
    
    Returns:
        Dictionary matching SimplifierOutput schema with:
        - level: Applied simplification level
        - sections: Organized text by section type
        - simplified_text: Full markdown-formatted text
        - entities: Extracted medical entities with positions
        - glossary: Alphabetical term definitions
        - highlights: Important sentences to emphasize
        - scores: Readability metrics comparison (original vs simplified)
        - protected_values: Preserved dosages, units, frequencies, lab values
    
    Raises:
        ValueError: If input format is invalid or level is unsupported
        RuntimeError: If external dependencies (scispaCy, API) fail
    """
    
    # Input validation and normalization
    if isinstance(text_or_sections, str):
        # Convert plain string to sections format
        sections_input = {"simple_explanation": text_or_sections}
        original_text = text_or_sections
    elif isinstance(text_or_sections, dict):
        sections_input = text_or_sections.copy()
        original_text = _combine_sections(sections_input)
    else:
        raise ValueError(f"text_or_sections must be str or dict, got {type(text_or_sections)}")
    
    # Validate level parameter
    valid_levels = ["basic", "intermediate", "advanced"]
    if level not in valid_levels:
        raise ValueError(f"level must be one of {valid_levels}, got '{level}'")
    
    # Ensure all expected section keys exist
    normalized_sections = {}
    for key in SECTION_KEYS:
        normalized_sections[key] = sections_input.get(key, "")
    
    # Step 1: Extract protected values
    protected_values = get_protected_values(original_text)
    
    # Step 2: Extract entities from original text (before masking)
    entities = extract_entities(original_text)
    
    # Step 3: Process each section through the simplification pipeline
    simplified_sections = {}
    llm_explanations = {}  # Collect explanations for glossary
    
    for key, text in normalized_sections.items():
        if not text.strip():
            simplified_sections[key] = ""
            continue
        
        # Mask protected values in this section
        masked_text, placeholder_map = mask_text(text)
        
        # Simplify the masked text
        simplified_masked = _get_llm_engine().simplify_text(masked_text, level)
        
        # Restore protected values
        simplified_text = restore_text(simplified_masked, placeholder_map)
        
        # Store the simplified section
        simplified_sections[key] = simplified_text
        
        # Collect LLM explanations for unknown terms (for glossary)
        section_entities = extract_entities(text)
        for entity in section_entities:
            if entity.text.lower() not in llm_explanations:
                explanation = _get_llm_engine().explain_term(entity.text)
                if explanation:
                    llm_explanations[entity.text.lower()] = explanation
    
    # Step 4: Generate combined simplified text
    simplified_combined = _combine_sections(simplified_sections)
    
    # Step 5: Validate protected values are preserved
    try:
        validate_protected_values(original_text, simplified_combined)
    except ValueError as e:
        # If validation fails, log warning but continue
        print(f"Warning: Protected value validation failed: {e}")
    
    # Step 6: Generate glossary from entities
    glossary = generate_glossary(entities, llm_explanations)
    
    # Step 7: Calculate readability scores
    original_scores = calculate_readability_scores(original_text)
    simplified_scores = calculate_readability_scores(simplified_combined)
    
    # Step 8: Detect instruction highlights
    highlights = identify_highlights(simplified_sections)
    
    # Step 9: Format as markdown
    simplified_markdown = format_markdown_output(simplified_sections, highlights)
    
    # Step 10: Update entity positions for final simplified text
    final_entities = extract_entities(simplified_markdown)
    
    # Create the output structure
    output = SimplifierOutput(
        level=level,
        sections=simplified_sections,
        simplified_text=simplified_markdown,
        entities=final_entities,
        glossary=glossary,
        highlights=highlights,
        scores={
            "original": {
                "flesch_reading_ease": original_scores["flesch_reading_ease"],
                "grade_level": original_scores["grade_level"]
            },
            "simplified": {
                "flesch_reading_ease": simplified_scores["flesch_reading_ease"], 
                "grade_level": simplified_scores["grade_level"]
            }
        },
        protected_values=protected_values
    )
    
    # Return as dict for JSON serialization
    return output.model_dump()


def _combine_sections(sections: Dict[str, str]) -> str:
    """Combine section dictionary into a single text string."""
    parts = []
    for key in SECTION_KEYS:
        text = sections.get(key, "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def simplify_rag_result(rag_result: dict, level: str = None) -> dict:
    """
    Simplify RAG pipeline output while preserving structure and references.
    
    This function accepts the exact output format from Person A's RAGPipeline.process()
    and processes it through the medical text simplification pipeline while maintaining
    all original chunk metadata and references.
    
    Args:
        rag_result: RAG pipeline output dict with format:
            {
              "reading_level": "Basic|Intermediate|Advanced",
              "total_chunks": int,
              "disclaimer": str,
              "results": [
                {
                  "chunk_id": str,
                  "original_chunk": str,
                  "section_title": str,
                  "chunk_index": int,
                  "simplified_output": {
                    "simple_explanation": str,
                    "important_instructions": [str],
                    "medication_guidance": [str],
                    "follow_up": [str]
                  },
                  "retrieved_references": [...]
                }
              ]
            }
        level: Target simplification level. Defaults to rag_result["reading_level"] normalized to lowercase.
    
    Returns:
        Enhanced RAG result dict with added simplification features:
        - Each chunk gets: entities, glossary, highlights, scores
        - Top level gets: aggregated glossary, scores, protected_values_check
        Validates against RagResultB schema.
    
    Raises:
        ValueError: If input format is invalid or level is unsupported
        KeyError: If required RAG result fields are missing
    """
    from .readability import readability_calculator
    
    # Validate input structure
    if not isinstance(rag_result, dict):
        raise ValueError(f"rag_result must be dict, got {type(rag_result)}")
    
    required_fields = ["reading_level", "total_chunks", "disclaimer", "results"]
    for field in required_fields:
        if field not in rag_result:
            raise KeyError(f"Required field '{field}' missing from rag_result")
    
    # Normalize level parameter
    if level is None:
        level = rag_result["reading_level"].strip().lower()
    else:
        level = level.strip().lower()
    
    # Validate level parameter
    valid_levels = ["basic", "intermediate", "advanced"]
    if level not in valid_levels:
        raise ValueError(f"level must be one of {valid_levels}, got '{level}'")
    
    # Initialize aggregation containers
    all_glossary_entries = []
    all_original_scores = []
    all_simplified_scores = []
    protected_values_missing = []
    
    # Process each chunk
    enhanced_results = []
    
    for chunk in rag_result["results"]:
        # Validate chunk structure
        required_chunk_fields = ["chunk_id", "original_chunk", "section_title", "chunk_index", "simplified_output", "retrieved_references"]
        for field in required_chunk_fields:
            if field not in chunk:
                raise KeyError(f"Required chunk field '{field}' missing")
        
        # Extract protected values from original chunk
        original_chunk = chunk["original_chunk"]
        protected_values = get_protected_values(original_chunk)
        
        # Extract entities from original chunk
        entities = extract_entities(original_chunk)
        
        # Process simplified_output sections
        simplified_output = chunk["simplified_output"]
        processed_sections = {}
        llm_explanations = {}
        
        # Process simple_explanation (string)
        simple_explanation = simplified_output.get("simple_explanation", "")
        if simple_explanation:
            processed_sections["simple_explanation"] = _process_text_section(simple_explanation, level)
        else:
            processed_sections["simple_explanation"] = ""
        
        # Process list sections
        for list_field in ["important_instructions", "medication_guidance", "follow_up"]:
            section_list = simplified_output.get(list_field, [])
            if isinstance(section_list, list):
                processed_list = []
                for item in section_list:
                    if isinstance(item, str) and item.strip():
                        processed_item = _process_text_section(item, level)
                        processed_list.append(processed_item)
                processed_sections[list_field] = processed_list
            else:
                processed_sections[list_field] = []
        
        # Collect LLM explanations for glossary
        section_entities = extract_entities(simple_explanation)
        for entity in section_entities:
            if entity.text.lower() not in llm_explanations:
                explanation = _get_llm_engine().explain_term(entity.text)
                if explanation:
                    llm_explanations[entity.text.lower()] = explanation
        
        # Generate glossary for this chunk
        chunk_glossary = generate_glossary(entities, llm_explanations)
        all_glossary_entries.extend(chunk_glossary)
        
        # Calculate readability scores
        original_scores = calculate_readability_scores(original_chunk)
        simplified_combined = _combine_simplified_output_sections(processed_sections)
        simplified_scores = calculate_readability_scores(simplified_combined)
        
        all_original_scores.append(original_scores)
        all_simplified_scores.append(simplified_scores)
        
        # Check protected values preservation
        for protected_value in protected_values:
            if protected_value not in simplified_combined:
                protected_values_missing.append(protected_value)
        
        # Identify highlights
        section_dict = {
            "simple_explanation": processed_sections["simple_explanation"],
            "important_instructions": "\n".join(processed_sections["important_instructions"]),
            "medication_guidance": "\n".join(processed_sections["medication_guidance"]),
            "follow_up": "\n".join(processed_sections["follow_up"])
        }
        highlights = identify_highlights(section_dict)
        
        # Build enhanced chunk result
        enhanced_chunk = {
            # Original fields preserved exactly
            "chunk_id": chunk["chunk_id"],
            "original_chunk": chunk["original_chunk"],
            "section_title": chunk["section_title"],
            "chunk_index": chunk["chunk_index"],
            "simplified_output": {
                "simple_explanation": processed_sections["simple_explanation"],
                "important_instructions": processed_sections["important_instructions"],
                "medication_guidance": processed_sections["medication_guidance"],
                "follow_up": processed_sections["follow_up"]
            },
            "retrieved_references": chunk["retrieved_references"],
            
            # New fields added by simplify_rag_result
            "entities": [entity.model_dump() if hasattr(entity, 'model_dump') else entity for entity in entities],
            "glossary": [entry.model_dump() if hasattr(entry, 'model_dump') else entry for entry in chunk_glossary],
            "highlights": [highlight if isinstance(highlight, dict) else highlight.model_dump() for highlight in highlights],
            "scores": {
                "original": {
                    "flesch_reading_ease": original_scores["flesch_reading_ease"],
                    "grade_level": original_scores["grade_level"]
                },
                "simplified": {
                    "flesch_reading_ease": simplified_scores["flesch_reading_ease"],
                    "grade_level": simplified_scores["grade_level"]
                }
            }
        }
        
        enhanced_results.append(enhanced_chunk)
    
    # Aggregate glossary (remove duplicates)
    glossary_dict = {}
    for entry in all_glossary_entries:
        key = entry.term.lower()
        if key not in glossary_dict:
            glossary_dict[key] = entry
    
    aggregated_glossary = sorted(glossary_dict.values(), key=lambda x: x.term.lower())
    
    # Calculate aggregate scores
    if all_original_scores:
        avg_original_flesch = sum(s["flesch_reading_ease"] for s in all_original_scores) / len(all_original_scores)
        avg_original_grade = sum(s["grade_level"] for s in all_original_scores) / len(all_original_scores)
    else:
        avg_original_flesch = avg_original_grade = 0.0
    
    if all_simplified_scores:
        avg_simplified_flesch = sum(s["flesch_reading_ease"] for s in all_simplified_scores) / len(all_simplified_scores)
        avg_simplified_grade = sum(s["grade_level"] for s in all_simplified_scores) / len(all_simplified_scores)
    else:
        avg_simplified_flesch = avg_simplified_grade = 0.0
    
    # Build final result
    enhanced_rag_result = {
        # Original top-level fields preserved exactly
        "reading_level": rag_result["reading_level"],
        "total_chunks": rag_result["total_chunks"],
        "disclaimer": rag_result["disclaimer"],
        "results": enhanced_results,
        
        # New top-level fields
        "glossary": [entry.model_dump() if hasattr(entry, 'model_dump') else entry for entry in aggregated_glossary],
        "scores": {
            "original": {
                "flesch_reading_ease": round(avg_original_flesch, 2),
                "grade_level": round(avg_original_grade, 2)
            },
            "simplified": {
                "flesch_reading_ease": round(avg_simplified_flesch, 2),
                "grade_level": round(avg_simplified_grade, 2)
            }
        },
        "protected_values_check": {
            "passed": len(protected_values_missing) == 0,
            "missing": list(set(protected_values_missing))  # Remove duplicates
        }
    }
    
    # Validate against schema and return
    from .schema import RagResultB
    validated_result = RagResultB(**enhanced_rag_result)
    return validated_result.model_dump()


def _process_text_section(text: str, level: str) -> str:
    """
    Process a single text section through the simplification pipeline.
    
    Args:
        text: Text content to process
        level: Target simplification level
        
    Returns:
        Processed text
    """
    if not text or not text.strip():
        return text
    
    # Mask protected values
    masked_text, placeholder_map = mask_text(text)
    
    # Check if LLM rewrite is needed
    compliance = readability_calculator.check_level_compliance(masked_text, level)
    if not compliance.get("grade_target_met", True):
        # Text is above target grade level - use LLM to simplify
        simplified_masked = _get_llm_engine().simplify_text(masked_text, level)
    else:
        # Text meets target - use rule-based processing only
        simplified_masked = _get_llm_engine()._rule_based_simplification(masked_text, level)
    
    # Restore protected values
    simplified_text = restore_text(simplified_masked, placeholder_map)
    
    return simplified_text


def _combine_simplified_output_sections(sections: Dict) -> str:
    """
    Combine simplified output sections into a single text string.
    
    Args:
        sections: Dictionary with simple_explanation (str) and list sections
        
    Returns:
        Combined text string
    """
    parts = []
    
    # Add simple explanation
    if sections.get("simple_explanation", "").strip():
        parts.append(sections["simple_explanation"])
    
    # Add list sections
    for field in ["important_instructions", "medication_guidance", "follow_up"]:
        items = sections.get(field, [])
        if isinstance(items, list) and items:
            # Join list items with newlines
            section_text = "\n".join(item for item in items if isinstance(item, str) and item.strip())
            if section_text:
                parts.append(section_text)
    
    return "\n\n".join(parts)

