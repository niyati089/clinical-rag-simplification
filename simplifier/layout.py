"""
Markdown layout formatting for medical text simplification output.

This module handles the conversion of simplified medical text into well-formatted
markdown with proper headings, bullet points, and highlighted instruction sentences.
"""

import re
from typing import Dict, List, Tuple


class LayoutFormatter:
    """
    Formatter for converting simplified medical text into structured markdown.
    """
    
    def __init__(self):
        """Initialize the layout formatter."""
        pass
    
    def format_sections(self, sections: Dict[str, str], highlights: List[Dict[str, str]]) -> str:
        """
        Format sections into structured markdown with headings and highlights.
        
        Args:
            sections: Dictionary with section content
            highlights: List of sentences to highlight with bold formatting
            
        Returns:
            Formatted markdown string
        """
        markdown_parts = []
        
        # Define section order and titles
        section_order = [
            ("simple_explanation", "## What This Means"),
            ("important_instructions", "## Important Instructions"),
            ("medication_guidance", "## Medication Information"),
            ("follow_up", "## Follow-up Care")
        ]
        
        for section_key, section_title in section_order:
            content = sections.get(section_key, "").strip()
            if not content:
                continue
                
            markdown_parts.append(section_title)
            markdown_parts.append("")  # Empty line after heading
            
            # Format content based on section type
            if section_key == "important_instructions":
                formatted_content = self._format_instructions(content, highlights)
            else:
                formatted_content = self._format_general_content(content, highlights, section_key)
            
            markdown_parts.append(formatted_content)
            markdown_parts.append("")  # Empty line after section
        
        return "\n".join(markdown_parts).strip()
    
    def _format_instructions(self, content: str, highlights: List[Dict[str, str]]) -> str:
        """
        Format instructions with bullet points and highlighting.
        
        Args:
            content: Raw instruction content
            highlights: Sentences to highlight
            
        Returns:
            Formatted instruction content
        """
        # Get highlight sentences for this section
        section_highlights = [
            h["sentence"] for h in highlights 
            if h.get("section") == "important_instructions"
        ]
        
        # Split content into sentences/points
        sentences = self._split_into_sentences(content)
        formatted_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if this sentence should be highlighted
            should_highlight = any(
                self._sentences_match(sentence, highlight)
                for highlight in section_highlights
            )
            
            # Format as bullet point with optional highlighting
            if should_highlight:
                formatted_sentences.append(f"- **{sentence}**")
            else:
                formatted_sentences.append(f"- {sentence}")
        
        return "\n".join(formatted_sentences)
    
    def _format_general_content(self, content: str, highlights: List[Dict[str, str]], section_key: str) -> str:
        """
        Format general content with paragraphs and highlighting.
        
        Args:
            content: Raw content
            highlights: Sentences to highlight
            section_key: Current section identifier
            
        Returns:
            Formatted content
        """
        # Get highlight sentences for this section
        section_highlights = [
            h["sentence"] for h in highlights 
            if h.get("section") == section_key
        ]
        
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        formatted_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Apply highlighting to sentences within the paragraph
            formatted_paragraph = self._apply_highlights_to_paragraph(
                paragraph, section_highlights
            )
            
            # Ensure short paragraphs (max ~3 sentences per paragraph)
            short_paragraphs = self._split_long_paragraph(formatted_paragraph)
            formatted_paragraphs.extend(short_paragraphs)
        
        return "\n\n".join(formatted_paragraphs)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, handling medical abbreviations.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Handle common medical abbreviations that shouldn't trigger sentence breaks
        text = text.replace("Dr.", "Dr<DOT>")
        text = text.replace("mg.", "mg<DOT>")
        text = text.replace("ml.", "ml<DOT>")
        text = text.replace("etc.", "etc<DOT>")
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore abbreviation periods and clean up
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.replace("<DOT>", ".").strip()
            if sentence and not sentence.endswith(('.', '!', '?')):
                # Add period if missing (likely the last sentence)
                if sentence:
                    sentence += "."
            if sentence:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _sentences_match(self, sentence1: str, sentence2: str) -> bool:
        """
        Check if two sentences match (allowing for minor formatting differences).
        
        Args:
            sentence1: First sentence
            sentence2: Second sentence
            
        Returns:
            True if sentences match
        """
        # Normalize both sentences for comparison
        norm1 = self._normalize_sentence(sentence1)
        norm2 = self._normalize_sentence(sentence2)
        
        return norm1 == norm2 or norm1 in norm2 or norm2 in norm1
    
    def _normalize_sentence(self, sentence: str) -> str:
        """
        Normalize sentence for comparison by removing punctuation and extra whitespace.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Normalized sentence
        """
        # Remove punctuation and extra whitespace
        normalized = re.sub(r'[^\w\s]', ' ', sentence.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _apply_highlights_to_paragraph(self, paragraph: str, highlights: List[str]) -> str:
        """
        Apply bold formatting to highlighted sentences within a paragraph.
        
        Args:
            paragraph: Input paragraph
            highlights: List of sentences to highlight
            
        Returns:
            Paragraph with highlighting applied
        """
        if not highlights:
            return paragraph
            
        # Split paragraph into sentences
        sentences = self._split_into_sentences(paragraph)
        formatted_sentences = []
        
        for sentence in sentences:
            should_highlight = any(
                self._sentences_match(sentence, highlight)
                for highlight in highlights
            )
            
            if should_highlight:
                formatted_sentences.append(f"**{sentence}**")
            else:
                formatted_sentences.append(sentence)
        
        return " ".join(formatted_sentences)
    
    def _split_long_paragraph(self, paragraph: str, max_sentences: int = 3) -> List[str]:
        """
        Split long paragraphs into shorter ones.
        
        Args:
            paragraph: Input paragraph
            max_sentences: Maximum sentences per paragraph
            
        Returns:
            List of shorter paragraphs
        """
        sentences = self._split_into_sentences(paragraph)
        
        if len(sentences) <= max_sentences:
            return [paragraph]
        
        # Split into chunks
        paragraphs = []
        for i in range(0, len(sentences), max_sentences):
            chunk = sentences[i:i + max_sentences]
            paragraphs.append(" ".join(chunk))
        
        return paragraphs
    
    def identify_instruction_highlights(self, sections: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Identify sentences that should be highlighted as important instructions.
        
        Args:
            sections: Dictionary with section content
            
        Returns:
            List of highlight dictionaries with section and sentence
        """
        highlights = []
        
        # Pattern words/phrases that indicate important instructions
        instruction_patterns = [
            r'\b(?:must|should|need to|have to|important to|crucial to)\b',
            r'\btake\s+\w+\s+(?:mg|ml|tablets|pills)',
            r'\b(?:avoid|do not|never|stop)\b',
            r'\b(?:call|contact|see)\s+(?:doctor|physician|your\s+doctor)',
            r'\b(?:emergency|urgent|immediately|right away)\b',
            r'\b(?:before|after)\s+(?:meals|eating|bed)',
            r'\bif\s+you\s+(?:experience|feel|have|notice)\b'
        ]
        
        for section_key, content in sections.items():
            if not content:
                continue
                
            sentences = self._split_into_sentences(content)
            
            for sentence in sentences:
                # Check if sentence matches instruction patterns
                for pattern in instruction_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        highlights.append({
                            "section": section_key,
                            "sentence": sentence.strip()
                        })
                        break  # Only add once per sentence
        
        return highlights


# Global formatter instance
layout_formatter = LayoutFormatter()


def format_markdown_output(sections: Dict[str, str], highlights: List[Dict[str, str]]) -> str:
    """
    Convenience function for formatting sections into markdown.
    
    Args:
        sections: Dictionary with section content
        highlights: List of sentences to highlight
        
    Returns:
        Formatted markdown string
    """
    return layout_formatter.format_sections(sections, highlights)


def identify_highlights(sections: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Convenience function for identifying instruction highlights.
    
    Args:
        sections: Dictionary with section content
        
    Returns:
        List of highlight dictionaries
    """
    return layout_formatter.identify_instruction_highlights(sections)