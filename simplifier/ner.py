"""
Named Entity Recognition for medical texts using scispaCy.

This module provides medical NER capabilities using the en_core_sci_md model
for diseases, drugs, procedures, and abbreviations, plus custom regex patterns
for lab values with units.
"""

import re
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from .schema import EntityModel

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for NER evaluation."""
    precision: float
    recall: float
    f1: float
    total_predicted: int
    total_actual: int
    total_correct: int


class NEREngine:
    """Named Entity Recognition engine for medical texts."""
    
    def __init__(self):
        """Initialize the NER engine with lazy loading."""
        self._nlp = None
        self._abbrev_detector = None
        self._initialized = False
        
        # Lab value patterns - matches values with units
        self._lab_patterns = [
            # HbA1c: 7.2%, A1C 6.8%, HbA1c: 7.2 (check this first before generic percentages)
            r'(?:HbA1[Cc]|A1[Cc])\s*(?:is\s+)?\d+(?:\.\d+)?%?',
            # Blood glucose: 120 mg/dL, 6.7 mmol/L
            r'\b\d+(?:\.\d+)?\s*(?:mg/dL|mmol/L)\b',
            # Blood pressure: 120/80 mmHg, 140/90
            r'\b\d{2,3}/\d{2,3}(?:\s*mmHg)?\b',
            # Temperature: 98.6°F, 37°C, 100.4 F
            r'\b\d+(?:\.\d+)?°?[FC]\b',
            # Lab values with units: 2.5 mg, 10 mL, 500 mcg
            r'\b\d+(?:\.\d+)?\s*(?:mg|mL|mcg|μg|g|kg|L|dL|IU|units?)\b',
            # Percentages: 95% oxygen saturation
            r'\b\d+(?:\.\d+)?%\b',
            # Heart rate, counts: 72 bpm, 15,000 WBC
            r'\b\d{1,3}(?:,\d{3})*\s*(?:bpm|beats?(?:\s+per\s+minute)?|WBC|RBC)\b',
        ]
        
        # Compile regex patterns
        self._lab_regex = re.compile('|'.join(self._lab_patterns), re.IGNORECASE)
    
    def _load_model(self) -> bool:
        """Lazy load the scispaCy model and abbreviation detector."""
        if self._initialized:
            return True
        
        # Check if we already tried and failed
        if hasattr(self, '_load_failed'):
            return False
            
        try:
            import spacy
            from scispacy.abbreviation import AbbreviationDetector
            
            # Load the scientific model
            self._nlp = spacy.load("en_core_sci_md")
            
            # Add abbreviation detector to pipeline
            self._abbrev_detector = AbbreviationDetector(self._nlp)
            self._nlp.add_pipe("abbreviation_detector", last=True)
            
            self._initialized = True
            logger.info("Successfully loaded en_core_sci_md model with abbreviation detector")
            return True
            
        except (ImportError, OSError) as e:
            # Mark as failed so we don't spam logs
            self._load_failed = True
            logger.warning(f"scispaCy model not available, using regex-only extraction")
            logger.info("For full NER capability, install: pip install spacy scispacy")
            logger.info("Then download model: python -m spacy download en_core_sci_md")
            return False
    
    def extract_entities(self, text: str) -> List[EntityModel]:
        """
        Extract named entities from medical text.
        
        Args:
            text: Input medical text
            
        Returns:
            List of EntityModel objects with type, text, start, and end positions
        """
        entities = []
        
        # Try to load and use scispaCy model
        model_loaded = self._load_model()
        
        if model_loaded:
            # Process with scispaCy
            doc = self._nlp(text)
            
            # Extract entities from scispaCy
            for ent in doc.ents:
                entity_type = self._map_scispacy_label(ent.label_)
                if entity_type:
                    entities.append(EntityModel(
                        text=ent.text,
                        type=entity_type,
                        start=ent.start_char,
                        end=ent.end_char
                    ))
            
            # Extract abbreviations
            if hasattr(doc._, "abbreviations"):
                for abbrev in doc._.abbreviations:
                    # Check if this span overlaps with existing entities
                    if not self._overlaps_with_existing(abbrev.start_char, abbrev.end_char, entities):
                        entities.append(EntityModel(
                            text=abbrev.text,
                            type="ABBREVIATION",
                            start=abbrev.start_char,
                            end=abbrev.end_char
                        ))
        
        # Extract lab values using regex (always works, even without scispaCy)
        lab_entities = self._extract_lab_values(text)
        for lab_entity in lab_entities:
            # Check if this span overlaps with existing entities
            if not self._overlaps_with_existing(lab_entity.start, lab_entity.end, entities):
                entities.append(lab_entity)
        
        # Sort by start position and return
        return sorted(entities, key=lambda e: e.start)
    
    def _map_scispacy_label(self, label: str) -> Optional[str]:
        """Map scispaCy entity labels to our schema types."""
        # Common scispaCy labels mapped to our types
        label_mapping = {
            # Diseases and conditions
            "DISEASE": "DISEASE",
            "DISORDER": "DISEASE", 
            "CONDITION": "DISEASE",
            "SYMPTOM": "DISEASE",
            "SYNDROME": "DISEASE",
            
            # Drugs and medications
            "DRUG": "DRUG",
            "MEDICATION": "DRUG",
            "PHARMACEUTICAL": "DRUG",
            "CHEMICAL": "DRUG",
            
            # Procedures and treatments
            "PROCEDURE": "PROCEDURE",
            "TREATMENT": "PROCEDURE",
            "THERAPY": "PROCEDURE",
            "SURGERY": "PROCEDURE",
            "TEST": "PROCEDURE",
        }
        
        return label_mapping.get(label.upper())
    
    def _extract_lab_values(self, text: str) -> List[EntityModel]:
        """Extract lab values and measurements using regex patterns."""
        entities = []
        
        for match in self._lab_regex.finditer(text):
            entities.append(EntityModel(
                text=match.group(),
                type="LAB_VALUE",
                start=match.start(),
                end=match.end()
            ))
        
        return entities
    
    def _overlaps_with_existing(self, start: int, end: int, existing_entities: List[EntityModel]) -> bool:
        """Check if a span overlaps with any existing entity."""
        for entity in existing_entities:
            if not (end <= entity.start or start >= entity.end):
                return True
        return False
    
    def evaluate_performance(self, labels_file: Optional[str] = None) -> EvaluationMetrics:
        """
        Evaluate NER performance against labeled test data.
        
        Args:
            labels_file: Path to JSON file with labeled data. Defaults to data/ner_labels.json
            
        Returns:
            EvaluationMetrics with precision, recall, and F1 scores
        """
        if labels_file is None:
            labels_file = "data/ner_labels.json"
        
        # Make path relative to project root
        if not os.path.isabs(labels_file):
            project_root = os.path.dirname(os.path.dirname(__file__))
            labels_file = os.path.join(project_root, labels_file)
        
        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Labels file not found: {labels_file}")
            return EvaluationMetrics(0.0, 0.0, 0.0, 0, 0, 0)
        
        total_predicted = 0
        total_actual = 0
        total_correct = 0
        
        for item in test_data:
            text = item["text"]
            expected_entities = item["entities"]
            
            # Extract entities using our NER system
            predicted_entities = self.extract_entities(text)
            
            # Convert expected entities to the same format
            expected_spans = set()
            for ent in expected_entities:
                expected_spans.add((ent["start"], ent["end"], ent["type"]))
            
            predicted_spans = set()
            for ent in predicted_entities:
                predicted_spans.add((ent.start, ent.end, ent.type))
            
            # Calculate metrics
            total_predicted += len(predicted_spans)
            total_actual += len(expected_spans)
            total_correct += len(predicted_spans & expected_spans)
        
        # Calculate precision, recall, F1
        precision = total_correct / total_predicted if total_predicted > 0 else 0.0
        recall = total_correct / total_actual if total_actual > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            total_predicted=total_predicted,
            total_actual=total_actual,
            total_correct=total_correct
        )


# Convenience function for easy import
def extract_entities(text: str) -> List[EntityModel]:
    """Extract named entities from medical text using the default NER engine."""
    engine = NEREngine()
    return engine.extract_entities(text)


def evaluate_ner_performance(labels_file: Optional[str] = None) -> dict:
    """
    Evaluate NER performance against labeled test data.
    
    Args:
        labels_file: Path to JSON file with labeled data. Defaults to data/ner_labels.json
        
    Returns:
        Dictionary with precision, recall, f1_score, and total_correct counts
    """
    engine = NEREngine()
    metrics = engine.evaluate_performance(labels_file)
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1,
        "total_correct": metrics.total_correct
    }