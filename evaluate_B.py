#!/usr/bin/env python3
"""
Evaluation script for medical text simplification (Component B).

Runs all sample documents through all three simplification levels and generates
a comprehensive results CSV with readability metrics and NER performance.

Usage:
    python evaluate_B.py [--offline] [--output results.csv]

Output CSV columns:
- doc: Document filename
- level: Simplification level (basic, intermediate, advanced)
- flesch_before: Original text Flesch Reading Ease score
- flesch_after: Simplified text Flesch Reading Ease score
- grade_before: Original text Flesch-Kincaid grade level
- grade_after: Simplified text Flesch-Kincaid grade level
- ner_f1: NER F1 score (when applicable)
- protected_count: Number of protected values preserved
- glossary_terms: Number of glossary terms generated
- highlights: Number of highlighted instruction sentences
"""

import sys
import csv
import json
import os
import traceback
from pathlib import Path
from typing import Dict, List, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from simplifier import simplify
    from simplifier.readability import calculate_readability_scores
    from simplifier.ner import evaluate_ner_performance
except ImportError as e:
    print(f"Error importing simplifier module: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


def find_sample_docs() -> List[Path]:
    """
    Find all sample document files.
    
    Returns:
        List of paths to sample documents
    """
    sample_dir = Path("data/sample_docs")
    
    if not sample_dir.exists():
        print(f"Error: Sample docs directory '{sample_dir}' not found.")
        return []
    
    # Find all .txt files
    doc_files = list(sample_dir.glob("*.txt"))
    doc_files.sort()
    
    if not doc_files:
        print(f"Warning: No .txt files found in '{sample_dir}'")
    
    return doc_files


def evaluate_document(doc_path: Path, level: str, offline_mode: bool = False) -> Dict[str, Any]:
    """
    Evaluate a single document at a specific level.
    
    Args:
        doc_path: Path to document file
        level: Simplification level
        offline_mode: Whether to run in offline mode
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Read document
        with open(doc_path, 'r', encoding='utf-8') as f:
            original_text = f.read().strip()
        
        # Calculate original readability
        original_scores = calculate_readability_scores(original_text)
        
        # Set offline mode if requested
        if offline_mode:
            os.environ['OFFLINE_MODE'] = '1'
        
        # Run simplification
        result = simplify(original_text, level=level)
        
        # Extract metrics
        simplified_scores = result["scores"]["simplified"]
        
        evaluation = {
            "doc": doc_path.name,
            "level": level,
            "flesch_before": original_scores["flesch_reading_ease"],
            "flesch_after": simplified_scores["flesch_reading_ease"],
            "grade_before": original_scores["grade_level"],
            "grade_after": simplified_scores["grade_level"],
            "ner_f1": 0.0,  # Will be calculated separately if labels available
            "protected_count": len(result["protected_values"]),
            "glossary_terms": len(result["glossary"]),
            "highlights": len(result["highlights"]),
            "entity_count": len(result["entities"]),
            "success": True,
            "error": None
        }
        
        # Try to calculate NER F1 score if labels are available
        try:
            ner_performance = evaluate_ner_performance()
            if ner_performance and "f1_score" in ner_performance:
                evaluation["ner_f1"] = ner_performance["f1_score"]
        except Exception as ner_error:
            print(f"Warning: Could not calculate NER F1 for {doc_path.name}: {ner_error}")
        
        return evaluation
        
    except Exception as e:
        print(f"Error processing {doc_path.name} at level {level}: {e}")
        return {
            "doc": doc_path.name,
            "level": level,
            "flesch_before": 0.0,
            "flesch_after": 0.0,
            "grade_before": 0.0,
            "grade_after": 0.0,
            "ner_f1": 0.0,
            "protected_count": 0,
            "glossary_terms": 0,
            "highlights": 0,
            "entity_count": 0,
            "success": False,
            "error": str(e)
        }


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate medical text simplification")
    
    parser.add_argument(
        '--output',
        default='results.csv',
        help='Output CSV file path (default: results.csv)'
    )
    
    parser.add_argument(
        '--offline',
        action='store_true',
        help='Run in offline mode (no LLM calls)'
    )
    
    parser.add_argument(
        '--levels',
        nargs='*',
        choices=['basic', 'intermediate', 'advanced'],
        default=['basic', 'intermediate', 'advanced'],
        help='Simplification levels to test'
    )
    
    parser.add_argument(
        '--docs',
        nargs='*',
        help='Specific documents to process (default: all in sample_docs/)'
    )
    
    args = parser.parse_args()
    
    # Find documents to process
    if args.docs:
        doc_files = [Path(doc) for doc in args.docs if Path(doc).exists()]
    else:
        doc_files = find_sample_docs()
    
    if not doc_files:
        print("No documents found to process.")
        sys.exit(1)
    
    print(f"Found {len(doc_files)} documents to process")
    print(f"Testing levels: {', '.join(args.levels)}")
    print(f"Offline mode: {args.offline}")
    
    # Prepare results
    results = []
    total_combinations = len(doc_files) * len(args.levels)
    current = 0
    
    # Process each document at each level
    for doc_path in doc_files:
        for level in args.levels:
            current += 1
            print(f"Processing {doc_path.name} at {level} level ({current}/{total_combinations})...")
            
            evaluation = evaluate_document(doc_path, level, args.offline)
            results.append(evaluation)
            
            # Show progress
            if evaluation["success"]:
                flesch_change = evaluation["flesch_after"] - evaluation["flesch_before"]
                grade_change = evaluation["grade_after"] - evaluation["grade_before"]
                print(f"  ✓ Flesch: {evaluation['flesch_before']:.1f} → {evaluation['flesch_after']:.1f} ({flesch_change:+.1f})")
                print(f"    Grade: {evaluation['grade_before']:.1f} → {evaluation['grade_after']:.1f} ({grade_change:+.1f})")
            else:
                print(f"  ✗ Error: {evaluation['error']}")
    
    # Write results to CSV
    output_path = Path(args.output)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'doc', 'level', 'flesch_before', 'flesch_after',
            'grade_before', 'grade_after', 'ner_f1', 'protected_count',
            'glossary_terms', 'highlights', 'entity_count', 'success', 'error'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults written to {output_path}")
    
    # Print summary statistics
    successful_results = [r for r in results if r["success"]]
    
    if successful_results:
        print(f"\nSummary ({len(successful_results)} successful evaluations):")
        
        # Calculate average improvements by level
        for level in args.levels:
            level_results = [r for r in successful_results if r["level"] == level]
            if not level_results:
                continue
                
            avg_flesch_before = sum(r["flesch_before"] for r in level_results) / len(level_results)
            avg_flesch_after = sum(r["flesch_after"] for r in level_results) / len(level_results)
            avg_grade_before = sum(r["grade_before"] for r in level_results) / len(level_results)
            avg_grade_after = sum(r["grade_after"] for r in level_results) / len(level_results)
            
            flesch_improvement = avg_flesch_after - avg_flesch_before
            grade_improvement = avg_grade_before - avg_grade_after  # Lower grade is better
            
            print(f"  {level.capitalize()} level:")
            print(f"    Flesch Reading Ease: {avg_flesch_before:.1f} → {avg_flesch_after:.1f} ({flesch_improvement:+.1f})")
            print(f"    Grade Level: {avg_grade_before:.1f} → {avg_grade_after:.1f} ({grade_improvement:+.1f})")
            
            avg_entities = sum(r["entity_count"] for r in level_results) / len(level_results)
            avg_glossary = sum(r["glossary_terms"] for r in level_results) / len(level_results)
            avg_highlights = sum(r["highlights"] for r in level_results) / len(level_results)
            
            print(f"    Avg entities: {avg_entities:.1f}, glossary: {avg_glossary:.1f}, highlights: {avg_highlights:.1f}")
    
    # Report any failures
    failed_results = [r for r in results if not r["success"]]
    if failed_results:
        print(f"\nFailed evaluations ({len(failed_results)}):")
        for result in failed_results:
            print(f"  {result['doc']} ({result['level']}): {result['error']}")


if __name__ == '__main__':
    main()