#!/usr/bin/env python3
"""
Demo script for medical text simplification.

Takes a text file and simplification level, runs the simplification process,
and outputs the result as JSON.

Usage:
    python run_demo.py <input_file> <level>
    
    input_file: Path to text file containing medical text
    level: Simplification level (basic, intermediate, advanced)

Example:
    python run_demo.py data/sample_docs/doc1_diabetes_basic.txt basic
"""

import sys
import json
import argparse
from pathlib import Path

# Add the current directory to Python path so we can import simplifier
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from simplifier import simplify
except ImportError as e:
    print(f"Error importing simplifier module: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Demonstrate medical text simplification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py data/sample_docs/doc1_diabetes_basic.txt basic
  python run_demo.py my_text.txt intermediate
  echo "Patient has acute myocardial infarction" | python run_demo.py - advanced
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Input text file path (use "-" for stdin)'
    )
    
    parser.add_argument(
        'level',
        choices=['basic', 'intermediate', 'advanced'],
        help='Simplification level'
    )
    
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty-print JSON output'
    )
    
    parser.add_argument(
        '--offline',
        action='store_true',
        help='Run in offline mode (no LLM calls)'
    )
    
    args = parser.parse_args()
    
    try:
        # Read input text
        if args.input_file == '-':
            text = sys.stdin.read().strip()
        else:
            input_path = Path(args.input_file)
            if not input_path.exists():
                print(f"Error: Input file '{args.input_file}' not found.")
                sys.exit(1)
            
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        
        if not text:
            print("Error: No input text provided.")
            sys.exit(1)
        
        print(f"Input text: {text[:100]}{'...' if len(text) > 100 else ''}", file=sys.stderr)
        print(f"Simplification level: {args.level}", file=sys.stderr)
        
        # Set environment for offline mode if requested
        if args.offline:
            os.environ['OFFLINE_MODE'] = '1'
        
        # Run simplification
        result = simplify(text, level=args.level)
        
        # Output result as JSON
        if args.pretty:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(result, ensure_ascii=False))
            
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"Error during simplification: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()