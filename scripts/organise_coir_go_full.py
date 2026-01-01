#!/usr/bin/env python3
"""
Script 2 (FULL): Organize Data Without Sampling
Processes the FULL CoIR Go dataset for comprehensive analysis.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import sys

# Configuration
RAW_DIR = Path("data/raw")
CODE_TO_TEXT_INPUT = RAW_DIR / "codesearchnet" / "consolidated.jsonl"
CODE_TO_CODE_INPUT = RAW_DIR / "codesearchnet-ccr" / "consolidated.jsonl"

CODE_TO_TEXT_DIR = Path("data/code-to-text")
CODE_TO_CODE_DIR = Path("data/code-to-code")

RANDOM_SEED = 42

def load_jsonl(file_path):
    """Load JSONL file into list of dictionaries."""
    print(f"Loading {file_path.name}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"  ✓ Loaded {len(data):,} examples")
    return data

def save_jsonl(data, file_path):
    """Save list of dictionaries to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"  ✓ Saved {len(data):,} samples to {file_path.name}")

def analyze_data(data, task_name):
    """Analyze and print statistics about the data."""
    print(f"\n[{task_name} Statistics]")
    print(f"  Total examples: {len(data):,}")
    
    # Analyze query and target lengths
    query_lengths = [len(d['query']) for d in data if 'query' in d]
    target_lengths = [len(d['target']) for d in data if 'target' in d]
    
    if query_lengths:
        print(f"  Query length: {min(query_lengths)}-{max(query_lengths)} chars")
        print(f"               (avg: {sum(query_lengths)/len(query_lengths):.0f} chars)")
    
    if target_lengths:
        print(f"  Target length: {min(target_lengths)}-{max(target_lengths)} chars")
        print(f"                (avg: {sum(target_lengths)/len(target_lengths):.0f} chars)")
    
    # Show example structure
    if data:
        print(f"  Example keys: {list(data[0].keys())}")

def show_example(data, task_name):
    """Show an example from the dataset."""
    if not data:
        return
    
    example = data[0]
    print(f"\n[{task_name} Example]")
    print(f"  Task type: {example.get('task_type', 'unknown')}")
    print(f"  Query (first 150 chars):")
    print(f"    {example['query'][:150]}...")
    print(f"  Target (first 150 chars):")
    print(f"    {example['target'][:150]}...")
    print()

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("CoIR Go FULL DATASET ORGANIZATION")
    print("=" * 80)
    print("\nProcessing ENTIRE datasets (NO SAMPLING):")
    print("  • Code-to-Text: All 8,122 examples")
    print("  • Code-to-Code: All 8,122 examples")
    print()
    
    # Check if input files exist
    if not CODE_TO_TEXT_INPUT.exists():
        print(f"\n✗ Error: Code-to-text data not found: {CODE_TO_TEXT_INPUT}")
        print("  Please run script 1 (download_coir_data.py) first.")
        sys.exit(1)
    
    if not CODE_TO_CODE_INPUT.exists():
        print(f"\n✗ Error: Code-to-code data not found: {CODE_TO_CODE_INPUT}")
        print("  Please run script 1 (download_coir_data.py) first.")
        sys.exit(1)
    
    # Process Code-to-Text
    print("\n" + "=" * 80)
    print("PROCESSING CODE-TO-TEXT (CodeSearchNet) - FULL DATASET")
    print("=" * 80)
    
    code_to_text_data = load_jsonl(CODE_TO_TEXT_INPUT)
    analyze_data(code_to_text_data, "Code-to-Text")
    show_example(code_to_text_data, "Code-to-Text")
    
    # Save FULL dataset
    print(f"\nSaving Code-to-Text data...")
    save_jsonl(code_to_text_data, CODE_TO_TEXT_DIR / "full_code_to_text.jsonl")
    
    # Process Code-to-Code
    print("\n" + "=" * 80)
    print("PROCESSING CODE-TO-CODE (CodeSearchNet-CCR) - FULL DATASET")
    print("=" * 80)
    
    code_to_code_data = load_jsonl(CODE_TO_CODE_INPUT)
    analyze_data(code_to_code_data, "Code-to-Code")
    show_example(code_to_code_data, "Code-to-Code")
    
    # Save FULL dataset
    print(f"\nSaving Code-to-Code data...")
    save_jsonl(code_to_code_data, CODE_TO_CODE_DIR / "full_code_to_code.jsonl")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✓ FULL DATASET ORGANIZATION COMPLETE")
    print("=" * 80)
    
    print(f"\n[Summary]")
    print(f"Code-to-Text:")
    print(f"  Full dataset:  {len(code_to_text_data):,} examples → {CODE_TO_TEXT_DIR / 'full_code_to_text.jsonl'}")
    print()
    print(f"Code-to-Code:")
    print(f"  Full dataset:  {len(code_to_code_data):,} examples → {CODE_TO_CODE_DIR / 'full_code_to_code.jsonl'}")
    print()
    
    print("Next steps:")
    print("  1. Run script 3 to parse ASTs and extract Go constructs:")
    print("     python scripts/parse_go_asts_full.py")
    print()

if __name__ == "__main__":
    main()
