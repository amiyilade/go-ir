#!/usr/bin/env python3
"""
Download CoIR Go Dataset
Downloads BOTH CodeSearchNet (code-to-text) and CodeSearchNet-CCR (code-to-code) 
for Go language from the CoIR benchmark.

Disclaimer: ChatGPT and Copilot were used to edit and enhance this script for better readability, error handling, and user feedback.
The author (me) implemented the core logic.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from collections import Counter
import sys

# Configuration
OUTPUT_DIR = Path("data/coir_go")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_codesearchnet_go():
    print("\n" + "=" * 80)
    print("DOWNLOADING CODESEARCHNET Go (Code-to-Text)")
    print("=" * 80)
    print("Task: Code Summary Retrieval - retrieve docstrings using code as query")
    print()
    
    try:
        dataset_name = "CoIR-Retrieval/CodeSearchNet"
        
        print("📥 Step 1/3: Loading Go corpus (code snippets with docstrings)...")
        corpus = load_dataset(dataset_name, "go-corpus", split="corpus")
        print(f"   ✓ Loaded {len(corpus):,} code snippets")
        
        print("\n📥 Step 2/3: Loading Go queries...")
        queries = load_dataset(dataset_name, "go-queries", split="queries")
        print(f"   ✓ Loaded {len(queries):,} queries")
        
        print("\n📥 Step 3/3: Loading relevance judgments...")
        qrels = load_dataset(dataset_name, "go-qrels", split="test")
        print(f"   ✓ Loaded {len(qrels):,} query-document pairs")
        
        output_dir = OUTPUT_DIR / "codesearchnet"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n💾 Saving to disk...")
        corpus.save_to_disk(output_dir / "corpus")
        queries.save_to_disk(output_dir / "queries")
        qrels.save_to_disk(output_dir / "qrels")
        print(f"   ✓ Saved to: {output_dir}")
        
        print("\n🔨 Creating consolidated dataset...")
        corpus_dict = {item['_id']: item['text'] for item in corpus}
        query_dict = {item['_id']: item['text'] for item in queries}
        
        examples = []
        for qrel in qrels:
            query_id = qrel['query-id']
            corpus_id = qrel['corpus-id']
            
            if query_id in query_dict and corpus_id in corpus_dict:
                examples.append({
                    'task_type': 'code-to-text',
                    'query_id': query_id,
                    'query': query_dict[query_id],  # This is our code
                    'corpus_id': corpus_id,
                    'target': corpus_dict[corpus_id],  # This is our docstring
                    'score': qrel['score']
                })        
        consolidated_file = output_dir / "consolidated.jsonl"
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')
        
        print(f"   ✓ Created {len(examples):,} consolidated examples")
        print(f"   ✓ Saved to: {consolidated_file}")
        
        return examples
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify dataset exists at: https://huggingface.co/CoIR-Retrieval/CodeSearchNet")
        return None

def download_codesearchnet_ccr_go():
    print("\n" + "=" * 80)
    print("DOWNLOADING CODESEARCHNET-CCR Go (Code-to-Code)")
    print("=" * 80)
    print("Task: Code Context Retrieval - retrieve code completion")
    print("Note: Code is split 40-70% into initial segment (query) and completion (target)")
    print()
    
    try:
        # CodeSearchNet-CCR uses the same base dataset but with different split
        dataset_name = "CoIR-Retrieval/CodeSearchNet-CCR"
        
        print("1/3: Loading Go corpus (code completions)...")
        corpus = load_dataset(dataset_name, "go-corpus", split="corpus")
        print(f"   ✓ Loaded {len(corpus):,} code segments")
        
        print("\n 2/3: Loading Go queries (initial code segments)...")
        queries = load_dataset(dataset_name, "go-queries", split="queries")
        print(f"   ✓ Loaded {len(queries):,} queries")
        
        print("\n 3/3: Loading relevance judgments...")
        qrels = load_dataset(dataset_name, "go-qrels", split="test")
        print(f"   ✓ Loaded {len(qrels):,} query-document pairs")
        
        output_dir = OUTPUT_DIR / "codesearchnet-ccr"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n Saving to disk...")
        corpus.save_to_disk(output_dir / "corpus")
        queries.save_to_disk(output_dir / "queries")
        qrels.save_to_disk(output_dir / "qrels")
        print(f"   ✓ Saved to: {output_dir}")
        
        print("\n Creating consolidated dataset...")
        corpus_dict = {item['_id']: item['text'] for item in corpus}
        query_dict = {item['_id']: item['text'] for item in queries}
        
        examples = []
        for qrel in qrels:
            query_id = qrel['query-id']
            corpus_id = qrel['corpus-id']
            
            if query_id in query_dict and corpus_id in corpus_dict:
                examples.append({
                    'task_type': 'code-to-code',
                    'query_id': query_id,
                    'query': query_dict[query_id],  # Initial code segment (40-70%)
                    'corpus_id': corpus_id,
                    'target': corpus_dict[corpus_id],  # Completion segment
                    'score': qrel['score']
                })
        
        consolidated_file = output_dir / "consolidated.jsonl"
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')
        
        print(f"   ✓ Created {len(examples):,} consolidated examples")
        print(f"   ✓ Saved to: {consolidated_file}")
        
        return examples
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify dataset exists at: https://huggingface.co/CoIR-Retrieval/CodeSearchNet-CCR")
        return None


def main():
    print("\n" + "=" * 80)
    print("CoIR Go DATASET DOWNLOADER")
    print("=" * 80)
    print("\nThis script downloads TWO datasets from CoIR benchmark:")
    print("  1. CodeSearchNet (Code-to-Text): Code → Documentation")
    print("  2. CodeSearchNet-CCR (Code-to-Code): Initial code → Completion")
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    code_to_text_examples = download_codesearchnet_go()
    
    if code_to_text_examples is None:
        print("\n✗ Failed to download CodeSearchNet. Aborting.")
        sys.exit(1)
    
    code_to_code_examples = download_codesearchnet_ccr_go()
    
    if code_to_code_examples is None:
        print("\n✗ Failed to download CodeSearchNet-CCR. Aborting.")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("CREATING COMBINED REFERENCE FILE")
    print("=" * 80)
    
    combined_file = OUTPUT_DIR / "coir_go_full.jsonl"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for ex in code_to_text_examples:
            f.write(json.dumps(ex) + '\n')
        for ex in code_to_code_examples:
            f.write(json.dumps(ex) + '\n')
    
    total_examples = len(code_to_text_examples) + len(code_to_code_examples)
    print(f"✓ Saved {total_examples:,} total examples to: {combined_file}")
    
    print("\n" + "=" * 80)
    print(" FINAL SUMMARY")
    print("=" * 80)
    print(f"Code-to-Text (CodeSearchNet):     {len(code_to_text_examples):,} examples")
    print(f"Code-to-Code (CodeSearchNet-CCR): {len(code_to_code_examples):,} examples")
    print(f"Total:                            {total_examples:,} examples")
    print()
    
    print("=" * 80)
    print("EXAMPLE: CODE-TO-TEXT")
    print("=" * 80)
    if code_to_text_examples:
        ex = code_to_text_examples[0]
        print(f"Query (Code):\n{ex['query'][:200]}...")
        print(f"\nTarget (Docstring):\n{ex['target'][:200]}...")
    print()
    
    print("=" * 80)
    print("EXAMPLE: CODE-TO-CODE")
    print("=" * 80)
    if code_to_code_examples:
        ex = code_to_code_examples[0]
        print(f"Query (Initial Segment):\n{ex['query'][:200]}...")
        print(f"\nTarget (Completion):\n{ex['target'][:200]}...")
    print()
    
    print("=" * 80)
    print("✓ DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\n Data location: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
