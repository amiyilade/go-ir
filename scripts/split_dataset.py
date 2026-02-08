#!/usr/bin/env python3
"""
Split Dataset into 300-Sample Chunks
Splits both code-to-text and code-to-code datasets into manageable chunks.
"""

import json
from pathlib import Path
from tqdm import tqdm

# Configuration
CODE_TO_TEXT_DIR = Path("data/code-to-text")
CODE_TO_CODE_DIR = Path("data/code-to-code")
CHUNKS_DIR = Path("data/chunks")

CHUNK_SIZE = 50

def split_dataset(input_file: Path, output_dir: Path, dataset_name: str):
    """Split a dataset into chunks of CHUNK_SIZE samples."""
    print(f"\n{'='*80}")
    print(f"SPLITTING: {dataset_name}")
    print(f"Chunk size: {CHUNK_SIZE} samples")
    print(f"{'='*80}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count total samples
    print("Counting samples...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_samples = sum(1 for _ in f)
    
    num_chunks = (total_samples + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Total samples: {total_samples:,}")
    print(f"Number of chunks: {num_chunks}")
    
    # Split into chunks
    chunk_id = 0
    samples_in_chunk = 0
    chunk_file = None
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_samples, desc="Splitting"):
            # Open new chunk file if needed
            if samples_in_chunk == 0:
                if chunk_file:
                    chunk_file.close()
                
                chunk_filename = output_dir / f"{dataset_name}_chunk_{chunk_id:02d}.jsonl"
                chunk_file = open(chunk_filename, 'w', encoding='utf-8')
            
            # Write to current chunk
            chunk_file.write(line)
            samples_in_chunk += 1
            
            # Close chunk when full
            if samples_in_chunk >= CHUNK_SIZE:
                chunk_file.close()
                chunk_file = None
                
                # Calculate actual size
                chunk_path = output_dir / f"{dataset_name}_chunk_{chunk_id:02d}.jsonl"
                size_mb = chunk_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ Chunk {chunk_id:02d}: {samples_in_chunk} samples ({size_mb:.1f} MB)")
                
                chunk_id += 1
                samples_in_chunk = 0
    
    # Close last chunk if open
    if chunk_file:
        chunk_file.close()
        chunk_path = output_dir / f"{dataset_name}_chunk_{chunk_id:02d}.jsonl"
        size_mb = chunk_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Chunk {chunk_id:02d}: {samples_in_chunk} samples ({size_mb:.1f} MB)")
    
    print(f"\n✓ Created {chunk_id + 1} chunks")
    print(f"  Location: {output_dir}/")
    
    return chunk_id + 1

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("DATASET CHUNKING - 300 SAMPLES PER CHUNK")
    print("=" * 80)
    print("\nThis will split your datasets into manageable 300-sample chunks.")
    print("Each chunk can be processed independently in separate Colab sessions.")
    print()
    
    # Check input files exist
    code_to_text_file = CODE_TO_TEXT_DIR / "full_code_to_text_with_asts.jsonl"
    code_to_code_file = CODE_TO_CODE_DIR / "full_code_to_code_with_asts.jsonl"
    
    if not code_to_text_file.exists():
        print(f"✗ Error: {code_to_text_file} not found")
        return
    
    if not code_to_code_file.exists():
        print(f"✗ Error: {code_to_code_file} not found")
        return
    
    # Create chunks directory
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Split code-to-text
    code_to_text_chunks_dir = CHUNKS_DIR / "code-to-text"
    num_c2t_chunks = split_dataset(
        code_to_text_file,
        code_to_text_chunks_dir,
        "code_to_text"
    )
    
    # Split code-to-code
    code_to_code_chunks_dir = CHUNKS_DIR / "code-to-code"
    num_c2c_chunks = split_dataset(
        code_to_code_file,
        code_to_code_chunks_dir,
        "code_to_code"
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ CHUNKING COMPLETE")
    print("=" * 80)
    print(f"\nCode-to-Text: {num_c2t_chunks} chunks")
    print(f"  Location: {code_to_text_chunks_dir}/")
    print(f"\nCode-to-Code: {num_c2c_chunks} chunks")
    print(f"  Location: {code_to_code_chunks_dir}/")
    print(f"\nTotal chunks to process: {num_c2t_chunks + num_c2c_chunks}")
    print(f"  ({num_c2t_chunks} code-to-text + {num_c2c_chunks} code-to-code)")
    print()
    print("Next step:")
    print("  python scripts/process_single_chunk.py --chunk 0 --task code-to-text --model unixcoder")
    print()

if __name__ == "__main__":
    main()
