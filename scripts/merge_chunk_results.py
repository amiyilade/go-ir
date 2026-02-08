#!/usr/bin/env python3
"""
Merge Chunk Results
Combines results from all processed chunks into final aggregated results.
Run this ONCE after all 28 chunks are complete.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Configuration
RESULTS_CHUNKS_DIR = Path("results/chunks")
FINAL_RESULTS_DIR = Path("results/final")

def merge_rq1_results(task_type: str, model_name: str):
    """Merge RQ1 (Attention-AST Alignment) results from all chunks."""
    print(f"\n📊 Merging RQ1 results for {task_type} / {model_name}...")
    
    chunks_dir = RESULTS_CHUNKS_DIR / task_type
    
    # Find all chunk directories
    chunk_dirs = sorted([d for d in chunks_dir.glob("chunk_*") if d.is_dir()])
    print(f"  Found {len(chunk_dirs)} chunks")
    
    # Collect all results
    all_layer_head_results = defaultdict(list)
    
    for chunk_dir in tqdm(chunk_dirs, desc="  Reading RQ1"):
        rq1_file = chunk_dir / f"{model_name}_rq1.json"
        
        if not rq1_file.exists():
            print(f"    ⚠️  Missing: {rq1_file}")
            continue
        
        with open(rq1_file, 'r', encoding='utf-8') as f:
            chunk_results = json.load(f)
        
        # Aggregate layer-head results
        if 'all_results' in chunk_results:
            for result in chunk_results['all_results']:
                layer = result['layer']
                head = result['head']
                key = f"layer_{layer}_head_{head}"
                all_layer_head_results[key].append(result)
    
    # Compute aggregated statistics
    print(f"  Computing aggregate statistics...")
    
    merged_results = []
    for key, results_list in all_layer_head_results.items():
        # Average across chunks
        mean_alignment = np.mean([r['mean_alignment_score'] for r in results_list])
        total_samples = sum(r['valid_samples'] for r in results_list)
        
        # Get layer/head from first result
        layer = results_list[0]['layer']
        head = results_list[0]['head']
        
        merged_results.append({
            'layer': layer,
            'head': head,
            'mean_alignment_score': mean_alignment,
            'valid_samples': total_samples,
            'num_chunks': len(results_list)
        })
    
    # Sort by alignment score
    merged_results.sort(key=lambda x: x['mean_alignment_score'], reverse=True)
    
    # Save merged results
    output = {
        'model_name': model_name,
        'task_type': task_type,
        'num_chunks_merged': len(chunk_dirs),
        'all_results': merged_results,
        'top_10_heads': merged_results[:10],
        'overall_stats': {
            'mean_alignment': np.mean([r['mean_alignment_score'] for r in merged_results]),
            'max_alignment': np.max([r['mean_alignment_score'] for r in merged_results]),
            'total_valid_samples': sum(r['valid_samples'] for r in merged_results)
        }
    }
    
    return output

def merge_rq2_results(task_type: str, model_name: str):
    """Merge RQ2 (Structural Probing) results from all chunks."""
    print(f"\n📊 Merging RQ2 results for {task_type} / {model_name}...")
    
    chunks_dir = RESULTS_CHUNKS_DIR / task_type
    chunk_dirs = sorted([d for d in chunks_dir.glob("chunk_*") if d.is_dir()])
    print(f"  Found {len(chunk_dirs)} chunks")
    
    # Collect results per layer
    layer_results = defaultdict(list)
    
    for chunk_dir in tqdm(chunk_dirs, desc="  Reading RQ2"):
        rq2_file = chunk_dir / f"{model_name}_rq2.json"
        
        if not rq2_file.exists():
            print(f"    ⚠️  Missing: {rq2_file}")
            continue
        
        with open(rq2_file, 'r', encoding='utf-8') as f:
            chunk_results = json.load(f)
        
        # Collect layer-wise correlations
        if 'all_layers' in chunk_results:
            for layer_result in chunk_results['all_layers']:
                layer = layer_result['layer']
                layer_results[layer].append(layer_result)
    
    # Compute aggregated statistics
    print(f"  Computing aggregate statistics...")
    
    merged_layers = []
    for layer in sorted(layer_results.keys()):
        results_list = layer_results[layer]
        
        # Average Spearman correlations
        mean_spearman = np.mean([r['spearman_correlation'] for r in results_list])
        total_samples = sum(r['num_samples'] for r in results_list)
        
        merged_layers.append({
            'layer': layer,
            'spearman_correlation': mean_spearman,
            'num_samples': total_samples,
            'num_chunks': len(results_list)
        })
    
    # Find best layer
    best_layer = max(merged_layers, key=lambda x: x['spearman_correlation'])
    
    output = {
        'model_name': model_name,
        'task_type': task_type,
        'num_chunks_merged': len(chunk_dirs),
        'all_layers': merged_layers,
        'best_layer': best_layer,
        'layer_summary': {
            f'layer_{layer}': r['spearman_correlation']
            for r in merged_layers
            for layer in [r['layer']]
        }
    }
    
    return output

def merge_rq3_results(task_type: str, model_name: str):
    """Merge RQ3 (Tree Induction) results from all chunks."""
    print(f"\n📊 Merging RQ3 results for {task_type} / {model_name}...")
    
    chunks_dir = RESULTS_CHUNKS_DIR / task_type
    chunk_dirs = sorted([d for d in chunks_dir.glob("chunk_*") if d.is_dir()])
    print(f"  Found {len(chunk_dirs)} chunks")
    
    # Collect results per layer
    layer_results = defaultdict(list)
    
    for chunk_dir in tqdm(chunk_dirs, desc="  Reading RQ3"):
        rq3_file = chunk_dir / f"{model_name}_rq3.json"
        
        if not rq3_file.exists():
            print(f"    ⚠️  Missing: {rq3_file}")
            continue
        
        with open(rq3_file, 'r', encoding='utf-8') as f:
            chunk_results = json.load(f)
        
        # Collect layer-wise F1 scores
        if 'all_layers' in chunk_results:
            for layer_result in chunk_results['all_layers']:
                layer = layer_result['layer']
                layer_results[layer].append(layer_result)
    
    # Compute aggregated statistics
    print(f"  Computing aggregate statistics...")
    
    merged_layers = []
    for layer in sorted(layer_results.keys()):
        results_list = layer_results[layer]
        
        # Average F1 scores
        mean_f1 = np.mean([r['mean_f1'] for r in results_list])
        total_samples = sum(r['num_samples'] for r in results_list)
        
        merged_layers.append({
            'layer': layer,
            'mean_f1': mean_f1,
            'num_samples': total_samples,
            'num_chunks': len(results_list)
        })
    
    # Find best layer
    best_layer = max(merged_layers, key=lambda x: x['mean_f1'])
    
    output = {
        'model_name': model_name,
        'task_type': task_type,
        'num_chunks_merged': len(chunk_dirs),
        'all_layers': merged_layers,
        'best_layer': best_layer,
        'layer_summary': {
            f'layer_{layer}': r['mean_f1']
            for r in merged_layers
            for layer in [r['layer']]
        }
    }
    
    return output

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("MERGING CHUNK RESULTS")
    print("=" * 80)
    print("\nThis will combine results from all processed chunks.")
    print()
    
    # Create output directory
    FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define all combinations to merge
    tasks = ['code-to-text', 'code-to-code']
    models = ['unixcoder']  # Add 'codebert' if you processed it
    rqs = ['rq1', 'rq2', 'rq3']
    
    # Track what was merged
    merged_count = 0
    
    for task in tasks:
        for model in models:
            print("\n" + "=" * 80)
            print(f"TASK: {task} | MODEL: {model}")
            print("=" * 80)
            
            # Check if chunks exist
            chunks_dir = RESULTS_CHUNKS_DIR / task
            if not chunks_dir.exists():
                print(f"  ⚠️  No chunks found for {task}")
                continue
            
            chunk_dirs = sorted([d for d in chunks_dir.glob("chunk_*") if d.is_dir()])
            if len(chunk_dirs) == 0:
                print(f"  ⚠️  No processed chunks found")
                continue
            
            print(f"  Found {len(chunk_dirs)} chunks to merge")
            
            # Merge RQ1
            try:
                rq1_merged = merge_rq1_results(task, model)
                output_file = FINAL_RESULTS_DIR / f"{task}_{model}_rq1.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(rq1_merged, f, indent=2)
                print(f"  ✓ Saved RQ1: {output_file.name}")
                merged_count += 1
            except Exception as e:
                print(f"  ✗ RQ1 merge failed: {e}")
            
            # Merge RQ2
            try:
                rq2_merged = merge_rq2_results(task, model)
                output_file = FINAL_RESULTS_DIR / f"{task}_{model}_rq2.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(rq2_merged, f, indent=2)
                print(f"  ✓ Saved RQ2: {output_file.name}")
                merged_count += 1
            except Exception as e:
                print(f"  ✗ RQ2 merge failed: {e}")
            
            # Merge RQ3
            try:
                rq3_merged = merge_rq3_results(task, model)
                output_file = FINAL_RESULTS_DIR / f"{task}_{model}_rq3.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(rq3_merged, f, indent=2)
                print(f"  ✓ Saved RQ3: {output_file.name}")
                merged_count += 1
            except Exception as e:
                print(f"  ✗ RQ3 merge failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ MERGING COMPLETE")
    print("=" * 80)
    print(f"\nMerged {merged_count} result files")
    print(f"Location: {FINAL_RESULTS_DIR}/")
    print()
    print("Next step:")
    print("  python scripts/visualizations_full.py")
    print()

if __name__ == "__main__":
    main()
