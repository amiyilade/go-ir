#!/usr/bin/env python3
"""
Script 5: Attention-AST Alignment Analysis (STREAMING VERSION)
Memory-efficient version that loads attention matrices on-demand.

Usage:
    python analyze_attention_ast_streaming.py --model unixcoder --task code-to-text
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse
import h5py

# Configuration
RESULTS_DIR = Path("results")
FEATURES_DIR = RESULTS_DIR / "features"
CODE_TO_TEXT_DIR = Path("data/code-to-text")
CODE_TO_CODE_DIR = Path("data/code-to-code")
OUTPUT_DIR = RESULTS_DIR / "ast_alignment"

# Parameters from paper
ATTENTION_THRESHOLD = 0.3
MIN_HIGH_CONFIDENCE_SCORES = 100

# Key heads to analyze
KEY_HEADS = [0, 3, 5, 7, 11]

class StreamingAttentionASTAligner:
    """Memory-efficient analyzer using streaming/lazy loading."""
    
    def __init__(self, features_file: Path):
        self.features_file = features_file
        self.alignment_results = defaultdict(list)
    
    def load_sample_index(self, ast_file: Path, task_type: str) -> Tuple[List[Dict], str]:
        """Build index of valid samples without loading attention matrices."""
        print(f"\nBuilding sample index...")
        print(f"  Features: {self.features_file.name}")
        print(f"  AST data: {ast_file.name}")
        
        # Determine code label
        code_label = 'code' if task_type == 'code-to-text' else 'initial_segment'
        
        # Load AST data
        print(f"  Loading AST data...")
        with open(ast_file, 'r', encoding='utf-8') as f:
            ast_data = [json.loads(line) for line in f]
        
        # Build sample index
        sample_index = []
        
        with h5py.File(self.features_file, 'r') as h5f:
            total_samples = h5f['metadata'].attrs['total_samples']
            
            print(f"  Indexing {total_samples} samples...")
            for sample_id in tqdm(range(total_samples), desc="  Progress"):
                if sample_id >= len(ast_data):
                    continue
                
                sample_key = f'sample_{sample_id}'
                if sample_key not in h5f or code_label not in h5f[sample_key]:
                    continue
                
                ast_info = ast_data[sample_id].get('ast_info', {}).get(code_label)
                if not ast_info:
                    continue
                
                # Store only metadata (no attention matrices!)
                sample_index.append({
                    'sample_id': sample_id,
                    'sample_key': sample_key,
                    'ast': ast_info
                })
        
        print(f"  ✓ Indexed {len(sample_index)} valid samples")
        return sample_index, code_label
    
    def get_attention_matrix(self, sample_key: str, code_label: str, 
                            layer_idx: int, head_idx: int) -> np.ndarray:
        """Load single attention matrix on-demand (lazy loading)."""
        with h5py.File(self.features_file, 'r') as h5f:
            path = f'{sample_key}/{code_label}/attention/layer_{layer_idx}/head_{head_idx}'
            if path in h5f:
                # Load as float16, convert to float32 for computation
                return h5f[path][:].astype('float32')
        return None
    
    def get_ast_parent_pairs(self, ast_tree: Dict) -> set:
        """Extract token pairs that share the same parent in AST."""
        parent_pairs = set()
        
        def traverse(node):
            if 'children' in node:
                children = node['children']
                leaf_indices = []
                
                for child in children:
                    indices = self._get_leaf_indices(child)
                    leaf_indices.extend(indices)
                
                # Create pairs of siblings
                for i in range(len(leaf_indices)):
                    for j in range(i + 1, len(leaf_indices)):
                        if abs(leaf_indices[i] - leaf_indices[j]) > 1:
                            parent_pairs.add((leaf_indices[i], leaf_indices[j]))
                            parent_pairs.add((leaf_indices[j], leaf_indices[i]))
                
                for child in children:
                    traverse(child)
        
        if ast_tree:
            traverse(ast_tree.get('ast_tree', {}))
        
        return parent_pairs
    
    def _get_leaf_indices(self, node, current_index=None) -> List[int]:
        """Get indices of all leaf nodes under this node."""
        if current_index is None:
            current_index = [0]
        
        indices = []
        
        if 'children' not in node or len(node.get('children', [])) == 0:
            indices.append(current_index[0])
            current_index[0] += 1
        else:
            for child in node['children']:
                indices.extend(self._get_leaf_indices(child, current_index))
        
        return indices
    
    def calculate_attention_variability(self, attention_matrices: List[np.ndarray],
                                       max_tokens: int = 10) -> float:
        """Calculate attention variability (position-based vs content-dependent)."""
        if len(attention_matrices) == 0:
            return 0.0
        
        truncated_attention = []
        for attn in attention_matrices:
            if attn.shape[0] >= max_tokens and attn.shape[1] >= max_tokens:
                truncated_attention.append(attn[:max_tokens, :max_tokens])
        
        if len(truncated_attention) == 0:
            return 0.0
        
        attention_array = np.stack(truncated_attention)
        mean_attention = np.mean(attention_array, axis=0)
        squared_diff = np.sum((attention_array - mean_attention) ** 2)
        total_attention = np.sum(attention_array)
        
        variability = squared_diff / total_attention if total_attention > 0 else 0.0
        return variability
    
    def calculate_alignment_score(self, attention_matrix: np.ndarray,
                                 ast_parent_pairs: set,
                                 threshold: float = ATTENTION_THRESHOLD) -> Dict:
        """Calculate proportion of high-attention pairs that align with AST."""
        seq_len = attention_matrix.shape[0]
        
        high_attention_pairs = []
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j and attention_matrix[i, j] > threshold:
                    high_attention_pairs.append((i, j))
        
        aligned_count = sum(1 for pair in high_attention_pairs if pair in ast_parent_pairs)
        
        total_high_attention = len(high_attention_pairs)
        alignment_score = aligned_count / total_high_attention if total_high_attention > 0 else 0.0
        
        return {
            'alignment_score': alignment_score,
            'aligned_pairs': aligned_count,
            'total_high_attention_pairs': total_high_attention,
            'total_ast_pairs': len(ast_parent_pairs)
        }
    
    def analyze_layer_head(self, sample_index: List[Dict], code_label: str,
                          layer_idx: int, head_idx: int) -> Dict:
        """Analyze specific layer-head using streaming (loads one matrix at a time)."""
        alignment_scores = []
        variability_matrices = []
        valid_samples = 0
        
        # Stream through samples (load attention on-demand)
        for sample in sample_index:
            # Load ONLY this one attention matrix (streaming!)
            attention_matrix = self.get_attention_matrix(
                sample['sample_key'], code_label, layer_idx, head_idx
            )
            
            if attention_matrix is None:
                continue
            
            # Get AST parent pairs
            ast_parent_pairs = self.get_ast_parent_pairs(sample['ast'])
            if len(ast_parent_pairs) == 0:
                continue
            
            # Calculate alignment
            alignment = self.calculate_alignment_score(attention_matrix, ast_parent_pairs)
            
            if alignment['total_high_attention_pairs'] >= MIN_HIGH_CONFIDENCE_SCORES:
                alignment_scores.append(alignment['alignment_score'])
                variability_matrices.append(attention_matrix)
                valid_samples += 1
            
            # Free memory immediately after use
            del attention_matrix
        
        if len(alignment_scores) == 0:
            return None
        
        # Calculate variability
        variability = self.calculate_attention_variability(variability_matrices)
        
        return {
            'layer': layer_idx,
            'head': head_idx,
            'mean_alignment_score': float(np.mean(alignment_scores)),
            'std_alignment_score': float(np.std(alignment_scores)),
            'variability': float(variability),
            'valid_samples': valid_samples,
            'head_type': 'content-dependent' if variability > 0.25 else 'position-based'
        }
    
    def analyze_all_layers_heads(self, sample_index: List[Dict], code_label: str,
                                 model_name: str) -> Dict:
        """Analyze all layers and heads using streaming."""
        print(f"\nAnalyzing all layers and heads...")
        
        # Determine number of layers from HDF5 metadata
        with h5py.File(self.features_file, 'r') as h5f:
            # Check first valid sample to get number of layers
            for sample in sample_index[:10]:
                sample_key = sample['sample_key']
                if f'{sample_key}/{code_label}/attention' in h5f:
                    num_layers = len(h5f[f'{sample_key}/{code_label}/attention'].keys())
                    break
        
        print(f"  Model: {num_layers} layers, {len(KEY_HEADS)} heads per layer")
        print(f"  Total combinations: {num_layers * len(KEY_HEADS)}")
        
        # Analyze each layer-head combination
        results = []
        total_combinations = num_layers * len(KEY_HEADS)
        
        with tqdm(total=total_combinations, desc="  Analyzing combinations") as pbar:
            for layer_idx in range(num_layers):
                for head_idx in KEY_HEADS:
                    result = self.analyze_layer_head(
                        sample_index, code_label, layer_idx, head_idx
                    )
                    if result:
                        results.append(result)
                    pbar.update(1)
        
        print(f"  ✓ Analyzed {len(results)} valid combinations")
        
        # Sort by alignment score
        results_sorted = sorted(results, key=lambda x: x['mean_alignment_score'], reverse=True)
        
        # Calculate per-layer statistics
        layer_stats = defaultdict(list)
        for result in results:
            layer_stats[result['layer']].append(result['mean_alignment_score'])
        
        layer_summary = {
            f'layer_{layer}': {
                'mean_alignment': float(np.mean(scores)),
                'max_alignment': float(np.max(scores)),
                'std_alignment': float(np.std(scores))
            }
            for layer, scores in layer_stats.items()
        }
        
        return {
            'model_name': model_name,
            'all_results': results,
            'top_10_heads': results_sorted[:10],
            'layer_summary': layer_summary,
            'overall_stats': {
                'mean_alignment': float(np.mean([r['mean_alignment_score'] for r in results])),
                'max_alignment': float(np.max([r['mean_alignment_score'] for r in results])),
                'content_dependent_heads': sum(1 for r in results if r['head_type'] == 'content-dependent'),
                'position_based_heads': sum(1 for r in results if r['head_type'] == 'position-based')
            }
        }

def main():
    parser = argparse.ArgumentParser(description='Attention-AST alignment analysis (streaming)')
    parser.add_argument('--model', type=str, required=True,
                       choices=['unixcoder', 'codebert'],
                       help='Model to analyze')
    parser.add_argument('--task', type=str, required=True,
                       choices=['code-to-text', 'code-to-code'],
                       help='Task type')
    parser.add_argument('--sample', action='store_true',
                       help='Use sample dataset (100 samples) instead of full')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("ATTENTION-AST ALIGNMENT ANALYSIS (RQ1)")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Task: {args.task}")
    print(f"Dataset: {'Sample (100)' if args.sample else 'Full'}")
    print(f"\nParameters (from paper):")
    print(f"  Attention threshold (θ): {ATTENTION_THRESHOLD}")
    print(f"  Min high-confidence scores: {MIN_HIGH_CONFIDENCE_SCORES}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set file paths
    if args.sample:
        prefix = "sample_"
        if args.task == 'code-to-text':
            ast_file = CODE_TO_TEXT_DIR / "sample_100_with_asts.jsonl"
        else:
            ast_file = CODE_TO_CODE_DIR / "sample_100_with_asts.jsonl"
    else:
        prefix = ""
        if args.task == 'code-to-text':
            ast_file = CODE_TO_TEXT_DIR / "full_code_to_text_with_asts.jsonl"
        else:
            ast_file = CODE_TO_CODE_DIR / "full_code_to_code_with_asts.jsonl"
    
    task_name = args.task.replace('-', '_')
    features_file = FEATURES_DIR / f"{prefix}{task_name}_{args.model}_optimized.h5"
    
    if not features_file.exists():
        features_file = FEATURES_DIR / f"{prefix}{task_name}_{args.model}.h5"
    
    print(f"\nLooking for features: {features_file}")
    
    if not features_file.exists():
        print(f"  ✗ Features file not found: {features_file}")
        return
    
    if not ast_file.exists():
        print(f"  ✗ AST file not found: {ast_file}")
        return
    
    # Initialize streaming analyzer
    aligner = StreamingAttentionASTAligner(features_file)
    
    # Build sample index (fast - no attention loading!)
    sample_index, code_label = aligner.load_sample_index(ast_file, args.task)
    
    if len(sample_index) == 0:
        print("  ⚠ No valid samples. Exiting...")
        return
    
    # Analyze all layers/heads (streaming - loads one matrix at a time)
    analysis_name = f"{args.model}_{args.task}"
    results = aligner.analyze_all_layers_heads(sample_index, code_label, analysis_name)
    
    # Save results
    output_file = OUTPUT_DIR / f"{analysis_name}_alignment_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  ✓ Results saved to {output_file.name}")
    
    # Print summary
    print(f"\n[Summary Statistics]")
    print(f"  Mean alignment score: {results['overall_stats']['mean_alignment']:.3f}")
    print(f"  Max alignment score: {results['overall_stats']['max_alignment']:.3f}")
    print(f"  Content-dependent heads: {results['overall_stats']['content_dependent_heads']}")
    print(f"  Position-based heads: {results['overall_stats']['position_based_heads']}")
    
    print(f"\n[Top 3 Aligned Heads]")
    for i, head in enumerate(results['top_10_heads'][:3]):
        print(f"  {i+1}. Layer {head['layer']}, Head {head['head']}: "
              f"{head['mean_alignment_score']:.3f} ({head['head_type']})")
    
    print("\n" + "=" * 80)
    print("✓ ALIGNMENT ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n")

if __name__ == "__main__":
    main()
