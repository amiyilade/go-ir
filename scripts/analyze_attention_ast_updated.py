#!/usr/bin/env python3
"""
Script 5: Attention-AST Alignment Analysis (UPDATED)
Analyzes how well attention weights align with AST syntactic structure.
Based on "What Do They Capture?" paper methodology.

NEW: Supports command-line arguments for model/task selection
Usage:
    python analyze_attention_ast.py --model unixcoder --task code-to-text [--sample]
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

# Key heads to analyze (from optimized extraction)
KEY_HEADS = [0, 3, 5, 7, 11]

class AttentionASTAligner:
    """Analyzes alignment between attention and AST structure."""
    
    def __init__(self):
        """Initialize the aligner."""
        self.alignment_results = defaultdict(list)
    
    def load_data(self, features_file: Path, ast_file: Path, task_type: str) -> List[Dict]:
        """Load features from HDF5 and corresponding AST data."""
        print(f"\nLoading data...")
        print(f"  Features: {features_file.name}")
        print(f"  AST data: {ast_file.name}")
        print(f"  Task type: {task_type}")
        
        # Load AST data
        with open(ast_file, 'r', encoding='utf-8') as f:
            ast_data = [json.loads(line) for line in f]
        
        # Load features from HDF5
        merged = []
        
        with h5py.File(features_file, 'r') as h5f:
            total_samples = h5f['metadata'].attrs['total_samples']
            
            # Determine code label based on task
            if task_type == 'code-to-text':
                code_label = 'code'
            else:
                code_label = 'initial_segment'
            
            for sample_id in range(total_samples):
                if sample_id >= len(ast_data):
                    continue
                
                sample_key = f'sample_{sample_id}'
                if sample_key not in h5f:
                    continue
                
                if code_label not in h5f[sample_key]:
                    continue
                
                label_grp = h5f[sample_key][code_label]
                
                # Extract attention weights
                attention_weights = {}
                if 'attention' in label_grp:
                    attn_grp = label_grp['attention']
                    for layer_key in attn_grp.keys():
                        layer_idx = int(layer_key.split('_')[1])
                        attention_weights[f'layer_{layer_idx}'] = {}
                        
                        for head_key in attn_grp[layer_key].keys():
                            head_idx = int(head_key.split('_')[1])
                            attention_weights[f'layer_{layer_idx}'][f'head_{head_idx}'] = \
                                attn_grp[layer_key][head_key][:].tolist()
                
                # Get AST info
                ast_info = ast_data[sample_id].get('ast_info', {}).get(code_label)
                
                if ast_info:
                    merged.append({
                        'sample_id': sample_id,
                        'task_type': task_type,
                        'code_label': code_label,
                        'features': {
                            'attention_weights': attention_weights
                        },
                        'ast': ast_info
                    })
        
        print(f"  ✓ Loaded {len(merged)} samples")
        return merged
    
    def get_ast_parent_pairs(self, ast_tree: Dict) -> set:
        """
        Extract token pairs that share the same parent in AST.
        Based on paper: "syntactic relation = same parent node"
        """
        parent_pairs = set()
        
        def traverse(node, parent_children=None):
            """Recursively collect sibling pairs (same parent)."""
            if 'children' in node:
                children = node['children']
                
                # Collect leaf node indices for this parent's children
                leaf_indices = []
                for child in children:
                    indices = self._get_leaf_indices(child)
                    leaf_indices.extend(indices)
                
                # Create pairs of siblings (shared parent)
                for i in range(len(leaf_indices)):
                    for j in range(i + 1, len(leaf_indices)):
                        # Exclude adjacent tokens (as per paper)
                        if abs(leaf_indices[i] - leaf_indices[j]) > 1:
                            parent_pairs.add((leaf_indices[i], leaf_indices[j]))
                            parent_pairs.add((leaf_indices[j], leaf_indices[i]))
                
                # Recurse on children
                for child in children:
                    traverse(child, children)
        
        if ast_tree:
            traverse(ast_tree.get('ast_tree', {}))
        
        return parent_pairs
    
    def _get_leaf_indices(self, node, current_index=None) -> List[int]:
        """Get indices of all leaf nodes under this node."""
        if current_index is None:
            current_index = [0]
        
        indices = []
        
        if 'children' not in node or len(node.get('children', [])) == 0:
            # Leaf node
            indices.append(current_index[0])
            current_index[0] += 1
        else:
            # Internal node
            for child in node['children']:
                indices.extend(self._get_leaf_indices(child, current_index))
        
        return indices
    
    def calculate_attention_variability(self, attention_matrices: List[np.ndarray],
                                       max_tokens: int = 10) -> float:
        """
        Calculate attention variability as defined in paper (Equation 5).
        High variability = content-dependent head
        Low variability = position-based head
        """
        if len(attention_matrices) == 0:
            return 0.0
        
        # Only use first N tokens
        truncated_attention = []
        for attn in attention_matrices:
            if attn.shape[0] >= max_tokens and attn.shape[1] >= max_tokens:
                truncated_attention.append(attn[:max_tokens, :max_tokens])
        
        if len(truncated_attention) == 0:
            return 0.0
        
        # Stack into array: [num_samples, max_tokens, max_tokens]
        attention_array = np.stack(truncated_attention)
        
        # Calculate mean attention across samples
        mean_attention = np.mean(attention_array, axis=0)
        
        # Calculate variability (Equation 5 from paper)
        squared_diff = np.sum((attention_array - mean_attention) ** 2)
        total_attention = np.sum(attention_array)
        
        variability = squared_diff / total_attention if total_attention > 0 else 0.0
        
        return variability
    
    def calculate_alignment_score(self, attention_matrix: np.ndarray,
                                 ast_parent_pairs: set,
                                 threshold: float = ATTENTION_THRESHOLD) -> Dict:
        """
        Calculate p_α(f) as defined in paper (Equation 4).
        Proportion of high-attention pairs that align with AST structure.
        """
        seq_len = attention_matrix.shape[0]
        
        # Find high-confidence attention pairs (α_i,j > θ)
        high_attention_pairs = []
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j and attention_matrix[i, j] > threshold:
                    high_attention_pairs.append((i, j))
        
        # Count alignments
        aligned_count = 0
        for pair in high_attention_pairs:
            if pair in ast_parent_pairs:
                aligned_count += 1
        
        # Calculate alignment proportion
        total_high_attention = len(high_attention_pairs)
        alignment_score = aligned_count / total_high_attention if total_high_attention > 0 else 0.0
        
        return {
            'alignment_score': alignment_score,
            'aligned_pairs': aligned_count,
            'total_high_attention_pairs': total_high_attention,
            'total_ast_pairs': len(ast_parent_pairs)
        }
    
    def analyze_layer_head(self, data: List[Dict], layer_idx: int, head_idx: int) -> Dict:
        """Analyze specific layer and head across dataset."""
        layer_key = f'layer_{layer_idx}'
        head_key = f'head_{head_idx}'
        
        alignment_scores = []
        variability_matrices = []
        valid_samples = 0
        
        for sample in data:
            features = sample['features']
            ast = sample['ast']
            
            if not ast or layer_key not in features['attention_weights']:
                continue
            
            if head_key not in features['attention_weights'][layer_key]:
                continue
            
            # Get attention matrix for this layer-head
            attention_matrix = np.array(
                features['attention_weights'][layer_key][head_key]
            )
            
            # Get AST parent pairs
            ast_parent_pairs = self.get_ast_parent_pairs(ast)
            
            if len(ast_parent_pairs) == 0:
                continue
            
            # Calculate alignment
            alignment = self.calculate_alignment_score(attention_matrix, ast_parent_pairs)
            
            # Only include if enough high-confidence attention scores
            if alignment['total_high_attention_pairs'] >= MIN_HIGH_CONFIDENCE_SCORES:
                alignment_scores.append(alignment['alignment_score'])
                variability_matrices.append(attention_matrix)
                valid_samples += 1
        
        if len(alignment_scores) == 0:
            return None
        
        # Calculate variability
        variability = self.calculate_attention_variability(variability_matrices)
        
        return {
            'layer': layer_idx,
            'head': head_idx,
            'mean_alignment_score': np.mean(alignment_scores),
            'std_alignment_score': np.std(alignment_scores),
            'variability': variability,
            'valid_samples': valid_samples,
            'head_type': 'content-dependent' if variability > 0.25 else 'position-based'
        }
    
    def analyze_all_layers_heads(self, data: List[Dict], model_name: str) -> Dict:
        """Analyze all layers and heads for a model."""
        print(f"\nAnalyzing all layers and heads for {model_name}...")
        
        # Get model dimensions
        sample_features = data[0]['features']
        num_layers = len(sample_features['attention_weights'])
        
        # Determine available heads (could be KEY_HEADS or all heads)
        sample_layer = sample_features['attention_weights']['layer_0']
        available_heads = [int(k.split('_')[1]) for k in sample_layer.keys()]
        
        print(f"  Model: {num_layers} layers, {len(available_heads)} heads per layer")
        print(f"  Available heads: {sorted(available_heads)}")
        
        # Analyze each layer-head combination
        results = []
        total_combinations = num_layers * len(available_heads)
        
        with tqdm(total=total_combinations, desc="  Analyzing layer-head combinations") as pbar:
            for layer_idx in range(num_layers):
                for head_idx in available_heads:
                    result = self.analyze_layer_head(data, layer_idx, head_idx)
                    if result:
                        results.append(result)
                    pbar.update(1)
        
        print(f"  ✓ Analyzed {len(results)} valid layer-head combinations")
        
        # Find best performing heads
        results_sorted = sorted(results, key=lambda x: x['mean_alignment_score'], reverse=True)
        
        # Calculate per-layer statistics
        layer_stats = defaultdict(list)
        for result in results:
            layer_stats[result['layer']].append(result['mean_alignment_score'])
        
        layer_summary = {
            f'layer_{layer}': {
                'mean_alignment': np.mean(scores),
                'max_alignment': np.max(scores),
                'std_alignment': np.std(scores)
            }
            for layer, scores in layer_stats.items()
        }
        
        return {
            'model_name': model_name,
            'all_results': results,
            'top_10_heads': results_sorted[:10],
            'layer_summary': layer_summary,
            'overall_stats': {
                'mean_alignment': np.mean([r['mean_alignment_score'] for r in results]),
                'max_alignment': np.max([r['mean_alignment_score'] for r in results]),
                'content_dependent_heads': sum(1 for r in results if r['head_type'] == 'content-dependent'),
                'position_based_heads': sum(1 for r in results if r['head_type'] == 'position-based')
            }
        }

def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze attention-AST alignment')
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
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    aligner = AttentionASTAligner()
    
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
    
    # Features file (HDF5)
    task_name = args.task.replace('-', '_')
    features_file = FEATURES_DIR / f"{prefix}{task_name}_{args.model}_optimized.h5"
    
    # Alternative: non-optimized file
    if not features_file.exists():
        features_file = FEATURES_DIR / f"{prefix}{task_name}_{args.model}.h5"
    
    print(f"\nLooking for features: {features_file}")
    
    if not features_file.exists():
        print(f"  ✗ Features file not found!")
        print(f"    Run feature extraction first:")
        print(f"    python scripts/extract_features_chunked_optimized.py \\")
        print(f"        --model {args.model} --task {args.task}")
        return
    
    if not ast_file.exists():
        print(f"  ✗ AST file not found: {ast_file}")
        return
    
    # Load data
    data = aligner.load_data(features_file, ast_file, args.task)
    
    if len(data) == 0:
        print("  ⚠ No valid data loaded. Exiting...")
        return
    
    # Analyze
    analysis_name = f"{args.model}_{args.task}"
    results = aligner.analyze_all_layers_heads(data, analysis_name)
    
    # Save results
    output_file = OUTPUT_DIR / f"{analysis_name}_alignment_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  ✓ Results saved to {output_file.name}")
    
    # Print summary
    print(f"\n  [Summary Statistics]")
    print(f"    Mean alignment score: {results['overall_stats']['mean_alignment']:.3f}")
    print(f"    Max alignment score: {results['overall_stats']['max_alignment']:.3f}")
    print(f"    Content-dependent heads: {results['overall_stats']['content_dependent_heads']}")
    print(f"    Position-based heads: {results['overall_stats']['position_based_heads']}")
    
    print(f"\n  [Top 3 Aligned Heads]")
    for i, head in enumerate(results['top_10_heads'][:3]):
        print(f"    {i+1}. Layer {head['layer']}, Head {head['head']}: "
              f"{head['mean_alignment_score']:.3f} ({head['head_type']})")
    
    print("\n" + "=" * 80)
    print("✓ ALIGNMENT ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print("\n")

if __name__ == "__main__":
    main()
