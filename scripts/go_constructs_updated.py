#!/usr/bin/env python3
"""
Script 8: Go-Specific Construct Analysis (UPDATED)
Analyzes how well models capture Go-specific language constructs.
This is a NOVEL contribution not present in the original paper.

NEW: Supports optimized HDF5 files with float16 and command-line arguments
Usage:
    python go_constructs.py --model unixcoder --task code-to-text [--sample]
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import argparse
import h5py

# Configuration
RESULTS_DIR = Path("results")
FEATURES_DIR = RESULTS_DIR / "features"
CODE_TO_TEXT_DIR = Path("data/code-to-text")
CODE_TO_CODE_DIR = Path("data/code-to-code")
OUTPUT_DIR = RESULTS_DIR / "go_constructs"

# Go-specific constructs to analyze
GO_CONSTRUCTS = [
    'goroutines',
    'channels',
    'defer',
    'error_patterns',
    'select_statements',
    'interfaces',
    'type_assertions',
    'context_usage'
]

# Key heads to analyze (from optimized extraction)
KEY_HEADS = [0, 3, 5, 7, 11]

class GoConstructAnalyzer:
    """Analyzes how well models capture Go-specific constructs."""
    
    def __init__(self):
        self.results = {}
    
    def load_data(self, features_file: Path, ast_file: Path, task_type: str) -> List[Dict]:
        """Load features from HDF5 and AST data with Go constructs."""
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
            
            # Determine code label
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
                go_constructs = ast_data[sample_id].get('go_constructs', {}).get(code_label)
                
                if not go_constructs:
                    continue
                
                # Extract attention weights (convert float16 to float32)
                attention_weights = {}
                if 'attention' in label_grp:
                    attn_grp = label_grp['attention']
                    for layer_key in attn_grp.keys():
                        layer_idx = int(layer_key.split('_')[1])
                        attention_weights[f'layer_{layer_idx}'] = {}
                        
                        layer_grp = attn_grp[layer_key]
                        for head_key in layer_grp.keys():
                            head_idx = int(head_key.split('_')[1])
                            attention_weights[f'layer_{layer_idx}'][f'head_{head_idx}'] = \
                                layer_grp[head_key][:].astype('float32')
                
                # Extract embeddings (convert float16 to float32)
                embeddings = {}
                if 'embeddings' in label_grp:
                    emb_grp = label_grp['embeddings']
                    for layer_key in emb_grp.keys():
                        layer_idx = int(layer_key.split('_')[1])
                        embeddings[f'layer_{layer_idx}'] = emb_grp[layer_key][:].astype('float32')
                
                merged.append({
                    'sample_id': sample_id,
                    'task_type': task_type,
                    'code_label': code_label,
                    'features': {
                        'attention_weights': attention_weights,
                        'embeddings': embeddings
                    },
                    'go_constructs': go_constructs
                })
        
        print(f"  ✓ Loaded {len(merged)} samples")
        return merged
    
    def get_construct_statistics(self, data: List[Dict]) -> Dict:
        """Get distribution statistics for each construct."""
        print("\nAnalyzing construct distribution...")
        
        construct_stats = {construct: {'count': 0, 'samples_with': 0, 'occurrences': []} 
                          for construct in GO_CONSTRUCTS}
        
        for sample in data:
            go_constructs = sample['go_constructs']
            
            for construct in GO_CONSTRUCTS:
                if construct in go_constructs and isinstance(go_constructs[construct], list):
                    occurrences = go_constructs[construct]
                    count = len(occurrences)
                    
                    if count > 0:
                        construct_stats[construct]['count'] += count
                        construct_stats[construct]['samples_with'] += 1
                        construct_stats[construct]['occurrences'].extend(occurrences)
        
        # Calculate percentages
        total_samples = len(data)
        for construct in GO_CONSTRUCTS:
            construct_stats[construct]['percentage'] = (
                construct_stats[construct]['samples_with'] / total_samples * 100
            )
        
        return construct_stats
    
    def analyze_construct_attention(self, data: List[Dict], construct_name: str,
                                   layer_idx: int, head_idx: int) -> Dict:
        """
        Analyze attention patterns for samples containing a specific Go construct.
        """
        results = {
            'construct': construct_name,
            'layer': layer_idx,
            'head': head_idx,
            'samples_with_construct': 0,
            'samples_without_construct': 0,
            'avg_attention_entropy_with': 0.0,
            'avg_attention_entropy_without': 0.0,
            'avg_max_attention_with': 0.0,
            'avg_max_attention_without': 0.0
        }
        
        entropy_with = []
        entropy_without = []
        max_attn_with = []
        max_attn_without = []
        
        for sample in data:
            go_constructs = sample['go_constructs']
            
            # Check if construct present
            has_construct = (construct_name in go_constructs and 
                           isinstance(go_constructs[construct_name], list) and
                           len(go_constructs[construct_name]) > 0)
            
            # Get attention matrix for this layer/head
            try:
                attention_weights = sample['features']['attention_weights']
                attention_matrix = attention_weights[f'layer_{layer_idx}'][f'head_{head_idx}']
            except (KeyError, IndexError):
                continue
            
            # Calculate attention metrics
            from scipy.stats import entropy
            avg_entropy = np.mean([entropy(row + 1e-10) for row in attention_matrix])
            max_attention = np.mean(np.max(attention_matrix, axis=1))
            
            if has_construct:
                entropy_with.append(avg_entropy)
                max_attn_with.append(max_attention)
                results['samples_with_construct'] += 1
            else:
                entropy_without.append(avg_entropy)
                max_attn_without.append(max_attention)
                results['samples_without_construct'] += 1
        
        # Aggregate results
        if entropy_with:
            results['avg_attention_entropy_with'] = float(np.mean(entropy_with))
            results['avg_max_attention_with'] = float(np.mean(max_attn_with))
        if entropy_without:
            results['avg_attention_entropy_without'] = float(np.mean(entropy_without))
            results['avg_max_attention_without'] = float(np.mean(max_attn_without))
        
        return results
    
    def analyze_construct_embeddings(self, data: List[Dict], construct_name: str,
                                    layer_idx: int) -> Dict:
        """
        Analyze embedding patterns for samples with vs without a specific Go construct.
        """
        results = {
            'construct': construct_name,
            'layer': layer_idx,
            'samples_with_construct': 0,
            'samples_without_construct': 0,
            'avg_embedding_norm_with': 0.0,
            'avg_embedding_norm_without': 0.0,
            'avg_embedding_std_with': 0.0,
            'avg_embedding_std_without': 0.0
        }
        
        norms_with = []
        norms_without = []
        stds_with = []
        stds_without = []
        
        for sample in data:
            go_constructs = sample['go_constructs']
            
            # Check if construct present
            has_construct = (construct_name in go_constructs and 
                           isinstance(go_constructs[construct_name], list) and
                           len(go_constructs[construct_name]) > 0)
            
            # Get embeddings for this layer
            try:
                embeddings = sample['features']['embeddings'][f'layer_{layer_idx}']
            except (KeyError, IndexError):
                continue
            
            # Calculate embedding statistics
            norms = np.linalg.norm(embeddings, axis=1)
            avg_norm = np.mean(norms)
            avg_std = np.mean(np.std(embeddings, axis=0))
            
            if has_construct:
                norms_with.append(avg_norm)
                stds_with.append(avg_std)
                results['samples_with_construct'] += 1
            else:
                norms_without.append(avg_norm)
                stds_without.append(avg_std)
                results['samples_without_construct'] += 1
        
        # Aggregate results
        if norms_with:
            results['avg_embedding_norm_with'] = float(np.mean(norms_with))
            results['avg_embedding_std_with'] = float(np.mean(stds_with))
        if norms_without:
            results['avg_embedding_norm_without'] = float(np.mean(norms_without))
            results['avg_embedding_std_without'] = float(np.mean(stds_without))
        
        return results
    
    def analyze_all_constructs(self, data: List[Dict], model_name: str) -> Dict:
        """Analyze all Go constructs across selected layers and heads."""
        print(f"\nAnalyzing Go constructs for {model_name}...")
        
        # Get construct statistics
        construct_stats = self.get_construct_statistics(data)
        
        # Get model dimensions
        sample_features = data[0]['features']
        num_layers = len(sample_features['attention_weights'])
        
        print(f"  Model: {num_layers} layers")
        print(f"  Analyzing heads: {KEY_HEADS}")
        
        # Analyze each construct
        construct_results = {}
        
        for construct in tqdm(GO_CONSTRUCTS, desc="  Analyzing constructs"):
            # Skip if too few samples
            if construct_stats[construct]['samples_with'] < 5:
                print(f"    Skipping {construct}: only {construct_stats[construct]['samples_with']} samples")
                continue
            
            print(f"\n  Analyzing: {construct}")
            print(f"    Present in {construct_stats[construct]['samples_with']} samples")
            print(f"    Total occurrences: {construct_stats[construct]['count']}")
            
            # Analyze attention for key layers and heads
            attention_results = []
            key_layers = [1, 4, 7, 10]
            
            for layer in key_layers:
                for head in KEY_HEADS:
                    # Check if this head exists in the data
                    if f'head_{head}' in sample_features['attention_weights'][f'layer_{layer}']:
                        attn_result = self.analyze_construct_attention(
                            data, construct, layer, head_idx=head
                        )
                        if attn_result['samples_with_construct'] > 0:
                            attention_results.append(attn_result)
            
            # Analyze embeddings for key layers
            embedding_results = []
            for layer in key_layers:
                emb_result = self.analyze_construct_embeddings(data, construct, layer)
                if emb_result['samples_with_construct'] > 0:
                    embedding_results.append(emb_result)
            
            construct_results[construct] = {
                'statistics': construct_stats[construct],
                'attention_analysis': attention_results,
                'embedding_analysis': embedding_results
            }
        
        return {
            'model_name': model_name,
            'construct_results': construct_results,
            'overall_statistics': construct_stats
        }

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Go construct analysis')
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
    print("GO-SPECIFIC CONSTRUCT ANALYSIS (RQ4 - NOVEL!)")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Task: {args.task}")
    print(f"Dataset: {'Sample (100)' if args.sample else 'Full'}")
    print(f"\nAnalyzing {len(GO_CONSTRUCTS)} Go-specific constructs:")
    for i, construct in enumerate(GO_CONSTRUCTS, 1):
        print(f"  {i}. {construct}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    analyzer = GoConstructAnalyzer()
    
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
    
    if not features_file.exists():
        print(f"  ✗ Features file not found: {features_file}")
        return
    
    if not ast_file.exists():
        print(f"  ✗ AST file not found: {ast_file}")
        return
    
    data = analyzer.load_data(features_file, ast_file, args.task)
    
    if len(data) == 0:
        print("  ⚠ No valid data. Exiting...")
        return
    
    analysis_name = f"{args.model}_{args.task}"
    results = analyzer.analyze_all_constructs(data, analysis_name)
    
    output_file = OUTPUT_DIR / f"{analysis_name}_go_constructs_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  ✓ Results saved to {output_file.name}")
    
    print("\n" + "=" * 80)
    print("✓ GO CONSTRUCT ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print(f"\nThis is your NOVEL contribution!")
    print(f"No prior work has analyzed Go-specific constructs like this.")
    print("\n")

if __name__ == "__main__":
    main()
