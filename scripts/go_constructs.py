#!/usr/bin/env python3
"""
NEW: Construct-Level Analysis (Proposal RQ1-3)
Analyzes how models encode Go-specific constructs with statistical tests.

Research Questions:
RQ1: What is the prevalence of different Go constructs?
RQ2: Do attention patterns differ for samples with vs without specific constructs?
RQ3: Are Go constructs encoded distinctly in internal representations?
"""

import json
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
from scipy import stats
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
RESULTS_DIR = Path("results")
FEATURES_DIR = RESULTS_DIR / "features"
CODE_TO_TEXT_DIR = Path("data/code-to-text")
CODE_TO_CODE_DIR = Path("data/code-to-code")
OUTPUT_DIR = RESULTS_DIR / "construct_analysis"

# Go constructs to analyze
GO_CONSTRUCTS = [
    'error_patterns',
    'goroutines',
    'channels',
    'defer',
    'interfaces',
    'type_assertions',
    'select_statements',
    'context_usage'
]

class ConstructAnalyzer:
    """Analyzes construct-specific encoding patterns."""
    
    def __init__(self):
        self.results = {}
    
    def load_data(self, features_file: Path, ast_file: Path, task_type: str) -> List[Dict]:
        """Load features from HDF5 and corresponding AST data."""
        print(f"
Loading data...")
        print(f"  Features: {features_file.name}")
        print(f"  AST data: {ast_file.name}")
        print(f"  Task type: {task_type}")
        
        # Load AST data (still JSON)
        with open(ast_file, 'r', encoding='utf-8') as f:
            ast_data = [json.loads(line) for line in f]
        
        # Determine code label
        code_label = 'code' if task_type == 'code-to-text' else 'initial_segment'
        
        # Load features from HDF5
        merged = []
        with h5py.File(features_file, 'r') as h5f:
            total_samples = h5f['metadata'].attrs['total_samples']
            
            for sample_id in range(total_samples):
                sample_key = f'sample_{sample_id}'
                if sample_key not in h5f or code_label not in h5f[sample_key]:
                    continue
                
                label_grp = h5f[sample_key][code_label]
                
                # Read attention (already numpy arrays!)
                attention_weights = {}
                attn_grp = label_grp['attention']
                for layer_key in attn_grp.keys():
                    attention_weights[layer_key] = {}
                    for head_key in attn_grp[layer_key].keys():
                        attention_weights[layer_key][head_key] = attn_grp[layer_key][head_key][:]
                
                # Read embeddings
                embeddings = {}
                emb_grp = label_grp['embeddings']
                for layer_key in emb_grp.keys():
                    embeddings[layer_key] = emb_grp[layer_key][:]
                
                # Merge with AST
                if sample_id < len(ast_data):
                    merged.append({
                        'sample_id': sample_id,
                        'task_type': task_type,
                        'code_label': code_label,
                        'features': {
                            'attention_weights': attention_weights,
                            'embeddings': embeddings
                        },
                        'ast': ast_data[sample_id].get('ast_info', {}).get(code_label),
                        'go_constructs': ast_data[sample_id].get('go_constructs', {}).get(code_label)
                    })
        
        print(f"  ✓ Loaded {len(merged)} samples")
        return merged
    
    def get_construct_prevalence(self, data: List[Dict]) -> Dict:
        """RQ1: Analyze prevalence of each Go construct."""
        print("\n[RQ1: Construct Prevalence Analysis]")
        
        prevalence = {}
        total_samples = len(data)
        
        for construct in GO_CONSTRUCTS:
            samples_with = 0
            total_occurrences = 0
            occurrence_distribution = []
            
            for sample in data:
                constructs = sample['go_constructs']
                if construct in constructs and isinstance(constructs[construct], list):
                    count = len(constructs[construct])
                    if count > 0:
                        samples_with += 1
                        total_occurrences += count
                        occurrence_distribution.append(count)
            
            prevalence[construct] = {
                'samples_with_construct': samples_with,
                'percentage': (samples_with / total_samples * 100) if total_samples > 0 else 0,
                'total_occurrences': total_occurrences,
                'avg_occurrences_when_present': (total_occurrences / samples_with) if samples_with > 0 else 0,
                'occurrence_distribution': occurrence_distribution
            }
            
            print(f"  {construct}:")
            print(f"    Present in: {samples_with}/{total_samples} ({prevalence[construct]['percentage']:.1f}%)")
            print(f"    Total occurrences: {total_occurrences}")
            if samples_with > 0:
                print(f"    Avg per sample: {prevalence[construct]['avg_occurrences_when_present']:.2f}")
        
        return prevalence
    
    def cohen_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        return d
    
    def analyze_attention_by_construct(self, data: List[Dict], construct: str, 
                                      layer_idx: int, head_idx: int) -> Dict:
        """RQ2: Compare attention patterns for samples with/without construct."""
        
        # Collect attention metrics
        entropy_with = []
        entropy_without = []
        max_attn_with = []
        max_attn_without = []
        
        for sample in data:
            go_constructs = sample['go_constructs']
            
            # Check if construct present
            has_construct = (construct in go_constructs and 
                           isinstance(go_constructs[construct], list) and
                           len(go_constructs[construct]) > 0)
            
            # Get attention matrix
            try:
                attention_weights = sample['features']['attention_weights']
                attention_matrix = np.array(
                    attention_weights[f'layer_{layer_idx}'][f'head_{head_idx}']
                )
            except (KeyError, IndexError):
                continue
            
            # Calculate metrics
            from scipy.stats import entropy as scipy_entropy
            
            # Attention entropy (measure of diffusion)
            avg_entropy = np.mean([scipy_entropy(row + 1e-10) for row in attention_matrix])
            
            # Max attention (measure of focus)
            max_attention = np.mean(np.max(attention_matrix, axis=1))
            
            if has_construct:
                entropy_with.append(avg_entropy)
                max_attn_with.append(max_attention)
            else:
                entropy_without.append(avg_entropy)
                max_attn_without.append(max_attention)
        
        # Statistical tests
        results = {
            'construct': construct,
            'layer': layer_idx,
            'head': head_idx,
            'n_with': len(entropy_with),
            'n_without': len(entropy_without)
        }
        
        if len(entropy_with) >= 5 and len(entropy_without) >= 5:
            # T-test for entropy
            t_stat_entropy, p_val_entropy = ttest_ind(entropy_with, entropy_without)
            cohen_d_entropy = self.cohen_d(np.array(entropy_with), np.array(entropy_without))
            
            # T-test for max attention
            t_stat_max, p_val_max = ttest_ind(max_attn_with, max_attn_without)
            cohen_d_max = self.cohen_d(np.array(max_attn_with), np.array(max_attn_without))
            
            results.update({
                'entropy': {
                    'mean_with': float(np.mean(entropy_with)),
                    'mean_without': float(np.mean(entropy_without)),
                    't_statistic': float(t_stat_entropy),
                    'p_value': float(p_val_entropy),
                    'cohen_d': float(cohen_d_entropy),
                    'significant': p_val_entropy < 0.05
                },
                'max_attention': {
                    'mean_with': float(np.mean(max_attn_with)),
                    'mean_without': float(np.mean(max_attn_without)),
                    't_statistic': float(t_stat_max),
                    'p_value': float(p_val_max),
                    'cohen_d': float(cohen_d_max),
                    'significant': p_val_max < 0.05
                }
            })
        
        return results
    
    def analyze_embeddings_by_construct(self, data: List[Dict], construct: str,
                                       layer_idx: int) -> Dict:
        """RQ3: Compare embedding patterns for samples with/without construct."""
        
        # Collect embedding metrics
        norms_with = []
        norms_without = []
        stds_with = []
        stds_without = []
        
        for sample in data:
            go_constructs = sample['go_constructs']
            
            # Check if construct present
            has_construct = (construct in go_constructs and 
                           isinstance(go_constructs[construct], list) and
                           len(go_constructs[construct]) > 0)
            
            # Get embeddings
            try:
                embeddings = np.array(sample['features']['embeddings'][f'layer_{layer_idx}'])
            except (KeyError, IndexError):
                continue
            
            # Calculate metrics
            # Average L2 norm across tokens
            norms = np.linalg.norm(embeddings, axis=1)
            avg_norm = np.mean(norms)
            
            # Standard deviation (measure of variability)
            avg_std = np.mean(np.std(embeddings, axis=0))
            
            if has_construct:
                norms_with.append(avg_norm)
                stds_with.append(avg_std)
            else:
                norms_without.append(avg_norm)
                stds_without.append(avg_std)
        
        # Statistical tests
        results = {
            'construct': construct,
            'layer': layer_idx,
            'n_with': len(norms_with),
            'n_without': len(norms_without)
        }
        
        if len(norms_with) >= 5 and len(norms_without) >= 5:
            # T-test for norms
            t_stat_norm, p_val_norm = ttest_ind(norms_with, norms_without)
            cohen_d_norm = self.cohen_d(np.array(norms_with), np.array(norms_without))
            
            # T-test for std
            t_stat_std, p_val_std = ttest_ind(stds_with, stds_without)
            cohen_d_std = self.cohen_d(np.array(stds_with), np.array(stds_without))
            
            results.update({
                'embedding_norm': {
                    'mean_with': float(np.mean(norms_with)),
                    'mean_without': float(np.mean(norms_without)),
                    't_statistic': float(t_stat_norm),
                    'p_value': float(p_val_norm),
                    'cohen_d': float(cohen_d_norm),
                    'significant': p_val_norm < 0.05
                },
                'embedding_std': {
                    'mean_with': float(np.mean(stds_with)),
                    'mean_without': float(np.mean(stds_without)),
                    't_statistic': float(t_stat_std),
                    'p_value': float(p_val_std),
                    'cohen_d': float(cohen_d_std),
                    'significant': p_val_std < 0.05
                }
            })
        
        return results
    
    def comprehensive_construct_analysis(self, data: List[Dict], model_name: str) -> Dict:
        """Run comprehensive analysis for all constructs."""
        print(f"\n[Comprehensive Construct Analysis: {model_name}]")
        
        # Get prevalence
        prevalence = self.get_construct_prevalence(data)
        
        # Analyze key layers (based on prior results)
        key_layers = [1, 4, 7, 10]
        key_head = 7  # Best performing head from prior analysis
        
        construct_results = {}
        
        for construct in tqdm(GO_CONSTRUCTS, desc="Analyzing constructs"):
            if prevalence[construct]['samples_with_construct'] < 10:
                print(f"  Skipping {construct}: only {prevalence[construct]['samples_with_construct']} samples")
                continue
            
            print(f"\n  Analyzing: {construct}")
            
            # Attention analysis across layers
            attention_results = []
            for layer in key_layers:
                attn_result = self.analyze_attention_by_construct(
                    data, construct, layer, key_head
                )
                if 'entropy' in attn_result:
                    attention_results.append(attn_result)
            
            # Embedding analysis across layers
            embedding_results = []
            for layer in key_layers:
                emb_result = self.analyze_embeddings_by_construct(
                    data, construct, layer
                )
                if 'embedding_norm' in emb_result:
                    embedding_results.append(emb_result)
            
            construct_results[construct] = {
                'prevalence': prevalence[construct],
                'attention_analysis': attention_results,
                'embedding_analysis': embedding_results,
                'significant_attention_layers': [
                    r['layer'] for r in attention_results 
                    if r.get('entropy', {}).get('significant') or r.get('max_attention', {}).get('significant')
                ],
                'significant_embedding_layers': [
                    r['layer'] for r in embedding_results
                    if r.get('embedding_norm', {}).get('significant') or r.get('embedding_std', {}).get('significant')
                ]
            }
            
            # Print summary
            if attention_results:
                sig_attn = len([r for r in attention_results 
                              if r.get('entropy', {}).get('significant')])
                print(f"    Significant attention layers: {sig_attn}/{len(attention_results)}")
            
            if embedding_results:
                sig_emb = len([r for r in embedding_results
                             if r.get('embedding_norm', {}).get('significant')])
                print(f"    Significant embedding layers: {sig_emb}/{len(embedding_results)}")
        
        return {
            'model_name': model_name,
            'prevalence_summary': prevalence,
            'construct_analyses': construct_results
        }


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("GO CONSTRUCT-LEVEL ANALYSIS (Proposal RQ1-3)")
    print("=" * 80)
    print("\nResearch Questions:")
    print("  RQ1: What is the prevalence of different Go constructs?")
    print("  RQ2: Do attention patterns differ for samples with vs without constructs?")
    print("  RQ3: Are Go constructs encoded distinctly in representations?")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ConstructAnalyzer()
    
    # Define analyses for both models
    analyses = []
    
    for model in ['unixcoder', 'codebert']:
        for task, task_type in [('code_to_text', 'code-to-text'), ('code_to_code', 'code-to-code')]:
            analyses.append({
                'name': f'{model}_{task}',
                'model': model,
                'features': RESULTS_DIR / f"model_outputs/{model}/{task}_full_{model}_features.jsonl",
                'ast': CODE_TO_TEXT_DIR / f"full_{task.replace('_', '-')}_with_asts.jsonl" if 'text' in task 
                       else CODE_TO_CODE_DIR / f"full_{task.replace('_', '-')}_with_asts.jsonl",
                'task_type': task_type
            })
    
    all_results = {}
    
    for analysis in analyses:
        print("\n" + "=" * 80)
        print(f"ANALYSIS: {analysis['name']}")
        print("=" * 80)
        
        # Check files
        if not analysis['features'].exists() or not analysis['ast'].exists():
            print(f"  ✗ Required files not found. Skipping...")
            continue
        
        # Load data
        data = analyzer.load_data(
            analysis['features'],
            analysis['ast'],
            analysis['task_type'],
            analysis['model']
        )
        
        if len(data) == 0:
            print("  ⚠ No valid data. Skipping...")
            continue
        
        # Run comprehensive analysis
        results = analyzer.comprehensive_construct_analysis(data, analysis['name'])
        all_results[analysis['name']] = results
        
        # Save individual results
        output_file = OUTPUT_DIR / f"{analysis['name']}_construct_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n  ✓ Results saved to {output_file.name}")
    
    # Save combined results
    if all_results:
        combined_output = OUTPUT_DIR / "all_construct_analysis.json"
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "=" * 80)
        print("✓ CONSTRUCT ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved in: {OUTPUT_DIR}/")
        print(f"\nKey Findings:")
        
        # Print summary of significant findings
        for name, result in all_results.items():
            print(f"\n{name}:")
            constructs = result['construct_analyses']
            for construct, analysis in constructs.items():
                sig_layers = len(analysis['significant_attention_layers']) + len(analysis['significant_embedding_layers'])
                if sig_layers > 0:
                    print(f"  {construct}: {sig_layers} significant layer(s)")
        
        print("\n")

if __name__ == "__main__":
    main()
