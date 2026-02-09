#!/usr/bin/env python3
"""
Script 7 (FULL): Syntax Tree Induction
Investigates whether models can induce syntax structure without supervision.
Uses Chu-Liu-Edmonds algorithm for maximum spanning tree induction.
Based on paper Section 4.3.

FULL DATASET VERSION - Supports both UniXcoder and CodeBERT.
"""

import json
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import networkx as nx
import sys

# Configuration
RESULTS_DIR = Path("results")
FEATURES_DIR = RESULTS_DIR / "features"
CODE_TO_TEXT_DIR = Path("data/code-to-text")
CODE_TO_CODE_DIR = Path("data/code-to-code")
OUTPUT_DIR = RESULTS_DIR / "tree_induction"

# Parameters
LAMBDA_BIAS = 1.0  # Right-skewness bias
MAX_SAMPLES_PER_TASK = 500  # Memory optimization

class TreeInducer:
    """Induces syntax trees from model representations."""
    
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
    
    def compute_syntactic_distances(self, embeddings: np.ndarray,
                                   distance_type: str = 'L2') -> np.ndarray:
        """Compute syntactic distances between adjacent tokens (Equation 8)."""
        seq_len = embeddings.shape[0]
        distances = []
        
        for i in range(seq_len - 1):
            r = embeddings[i]
            s = embeddings[i + 1]
            
            if distance_type == 'L1':
                dist = np.sum(np.abs(r - s))
            elif distance_type == 'L2':
                dist = np.sqrt(np.sum((r - s) ** 2))
            else:
                raise ValueError(f"Unknown distance type: {distance_type}")
            
            distances.append(dist)
        
        return np.array(distances)
    
    def apply_right_skewness_bias(self, distances: np.ndarray,
                                  lambda_param: float = LAMBDA_BIAS) -> np.ndarray:
        """Apply right-skewness bias (Equation 9 from paper)."""
        m = len(distances) + 1
        avg_dist = np.mean(distances)
        
        biased_distances = distances.copy()
        
        for i in range(len(distances)):
            i_paper = i + 1
            
            if i_paper > 1:
                bias_term = lambda_param * avg_dist * (1 - 1 / ((m - 1) * (i_paper - 1)))
                biased_distances[i] += bias_term
        
        return biased_distances
    
    def chu_liu_edmonds(self, distances: np.ndarray) -> List[int]:
        """Chu-Liu-Edmonds algorithm for maximum spanning tree."""
        n = len(distances) + 1
        
        G = nx.DiGraph()
        
        for i in range(n):
            G.add_node(i)
        
        # Add edges between adjacent tokens
        for i in range(n - 1):
            weight = -distances[i]
            G.add_edge(i, i + 1, weight=weight)
            G.add_edge(i + 1, i, weight=weight)
        
        # Add edges between non-adjacent tokens
        for i in range(n):
            for j in range(i + 2, n):
                avg_weight = -np.mean(distances[i:j])
                G.add_edge(i, j, weight=avg_weight)
                G.add_edge(j, i, weight=avg_weight)
        
        try:
            mst = nx.maximum_spanning_arborescence(G)
            
            parent_indices = [-1] * n
            parent_indices[0] = -1
            
            for i in range(1, n):
                predecessors = list(mst.predecessors(i))
                if predecessors:
                    parent_indices[i] = predecessors[0]
                else:
                    parent_indices[i] = 0
            
            return parent_indices
            
        except Exception as e:
            return self._right_branching_tree(n)
    
    def _right_branching_tree(self, n: int) -> List[int]:
        """Create right-branching tree as fallback."""
        parents = [-1] + list(range(n - 1))
        return parents
    
    def tree_to_intermediate_nodes(self, parent_indices: List[int]) -> Set[Tuple[int, int]]:
        """Convert tree to set of intermediate nodes."""
        n = len(parent_indices)
        intermediate_nodes = set()
        
        children_by_parent = {}
        for i, parent in enumerate(parent_indices):
            if parent != -1:
                if parent not in children_by_parent:
                    children_by_parent[parent] = []
                children_by_parent[parent].append(i)
        
        for parent, children in children_by_parent.items():
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    intermediate_nodes.add((children[i], children[j]))
                    intermediate_nodes.add((children[j], children[i]))
        
        return intermediate_nodes
    
    def calculate_f1_score(self, induced_tree: Set[Tuple[int, int]],
                          gold_tree: Set[Tuple[int, int]]) -> Dict:
        """Calculate F1 score (Equation 10 from paper)."""
        if len(gold_tree) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        intersection = induced_tree & gold_tree
        
        precision = len(intersection) / len(induced_tree) if len(induced_tree) > 0 else 0.0
        recall = len(intersection) / len(gold_tree)
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'intersection_size': len(intersection),
            'induced_size': len(induced_tree),
            'gold_size': len(gold_tree)
        }
    
    def extract_gold_tree_structure(self, ast: Dict) -> Set[Tuple[int, int]]:
        """Extract intermediate nodes from gold AST."""
        leaf_nodes = ast.get('leaf_nodes', [])
        ast_tree = ast.get('ast_tree', {})
        
        intermediate_nodes = set()
        
        def traverse(node, leaf_index=None):
            if leaf_index is None:
                leaf_index = [0]
            
            if 'children' not in node or len(node.get('children', [])) == 0:
                my_index = leaf_index[0]
                leaf_index[0] += 1
                return [my_index]
            else:
                all_child_leaves = []
                for child in node['children']:
                    child_leaves = traverse(child, leaf_index)
                    all_child_leaves.append(child_leaves)
                
                for i in range(len(all_child_leaves)):
                    for j in range(i + 1, len(all_child_leaves)):
                        for leaf_i in all_child_leaves[i]:
                            for leaf_j in all_child_leaves[j]:
                                intermediate_nodes.add((leaf_i, leaf_j))
                                intermediate_nodes.add((leaf_j, leaf_i))
                
                flat_leaves = []
                for child_leaves in all_child_leaves:
                    flat_leaves.extend(child_leaves)
                return flat_leaves
        
        traverse(ast_tree)
        return intermediate_nodes
    
    def induce_tree_for_sample(self, sample: Dict, layer_idx: int,
                               use_bias: bool = True) -> Dict:
        """Induce tree for a single sample."""
        layer_key = f'layer_{layer_idx}'
        
        if layer_key not in sample['features']['embeddings']:
            return None
        
        embeddings = np.array(sample['features']['embeddings'][layer_key])
        
        distances = self.compute_syntactic_distances(embeddings, distance_type='L2')
        
        if use_bias:
            distances = self.apply_right_skewness_bias(distances)
        
        parent_indices = self.chu_liu_edmonds(distances)
        
        induced_intermediate = self.tree_to_intermediate_nodes(parent_indices)
        
        gold_intermediate = self.extract_gold_tree_structure(sample['ast'])
        
        f1_scores = self.calculate_f1_score(induced_intermediate, gold_intermediate)
        
        return {
            'sample_id': sample['sample_id'],
            'induced_parent_indices': parent_indices,
            'f1_scores': f1_scores
        }
    
    def analyze_layer(self, data: List[Dict], layer_idx: int,
                     model_name: str) -> Dict:
        """Analyze tree induction for specific layer."""
        print(f"\n  Analyzing layer {layer_idx}...")
        
        results = []
        
        for sample in tqdm(data, desc=f"    Inducing trees", leave=False):
            result = self.induce_tree_for_sample(sample, layer_idx, use_bias=True)
            if result:
                results.append(result)
        
        if len(results) == 0:
            return None
        
        f1_scores = [r['f1_scores']['f1'] for r in results]
        
        print(f"    Mean F1: {np.mean(f1_scores):.3f}")
        
        return {
            'layer': layer_idx,
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'all_f1_scores': f1_scores,
            'num_samples': len(results)
        }
    
    def analyze_all_layers(self, data: List[Dict], model_name: str) -> Dict:
        """Analyze tree induction for all layers."""
        print(f"\nAnalyzing tree induction for {model_name}...")
        
        num_layers = len(data[0]['features']['embeddings'])
        print(f"  Model has {num_layers} layers")
        print(f"  Processing {len(data)} samples")
        
        results = []
        for layer_idx in range(num_layers):
            result = self.analyze_layer(data, layer_idx, model_name)
            if result:
                results.append(result)
        
        best_layer = max(results, key=lambda x: x['mean_f1'])
        
        return {
            'model_name': model_name,
            'all_layers': results,
            'best_layer': best_layer,
            'layer_summary': {
                f'layer_{r["layer"]}': r['mean_f1']
                for r in results
            }
        }

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("SYNTAX TREE INDUCTION ANALYSIS (RQ3) - FULL DATASET")
    print("=" * 80)
    print(f"\nAlgorithm: Chu-Liu-Edmonds maximum spanning tree")
    print(f"Right-skewness bias (λ): {LAMBDA_BIAS}")
    print(f"Max samples per task: {MAX_SAMPLES_PER_TASK} (memory limit)")
    print(f"\nAnalyzing: UniXcoder and CodeBERT")
    print(f"Tasks: Code-to-Text and Code-to-Code")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    inducer = TreeInducer()
    
    analyses = []
    for model in ['unixcoder', 'codebert']:
        analyses.extend([
            {
                'name': f'{model}_code_to_text',
                'features': RESULTS_DIR / f"model_outputs/{model}/code_to_text_full_{model}_features.jsonl",
                'ast': CODE_TO_TEXT_DIR / "full_code_to_text_with_asts.jsonl",
                'task_type': 'code-to-text'
            },
            {
                'name': f'{model}_code_to_code',
                'features': RESULTS_DIR / f"model_outputs/{model}/code_to_code_full_{model}_features.jsonl",
                'ast': CODE_TO_CODE_DIR / "full_code_to_code_with_asts.jsonl",
                'task_type': 'code-to-code'
            }
        ])
    
    all_results = {}
    
    for analysis in analyses:
        print("\n" + "=" * 80)
        print(f"ANALYSIS: {analysis['name']}")
        print("=" * 80)
        
        if not analysis['features'].exists() or not analysis['ast'].exists():
            print(f"  ✗ Required files not found. Skipping...")
            continue
        
        data = inducer.load_data(
            analysis['features'],
            analysis['ast'],
            analysis['task_type'],
            max_samples=MAX_SAMPLES_PER_TASK
        )
        
        if len(data) == 0:
            print("  ⚠ No valid data. Skipping...")
            continue
        
        results = inducer.analyze_all_layers(data, analysis['name'])
        all_results[analysis['name']] = results
        
        output_file = OUTPUT_DIR / f"{analysis['name']}_tree_induction_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n  ✓ Results saved to {output_file.name}")
        print(f"\n  [Best Layer for Tree Induction]")
        best = results['best_layer']
        print(f"    Layer {best['layer']}: F1 = {best['mean_f1']:.3f} ± {best['std_f1']:.3f}")
    
    if all_results:
        combined_output = OUTPUT_DIR / "all_tree_induction_results.json"
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "=" * 80)
        print("✓ TREE INDUCTION COMPLETE")
        print("=" * 80)
        print(f"\nResults saved in: {OUTPUT_DIR}/")
        print(f"\nNext steps:")
        print(f"  1. Compare F1 scores across layers and models")
        print(f"  2. Run construct_analysis.py for construct-level analysis")
        print("\n")

if __name__ == "__main__":
    main()
