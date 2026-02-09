#!/usr/bin/env python3
"""
Script 6 (FULL): Structural Probing Analysis
Investigates whether syntax structure is encoded in contextual embeddings.
Based on "What Do They Capture?" paper methodology (Section 4.2).

FULL DATASET VERSION - Supports both UniXcoder and CodeBERT.
"""

import json
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import sys

# Configuration
RESULTS_DIR = Path("results")
FEATURES_DIR = RESULTS_DIR / "features"
CODE_TO_TEXT_DIR = Path("data/code-to-text")
CODE_TO_CODE_DIR = Path("data/code-to-code")
OUTPUT_DIR = RESULTS_DIR / "structural_probing"

# Training parameters
MAX_CODE_LENGTH = 100
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32

class StructuralProbe(nn.Module):
    """Linear transformation probe (Equation 6 from paper)."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.B = nn.Parameter(torch.randn(hidden_size, hidden_size))
    
    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """Compute squared distance d_B(h_i, h_j)^2."""
        diff = h_i - h_j
        transformed = torch.matmul(diff, self.B)
        squared_dist = torch.sum(transformed ** 2, dim=1)
        return squared_dist

class StructuralProbingAnalyzer:
    """Analyzes whether syntax structure is encoded in embeddings."""
    
    def __init__(self):
        self.results = {}
    
    def load_data(self, features_file: Path, ast_file: Path, task_type: str) -> List[Dict]:
        """Load features and AST data."""
        print(f"\nLoading data...")
        print(f"  Features: {features_file.name}")
        print(f"  AST data: {ast_file.name}")
        print(f"  Task type: {task_type}")
        
        with open(features_file, 'r', encoding='utf-8') as f:
            features = [json.loads(line) for line in f]
        
        with open(ast_file, 'r', encoding='utf-8') as f:
            ast_data = [json.loads(line) for line in f]
        
        merged = []
        for feat in features:
            sample_id = feat['sample_id']
            if sample_id < len(ast_data):
                code_label = 'code' if task_type == 'code-to-text' else 'initial_segment'
                
                if code_label in feat.get('features', {}):
                    ast_info = ast_data[sample_id].get('ast_info', {}).get(code_label)
                    if ast_info:
                        merged.append({
                            'sample_id': sample_id,
                            'task_type': task_type,
                            'code_label': code_label,
                            'features': feat['features'][code_label],
                            'ast': ast_info
                        })
        
        filtered = [s for s in merged if s['ast'] and 
                   len(s['ast'].get('leaf_nodes', [])) <= MAX_CODE_LENGTH]
        
        print(f"  ✓ Loaded {len(filtered)} samples (max length: {MAX_CODE_LENGTH})")
        return filtered
    
    def get_ast_distances(self, ast_tree: Dict) -> np.ndarray:
        """Compute tree distance matrix from AST."""
        leaf_nodes = ast_tree.get('leaf_nodes', [])
        num_nodes = len(leaf_nodes)
        
        distances = np.full((num_nodes, num_nodes), num_nodes * 2, dtype=float)
        np.fill_diagonal(distances, 0)
        
        parent_map = self._build_parent_map(ast_tree.get('ast_tree', {}))
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = self._tree_distance(i, j, parent_map)
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _build_parent_map(self, tree: Dict, parent_id=None, 
                         node_map=None, current_id=None) -> Dict:
        """Build mapping from node to parent."""
        if node_map is None:
            node_map = {}
            current_id = [0]
        
        my_id = current_id[0]
        current_id[0] += 1
        
        node_map[my_id] = {
            'parent': parent_id,
            'type': tree.get('type'),
            'is_leaf': 'children' not in tree or len(tree.get('children', [])) == 0
        }
        
        if 'children' in tree:
            for child in tree['children']:
                self._build_parent_map(child, my_id, node_map, current_id)
        
        return node_map
    
    def _tree_distance(self, node_i: int, node_j: int, parent_map: Dict) -> int:
        """Calculate tree distance between two leaf nodes."""
        path_i = []
        current = node_i
        while current is not None and current in parent_map:
            path_i.append(current)
            current = parent_map[current]['parent']
        
        path_j = []
        current = node_j
        while current is not None and current in parent_map:
            path_j.append(current)
            current = parent_map[current]['parent']
        
        path_i_set = set(path_i)
        lca = None
        for node in path_j:
            if node in path_i_set:
                lca = node
                break
        
        if lca is None:
            return len(path_i) + len(path_j)
        
        dist_i_to_lca = path_i.index(lca)
        dist_j_to_lca = path_j.index(lca)
        
        return dist_i_to_lca + dist_j_to_lca
    
    def prepare_training_data(self, data: List[Dict], layer_idx: int) -> Tuple:
        """Prepare training data for structural probe."""
        layer_key = f'layer_{layer_idx}'
        
        embeddings_list = []
        distances_list = []
        lengths_list = []
        
        for sample in data:
            if layer_key not in sample['features']['embeddings']:
                continue
            
            embeddings = np.array(sample['features']['embeddings'][layer_key])
            ast_distances = self.get_ast_distances(sample['ast'])
            
            min_len = min(embeddings.shape[0], ast_distances.shape[0])
            
            embeddings_list.append(embeddings[:min_len])
            distances_list.append(ast_distances[:min_len, :min_len])
            lengths_list.append(min_len)
        
        return embeddings_list, distances_list, lengths_list
    
    def train_probe(self, embeddings_list: List[np.ndarray], 
                   distances_list: List[np.ndarray],
                   hidden_size: int) -> StructuralProbe:
        """Train structural probe on data."""
        print(f"\n  Training structural probe...")
        
        probe = StructuralProbe(hidden_size)
        optimizer = optim.Adam(probe.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        
        for epoch in range(EPOCHS):
            total_loss = 0.0
            num_batches = 0
            
            for embeddings, distances in zip(embeddings_list, distances_list):
                seq_len = embeddings.shape[0]
                
                emb_tensor = torch.FloatTensor(embeddings)
                dist_tensor = torch.FloatTensor(distances)
                
                pairs = []
                target_dists = []
                
                for i in range(seq_len):
                    for j in range(seq_len):
                        if i != j:
                            pairs.append((i, j))
                            target_dists.append(distances[i, j])
                
                if len(pairs) == 0:
                    continue
                
                for batch_start in range(0, len(pairs), BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, len(pairs))
                    batch_pairs = pairs[batch_start:batch_end]
                    batch_targets = target_dists[batch_start:batch_end]
                    
                    h_i = torch.stack([emb_tensor[i] for i, j in batch_pairs])
                    h_j = torch.stack([emb_tensor[j] for i, j in batch_pairs])
                    targets = torch.FloatTensor(batch_targets)
                    
                    optimizer.zero_grad()
                    pred_dists = probe(h_i, h_j)
                    
                    loss = criterion(pred_dists, targets ** 2)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                print(f"    Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
        
        print(f"  ✓ Training complete")
        return probe
    
    def evaluate_probe(self, probe: StructuralProbe, 
                      embeddings_list: List[np.ndarray],
                      distances_list: List[np.ndarray]) -> Dict:
        """Evaluate probe using Spearman correlation."""
        print(f"\n  Evaluating probe...")
        
        probe.eval()
        correlations = []
        
        with torch.no_grad():
            for embeddings, gold_distances in zip(embeddings_list, distances_list):
                seq_len = embeddings.shape[0]
                
                emb_tensor = torch.FloatTensor(embeddings)
                
                pred_distances = np.zeros((seq_len, seq_len))
                
                for i in range(seq_len):
                    for j in range(seq_len):
                        if i != j:
                            h_i = emb_tensor[i].unsqueeze(0)
                            h_j = emb_tensor[j].unsqueeze(0)
                            pred_dist = probe(h_i, h_j).item()
                            pred_distances[i, j] = np.sqrt(pred_dist)
                
                gold_flat = gold_distances.flatten()
                pred_flat = pred_distances.flatten()
                
                if len(gold_flat) > 1:
                    corr, _ = spearmanr(gold_flat, pred_flat)
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        print(f"  ✓ Evaluated on {len(correlations)} sequences")
        
        return {
            'mean_spearman': np.mean(correlations) if correlations else 0.0,
            'std_spearman': np.std(correlations) if correlations else 0.0,
            'correlations': correlations
        }
    
    def analyze_layer(self, data: List[Dict], layer_idx: int, 
                     hidden_size: int) -> Dict:
        """Analyze a specific layer."""
        print(f"\nAnalyzing layer {layer_idx}...")
        
        embeddings_list, distances_list, lengths_list = self.prepare_training_data(
            data, layer_idx
        )
        
        if len(embeddings_list) == 0:
            print("  ⚠ No valid data for this layer")
            return None
        
        print(f"  Valid samples: {len(embeddings_list)}")
        
        train_emb, test_emb, train_dist, test_dist = train_test_split(
            embeddings_list, distances_list, test_size=0.2, random_state=42
        )
        
        probe = self.train_probe(train_emb, train_dist, hidden_size)
        
        test_results = self.evaluate_probe(probe, test_emb, test_dist)
        
        print(f"  Mean Spearman correlation: {test_results['mean_spearman']:.3f}")
        
        return {
            'layer': layer_idx,
            'spearman_correlation': test_results['mean_spearman'],
            'std_correlation': test_results['std_spearman'],
            'num_samples': len(embeddings_list)
        }
    
    def analyze_all_layers(self, data: List[Dict], model_name: str) -> Dict:
        """Analyze all layers for a model."""
        print(f"\nAnalyzing all layers for {model_name}...")
        
        sample_features = data[0]['features']
        num_layers = len(sample_features['embeddings'])
        hidden_size = sample_features['model_info']['hidden_size']
        
        print(f"  Model: {num_layers} layers, hidden size: {hidden_size}")
        
        results = []
        for layer_idx in range(num_layers):
            result = self.analyze_layer(data, layer_idx, hidden_size)
            if result:
                results.append(result)
        
        best_layer = max(results, key=lambda x: x['spearman_correlation'])
        
        return {
            'model_name': model_name,
            'all_layers': results,
            'best_layer': best_layer,
            'layer_summary': {
                f'layer_{r["layer"]}': r['spearman_correlation']
                for r in results
            }
        }

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("STRUCTURAL PROBING ANALYSIS (RQ2) - FULL DATASET")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Max code length: {MAX_CODE_LENGTH} tokens")
    print(f"  Training epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"\nAnalyzing: UniXcoder and CodeBERT")
    print(f"Tasks: Code-to-Text and Code-to-Code")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    analyzer = StructuralProbingAnalyzer()
    
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
        
        data = analyzer.load_data(
            analysis['features'], 
            analysis['ast'],
            analysis['task_type']
        )
        
        if len(data) == 0:
            print("  ⚠ No valid data. Skipping...")
            continue
        
        results = analyzer.analyze_all_layers(data, analysis['name'])
        all_results[analysis['name']] = results
        
        output_file = OUTPUT_DIR / f"{analysis['name']}_probing_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n  ✓ Results saved to {output_file.name}")
        print(f"\n  [Best Layer]")
        best = results['best_layer']
        print(f"    Layer {best['layer']}: Spearman = {best['spearman_correlation']:.3f}")
    
    if all_results:
        combined_output = OUTPUT_DIR / "all_probing_results.json"
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "=" * 80)
        print("✓ STRUCTURAL PROBING COMPLETE")
        print("=" * 80)
        print(f"\nResults saved in: {OUTPUT_DIR}/")
        print(f"\nNext steps:")
        print(f"  1. Compare Spearman correlations across models")
        print(f"  2. Run tree_induction_full.py for RQ3 analysis")
        print("\n")

if __name__ == "__main__":
    main()
