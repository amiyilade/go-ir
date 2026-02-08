#!/usr/bin/env python3
"""
Process Single Chunk - COMPLETE SELF-CONTAINED VERSION
Extracts features, runs all 3 analyses (RQ1, RQ2, RQ3), saves results, cleans up.

This script includes all analysis logic directly - no external dependencies needed.
Run this 28 times (once per chunk) across multiple sessions.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import gc
import sys
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import networkx as nx

# Configuration
CHUNKS_DIR = Path("data/chunks")
RESULTS_CHUNKS_DIR = Path("results/chunks")
TEMP_DIR = Path("/content/temp_chunk")

# Model configurations
MODELS = {
    'unixcoder': {
        'name': 'microsoft/unixcoder-base',
        'max_length': 512,
        'trust_remote_code': False
    },
    'codebert': {
        'name': 'microsoft/codebert-base',
        'max_length': 512,
        'trust_remote_code': False
    }
}

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RQ1 Parameters
ATTENTION_THRESHOLD = 0.3
MIN_HIGH_CONFIDENCE_SCORES = 100

# RQ2 Parameters
MAX_CODE_LENGTH = 100
EPOCHS = 50
LEARNING_RATE = 0.001
PROBE_BATCH_SIZE = 32

# RQ3 Parameters
LAMBDA_BIAS = 1.0
MAX_SAMPLES_RQ3 = 30  # Memory constraint

# ============================================================================
# MODEL ANALYZER - FEATURE EXTRACTION
# ============================================================================

class ModelAnalyzer:
    """Extracts attention weights and embeddings."""
    
    def __init__(self, model_name: str, model_config: dict):
        print(f"\n⚡ Loading {model_name}...")
        
        self.model_name = model_name
        self.max_length = model_config['max_length']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        self.model = AutoModel.from_pretrained(
            model_config['name'],
            output_attentions=True,
            output_hidden_states=True
        )
        
        self.model.to(DEVICE)
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        
        print(f"  ✓ Model loaded ({self.num_layers} layers, {self.num_heads} heads)")
    
    def extract_features(self, code: str, sample_id: int = -1):
        """Extract attention weights and embeddings for code."""
        try:
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(DEVICE)
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract attention
            attentions = outputs.attentions
            attention_dict = {}
            for layer_idx, layer_attn in enumerate(attentions):
                layer_attn = layer_attn.squeeze(0).cpu().numpy()
                attention_dict[f'layer_{layer_idx}'] = {
                    f'head_{head_idx}': layer_attn[head_idx].tolist()
                    for head_idx in range(layer_attn.shape[0])
                }
            
            # Extract embeddings
            hidden_states = outputs.hidden_states
            embeddings_dict = {}
            for layer_idx, layer_hidden in enumerate(hidden_states):
                layer_hidden = layer_hidden.squeeze(0).cpu().numpy()
                embeddings_dict[f'layer_{layer_idx}'] = layer_hidden.tolist()
            
            return {
                'tokens': tokens,
                'attention_weights': attention_dict,
                'embeddings': embeddings_dict,
                'seq_length': len(tokens),
                'model_info': {
                    'model_name': self.model_name,
                    'num_layers': self.num_layers,
                    'num_heads': self.num_heads,
                    'hidden_size': self.hidden_size
                }
            }
        
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed for sample {sample_id}: {e}")

# ============================================================================
# RQ1: ATTENTION-AST ALIGNMENT ANALYZER
# ============================================================================

class AttentionASTAligner:
    """Analyzes alignment between attention and AST structure."""
    
    def __init__(self):
        self.alignment_results = defaultdict(list)
    
    def get_ast_parent_pairs(self, ast_tree: dict) -> set:
        """Extract token pairs that share the same parent in AST."""
        parent_pairs = set()
        
        def traverse(node, parent_children=None):
            if 'children' in node:
                children = node['children']
                
                # Collect leaf indices for this parent's children
                leaf_indices = []
                for child in children:
                    indices = self._get_leaf_indices(child)
                    leaf_indices.extend(indices)
                
                # Create pairs of siblings (shared parent)
                for i in range(len(leaf_indices)):
                    for j in range(i + 1, len(leaf_indices)):
                        if abs(leaf_indices[i] - leaf_indices[j]) > 1:
                            parent_pairs.add((leaf_indices[i], leaf_indices[j]))
                            parent_pairs.add((leaf_indices[j], leaf_indices[i]))
                
                # Recurse
                for child in children:
                    traverse(child, children)
        
        if ast_tree:
            traverse(ast_tree.get('ast_tree', {}))
        
        return parent_pairs
    
    def _get_leaf_indices(self, node, current_index=None):
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
    
    def calculate_attention_variability(self, attention_matrices, max_tokens=10):
        """Calculate attention variability (Equation 5 from paper)."""
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
    
    def calculate_alignment_score(self, attention_matrix, ast_parent_pairs, 
                                 threshold=ATTENTION_THRESHOLD):
        """Calculate p_α(f) - proportion of high-attention pairs aligning with AST."""
        seq_len = attention_matrix.shape[0]
        
        # Find high-confidence attention pairs
        high_attention_pairs = []
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j and attention_matrix[i, j] > threshold:
                    high_attention_pairs.append((i, j))
        
        # Count alignments
        aligned_count = sum(1 for pair in high_attention_pairs if pair in ast_parent_pairs)
        
        total_high_attention = len(high_attention_pairs)
        alignment_score = aligned_count / total_high_attention if total_high_attention > 0 else 0.0
        
        return {
            'alignment_score': alignment_score,
            'aligned_pairs': aligned_count,
            'total_high_attention_pairs': total_high_attention,
            'total_ast_pairs': len(ast_parent_pairs)
        }
    
    def analyze_layer_head(self, data, layer_idx, head_idx):
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
            
            attention_matrix = np.array(
                features['attention_weights'][layer_key][head_key]
            )
            
            ast_parent_pairs = self.get_ast_parent_pairs(ast)
            
            if len(ast_parent_pairs) == 0:
                continue
            
            alignment = self.calculate_alignment_score(attention_matrix, ast_parent_pairs)
            
            if alignment['total_high_attention_pairs'] >= MIN_HIGH_CONFIDENCE_SCORES:
                alignment_scores.append(alignment['alignment_score'])
                variability_matrices.append(attention_matrix)
                valid_samples += 1
        
        if len(alignment_scores) == 0:
            return None
        
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
    
    def analyze_all_layers_heads(self, data, model_name):
        """Analyze all layers and heads for a model."""
        sample_features = data[0]['features']
        num_layers = len(sample_features['attention_weights'])
        num_heads = len(sample_features['attention_weights']['layer_0'])
        
        results = []
        total_combinations = num_layers * num_heads
        
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                result = self.analyze_layer_head(data, layer_idx, head_idx)
                if result:
                    results.append(result)
        
        results_sorted = sorted(results, key=lambda x: x['mean_alignment_score'], reverse=True)
        
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

# ============================================================================
# RQ2: STRUCTURAL PROBING ANALYZER
# ============================================================================

class StructuralProbe(nn.Module):
    """Linear transformation probe for syntax structure."""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.B = nn.Parameter(torch.randn(hidden_size, hidden_size))
    
    def forward(self, h_i, h_j):
        """Compute squared distance d_B(h_i, h_j)^2."""
        diff = h_i - h_j
        transformed = torch.matmul(diff, self.B)
        squared_dist = torch.sum(transformed ** 2, dim=1)
        return squared_dist

class StructuralProbingAnalyzer:
    """Analyzes whether syntax structure is encoded in embeddings."""
    
    def __init__(self):
        self.results = {}
    
    def get_ast_distances(self, ast_tree):
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
    
    def _build_parent_map(self, tree, parent_id=None, node_map=None, current_id=None):
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
    
    def _tree_distance(self, node_i, node_j, parent_map):
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
    
    def prepare_training_data(self, data, layer_idx):
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
    
    def train_probe(self, embeddings_list, distances_list, hidden_size):
        """Train structural probe."""
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
                
                for batch_start in range(0, len(pairs), PROBE_BATCH_SIZE):
                    batch_end = min(batch_start + PROBE_BATCH_SIZE, len(pairs))
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
        
        return probe
    
    def evaluate_probe(self, probe, embeddings_list, distances_list):
        """Evaluate probe using Spearman correlation."""
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
        
        return {
            'mean_spearman': np.mean(correlations) if correlations else 0.0,
            'std_spearman': np.std(correlations) if correlations else 0.0,
            'correlations': correlations
        }
    
    def analyze_layer(self, data, layer_idx, hidden_size):
        """Analyze a specific layer."""
        embeddings_list, distances_list, lengths_list = self.prepare_training_data(data, layer_idx)
        
        if len(embeddings_list) == 0:
            return None
        
        train_emb, test_emb, train_dist, test_dist = train_test_split(
            embeddings_list, distances_list, test_size=0.2, random_state=42
        )
        
        probe = self.train_probe(train_emb, train_dist, hidden_size)
        test_results = self.evaluate_probe(probe, test_emb, test_dist)
        
        return {
            'layer': layer_idx,
            'spearman_correlation': test_results['mean_spearman'],
            'std_correlation': test_results['std_spearman'],
            'num_samples': len(embeddings_list)
        }
    
    def analyze_all_layers(self, data, model_name):
        """Analyze all layers for a model."""
        sample_features = data[0]['features']
        num_layers = len(sample_features['embeddings'])
        hidden_size = sample_features['model_info']['hidden_size']
        
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

# ============================================================================
# RQ3: TREE INDUCTION ANALYZER
# ============================================================================

class TreeInducer:
    """Induces syntax trees from model representations."""
    
    def __init__(self):
        self.results = {}
    
    def compute_syntactic_distances(self, embeddings, distance_type='L2'):
        """Compute syntactic distances between adjacent tokens."""
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
    
    def apply_right_skewness_bias(self, distances, lambda_param=LAMBDA_BIAS):
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
    
    def chu_liu_edmonds(self, distances):
        """Chu-Liu-Edmonds algorithm for maximum spanning tree."""
        n = len(distances) + 1
        
        G = nx.DiGraph()
        
        for i in range(n):
            G.add_node(i)
        
        for i in range(n - 1):
            weight = -distances[i]
            G.add_edge(i, i + 1, weight=weight)
            G.add_edge(i + 1, i, weight=weight)
        
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
    
    def _right_branching_tree(self, n):
        """Create right-branching tree as fallback."""
        return [-1] + list(range(n - 1))
    
    def tree_to_intermediate_nodes(self, parent_indices):
        """Convert tree to set of intermediate nodes (sibling pairs)."""
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
    
    def calculate_f1_score(self, induced_tree, gold_tree):
        """Calculate F1 score between induced and gold trees."""
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
    
    def extract_gold_tree_structure(self, ast):
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
    
    def induce_tree_for_sample(self, sample, layer_idx, use_bias=True):
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
    
    def analyze_layer(self, data, layer_idx, model_name):
        """Analyze tree induction for specific layer."""
        results = []
        
        for sample in data:
            result = self.induce_tree_for_sample(sample, layer_idx, use_bias=True)
            if result:
                results.append(result)
        
        if len(results) == 0:
            return None
        
        f1_scores = [r['f1_scores']['f1'] for r in results]
        
        return {
            'layer': layer_idx,
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'all_f1_scores': f1_scores,
            'num_samples': len(results)
        }
    
    def analyze_all_layers(self, data, model_name):
        """Analyze tree induction for all layers."""
        num_layers = len(data[0]['features']['embeddings'])
        
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

# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def extract_chunk_features(chunk_file, model_analyzer, task_type, output_file):
    """Extract features for one chunk and save to disk."""
    print(f"\n{'='*80}")
    print(f"EXTRACTING FEATURES: {chunk_file.name}")
    print(f"{'='*80}")
    
    with open(chunk_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Samples in chunk: {len(data)}")
    
    if task_type == 'code-to-text':
        code_fields = [('query', 'code')]
    else:
        code_fields = [('query', 'initial_segment'), ('target', 'completion')]
    
    all_samples = []
    for i, sample in enumerate(data):
        for field_name, label in code_fields:
            code = sample.get(field_name, '')
            if code and isinstance(code, str):
                all_samples.append({
                    'sample_id': i,
                    'code': code,
                    'label': label,
                    'task_type': task_type
                })
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text('')  # Clear file
    
    results_dict = {}
    total_samples_written = 0
    WRITE_EVERY = 100  # Write to disk every 100 samples and clear RAM
    
    with tqdm(total=len(all_samples), desc="Extracting", unit="sample") as pbar:
        for batch_start in range(0, len(all_samples), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(all_samples))
            batch = all_samples[batch_start:batch_end]
            
            batch_codes = [item['code'] for item in batch]
            
            try:
                inputs = model_analyzer.tokenizer(
                    batch_codes,
                    return_tensors="pt",
                    max_length=model_analyzer.max_length,
                    truncation=True,
                    padding=True
                ).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model_analyzer.model(**inputs)
                
                for idx, item in enumerate(batch):
                    sample_id = item['sample_id']
                    label = item['label']
                    
                    if sample_id not in results_dict:
                        results_dict[sample_id] = {
                            'sample_id': sample_id,
                            'task_type': item['task_type'],
                            'features': {}
                        }
                    
                    tokens = model_analyzer.tokenizer.convert_ids_to_tokens(
                        inputs['input_ids'][idx]
                    )
                    
                    attention_dict = {}
                    for layer_idx, layer_attn in enumerate(outputs.attentions):
                        layer_attn_cpu = layer_attn[idx].cpu().numpy()
                        attention_dict[f'layer_{layer_idx}'] = {
                            f'head_{head_idx}': layer_attn_cpu[head_idx].tolist()
                            for head_idx in range(layer_attn_cpu.shape[0])
                        }
                        del layer_attn_cpu
                    
                    embeddings_dict = {}
                    for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
                        layer_hidden_cpu = layer_hidden[idx].cpu().numpy()
                        embeddings_dict[f'layer_{layer_idx}'] = layer_hidden_cpu.tolist()
                        del layer_hidden_cpu
                    
                    results_dict[sample_id]['features'][label] = {
                        'tokens': tokens,
                        'attention_weights': attention_dict,
                        'embeddings': embeddings_dict,
                        'seq_length': len(tokens),
                        'model_info': {
                            'model_name': model_analyzer.model_name,
                            'num_layers': model_analyzer.num_layers,
                            'num_heads': model_analyzer.num_heads,
                            'hidden_size': model_analyzer.hidden_size
                        }
                    }
                    
                    pbar.update(1)
                    
                    # WRITE EVERY 100 SAMPLES AND CLEAR RAM
                    if len(results_dict) >= WRITE_EVERY:
                        # Write accumulated results to disk (append mode)
                        with open(output_file, 'a', encoding='utf-8') as f:
                            for sid in sorted(results_dict.keys()):
                                json.dump(results_dict[sid], f, ensure_ascii=False)
                                f.write('\n')
                        
                        total_samples_written += len(results_dict)
                        
                        # CRITICAL: Clear dictionary to free RAM
                        results_dict.clear()
                        
                        # Force garbage collection
                        gc.collect()
                        
                        # Get current RAM usage (try psutil, fallback if not available)
                        try:
                            import psutil
                            ram_gb = psutil.virtual_memory().used / (1024**3)
                            print(f"\n  💾 Written {total_samples_written} samples, RAM: {ram_gb:.1f} GB")
                        except ImportError:
                            print(f"\n  💾 Written {total_samples_written} samples")
                
                del outputs, inputs
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            
            except Exception as e:
                print(f"\n✗ Error in batch: {e}")
                continue
    
    # Write remaining samples (< 100)
    if len(results_dict) > 0:
        print(f"\n💾 Writing final {len(results_dict)} samples...")
        with open(output_file, 'a', encoding='utf-8') as f:
            for sample_id in sorted(results_dict.keys()):
                json.dump(results_dict[sample_id], f, ensure_ascii=False)
                f.write('\n')
        
        total_samples_written += len(results_dict)
        results_dict.clear()
        gc.collect()
    
    size_gb = output_file.stat().st_size / (1024**3)
    print(f"  ✓ Saved {total_samples_written} samples ({size_gb:.2f} GB)")
    
    return total_samples_written

def run_rq1_analysis(features_file, ast_file, task_type, model_name, output_file):
    """Run RQ1: Attention-AST Alignment analysis."""
    print(f"\n📊 Running RQ1: Attention-AST Alignment...")
    
    # Load features
    with open(features_file, 'r', encoding='utf-8') as f:
        features_data = [json.loads(line) for line in f]
    
    # Load AST data
    with open(ast_file, 'r', encoding='utf-8') as f:
        ast_data = [json.loads(line) for line in f]
    
    # Merge data
    merged = []
    for feat in features_data:
        sample_id = feat['sample_id']
        if sample_id < len(ast_data):
            if task_type == 'code-to-text':
                code_label = 'code'
            else:
                code_label = 'initial_segment'
            
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
    
    print(f"  Analyzing {len(merged)} samples...")
    
    # Run analysis
    aligner = AttentionASTAligner()
    results = aligner.analyze_all_layers_heads(merged, model_name)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved RQ1 results")
    print(f"    Mean alignment: {results['overall_stats']['mean_alignment']:.3f}")

def run_rq2_analysis(features_file, ast_file, task_type, model_name, output_file):
    """Run RQ2: Structural Probing analysis."""
    print(f"\n📊 Running RQ2: Structural Probing...")
    
    # Load features
    with open(features_file, 'r', encoding='utf-8') as f:
        features_data = [json.loads(line) for line in f]
    
    # Load AST data
    with open(ast_file, 'r', encoding='utf-8') as f:
        ast_data = [json.loads(line) for line in f]
    
    # Merge and filter by length
    merged = []
    for feat in features_data:
        sample_id = feat['sample_id']
        if sample_id < len(ast_data):
            if task_type == 'code-to-text':
                code_label = 'code'
            else:
                code_label = 'initial_segment'
            
            if code_label in feat.get('features', {}):
                ast_info = ast_data[sample_id].get('ast_info', {}).get(code_label)
                if ast_info and len(ast_info.get('leaf_nodes', [])) <= MAX_CODE_LENGTH:
                    merged.append({
                        'sample_id': sample_id,
                        'task_type': task_type,
                        'code_label': code_label,
                        'features': feat['features'][code_label],
                        'ast': ast_info
                    })
    
    print(f"  Analyzing {len(merged)} samples (max length: {MAX_CODE_LENGTH})...")
    
    # Run analysis
    analyzer = StructuralProbingAnalyzer()
    results = analyzer.analyze_all_layers(merged, model_name)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved RQ2 results")
    print(f"    Best layer {results['best_layer']['layer']}: ρ = {results['best_layer']['spearman_correlation']:.3f}")

def run_rq3_analysis(features_file, ast_file, task_type, model_name, output_file):
    """Run RQ3: Tree Induction analysis."""
    print(f"\n📊 Running RQ3: Tree Induction...")
    
    # Load features
    with open(features_file, 'r', encoding='utf-8') as f:
        features_data = [json.loads(line) for line in f]
    
    # Load AST data
    with open(ast_file, 'r', encoding='utf-8') as f:
        ast_data = [json.loads(line) for line in f]
    
    # Merge data
    merged = []
    for feat in features_data:
        sample_id = feat['sample_id']
        if sample_id < len(ast_data):
            if task_type == 'code-to-text':
                code_label = 'code'
            else:
                code_label = 'initial_segment'
            
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
    
    # Limit samples for memory
    if len(merged) > MAX_SAMPLES_RQ3:
        import random
        random.seed(42)
        merged = random.sample(merged, MAX_SAMPLES_RQ3)
    
    print(f"  Analyzing {len(merged)} samples (limited for memory)...")
    
    # Run analysis
    inducer = TreeInducer()
    results = inducer.analyze_all_layers(merged, model_name)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved RQ3 results")
    print(f"    Best layer {results['best_layer']['layer']}: F1 = {results['best_layer']['mean_f1']:.3f}")

def process_chunk(chunk_id, task_type, model_name):
    """Process a single chunk: extract → analyze → save → cleanup."""
    print("\n" + "=" * 80)
    print(f"PROCESSING CHUNK {chunk_id:02d}")
    print(f"Task: {task_type}")
    print(f"Model: {model_name}")
    print("=" * 80)
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage("/content")
    free_gb = free / (1024**3)
    print(f"\n💾 Disk space: {free_gb:.1f} GB free")
    
    if free_gb < 130:
        print(f"⚠️  Warning: Less than 130 GB free. Chunk might fail!")
    
    # Setup paths
    chunk_file = CHUNKS_DIR / task_type / f"{task_type.replace('-', '_')}_chunk_{chunk_id:02d}.jsonl"
    ast_file = Path(f"data/{task_type}/full_{task_type.replace('-', '_')}_with_asts.jsonl")
    
    if not chunk_file.exists():
        print(f"✗ Error: Chunk file not found: {chunk_file}")
        return False
    
    if not ast_file.exists():
        print(f"✗ Error: AST file not found: {ast_file}")
        return False
    
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_features = TEMP_DIR / f"chunk_{chunk_id:02d}_features.jsonl"
    
    results_dir = RESULTS_CHUNKS_DIR / task_type / f"chunk_{chunk_id:02d}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Extract features
        model_analyzer = ModelAnalyzer(model_name, MODELS[model_name])
        num_samples = extract_chunk_features(
            chunk_file,
            model_analyzer,
            task_type,
            temp_features
        )
        
        del model_analyzer
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        # Check disk
        total, used, free = shutil.disk_usage("/content")
        free_gb = free / (1024**3)
        print(f"\n💾 Disk after extraction: {free_gb:.1f} GB free")
        
        # Step 2: Run RQ1
        run_rq1_analysis(
            temp_features, ast_file, task_type, model_name,
            results_dir / f"{model_name}_rq1.json"
        )
        
        # Step 3: Run RQ2
        run_rq2_analysis(
            temp_features, ast_file, task_type, model_name,
            results_dir / f"{model_name}_rq2.json"
        )
        
        # Step 4: Run RQ3
        run_rq3_analysis(
            temp_features, ast_file, task_type, model_name,
            results_dir / f"{model_name}_rq3.json"
        )
        
        # Step 5: Cleanup
        print(f"\n🗑️  Deleting temporary features...")
        size_gb = temp_features.stat().st_size / (1024**3)
        temp_features.unlink()
        print(f"  ✓ Freed {size_gb:.2f} GB")
        
        total, used, free = shutil.disk_usage("/content")
        free_gb = free / (1024**3)
        print(f"\n💾 Disk after cleanup: {free_gb:.1f} GB free")
        
        print(f"\n✓ Chunk {chunk_id:02d} complete!")
        print(f"  Results saved to: {results_dir}/")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing chunk: {e}")
        import traceback
        traceback.print_exc()
        
        if temp_features.exists():
            temp_features.unlink()
        
        return False

def main():
    parser = argparse.ArgumentParser(description='Process a single chunk')
    parser.add_argument('--chunk', type=int, required=True, help='Chunk ID (0-27)')
    parser.add_argument('--task', type=str, required=True, 
                       choices=['code-to-text', 'code-to-code'],
                       help='Task type')
    parser.add_argument('--model', type=str, required=True,
                       choices=['unixcoder', 'codebert'],
                       help='Model to use')
    
    args = parser.parse_args()
    
    success = process_chunk(args.chunk, args.task, args.model)
    
    if success:
        print("\n" + "=" * 80)
        print("✓ SUCCESS")
        print("=" * 80)
        print(f"\nChunk {args.chunk:02d} processed successfully!")
        print(f"\nNext: python scripts/process_single_chunk.py --chunk {args.chunk + 1} --task {args.task} --model {args.model}")
    else:
        print("\n" + "=" * 80)
        print("✗ FAILED")
        print("=" * 80)
        print(f"\nChunk {args.chunk:02d} failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
