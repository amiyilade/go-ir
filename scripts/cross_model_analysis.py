#!/usr/bin/env python3
"""
NEW: Cross-Model Comparison Analysis (Proposal Phase 3)
Compares UniXcoder (AST-augmented) vs CodeBERT (code-only) encoding strategies.

Implements:
- Centered Kernel Alignment (CKA) for representational similarity
- PCA/t-SNE visualizations of construct clusters
- Classification accuracy per layer
- Model agreement analysis
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cosine

# Configuration
RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "cross_model_analysis"

# Constructs to analyze
GO_CONSTRUCTS = [
    'error_patterns',
    'goroutines',
    'channels',
    'defer'
]

class CrossModelAnalyzer:
    """Analyzes differences between model architectures."""
    
    def __init__(self):
        self.results = {}
    
    def load_paired_data(self, unixcoder_file: Path, codebert_file: Path, 
                        ast_file: Path, task_type: str) -> List[Dict]:
        """Load data from both models for the same samples."""
        print(f"\nLoading paired data for {task_type}...")
        
        # Load both model outputs
        with open(unixcoder_file, 'r') as f:
            unixcoder_data = {item['sample_id']: item for item in (json.loads(line) for line in f)}
        
        with open(codebert_file, 'r') as f:
            codebert_data = {item['sample_id']: item for item in (json.loads(line) for line in f)}
        
        # Load AST data
        with open(ast_file, 'r') as f:
            ast_data = [json.loads(line) for line in f]
        
        # Find common samples
        common_ids = set(unixcoder_data.keys()) & set(codebert_data.keys())
        
        code_label = 'code' if task_type == 'code-to-text' else 'initial_segment'
        
        paired = []
        for sample_id in sorted(common_ids):
            if sample_id < len(ast_data):
                go_constructs = ast_data[sample_id].get('go_constructs', {}).get(code_label)
                
                if (code_label in unixcoder_data[sample_id]['features'] and
                    code_label in codebert_data[sample_id]['features'] and
                    go_constructs):
                    
                    paired.append({
                        'sample_id': sample_id,
                        'task_type': task_type,
                        'unixcoder_features': unixcoder_data[sample_id]['features'][code_label],
                        'codebert_features': codebert_data[sample_id]['features'][code_label],
                        'go_constructs': go_constructs
                    })
        
        print(f"  ✓ Loaded {len(paired)} paired samples")
        return paired
    
    def compute_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Centered Kernel Alignment (CKA) between two representations.
        
        CKA measures similarity between neural network representations.
        Based on: "Similarity of Neural Network Representations Revisited" (Kornblith et al., 2019)
        
        Args:
            X: [n_samples, dim_x] - representations from model 1
            Y: [n_samples, dim_y] - representations from model 2
        
        Returns:
            CKA score (0 to 1, higher = more similar)
        """
        def centering(K):
            """Center a kernel matrix."""
            n = K.shape[0]
            unit = np.ones([n, n])
            I = np.eye(n)
            H = I - unit / n
            return np.dot(np.dot(H, K), H)
        
        def linear_kernel(X):
            """Compute linear kernel."""
            return np.dot(X, X.T)
        
        def hsic(K, L):
            """Compute Hilbert-Schmidt Independence Criterion."""
            return np.sum(centering(K) * centering(L))
        
        # Compute kernels
        K = linear_kernel(X)
        L = linear_kernel(Y)
        
        # Compute CKA
        hsic_kl = hsic(K, L)
        hsic_kk = hsic(K, K)
        hsic_ll = hsic(L, L)
        
        cka = hsic_kl / np.sqrt(hsic_kk * hsic_ll)
        
        return float(cka)
    
    def layer_wise_cka(self, data: List[Dict], max_samples: int = 1000) -> Dict:
        """Compute CKA between UniXcoder and CodeBERT for each layer."""
        print("\n[Computing Layer-wise CKA]")
        
        # Limit samples for computational efficiency
        if len(data) > max_samples:
            print(f"  Using {max_samples} samples (from {len(data)})")
            data = data[:max_samples]
        
        # Get number of layers (should be 13: 0-12)
        num_layers = len(data[0]['unixcoder_features']['embeddings'])
        
        cka_scores = {}
        
        for layer_idx in tqdm(range(num_layers), desc="Computing CKA"):
            # Collect embeddings from both models
            unixcoder_embs = []
            codebert_embs = []
            
            for sample in data:
                try:
                    unixcoder_emb = np.array(sample['unixcoder_features']['embeddings'][f'layer_{layer_idx}'])
                    codebert_emb = np.array(sample['codebert_features']['embeddings'][f'layer_{layer_idx}'])
                    
                    # Use [CLS] token embedding (first token)
                    unixcoder_embs.append(unixcoder_emb[0])
                    codebert_embs.append(codebert_emb[0])
                except (KeyError, IndexError):
                    continue
            
            if len(unixcoder_embs) > 10:
                X = np.array(unixcoder_embs)  # [n_samples, hidden_dim]
                Y = np.array(codebert_embs)
                
                cka = self.compute_cka(X, Y)
                cka_scores[f'layer_{layer_idx}'] = cka
                
                print(f"  Layer {layer_idx}: CKA = {cka:.4f}")
        
        return cka_scores
    
    def visualize_construct_clusters(self, data: List[Dict], construct: str, 
                                    layer_idx: int, output_dir: Path):
        """Create PCA and t-SNE visualizations for construct encoding."""
        print(f"\n[Visualizing {construct} at Layer {layer_idx}]")
        
        # Collect embeddings and labels
        unixcoder_embs = []
        codebert_embs = []
        labels = []  # 1 if construct present, 0 otherwise
        
        for sample in data:
            go_constructs = sample['go_constructs']
            has_construct = (construct in go_constructs and 
                           isinstance(go_constructs[construct], list) and
                           len(go_constructs[construct]) > 0)
            
            try:
                unixcoder_emb = np.array(sample['unixcoder_features']['embeddings'][f'layer_{layer_idx}'])
                codebert_emb = np.array(sample['codebert_features']['embeddings'][f'layer_{layer_idx}'])
                
                # Use mean pooling
                unixcoder_embs.append(np.mean(unixcoder_emb, axis=0))
                codebert_embs.append(np.mean(codebert_emb, axis=0))
                labels.append(1 if has_construct else 0)
            except (KeyError, IndexError):
                continue
        
        if len(unixcoder_embs) < 50:
            print(f"  Skipping: insufficient samples ({len(unixcoder_embs)})")
            return
        
        unixcoder_embs = np.array(unixcoder_embs)
        codebert_embs = np.array(codebert_embs)
        labels = np.array(labels)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # PCA - UniXcoder
        pca = PCA(n_components=2)
        unixcoder_pca = pca.fit_transform(unixcoder_embs)
        
        axes[0, 0].scatter(unixcoder_pca[labels==0, 0], unixcoder_pca[labels==0, 1], 
                          alpha=0.5, label='Without', s=30)
        axes[0, 0].scatter(unixcoder_pca[labels==1, 0], unixcoder_pca[labels==1, 1],
                          alpha=0.7, label='With', s=30)
        axes[0, 0].set_title(f'UniXcoder PCA - {construct} (Layer {layer_idx})')
        axes[0, 0].set_xlabel('PC1')
        axes[0, 0].set_ylabel('PC2')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # PCA - CodeBERT
        codebert_pca = pca.fit_transform(codebert_embs)
        
        axes[0, 1].scatter(codebert_pca[labels==0, 0], codebert_pca[labels==0, 1],
                          alpha=0.5, label='Without', s=30)
        axes[0, 1].scatter(codebert_pca[labels==1, 0], codebert_pca[labels==1, 1],
                          alpha=0.7, label='With', s=30)
        axes[0, 1].set_title(f'CodeBERT PCA - {construct} (Layer {layer_idx})')
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # t-SNE - UniXcoder (using subset for speed)
        max_tsne = min(500, len(unixcoder_embs))
        indices = np.random.choice(len(unixcoder_embs), max_tsne, replace=False)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        unixcoder_tsne = tsne.fit_transform(unixcoder_embs[indices])
        
        axes[1, 0].scatter(unixcoder_tsne[labels[indices]==0, 0], unixcoder_tsne[labels[indices]==0, 1],
                          alpha=0.5, label='Without', s=30)
        axes[1, 0].scatter(unixcoder_tsne[labels[indices]==1, 0], unixcoder_tsne[labels[indices]==1, 1],
                          alpha=0.7, label='With', s=30)
        axes[1, 0].set_title(f'UniXcoder t-SNE - {construct} (Layer {layer_idx})')
        axes[1, 0].set_xlabel('t-SNE 1')
        axes[1, 0].set_ylabel('t-SNE 2')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # t-SNE - CodeBERT
        codebert_tsne = tsne.fit_transform(codebert_embs[indices])
        
        axes[1, 1].scatter(codebert_tsne[labels[indices]==0, 0], codebert_tsne[labels[indices]==0, 1],
                          alpha=0.5, label='Without', s=30)
        axes[1, 1].scatter(codebert_tsne[labels[indices]==1, 0], codebert_tsne[labels[indices]==1, 1],
                          alpha=0.7, label='With', s=30)
        axes[1, 1].set_title(f'CodeBERT t-SNE - {construct} (Layer {layer_idx})')
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_file = output_dir / f'{construct}_layer{layer_idx}_visualization.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved visualization to {output_file.name}")
    
    def classification_accuracy_per_layer(self, data: List[Dict], construct: str) -> Dict:
        """
        Test if layer representations can distinguish presence of construct.
        Uses logistic regression with cross-validation.
        """
        print(f"\n[Classification Accuracy for {construct}]")
        
        results = {
            'construct': construct,
            'unixcoder': {},
            'codebert': {}
        }
        
        num_layers = len(data[0]['unixcoder_features']['embeddings'])
        
        for layer_idx in tqdm(range(num_layers), desc="Testing layers"):
            # Collect embeddings and labels
            unixcoder_embs = []
            codebert_embs = []
            labels = []
            
            for sample in data:
                go_constructs = sample['go_constructs']
                has_construct = (construct in go_constructs and
                               isinstance(go_constructs[construct], list) and
                               len(go_constructs[construct]) > 0)
                
                try:
                    unixcoder_emb = np.array(sample['unixcoder_features']['embeddings'][f'layer_{layer_idx}'])
                    codebert_emb = np.array(sample['codebert_features']['embeddings'][f'layer_{layer_idx}'])
                    
                    # Mean pooling
                    unixcoder_embs.append(np.mean(unixcoder_emb, axis=0))
                    codebert_embs.append(np.mean(codebert_emb, axis=0))
                    labels.append(1 if has_construct else 0)
                except (KeyError, IndexError):
                    continue
            
            if len(unixcoder_embs) < 50:
                continue
            
            X_unixcoder = np.array(unixcoder_embs)
            X_codebert = np.array(codebert_embs)
            y = np.array(labels)
            
            # Skip if too imbalanced
            if np.sum(y) < 10 or np.sum(1-y) < 10:
                continue
            
            # Cross-validated accuracy
            clf = LogisticRegression(max_iter=1000, random_state=42)
            
            unixcoder_scores = cross_val_score(clf, X_unixcoder, y, cv=5, scoring='accuracy')
            codebert_scores = cross_val_score(clf, X_codebert, y, cv=5, scoring='accuracy')
            
            results['unixcoder'][f'layer_{layer_idx}'] = {
                'mean_accuracy': float(np.mean(unixcoder_scores)),
                'std_accuracy': float(np.std(unixcoder_scores))
            }
            
            results['codebert'][f'layer_{layer_idx}'] = {
                'mean_accuracy': float(np.mean(codebert_scores)),
                'std_accuracy': float(np.std(codebert_scores))
            }
        
        # Find best layers
        if results['unixcoder']:
            best_unixcoder = max(results['unixcoder'].items(), 
                               key=lambda x: x[1]['mean_accuracy'])
            best_codebert = max(results['codebert'].items(),
                              key=lambda x: x[1]['mean_accuracy'])
            
            print(f"  UniXcoder best: {best_unixcoder[0]} ({best_unixcoder[1]['mean_accuracy']:.3f})")
            print(f"  CodeBERT best: {best_codebert[0]} ({best_codebert[1]['mean_accuracy']:.3f})")
        
        return results
    
    def model_agreement_analysis(self, data: List[Dict]) -> Dict:
        """Analyze where models agree vs disagree on construct encoding."""
        print("\n[Model Agreement Analysis]")
        
        # For each construct, measure agreement in predictions
        agreement = {}
        
        for construct in GO_CONSTRUCTS:
            # Use Layer 7 (syntax layer from prior analysis)
            layer_idx = 7
            
            unixcoder_embs = []
            codebert_embs = []
            labels = []
            
            for sample in data:
                go_constructs = sample['go_constructs']
                has_construct = (construct in go_constructs and
                               isinstance(go_constructs[construct], list) and
                               len(go_constructs[construct]) > 0)
                
                try:
                    unixcoder_emb = np.array(sample['unixcoder_features']['embeddings'][f'layer_{layer_idx}'])
                    codebert_emb = np.array(sample['codebert_features']['embeddings'][f'layer_{layer_idx}'])
                    
                    unixcoder_embs.append(np.mean(unixcoder_emb, axis=0))
                    codebert_embs.append(np.mean(codebert_emb, axis=0))
                    labels.append(1 if has_construct else 0)
                except (KeyError, IndexError):
                    continue
            
            if len(unixcoder_embs) < 50:
                continue
            
            X_uni = np.array(unixcoder_embs)
            X_code = np.array(codebert_embs)
            y = np.array(labels)
            
            # Train classifiers
            clf_uni = LogisticRegression(max_iter=1000, random_state=42)
            clf_code = LogisticRegression(max_iter=1000, random_state=42)
            
            clf_uni.fit(X_uni, y)
            clf_code.fit(X_code, y)
            
            # Get predictions
            pred_uni = clf_uni.predict(X_uni)
            pred_code = clf_code.predict(X_code)
            
            # Calculate agreement
            agreement_rate = np.mean(pred_uni == pred_code)
            
            # Where they agree, are they correct?
            agree_mask = (pred_uni == pred_code)
            correct_when_agree = np.mean((pred_uni == y)[agree_mask]) if np.sum(agree_mask) > 0 else 0
            
            agreement[construct] = {
                'agreement_rate': float(agreement_rate),
                'accuracy_when_agree': float(correct_when_agree),
                'samples_analyzed': len(unixcoder_embs)
            }
            
            print(f"  {construct}:")
            print(f"    Agreement: {agreement_rate:.3f}")
            print(f"    Accuracy when agree: {correct_when_agree:.3f}")
        
        return agreement


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("CROSS-MODEL ANALYSIS: UniXcoder vs CodeBERT")
    print("=" * 80)
    print("\nAnalyses:")
    print("  1. Centered Kernel Alignment (CKA)")
    print("  2. PCA/t-SNE Visualizations")
    print("  3. Classification Accuracy per Layer")
    print("  4. Model Agreement Analysis")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    viz_dir = OUTPUT_DIR / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = CrossModelAnalyzer()
    
    # Define data sources
    analyses = [
        {
            'name': 'code_to_text',
            'task_type': 'code-to-text',
            'unixcoder': RESULTS_DIR / "model_outputs/unixcoder/code_to_text_full_unixcoder_features.jsonl",
            'codebert': RESULTS_DIR / "model_outputs/codebert/code_to_text_full_codebert_features.jsonl",
            'ast': Path("data/code-to-text/full_code-to-text_with_asts.jsonl")
        },
        {
            'name': 'code_to_code',
            'task_type': 'code-to-code',
            'unixcoder': RESULTS_DIR / "model_outputs/unixcoder/code_to_code_full_unixcoder_features.jsonl",
            'codebert': RESULTS_DIR / "model_outputs/codebert/code_to_code_full_codebert_features.jsonl",
            'ast': Path("data/code-to-code/full_code-to-code_with_asts.jsonl")
        }
    ]
    
    all_results = {}
    
    for analysis in analyses:
        print("\n" + "=" * 80)
        print(f"ANALYSIS: {analysis['name']}")
        print("=" * 80)
        
        # Check files
        if not all([analysis['unixcoder'].exists(), 
                   analysis['codebert'].exists(),
                   analysis['ast'].exists()]):
            print(f"  ✗ Required files not found. Skipping...")
            continue
        
        # Load paired data
        data = analyzer.load_paired_data(
            analysis['unixcoder'],
            analysis['codebert'],
            analysis['ast'],
            analysis['task_type']
        )
        
        if len(data) == 0:
            print("  ⚠ No valid paired data. Skipping...")
            continue
        
        results = {}
        
        # 1. CKA Analysis
        results['cka_scores'] = analyzer.layer_wise_cka(data)
        
        # 2. Visualizations for key constructs
        for construct in ['error_patterns', 'goroutines']:
            analyzer.visualize_construct_clusters(data, construct, 7, viz_dir)
        
        # 3. Classification accuracy
        classification_results = {}
        for construct in GO_CONSTRUCTS:
            classification_results[construct] = analyzer.classification_accuracy_per_layer(data, construct)
        results['classification_accuracy'] = classification_results
        
        # 4. Model agreement
        results['model_agreement'] = analyzer.model_agreement_analysis(data)
        
        all_results[analysis['name']] = results
        
        # Save results
        output_file = OUTPUT_DIR / f"{analysis['name']}_cross_model_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n  ✓ Results saved to {output_file.name}")
    
    # Save combined results
    if all_results:
        combined_output = OUTPUT_DIR / "all_cross_model_analysis.json"
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "=" * 80)
        print("✓ CROSS-MODEL ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved in: {OUTPUT_DIR}/")
        print(f"Visualizations in: {viz_dir}/")
        print("\n")

if __name__ == "__main__":
    main()
