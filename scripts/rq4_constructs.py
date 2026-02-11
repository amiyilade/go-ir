#!/usr/bin/env python3
"""
Script 8: RQ4 — Construct Embedding Analysis
Part A: Linear classification probes — can layer embeddings detect construct presence?
Part B: PCA / t-SNE — visualise construct clusters in embedding space

Usage:
    python 8_rq4_constructs.py --model unixcoder --task code-to-text

Output:
    results/rq4_{task}_{model}.json
    figures/rq4_tsne_{task}_{model}_{construct}.png  (one per construct)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import stream_samples, load_metadata, get_embedding, has_construct

DATA_DIR     = Path("data")
FEATURES_DIR = Path("data/features")
RESULTS_DIR  = Path("results")
FIGURES_DIR  = Path("figures")

CONSTRUCTS = [
    'goroutines', 'channels', 'defer', 'error_patterns',
    'select_statements', 'interfaces', 'type_assertions', 'context_usage'
]

# Key layers to analyse for construct encoding
KEY_LAYERS = [0, 1, 3, 5, 7, 9, 11, 12]

# Minimum samples in minority class for a construct to be analysed
MIN_POSITIVE = 10


def mean_pool(emb: np.ndarray) -> np.ndarray:
    """Mean-pool over sequence dimension → [hidden_size]."""
    return emb.astype(np.float32).mean(axis=0)


def cls_token(emb: np.ndarray) -> np.ndarray:
    """CLS token (first token) → [hidden_size]."""
    return emb[0].astype(np.float32)


# -----------------------------------------------------------------------
# Part A: Classification probes
# -----------------------------------------------------------------------

def probe_construct(samples: list, construct: str,
                    num_layers: int, pooling: str = 'mean') -> dict:
    """
    For each key layer, train a linear probe to predict whether
    the construct is present. Returns per-layer AUROC (5-fold CV).
    """
    # Build labels
    labels = np.array([1 if has_construct(s, construct) else 0
                        for s in samples])
    n_pos = labels.sum()
    n_neg = (labels == 0).sum()

    if n_pos < MIN_POSITIVE or n_neg < MIN_POSITIVE:
        return {'skipped': True,
                'reason': f'too few positives ({n_pos}) or negatives ({n_neg})'}

    layer_results = []

    for layer in KEY_LAYERS:
        if layer >= num_layers:
            continue

        # Build feature matrix
        X_list = []
        for s in samples:
            emb = get_embedding(s, layer)
            if emb is None:
                X_list.append(None)
                continue
            if pooling == 'mean':
                X_list.append(mean_pool(emb))
            else:
                X_list.append(cls_token(emb))

        # Filter out None
        valid_mask = [x is not None for x in X_list]
        X  = np.stack([x for x in X_list if x is not None])
        y  = labels[valid_mask]

        if y.sum() < MIN_POSITIVE:
            layer_results.append({'layer': layer, 'auroc': 0.5, 'n': len(y)})
            continue

        # Standardise + logistic regression (5-fold)
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)
        clf    = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        try:
            scores = cross_val_score(clf, X_sc, y, cv=cv,
                                     scoring='roc_auc', n_jobs=-1)
            auroc  = float(np.mean(scores))
            std    = float(np.std(scores))
        except Exception:
            auroc, std = 0.5, 0.0

        layer_results.append({
            'layer': layer,
            'auroc': round(auroc, 4),
            'std':   round(std, 4),
            'n_positive': int(y.sum()),
            'n_negative': int((y == 0).sum()),
        })

    best = max(layer_results, key=lambda x: x.get('auroc', 0)) if layer_results else {}
    return {
        'layer_results': layer_results,
        'best_layer':    best,
        'n_positive':    int(n_pos),
        'n_negative':    int(n_neg),
    }


def run_part_a(samples: list, meta: dict) -> dict:
    print("\n  Part A: Classification Probes...")
    results = {}
    for construct in tqdm(CONSTRUCTS, desc="    Probing constructs"):
        res = probe_construct(samples, construct, meta['num_layers'])
        results[construct] = res
        if not res.get('skipped'):
            best = res.get('best_layer', {})
            print(f"    {construct:<22}: best layer={best.get('layer','?')} "
                  f"AUROC={best.get('auroc', 0):.3f}")
    return results


# -----------------------------------------------------------------------
# Part B: PCA (and t-SNE if sklearn has it)
# -----------------------------------------------------------------------

def run_part_b_pca(samples: list, meta: dict,
                   output_dir: Path,
                   model: str, task: str) -> dict:
    """
    PCA at layer 7. For each construct:
    - Compute 2D PCA of mean-pooled embeddings
    - Separate WITH vs WITHOUT construct
    - Store centroid distance and variance ratio for reporting
    """
    print("\n  Part B: PCA/t-SNE construct visualisation...")
    output_dir.mkdir(parents=True, exist_ok=True)

    layer = 7   # syntax layer
    if layer >= meta['num_layers']:
        layer = meta['num_layers'] // 2

    # Build embedding matrix (mean-pooled, layer 7)
    embeddings = []
    valid_samples = []
    for s in samples:
        emb = get_embedding(s, layer)
        if emb is not None:
            embeddings.append(mean_pool(emb))
            valid_samples.append(s)

    if len(embeddings) < 20:
        return {'error': 'not enough samples for PCA'}

    X = np.stack(embeddings).astype(np.float32)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # Fit PCA once on all data
    pca = PCA(n_components=50, random_state=42)
    X_pca_50 = pca.fit_transform(X_sc)

    # 2D for visualisation
    pca2 = PCA(n_components=2, random_state=42)
    X_2d = pca2.fit_transform(X_sc)

    var_explained = float(pca2.explained_variance_ratio_.sum())

    construct_results = {}
    for construct in CONSTRUCTS:
        labels = np.array([1 if has_construct(s, construct) else 0
                            for s in valid_samples])
        n_pos = labels.sum()
        if n_pos < MIN_POSITIVE or (len(labels) - n_pos) < MIN_POSITIVE:
            construct_results[construct] = {'skipped': True}
            continue

        pos_pts = X_2d[labels == 1]
        neg_pts = X_2d[labels == 0]

        centroid_pos = pos_pts.mean(axis=0)
        centroid_neg = neg_pts.mean(axis=0)
        centroid_dist = float(np.linalg.norm(centroid_pos - centroid_neg))

        # Try t-SNE if available (skip gracefully if not)
        tsne_results = None
        try:
            from sklearn.manifold import TSNE
            # Run t-SNE on top-50 PCA components (much faster)
            n_tsne = min(len(X_pca_50), 500)
            idx    = np.random.RandomState(42).choice(len(X_pca_50), n_tsne, replace=False)
            X_sub  = X_pca_50[idx]
            lab_sub = labels[idx]

            tsne = TSNE(n_components=2, random_state=42,
                        perplexity=min(30, n_tsne // 5),
                        n_iter=500, verbose=0)
            X_tsne = tsne.fit_transform(X_sub)

            pos_t = X_tsne[lab_sub == 1]
            neg_t = X_tsne[lab_sub == 0]
            tsne_centroid_dist = float(np.linalg.norm(
                pos_t.mean(axis=0) - neg_t.mean(axis=0)))

            tsne_results = {
                'centroid_distance': tsne_centroid_dist,
                'n_tsne_samples':    n_tsne,
            }

            # Save t-SNE plot
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(X_tsne[lab_sub==0, 0], X_tsne[lab_sub==0, 1],
                           c='steelblue', alpha=0.4, s=8, label=f'without {construct}')
                ax.scatter(X_tsne[lab_sub==1, 0], X_tsne[lab_sub==1, 1],
                           c='crimson', alpha=0.7, s=12, label=f'with {construct}')
                ax.set_title(f't-SNE Layer 7 — {construct}\n{model} {task}')
                ax.legend(fontsize=8)
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                plt.tight_layout()
                fig_path = output_dir / f"rq4_tsne_{task.replace('-','_')}_{model}_{construct}.png"
                plt.savefig(fig_path, dpi=150)
                plt.close()
            except Exception:
                pass  # matplotlib not required for results

        except ImportError:
            pass

        construct_results[construct] = {
            'n_positive':        int(n_pos),
            'n_negative':        int(len(labels) - n_pos),
            'pca_centroid_distance': centroid_dist,
            'pca_var_explained':     var_explained,
            'tsne':                  tsne_results,
        }

    return {
        'layer_analysed':    layer,
        'pca_var_explained': var_explained,
        'construct_results': construct_results,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['unixcoder', 'codebert'])
    parser.add_argument('--task',  required=True, choices=['code-to-text', 'code-to-code'])
    args = parser.parse_args()

    task_key  = args.task.replace('-', '_')
    h5_path   = FEATURES_DIR / f"{task_key}_{args.model}.h5"

    if args.task == 'code-to-text':
        jsonl_path = DATA_DIR / "code-to-text/stratified_2k_code_to_text_with_asts.jsonl"
    else:
        jsonl_path = DATA_DIR / "code-to-code/stratified_2k_code_to_code_with_asts.jsonl"

    print("\n" + "="*70)
    print(f"SCRIPT 8: RQ4 CONSTRUCT ENCODING")
    print(f"  Model: {args.model}  Task: {args.task}")
    print("="*70)

    if not h5_path.exists():
        print(f"\n✗ HDF5 not found: {h5_path}. Run Script 4 first.")
        return

    meta = load_metadata(h5_path)
    print(f"\n  {meta['num_layers']} layers, hidden={meta['hidden_size']}")

    print("\n  Loading samples...")
    samples = list(stream_samples(
        h5_path, jsonl_path,
        embedding_layers=KEY_LAYERS + [7]  # ensure layer 7 loaded for PCA
    ))
    print(f"  Loaded {len(samples)}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    part_a = run_part_a(samples, meta)
    part_b = run_part_b_pca(samples, meta, FIGURES_DIR, args.model, args.task)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"rq4_{task_key}_{args.model}.json"
    with open(output_path, 'w') as f:
        json.dump({
            'model':  args.model,
            'task':   args.task,
            'part_a_classification_probes': part_a,
            'part_b_pca_tsne':              part_b,
        }, f, indent=2)

    print(f"\n✔ Saved: {output_path}")
    print(f"  Figures: {FIGURES_DIR}/rq4_tsne_{task_key}_{args.model}_*.png")


if __name__ == "__main__":
    main()
