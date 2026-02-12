#!/usr/bin/env python3
"""
RQ4 — Construct Embedding Analysis (DATA-ONLY)

Part A: Linear classification probes — can layer embeddings detect construct presence?
Part B: PCA + t-SNE (data export) — export 2D coords for qualitative visualisation

Output:
    results/rq4_{task_key}_{model}.json

Disclaimer: ChatGPT and Copilot were used to edit and enhance this script for better readability, error handling, and user feedback.
The author (me) implemented the core logic.
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import stream_samples, load_metadata, get_embedding, has_construct

DATA_DIR     = Path("data")
FEATURES_DIR = Path("data/features")
RESULTS_DIR  = Path("results")

CONSTRUCTS = [
    "goroutines", "channels", "defer", "error_patterns",
    "select_statements", "interfaces", "type_assertions", "context_usage"
]

KEY_LAYERS = [0, 1, 3, 5, 7, 9, 11, 12]

DEFAULT_VIS_LAYER = 7

MIN_POSITIVE = 10

TSNE_MAX_POINTS = 500


def mean_pool(emb: np.ndarray) -> np.ndarray:
    """Mean-pool over sequence dimension → [hidden_size]."""
    return emb.astype(np.float32).mean(axis=0)


def cls_token(emb: np.ndarray) -> np.ndarray:
    """CLS token (first token) → [hidden_size]."""
    return emb[0].astype(np.float32)

def probe_construct(
    samples: list,
    construct: str,
    num_layers: int,
    pooling: str = "mean",
) -> dict:
    labels = np.array([1 if has_construct(s, construct) else 0 for s in samples], dtype=int)
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())

    if n_pos < MIN_POSITIVE or n_neg < MIN_POSITIVE:
        return {
            "skipped": True,
            "reason": f"too few positives ({n_pos}) or negatives ({n_neg})",
            "n_positive": n_pos,
            "n_negative": n_neg,
            "layer_results": [],
            "best_layer": {},
        }

    layer_results = []

    for layer in KEY_LAYERS:
        if layer >= num_layers:
            continue

        X_list = []
        for s in samples:
            emb = get_embedding(s, layer)
            if emb is None:
                X_list.append(None)
                continue
            X_list.append(mean_pool(emb) if pooling == "mean" else cls_token(emb))

        valid_mask = np.array([x is not None for x in X_list], dtype=bool)
        if valid_mask.sum() < 2 * MIN_POSITIVE:
            layer_results.append({
                "layer": layer,
                "auroc": 0.5,
                "std": 0.0,
                "n_positive": int(labels[valid_mask].sum()),
                "n_negative": int((labels[valid_mask] == 0).sum()),
            })
            continue

        X = np.stack([x for x in X_list if x is not None]).astype(np.float32)
        y = labels[valid_mask]

        if int(y.sum()) < MIN_POSITIVE or int((y == 0).sum()) < MIN_POSITIVE:
            layer_results.append({
                "layer": layer,
                "auroc": 0.5,
                "std": 0.0,
                "n_positive": int(y.sum()),
                "n_negative": int((y == 0).sum()),
            })
            continue

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        try:
            scores = cross_val_score(clf, X_sc, y, cv=cv, scoring="roc_auc", n_jobs=-1)
            auroc  = float(np.mean(scores))
            std    = float(np.std(scores))
        except Exception:
            auroc, std = 0.5, 0.0

        layer_results.append({
            "layer": layer,
            "auroc": round(auroc, 4),
            "std":   round(std, 4),
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
        })

    best = max(layer_results, key=lambda x: x.get("auroc", 0.0), default={})
    return {
        "layer_results": layer_results,
        "best_layer": best,
        "n_positive": n_pos,
        "n_negative": n_neg,
    }


def run_part_a(samples: list, meta: dict) -> dict:
    print("\n  Part A: Classification Probes...")
    results = {}
    for construct in tqdm(CONSTRUCTS, desc="    Probing constructs"):
        res = probe_construct(samples, construct, meta["num_layers"])
        results[construct] = res
        if not res.get("skipped"):
            best = res.get("best_layer", {})
            print(
                f"    {construct:<22}: best layer={best.get('layer','?')} "
                f"AUROC={best.get('auroc', 0):.3f}"
            )
    return results

def run_part_b_pca_tsne(samples: list, meta: dict) -> dict:
    print("\n  Part B: PCA/t-SNE construct data export...")

    layer = DEFAULT_VIS_LAYER
    if layer >= meta["num_layers"]:
        layer = max(0, meta["num_layers"] // 2)

    embeddings = []
    valid_samples = []
    for s in samples:
        emb = get_embedding(s, layer)
        if emb is not None:
            embeddings.append(mean_pool(emb))
            valid_samples.append(s)

    if len(embeddings) < 20:
        return {"error": "not enough samples for PCA/t-SNE", "layer_analysed": layer}

    X = np.stack(embeddings).astype(np.float32)
    X_sc = StandardScaler().fit_transform(X)

    pca50 = PCA(n_components=50, random_state=42)
    X_pca_50 = pca50.fit_transform(X_sc)

    pca2 = PCA(n_components=2, random_state=42)
    _X_pca_2 = pca2.fit_transform(X_sc)
    pca2_var_explained = float(pca2.explained_variance_ratio_.sum())

    try:
        from sklearn.manifold import TSNE
        tsne_available = True
    except Exception:
        TSNE = None
        tsne_available = False

    rng = np.random.RandomState(42)

    construct_results = {}
    for construct in CONSTRUCTS:
        labels = np.array([1 if has_construct(s, construct) else 0 for s in valid_samples], dtype=int)
        n_pos = int(labels.sum())
        n_neg = int((labels == 0).sum())

        if n_pos < MIN_POSITIVE or n_neg < MIN_POSITIVE:
            construct_results[construct] = {
                "skipped": True,
                "n_positive": n_pos,
                "n_negative": n_neg,
            }
            continue

        pos_pts = _X_pca_2[labels == 1]
        neg_pts = _X_pca_2[labels == 0]
        centroid_dist_pca2 = float(np.linalg.norm(pos_pts.mean(axis=0) - neg_pts.mean(axis=0)))

        tsne_block = None
        if tsne_available:
            n = len(X_pca_50)
            n_tsne = min(n, TSNE_MAX_POINTS)
            idx = rng.choice(n, n_tsne, replace=False)

            X_sub = X_pca_50[idx]
            y_sub = labels[idx]

            perplexity = min(30, max(5, n_tsne // 5))
            perplexity = min(perplexity, n_tsne - 1)

            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=perplexity,
                max_iter=500,
                verbose=0,
                init="pca",
                learning_rate="auto",
            )

            try:
                X_tsne = tsne.fit_transform(X_sub)
                pos_t = X_tsne[y_sub == 1]
                neg_t = X_tsne[y_sub == 0]
                centroid_dist_tsne = float(np.linalg.norm(pos_t.mean(axis=0) - neg_t.mean(axis=0)))

                tsne_block = {
                    "n_tsne_samples": int(n_tsne),
                    "perplexity": float(perplexity),
                    "centroid_distance": centroid_dist_tsne,
                    "coords": {
                        "x": X_tsne[:, 0].astype(float).tolist(),
                        "y": X_tsne[:, 1].astype(float).tolist(),
                        "labels": y_sub.astype(int).tolist(),
                    },
                }
            except Exception as e:
                tsne_block = {"error": f"tsne_failed: {type(e).__name__}"}

        construct_results[construct] = {
            "skipped": False,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "pca2_var_explained": pca2_var_explained,
            "pca2_centroid_distance": centroid_dist_pca2,
            "tsne": tsne_block, 
        }

    return {
        "layer_analysed": layer,
        "pca2_var_explained": pca2_var_explained,
        "construct_results": construct_results,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["unixcoder", "codebert"])
    parser.add_argument("--task",  required=True, choices=["code-to-text", "code-to-code"])
    args = parser.parse_args()

    task_key = args.task.replace("-", "_")
    h5_path  = FEATURES_DIR / f"{task_key}_{args.model}.h5"

    if args.task == "code-to-text":
        jsonl_path = DATA_DIR / "code-to-text/stratified_2k_code_to_text_with_asts.jsonl"
    else:
        jsonl_path = DATA_DIR / "code-to-code/stratified_2k_code_to_code_with_asts.jsonl"

    print("\n" + "=" * 70)
    print("SCRIPT 8: RQ4 CONSTRUCT ENCODING (DATA-ONLY)")
    print(f"  Model: {args.model}  Task: {args.task}")
    print("=" * 70)

    if not h5_path.exists():
        print(f"\n✗ HDF5 not found: {h5_path}. Run extract_features.py first.")
        return

    meta = load_metadata(h5_path)
    print(f"\n  {meta['num_layers']} layers, hidden={meta['hidden_size']}")

    print("\n  Loading samples...")
    vis_layer = DEFAULT_VIS_LAYER
    if vis_layer >= meta["num_layers"]:
        vis_layer = max(0, meta["num_layers"] // 2)

    embedding_layers = sorted(set(KEY_LAYERS + [vis_layer]))
    samples = list(stream_samples(h5_path, jsonl_path, embedding_layers=embedding_layers))
    print(f"  Loaded {len(samples)}")

    part_a = run_part_a(samples, meta)
    part_b = run_part_b_pca_tsne(samples, meta)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"rq4_{task_key}_{args.model}.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "task": args.task,
            "part_a_classification_probes": part_a,
            "part_b_pca_tsne": part_b,
        }, f, indent=2)

    print(f"\n✔ Saved: {output_path}")
    if part_b.get("error"):
        print(f"  Note: Part B error: {part_b['error']}")


if __name__ == "__main__":
    main()
