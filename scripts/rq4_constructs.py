#!/usr/bin/env python3
"""
Script 8: RQ4 — Go Construct Encoding  (Novel contribution)
Part A: Classification probes — can each layer's embeddings predict
        whether a Go construct is present? (accuracy + F1 per layer)
Part B: PCA variance explained + t-SNE coordinates saved for plotting.

Input:  data/features/{task}_{model}.h5
        data/stratified_2k_{task}_with_asts.jsonl
Output: results/rq4_{task}_{model}.json

Usage:
    python scripts/rq4_constructs.py --model unixcoder --task code-to-text
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import stream_samples, load_metadata

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")

GO_CONSTRUCTS = [
    "goroutines", "channels", "defer", "error_patterns",
    "select_statements", "interfaces", "type_assertions", "context_usage",
]

KEY_LAYERS  = [0, 1, 3, 5, 7, 9, 11, 12]   # subset for speed
N_SPLITS_CV = 5      # cross-validation folds
TSNE_PERP   = 30
TSNE_MAX_SAMPLES = 500   # cap t-SNE inputs (slow otherwise)
PCA_COMPONENTS   = 50


def pool_embedding(emb: np.ndarray) -> np.ndarray:
    """Mean-pool across sequence dimension. (seq, hidden) → (hidden,)."""
    return emb.mean(axis=0)


def run_classification_probe(X: np.ndarray, y: np.ndarray,
                              construct: str, layer: int) -> dict:
    """
    Logistic regression probe with stratified 5-fold CV.
    Returns accuracy and F1 averaged across folds.
    """
    if sum(y) < 2 or sum(1 - y) < 2:
        return None  # need at least 2 of each class

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
    accs, f1s = [], []

    for tr, te in skf.split(X_sc, y):
        clf = LogisticRegression(max_iter=500, solver="lbfgs",
                                  class_weight="balanced")
        clf.fit(X_sc[tr], y[tr])
        pred = clf.predict(X_sc[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, zero_division=0))

    return {
        "layer":    layer,
        "accuracy": round(float(np.mean(accs)), 4),
        "acc_std":  round(float(np.std(accs)),  4),
        "f1":       round(float(np.mean(f1s)),  4),
        "f1_std":   round(float(np.std(f1s)),   4),
        "n_pos":    int(sum(y)),
        "n_neg":    int(len(y) - sum(y)),
    }


def run_pca(X: np.ndarray) -> dict:
    """Fit PCA, return variance explained by top 2/5/10 components."""
    n_comp = min(PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
    if n_comp < 2:
        return {}
    pca = PCA(n_components=n_comp)
    pca.fit(StandardScaler().fit_transform(X))
    ev = pca.explained_variance_ratio_
    return {
        "var_top2":   round(float(ev[:2].sum()),  4),
        "var_top5":   round(float(ev[:5].sum()),  4),
        "var_top10":  round(float(ev[:min(10, n_comp)].sum()), 4),
        "var_ratio":  [round(float(v), 5) for v in ev],
    }


def run_tsne(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Run t-SNE on a capped subset. Return coordinates + labels for plotting.
    We save the coordinates so visualize.py can render them without
    re-running the expensive embedding collection.
    """
    n = min(TSNE_MAX_SAMPLES, X.shape[0])
    idx = np.random.default_rng(42).choice(X.shape[0], n, replace=False)
    Xs  = StandardScaler().fit_transform(X[idx])
    perp = min(TSNE_PERP, n // 3 - 1)
    if perp < 2:
        return {}
    coords = TSNE(n_components=2, perplexity=perp,
                  random_state=42, n_iter=500).fit_transform(Xs)
    return {
        "x":      [round(float(v), 4) for v in coords[:, 0]],
        "y":      [round(float(v), 4) for v in coords[:, 1]],
        "labels": [int(v) for v in labels[idx]],
        "n":      n,
    }


# ── main ───────────────────────────────────────────────────────────────────

def analyse(model: str, task: str):
    task_slug  = task.replace("-", "_")
    h5_path    = DATA_DIR / "features" / f"{task}_{model}.h5"
    jsonl_path = DATA_DIR / f"stratified_2k_{task_slug}_with_asts.jsonl"

    if not h5_path.exists() or not jsonl_path.exists():
        print("✗ Required files not found")
        return

    meta    = load_metadata(h5_path)
    n_total = meta["num_samples"]
    n_lay   = meta["num_layers"]
    print(f"  {n_total} samples, {n_lay} layers")

    # ── collect pooled embeddings per layer ───────────────────────────────
    # layer_embs[layer] → list of (hidden,) arrays
    layer_embs:   dict = defaultdict(list)
    # labels[layer][construct] → list of 0/1
    labels_dict:  dict = defaultdict(lambda: defaultdict(list))

    for sample in tqdm(stream_samples(h5_path, jsonl_path),
                       desc="  Collecting", total=n_total):
        constructs = sample["record"].get("go_constructs", {})

        for layer_idx in KEY_LAYERS:
            if layer_idx > n_lay:
                continue
            emb    = sample["embeddings"][layer_idx]   # (seq, hidden)
            pooled = pool_embedding(emb)
            layer_embs[layer_idx].append(pooled)

            for c in GO_CONSTRUCTS:
                has = int(constructs.get(c, 0) > 0)
                labels_dict[layer_idx][c].append(has)

    # Convert to arrays
    for layer_idx in KEY_LAYERS:
        if layer_embs[layer_idx]:
            layer_embs[layer_idx] = np.stack(layer_embs[layer_idx])

    # ── Part A: classification probes ─────────────────────────────────────
    print("\n  Part A: Classification probes …")
    probe_results: dict = {}

    for c in tqdm(GO_CONSTRUCTS, desc="  Constructs"):
        c_results = []
        for layer_idx in KEY_LAYERS:
            if layer_idx > n_lay:
                continue
            X = layer_embs[layer_idx]
            if not isinstance(X, np.ndarray):
                continue
            y = np.array(labels_dict[layer_idx][c])

            r = run_classification_probe(X, y, c, layer_idx)
            if r:
                c_results.append(r)

        if c_results:
            best = max(c_results, key=lambda x: x["f1"])
            probe_results[c] = {
                "all_layers":  c_results,
                "best_layer":  best,
                "layer_summary": {f"layer_{r['layer']}": r["f1"] for r in c_results},
            }
            print(f"    {c:<22} best layer={best['layer']}  "
                  f"F1={best['f1']:.3f}  acc={best['accuracy']:.3f}")

    # ── Part B: PCA + t-SNE ───────────────────────────────────────────────
    print("\n  Part B: PCA + t-SNE …")

    # Layer 7 is the syntax layer — focus visualisation here, plus layer 1 for contrast
    vis_layers = [l for l in [1, 7] if l in layer_embs and isinstance(layer_embs[l], np.ndarray)]

    pca_results  = {}
    tsne_results = {}

    for layer_idx in vis_layers:
        X = layer_embs[layer_idx]
        print(f"    Layer {layer_idx}: PCA …", end="", flush=True)
        pca_results[f"layer_{layer_idx}"] = run_pca(X)
        print(" t-SNE …", end="", flush=True)

        # t-SNE per construct (only the most prevalent ones to keep output size down)
        tsne_results[f"layer_{layer_idx}"] = {}
        for c in ["error_patterns", "goroutines", "channels", "defer"]:
            y = np.array(labels_dict[layer_idx][c])
            if sum(y) < 5:
                continue
            tsne_results[f"layer_{layer_idx}"][c] = run_tsne(X, y)
        print(" done")

    # ── save ───────────────────────────────────────────────────────────────
    out = {
        "model":  model,
        "task":   task,
        "part_a_classification_probes": probe_results,
        "part_b_representation": {
            "pca":  pca_results,
            "tsne": tsne_results,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"rq4_{task_slug}_{model}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  ✓ Saved to {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["unixcoder", "codebert"])
    parser.add_argument("--task",  required=True,
                        choices=["code_to_text", "code_to_code"])
    args = parser.parse_args()

    print("=" * 60)
    print(f"SCRIPT 8: RQ4 — GO CONSTRUCT ENCODING")
    print(f"  model={args.model}  task={args.task}")
    print("=" * 60)

    analyse(args.model, args.task)


if __name__ == "__main__":
    main()
