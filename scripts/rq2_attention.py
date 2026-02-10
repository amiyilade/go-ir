#!/usr/bin/env python3
"""
Script 6: RQ2 — Attention Analysis
Part A (Wan et al.): Attention-AST alignment per layer/head.
Part B (Novel):      Attention entropy & focus differ for samples
                     with vs without each Go construct.

Input:  data/features/{task}_{model}.h5
        data/stratified_2k_{task}_with_asts.jsonl
Output: results/rq2_{task}_{model}.json

Usage:
    python scripts/rq2_attention.py --model unixcoder --task code-to-text
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import entropy as scipy_entropy, ttest_ind
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import stream_samples, load_metadata, KEY_HEADS

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")

ATTN_THRESHOLD        = 0.3   # high-confidence attention (Wan et al.)
MIN_HIGH_CONF_PAIRS   = 10    # min pairs needed per sample to count
GO_CONSTRUCTS = [
    "goroutines", "channels", "defer", "error_patterns",
    "select_statements", "interfaces", "type_assertions", "context_usage",
]


# ── AST helpers ────────────────────────────────────────────────────────────

def get_ast_parent_pairs(ast_tree: dict) -> set:
    """Return set of (i,j) token-index pairs sharing an AST parent node."""
    pairs = set()
    leaf_counter = [0]

    def leaf_indices(node):
        if "children" not in node or not node["children"]:
            idx = leaf_counter[0]
            leaf_counter[0] += 1
            return [idx]
        indices = []
        for c in node["children"]:
            indices.extend(leaf_indices(c))
        return indices

    def traverse(node):
        if "children" in node and node["children"]:
            child_leaves = [leaf_indices(c) for c in node["children"]]
            flat = [i for grp in child_leaves for i in grp]
            for i in range(len(flat)):
                for j in range(i + 1, len(flat)):
                    if abs(flat[i] - flat[j]) > 1:   # exclude adjacent (Wan et al.)
                        pairs.add((flat[i], flat[j]))
                        pairs.add((flat[j], flat[i]))
            for c in node["children"]:
                traverse(c)

    leaf_counter[0] = 0
    traverse(ast_tree)
    return pairs


def alignment_score(attn: np.ndarray, parent_pairs: set) -> float:
    """Proportion of high-confidence attention pairs that align with AST."""
    seq = attn.shape[0]
    high = [(i, j) for i in range(seq) for j in range(seq)
            if i != j and attn[i, j] > ATTN_THRESHOLD]
    if len(high) < MIN_HIGH_CONF_PAIRS:
        return np.nan
    aligned = sum(1 for p in high if p in parent_pairs)
    return aligned / len(high)


def attention_variability(matrices: list) -> float:
    """Attention variability across samples (Wan et al. content-dep metric)."""
    if not matrices:
        return 0.0
    n = min(10, min(m.shape[0] for m in matrices))
    arr = np.stack([m[:n, :n] for m in matrices if m.shape[0] >= n])
    if len(arr) == 0:
        return 0.0
    mu  = arr.mean(axis=0)
    tot = arr.sum()
    return float(np.sum((arr - mu) ** 2) / tot) if tot > 0 else 0.0


# ── Part B helpers ─────────────────────────────────────────────────────────

def attn_entropy(attn: np.ndarray) -> float:
    """Mean per-row entropy of an attention matrix."""
    return float(np.mean([scipy_entropy(row + 1e-10) for row in attn]))


def attn_max_focus(attn: np.ndarray) -> float:
    """Mean max-attention value per row (how focused the head is)."""
    return float(np.mean(np.max(attn, axis=1)))


# ── main analysis ──────────────────────────────────────────────────────────

def analyse(model: str, task: str):
    task_slug = task.replace("-", "_")
    h5_path   = DATA_DIR / "features" / f"{task}_{model}.h5"
    jsonl_path = DATA_DIR / f"stratified_2k_{task_slug}_with_asts.jsonl"

    if not h5_path.exists():
        print(f"✗ HDF5 not found: {h5_path}")
        return
    if not jsonl_path.exists():
        print(f"✗ JSONL not found: {jsonl_path}")
        return

    meta = load_metadata(h5_path)
    n_layers    = meta["num_layers"]
    n_key_heads = len(meta["key_heads"])
    print(f"  Model: {n_layers} layers, key heads: {meta['key_heads']}")

    # ── storage ────────────────────────────────────────────────────────────
    # Part A: alignment[layer][key_head_idx] → list of scores
    alignment      = defaultdict(lambda: defaultdict(list))
    # variability matrices
    var_matrices   = defaultdict(lambda: defaultdict(list))

    # Part B: for each construct, per layer → {with: [], without: []}
    ent_by_construct  = defaultdict(lambda: defaultdict(lambda: {"with": [], "without": []}))
    foc_by_construct  = defaultdict(lambda: defaultdict(lambda: {"with": [], "without": []}))

    n_valid = 0

    for sample in tqdm(stream_samples(h5_path, jsonl_path),
                       desc="  Samples", total=meta["num_samples"]):

        rec      = sample["record"]
        ast_info = rec.get("ast_info", {})
        ast_tree = ast_info.get("ast_tree") if isinstance(ast_info, dict) else None
        if ast_tree is None:
            continue

        parent_pairs = get_ast_parent_pairs(ast_tree)
        constructs   = rec.get("go_constructs", {})

        for layer_idx in range(n_layers):
            for kh_idx in range(n_key_heads):
                attn = sample["attentions"][layer_idx][kh_idx]

                # Part A
                score = alignment_score(attn, parent_pairs)
                if not np.isnan(score):
                    alignment[layer_idx][kh_idx].append(score)
                    var_matrices[layer_idx][kh_idx].append(attn)

                # Part B — only head 7 (key_idx 3) to keep output size manageable
                if kh_idx == KEY_HEADS.index(7) if 7 in KEY_HEADS else -1:
                    ent = attn_entropy(attn)
                    foc = attn_max_focus(attn)
                    for c in GO_CONSTRUCTS:
                        has = constructs.get(c, 0) > 0
                        bucket = "with" if has else "without"
                        ent_by_construct[c][layer_idx][bucket].append(ent)
                        foc_by_construct[c][layer_idx][bucket].append(foc)

        n_valid += 1

    print(f"  Valid samples processed: {n_valid}")

    # ── Part A results ─────────────────────────────────────────────────────
    part_a_results = []
    for layer_idx in range(n_layers):
        for kh_idx in range(n_key_heads):
            scores = alignment[layer_idx][kh_idx]
            if not scores:
                continue
            var = attention_variability(var_matrices[layer_idx][kh_idx])
            part_a_results.append({
                "layer":             layer_idx,
                "head_id":           KEY_HEADS[kh_idx],
                "head_key_idx":      kh_idx,
                "mean_alignment":    round(float(np.mean(scores)), 4),
                "std_alignment":     round(float(np.std(scores)),  4),
                "variability":       round(var, 4),
                "head_type":         "content-dependent" if var > 0.25 else "position-based",
                "n_samples":         len(scores),
            })

    part_a_results.sort(key=lambda x: x["mean_alignment"], reverse=True)

    layer_summary_a = {}
    by_layer = defaultdict(list)
    for r in part_a_results:
        by_layer[r["layer"]].append(r["mean_alignment"])
    for l, vals in by_layer.items():
        layer_summary_a[f"layer_{l}"] = {
            "mean": round(float(np.mean(vals)), 4),
            "max":  round(float(np.max(vals)), 4),
        }

    # ── Part B results ─────────────────────────────────────────────────────
    part_b_results = {}
    for c in GO_CONSTRUCTS:
        c_res = {}
        for layer_idx in range(n_layers):
            ent_w = ent_by_construct[c][layer_idx]["with"]
            ent_o = ent_by_construct[c][layer_idx]["without"]
            foc_w = foc_by_construct[c][layer_idx]["with"]
            foc_o = foc_by_construct[c][layer_idx]["without"]

            if not ent_w or not ent_o:
                continue

            ent_p = ttest_ind(ent_w, ent_o, equal_var=False).pvalue if len(ent_w) > 1 and len(ent_o) > 1 else 1.0
            foc_p = ttest_ind(foc_w, foc_o, equal_var=False).pvalue if len(foc_w) > 1 and len(foc_o) > 1 else 1.0

            c_res[f"layer_{layer_idx}"] = {
                "entropy_with":    round(float(np.mean(ent_w)), 4),
                "entropy_without": round(float(np.mean(ent_o)), 4),
                "entropy_diff":    round(float(np.mean(ent_w)) - float(np.mean(ent_o)), 4),
                "entropy_pvalue":  round(float(ent_p), 4),
                "focus_with":      round(float(np.mean(foc_w)), 4),
                "focus_without":   round(float(np.mean(foc_o)), 4),
                "focus_diff":      round(float(np.mean(foc_w)) - float(np.mean(foc_o)), 4),
                "focus_pvalue":    round(float(foc_p), 4),
                "n_with":          len(ent_w),
                "n_without":       len(ent_o),
            }
        part_b_results[c] = c_res

    # ── save ───────────────────────────────────────────────────────────────
    out = {
        "model":        model,
        "task":         task,
        "n_samples":    n_valid,
        "part_a_alignment": {
            "all_heads":     part_a_results,
            "top_10":        part_a_results[:10],
            "layer_summary": layer_summary_a,
            "overall": {
                "mean": round(float(np.mean([r["mean_alignment"] for r in part_a_results])), 4),
                "max":  round(float(np.max([r["mean_alignment"]  for r in part_a_results])), 4),
                "content_dependent": sum(1 for r in part_a_results if r["head_type"] == "content-dependent"),
                "position_based":    sum(1 for r in part_a_results if r["head_type"] == "position-based"),
            },
        },
        "part_b_construct_attention": part_b_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"rq2_{task_slug}_{model}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  [Part A — top 3 heads]")
    for r in part_a_results[:3]:
        print(f"    Layer {r['layer']}, Head {r['head_id']}: "
              f"{r['mean_alignment']:.3f}  ({r['head_type']})")

    print(f"\n  ✓ Saved to {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["unixcoder", "codebert"])
    parser.add_argument("--task",  required=True,
                        choices=["code-to-text", "code-to-code"])
    args = parser.parse_args()

    print("=" * 60)
    print(f"SCRIPT 6: RQ2 — ATTENTION ANALYSIS")
    print(f"  model={args.model}  task={args.task}")
    print("=" * 60)

    analyse(args.model, args.task)


if __name__ == "__main__":
    main()
