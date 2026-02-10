#!/usr/bin/env python3
"""
Script 7: RQ3 — Structural Probing & Tree Induction  (Wan et al.)
Part A: Linear structural probe — Spearman correlation per layer.
Part B: Unsupervised tree induction via Chu-Liu-Edmonds — F1 per layer.

Input:  data/features/{task}_{model}.h5
        data/stratified_2k_{task}_with_asts.jsonl
Output: results/rq3_{task}_{model}.json

Usage:
    python scripts/rq3_probing.py --model unixcoder --task code-to-text
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import stream_samples, load_metadata

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")

# Probe hyperparameters (Wan et al.)
MAX_SEQ_LEN  = 100
PROBE_EPOCHS = 50
LR           = 1e-3
BATCH_SIZE   = 32

# Tree induction
LAMBDA_BIAS  = 1.0
MAX_TREE_SAMPLES = 50   # O(n²) memory — cap for tractability


# ── Part A: Structural Probe ───────────────────────────────────────────────

class StructuralProbe(nn.Module):
    """Linear probe: d_B(h_i, h_j)² = (B(h_i−h_j))ᵀ (B(h_i−h_j))."""
    def __init__(self, hidden: int):
        super().__init__()
        self.B = nn.Parameter(torch.randn(hidden, hidden) * 0.01)

    def forward(self, hi, hj):
        diff = hi - hj
        t    = diff @ self.B
        return (t * t).sum(dim=1)


def get_ast_distances(ast: dict) -> np.ndarray:
    """Compute tree-distance matrix between leaf tokens."""
    leaves = ast.get("leaf_nodes", [])
    n = len(leaves)
    if n == 0:
        return np.zeros((0, 0))

    # Build parent map from serialised tree
    parent_map: dict = {}
    _build_parent_map(ast.get("ast_tree", {}), None, parent_map, [0])

    D = np.full((n, n), n * 2, dtype=float)
    np.fill_diagonal(D, 0)

    for i in range(n):
        for j in range(i + 1, n):
            d = _tree_dist(i, j, parent_map)
            D[i, j] = D[j, i] = d
    return D


def _build_parent_map(node: dict, parent, pm: dict, ctr: list):
    my_id = ctr[0]; ctr[0] += 1
    pm[my_id] = {"parent": parent, "is_leaf": "children" not in node or not node.get("children")}
    for child in node.get("children", []):
        _build_parent_map(child, my_id, pm, ctr)


def _tree_dist(i: int, j: int, pm: dict) -> int:
    def ancestors(n):
        path = []
        while n is not None and n in pm:
            path.append(n)
            n = pm[n]["parent"]
        return path
    pi, pj = ancestors(i), ancestors(j)
    si = set(pi)
    for k, node in enumerate(pj):
        if node in si:
            return pi.index(node) + k
    return len(pi) + len(pj)


def train_probe(emb_list, dist_list, hidden):
    probe = StructuralProbe(hidden)
    opt   = optim.Adam(probe.parameters(), lr=LR)
    crit  = nn.MSELoss()

    for _ in range(PROBE_EPOCHS):
        for emb, dist in zip(emb_list, dist_list):
            n    = emb.shape[0]
            embt = torch.FloatTensor(emb)
            for bs in range(0, n * n, BATCH_SIZE):
                pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
                batch = pairs[bs:bs + BATCH_SIZE]
                if not batch:
                    break
                hi  = torch.stack([embt[i] for i, j in batch])
                hj  = torch.stack([embt[j] for i, j in batch])
                tgt = torch.FloatTensor([dist[i, j] ** 2 for i, j in batch])
                opt.zero_grad()
                loss = crit(probe(hi, hj), tgt)
                loss.backward()
                opt.step()
    return probe


def eval_probe(probe, emb_list, dist_list):
    probe.eval()
    corrs = []
    with torch.no_grad():
        for emb, gold in zip(emb_list, dist_list):
            n    = emb.shape[0]
            embt = torch.FloatTensor(emb)
            pred = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        hi = embt[i].unsqueeze(0)
                        hj = embt[j].unsqueeze(0)
                        pred[i, j] = probe(hi, hj).item() ** 0.5
            c, _ = spearmanr(gold.flatten(), pred.flatten())
            if not np.isnan(c):
                corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0, float(np.std(corrs)) if corrs else 0.0


# ── Part B: Tree Induction ─────────────────────────────────────────────────

def syntactic_distances_l2(emb: np.ndarray) -> np.ndarray:
    """Adjacent L2 distances. Shape: (seq-1,)."""
    return np.sqrt(np.sum((emb[1:] - emb[:-1]) ** 2, axis=1))


def apply_right_bias(dists: np.ndarray, lam: float = LAMBDA_BIAS) -> np.ndarray:
    """Right-skewness bias from Wan et al. Eq. 9."""
    m   = len(dists) + 1
    avg = np.mean(dists)
    out = dists.copy()
    for i in range(1, len(dists)):      # 1-based in paper, 0-based here
        out[i] += lam * avg * (1 - 1 / ((m - 1) * i))
    return out


def induce_tree(dists: np.ndarray) -> List[int]:
    """Chu-Liu-Edmonds → parent list."""
    n = len(dists) + 1
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n - 1):
        w = -dists[i]
        G.add_edge(i, i + 1, weight=w)
        G.add_edge(i + 1, i, weight=w)
    for i in range(n):
        for j in range(i + 2, n):
            w = -float(np.mean(dists[i:j]))
            G.add_edge(i, j, weight=w)
            G.add_edge(j, i, weight=w)
    try:
        mst = nx.maximum_spanning_arborescence(G)
        parents = [-1] * n
        for i in range(1, n):
            preds = list(mst.predecessors(i))
            parents[i] = preds[0] if preds else 0
        return parents
    except Exception:
        return [-1] + list(range(n - 1))  # right-branching fallback


def tree_siblings(parents: List[int]) -> set:
    children = defaultdict(list)
    for i, p in enumerate(parents):
        if p != -1:
            children[p].append(i)
    pairs = set()
    for ch in children.values():
        for i in range(len(ch)):
            for j in range(i + 1, len(ch)):
                pairs.add((ch[i], ch[j]))
                pairs.add((ch[j], ch[i]))
    return pairs


def gold_siblings(ast: dict) -> set:
    pairs = set()
    def traverse(node):
        if "children" not in node or not node["children"]:
            return [None]   # leaf placeholder
        child_groups = [traverse(c) for c in node["children"]]
        # pairs between children
        for i in range(len(child_groups)):
            for j in range(i + 1, len(child_groups)):
                for li in child_groups[i]:
                    for lj in child_groups[j]:
                        if li is not None and lj is not None:
                            pairs.add((li, lj))
                            pairs.add((lj, li))
        return [l for g in child_groups for l in g]

    ctr = [0]
    def traverse_with_idx(node):
        if "children" not in node or not node["children"]:
            idx = ctr[0]; ctr[0] += 1
            return [idx]
        groups = []
        for c in node["children"]:
            g = traverse_with_idx(c)
            groups.append(g)
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                for li in groups[i]:
                    for lj in groups[j]:
                        pairs.add((li, lj))
                        pairs.add((lj, li))
        return [l for g in groups for l in g]

    traverse_with_idx(ast.get("ast_tree", {}))
    return pairs


def f1_score(induced: set, gold: set) -> float:
    if not gold:
        return 0.0
    inter = induced & gold
    p = len(inter) / len(induced) if induced else 0.0
    r = len(inter) / len(gold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── main ───────────────────────────────────────────────────────────────────

def analyse(model: str, task: str):
    task_slug  = task.replace("-", "_")
    h5_path    = DATA_DIR / "features" / f"{task}_{model}.h5"
    jsonl_path = DATA_DIR / f"stratified_2k_{task_slug}_with_asts.jsonl"

    if not h5_path.exists() or not jsonl_path.exists():
        print("✗ Required files not found")
        return

    meta   = load_metadata(h5_path)
    n_lay  = meta["num_layers"]
    hidden = meta["hidden_size"]
    print(f"  {n_lay} layers, hidden={hidden}")

    # ── collect per-layer data (Part A) ───────────────────────────────────
    probe_data: dict = defaultdict(lambda: {"emb": [], "dist": []})
    tree_data:  List = []

    for sample in tqdm(stream_samples(h5_path, jsonl_path, max_seq_len=MAX_SEQ_LEN),
                       desc="  Loading", total=meta["num_samples"]):
        ast_info = sample["record"].get("ast_info", {})
        if not isinstance(ast_info, dict) or "ast_tree" not in ast_info:
            continue

        D = get_ast_distances(ast_info)
        if D.shape[0] == 0:
            continue

        for layer_idx in range(n_lay):
            emb = sample["embeddings"][layer_idx]
            n   = min(emb.shape[0], D.shape[0])
            if n < 3:
                continue
            probe_data[layer_idx]["emb"].append(emb[:n])
            probe_data[layer_idx]["dist"].append(D[:n, :n])

        if len(tree_data) < MAX_TREE_SAMPLES:
            tree_data.append({
                "embeddings": sample["embeddings"],
                "ast_info":   ast_info,
            })

    # ── Part A: train + eval probe per layer ──────────────────────────────
    probe_results = []
    for layer_idx in tqdm(range(n_lay), desc="  Probing layers"):
        emb_all  = probe_data[layer_idx]["emb"]
        dist_all = probe_data[layer_idx]["dist"]
        if len(emb_all) < 5:
            continue

        emb_tr, emb_te, dist_tr, dist_te = train_test_split(
            emb_all, dist_all, test_size=0.2, random_state=42)

        probe = train_probe(emb_tr, dist_tr, hidden)
        rho, std = eval_probe(probe, emb_te, dist_te)

        probe_results.append({
            "layer":      layer_idx,
            "spearman":   round(rho, 4),
            "std":        round(std, 4),
            "n_samples":  len(emb_all),
        })

    best_probe = max(probe_results, key=lambda x: x["spearman"]) if probe_results else {}

    # ── Part B: tree induction per layer ──────────────────────────────────
    tree_results = []
    for layer_idx in range(n_lay):
        f1s = []
        for s in tree_data:
            emb  = s["embeddings"][layer_idx]
            ast  = s["ast_info"]
            if emb.shape[0] < 3:
                continue
            dists   = syntactic_distances_l2(emb)
            dists   = apply_right_bias(dists)
            parents = induce_tree(dists)
            ind_sib = tree_siblings(parents)
            gld_sib = gold_siblings(ast)
            f1s.append(f1_score(ind_sib, gld_sib))

        if f1s:
            tree_results.append({
                "layer":    layer_idx,
                "mean_f1":  round(float(np.mean(f1s)), 4),
                "std_f1":   round(float(np.std(f1s)),  4),
                "n_samples": len(f1s),
            })

    best_tree = max(tree_results, key=lambda x: x["mean_f1"]) if tree_results else {}

    # ── save ───────────────────────────────────────────────────────────────
    out = {
        "model":  model, "task": task,
        "part_a_structural_probing": {
            "all_layers":  probe_results,
            "best_layer":  best_probe,
            "layer_summary": {f"layer_{r['layer']}": r["spearman"] for r in probe_results},
        },
        "part_b_tree_induction": {
            "all_layers":  tree_results,
            "best_layer":  best_tree,
            "layer_summary": {f"layer_{r['layer']}": r["mean_f1"] for r in tree_results},
            "n_samples_used": len(tree_data),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"rq3_{task_slug}_{model}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    if best_probe:
        print(f"\n  [Probing] Best: Layer {best_probe['layer']} ρ={best_probe['spearman']:.3f}")
    if best_tree:
        print(f"  [Tree]    Best: Layer {best_tree['layer']} F1={best_tree['mean_f1']:.3f}")
    print(f"  ✓ Saved to {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["unixcoder", "codebert"])
    parser.add_argument("--task",  required=True,
                        choices=["code-to-text", "code-to-code"])
    args = parser.parse_args()

    print("=" * 60)
    print(f"SCRIPT 7: RQ3 — STRUCTURAL PROBING + TREE INDUCTION")
    print(f"  model={args.model}  task={args.task}")
    print("=" * 60)

    analyse(args.model, args.task)


if __name__ == "__main__":
    main()
