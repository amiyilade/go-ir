#!/usr/bin/env python3
"""
Script 7: RQ3 — Structural Probing + Tree Induction  (vectorized, fast)

Key fixes vs original:
  - AST distances computed ONCE, reused across all 12 layers
  - Pairs built with torch.triu_indices (no Python nested loops)
  - Entire probe forward pass in ONE batched call per sample
  - MPS/CUDA used when available
  - Reduced defaults: 20 epochs, max 400 train samples, max 80 tokens

Usage:
    python 7_rq3_probing.py --model unixcoder --task code-to-text
Output:
    results/rq3_{task}_{model}.json
"""

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import stream_samples, load_metadata, get_embedding

DATA_DIR     = Path("data")
FEATURES_DIR = Path("data/features")
RESULTS_DIR  = Path("results")

MAX_SEQ_LEN           = 80
PROBE_EPOCHS          = 20
PROBE_LR              = 1e-3
MAX_TRAIN_SAMPLES     = 2000
MAX_INDUCTION_SAMPLES = 50
LAMBDA_BIAS           = 1.0

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()  else
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cpu")
)


# ── Probe ──────────────────────────────────────────────────────────────────────
class StructuralProbe(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.B = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)

    def forward_pairs(self, H, idx_i, idx_j):
        diff = H[idx_i] - H[idx_j]
        return (diff @ self.B ** 2).sum(dim=1)


# ── AST helpers ────────────────────────────────────────────────────────────────
def get_tree_distances(ast_info):
    leaf_nodes = ast_info.get('leaf_nodes', [])
    n = len(leaf_nodes)
    if n == 0:
        return np.array([])
    parent_map = {}
    _build_parent_map(ast_info.get('ast_tree', {}), None, parent_map, [0])
    dist_mat = np.full((n, n), n * 2, dtype=np.float32)
    np.fill_diagonal(dist_mat, 0)
    for i in range(n):
        for j in range(i + 1, n):
            d = _tree_distance(i, j, parent_map)
            dist_mat[i, j] = d; dist_mat[j, i] = d
    return dist_mat

def _build_parent_map(node, parent_id, mapping, counter):
    my_id = counter[0]; counter[0] += 1
    mapping[my_id] = {'parent': parent_id}
    for child in node.get('children', []):
        _build_parent_map(child, my_id, mapping, counter)

def _tree_distance(i, j, parent_map):
    path_i = []
    cur = i
    while cur is not None and cur in parent_map:
        path_i.append(cur); cur = parent_map[cur]['parent']
    set_i = set(path_i)
    path_j = []; cur = j; lca = None
    while cur is not None and cur in parent_map:
        path_j.append(cur)
        if cur in set_i: lca = cur; break
        cur = parent_map[cur]['parent']
    if lca is None:
        return len(path_i) + len(path_j)
    return path_i.index(lca) + path_j.index(lca)

def gold_pairs_from_ast(ast_info):
    pairs = set(); leaf_counter = [0]
    def traverse(node):
        if 'children' not in node or not node.get('children'):
            idx = leaf_counter[0]; leaf_counter[0] += 1; return [idx]
        groups = [traverse(c) for c in node['children']]
        for ii in range(len(groups)):
            for jj in range(ii + 1, len(groups)):
                for a in groups[ii]:
                    for b in groups[jj]:
                        pairs.add((a, b)); pairs.add((b, a))
        return [x for g in groups for x in g]
    if ast_info.get('ast_tree'):
        traverse(ast_info['ast_tree'])
    return pairs


# ── Part A ─────────────────────────────────────────────────────────────────────
def run_part_a(samples, meta):
    print(f"\n  Part A: Structural Probing  "
          f"(epochs={PROBE_EPOCHS}, max_train={MAX_TRAIN_SAMPLES}, device={DEVICE})")

    num_layers  = meta['num_layers']
    hidden_size = meta['hidden_size']

    # Pre-compute AST distances ONCE for all samples
    print("    Pre-computing AST distances...", end=' ', flush=True)
    ast_cache = {}
    for s in samples:
        ai = s.get('ast_info')
        if not ai: continue
        dm = get_tree_distances(ai)
        if dm.ndim < 2 or dm.shape[0] < 3: continue
        t = min(dm.shape[0], MAX_SEQ_LEN)
        ast_cache[id(s)] = dm[:t, :t]
    print(f"{len(ast_cache)} cached")

    layer_results = []
    for layer in tqdm(range(num_layers), desc="  Probing layers"):
        emb_list, dist_list = [], []
        for s in samples:
            dm = ast_cache.get(id(s))
            if dm is None: continue
            emb = get_embedding(s, layer)
            if emb is None: continue
            t = min(emb.shape[0], dm.shape[0])
            if t < 3: continue
            emb_list.append(emb[:t].astype(np.float32))
            dist_list.append(dm[:t, :t])

        if len(emb_list) < 5:
            layer_results.append({'layer': layer, 'spearman': 0.0, 'std': 0.0, 'n': 0})
            continue

        # Cap training size
        if len(emb_list) > MAX_TRAIN_SAMPLES + 100:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(emb_list), MAX_TRAIN_SAMPLES + 100, replace=False)
            emb_list  = [emb_list[i]  for i in idx]
            dist_list = [dist_list[i] for i in idx]

        tr_emb, te_emb, tr_dist, te_dist = train_test_split(
            emb_list, dist_list, test_size=0.2, random_state=42)

        probe = StructuralProbe(hidden_size).to(DEVICE)
        opt   = optim.Adam(probe.parameters(), lr=PROBE_LR)
        crit  = nn.MSELoss()

        for _ep in range(PROBE_EPOCHS):
            probe.train()
            for emb, dist in zip(tr_emb, tr_dist):
                n = emb.shape[0]
                H = torch.from_numpy(emb).to(DEVICE)
                # Build ALL pairs vectorised — no Python loop
                ii, jj = torch.triu_indices(n, n, offset=1, device=DEVICE)
                idx_i  = torch.cat([ii, jj])
                idx_j  = torch.cat([jj, ii])
                tgt    = torch.from_numpy(
                    dist[idx_i.cpu().numpy(), idx_j.cpu().numpy()]).to(DEVICE)
                opt.zero_grad()
                pred = probe.forward_pairs(H, idx_i, idx_j)
                loss = crit(pred, tgt ** 2)
                loss.backward(); opt.step()

        probe.eval()
        correlations = []
        with torch.no_grad():
            for emb, dist in zip(te_emb, te_dist):
                n = emb.shape[0]
                H = torch.from_numpy(emb).to(DEVICE)
                ii, jj = torch.triu_indices(n, n, offset=1, device=DEVICE)
                idx_i  = torch.cat([ii, jj]); idx_j = torch.cat([jj, ii])
                pred_d2 = probe.forward_pairs(H, idx_i, idx_j).cpu().numpy()
                pred_d  = np.sqrt(np.maximum(pred_d2, 0))
                gold_d  = dist[idx_i.cpu().numpy(), idx_j.cpu().numpy()]
                corr, _ = spearmanr(gold_d, pred_d)
                if not np.isnan(corr): correlations.append(float(corr))

        row = {
            'layer':    layer,
            'spearman': float(np.mean(correlations)) if correlations else 0.0,
            'std':      float(np.std(correlations))  if correlations else 0.0,
            'n':        len(correlations),
        }
        layer_results.append(row)
        tqdm.write(f"    Layer {layer:2d}: ρ={row['spearman']:.3f}  (n={row['n']})")

    best = max(layer_results, key=lambda x: x['spearman'])
    return {'layer_results': layer_results, 'best_layer': best,
            'layer_summary': {f"layer_{r['layer']}": r['spearman'] for r in layer_results}}


# ── Part B ─────────────────────────────────────────────────────────────────────
def compute_distances(emb):
    diffs = emb[:-1] - emb[1:]
    return np.sqrt((diffs ** 2).sum(axis=1))

def apply_bias(distances, lam=LAMBDA_BIAS):
    m = len(distances) + 1; avg = np.mean(distances); out = distances.copy()
    for i in range(len(distances)):
        ip = i + 1
        if ip > 1: out[i] += lam * avg * (1 - 1.0 / ((m - 1) * ip))
    return out

def induce_tree(distances):
    n = len(distances) + 1
    G = nx.DiGraph()
    for i in range(n): G.add_node(i)
    for i in range(n - 1):
        w = -distances[i]; G.add_edge(i, i+1, weight=w); G.add_edge(i+1, i, weight=w)
    for i in range(n):
        for j in range(i + 2, n):
            w = -float(np.mean(distances[i:j]))
            G.add_edge(i, j, weight=w); G.add_edge(j, i, weight=w)
    try:
        mst = nx.maximum_spanning_arborescence(G)
        parents = [-1] * n
        for i in range(1, n):
            preds = list(mst.predecessors(i)); parents[i] = preds[0] if preds else 0
        return parents
    except Exception:
        return [-1] + list(range(n - 1))

def tree_to_pairs(parents):
    children_of = defaultdict(list)
    for i, p in enumerate(parents):
        if p != -1: children_of[p].append(i)
    pairs = set()
    for ch in children_of.values():
        for ii in range(len(ch)):
            for jj in range(ii + 1, len(ch)):
                pairs.add((ch[ii], ch[jj])); pairs.add((ch[jj], ch[ii]))
    return pairs

def f1_score(induced, gold):
    if not gold: return 0.0
    inter = induced & gold
    p = len(inter) / len(induced) if induced else 0.0
    r = len(inter) / len(gold)
    return 2 * p * r / (p + r) if (p + r) else 0.0

def run_part_b(samples, meta):
    print(f"\n  Part B: Tree Induction (≤{MAX_INDUCTION_SAMPLES} samples)...")
    num_layers = meta['num_layers']
    valid = [s for s in samples if s.get('ast_info')][:MAX_INDUCTION_SAMPLES]
    print(f"    Using {len(valid)} samples")
    layer_results = []
    for layer in tqdm(range(num_layers), desc="  Inducing trees"):
        f1s = []
        for s in valid:
            emb = get_embedding(s, layer)
            if emb is None or emb.shape[0] < 2: continue
            dists   = apply_bias(compute_distances(emb))
            parents = induce_tree(dists)
            f1s.append(f1_score(tree_to_pairs(parents), gold_pairs_from_ast(s['ast_info'])))
        row = {'layer': layer,
               'mean_f1': float(np.mean(f1s)) if f1s else 0.0,
               'std_f1':  float(np.std(f1s))  if f1s else 0.0,
               'n': len(f1s)}
        layer_results.append(row)
        tqdm.write(f"    Layer {layer:2d}: F1={row['mean_f1']:.3f}")
    best = max(layer_results, key=lambda x: x['mean_f1'])
    return {'layer_results': layer_results, 'best_layer': best,
            'layer_summary': {f"layer_{r['layer']}": r['mean_f1'] for r in layer_results}}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['unixcoder', 'codebert'])
    parser.add_argument('--task',  required=True, choices=['code-to-text', 'code-to-code'])
    args = parser.parse_args()

    task_key   = args.task.replace('-', '_')
    h5_path    = FEATURES_DIR / f"{task_key}_{args.model}.h5"
    jsonl_path = DATA_DIR / args.task / f"stratified_2k_{task_key}_with_asts.jsonl"

    print("\n" + "="*60)
    print(f"SCRIPT 7: RQ3 — STRUCTURAL PROBING + TREE INDUCTION")
    print(f"  model={args.model}  task={args.task}")
    print("="*60)

    if not h5_path.exists():
        print(f"\n✗ HDF5 not found: {h5_path}. Run Script 4 first."); return

    meta = load_metadata(h5_path)
    print(f"\n  {meta['num_layers']} layers, hidden={meta['hidden_size']}")

    print("\n  Loading samples...")
    samples = list(tqdm(stream_samples(h5_path, jsonl_path), desc="  Loading "))
    print(f"  Loaded {len(samples)}")

    part_a = run_part_a(samples, meta)
    part_b = run_part_b(samples, meta)

    print(f"\n  ── Summary ──")
    print(f"  Probing   best: Layer {part_a['best_layer']['layer']}  "
          f"(ρ={part_a['best_layer']['spearman']:.3f})")
    print(f"  Induction best: Layer {part_b['best_layer']['layer']}  "
          f"(F1={part_b['best_layer']['mean_f1']:.3f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"rq3_{task_key}_{args.model}.json"
    with open(out, 'w') as f:
        json.dump({'model': args.model, 'task': args.task,
                   'part_a_structural_probing': part_a,
                   'part_b_tree_induction':     part_b}, f, indent=2)
    print(f"\n✔ Saved: {out}")

if __name__ == "__main__":
    main()
