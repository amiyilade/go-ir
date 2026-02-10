#!/usr/bin/env python3
"""
Script 6: RQ2 — Attention Analysis
Part A: Attention-AST alignment (Wan et al. methodology)
Part B: Attention entropy/focus — samples WITH vs WITHOUT each Go construct

Usage:
    python 6_rq2_attention.py --model unixcoder --task code-to-text
    python 6_rq2_attention.py --model codebert  --task code-to-code

Output:
    results/rq2_{task}_{model}.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import (
    stream_samples, load_metadata, get_attention_matrix,
    has_construct, count_construct
)

DATA_DIR    = Path("data")
FEATURES_DIR = Path("data/features")
RESULTS_DIR  = Path("results")

CONSTRUCTS = [
    'goroutines', 'channels', 'defer', 'error_patterns',
    'select_statements', 'interfaces', 'type_assertions', 'context_usage'
]
KEY_HEADS = [0, 3, 5, 7, 11]

# Wan et al. parameters
ATTENTION_THRESHOLD     = 0.3   # θ: high-confidence attention
MIN_HIGH_CONF_PAIRS     = 10    # minimum pairs required for alignment score


# -----------------------------------------------------------------------
# Part A: Attention-AST Alignment
# -----------------------------------------------------------------------

def get_ast_parent_pairs(ast_info: dict) -> set:
    """Extract (i,j) pairs where tokens i,j share the same AST parent node."""
    if not ast_info:
        return set()

    pairs = set()
    leaf_counter = [0]

    def get_leaf_indices(node) -> list:
        if 'children' not in node or not node.get('children'):
            idx = leaf_counter[0]
            leaf_counter[0] += 1
            return [idx]
        indices = []
        for child in node['children']:
            indices.extend(get_leaf_indices(child))
        return indices

    def traverse(node):
        if 'children' not in node:
            return
        children = node['children']
        sibling_leaves = []
        for child in children:
            sibling_leaves.append(get_leaf_indices(child))

        # pairs between siblings (non-adjacent, per paper)
        flat = [idx for grp in sibling_leaves for idx in grp]
        for ii in range(len(flat)):
            for jj in range(ii + 1, len(flat)):
                if abs(flat[ii] - flat[jj]) > 1:
                    pairs.add((flat[ii], flat[jj]))
                    pairs.add((flat[jj], flat[ii]))

        for child in children:
            traverse(child)

    ast_tree = ast_info.get('ast_tree', {})
    if ast_tree:
        traverse(ast_tree)

    return pairs


def alignment_score_for_head(attn_matrix: np.ndarray,
                              ast_pairs: set,
                              threshold: float = ATTENTION_THRESHOLD) -> dict:
    """p_α(f) as in Wan et al. Equation 4."""
    seq_len = attn_matrix.shape[0]
    high_conf = [(i, j) for i in range(seq_len)
                         for j in range(seq_len)
                         if i != j and attn_matrix[i, j] > threshold]

    if len(high_conf) < MIN_HIGH_CONF_PAIRS:
        return None

    aligned = sum(1 for p in high_conf if p in ast_pairs)
    return {
        'score':       aligned / len(high_conf),
        'aligned':     aligned,
        'total_hc':    len(high_conf),
        'total_ast':   len(ast_pairs),
    }


def attention_variability(matrices: list, max_tok: int = 10) -> float:
    """Equation 5 from Wan et al. High = content-dependent, low = position-based."""
    trimmed = [m[:max_tok, :max_tok] for m in matrices
               if m.shape[0] >= max_tok and m.shape[1] >= max_tok]
    if not trimmed:
        return 0.0
    stack = np.stack(trimmed).astype(np.float32)
    mean  = np.mean(stack, axis=0)
    total = np.sum(stack)
    if total == 0:
        return 0.0
    return float(np.sum((stack - mean) ** 2) / total)


def run_part_a(samples: list, meta: dict) -> dict:
    """Attention-AST alignment across all layer-head combinations."""
    print("\n  Part A: Attention-AST Alignment...")

    num_layers = meta['num_layers']
    results_by_head = defaultdict(lambda: {'scores': [], 'matrices': []})

    for sample in tqdm(samples, desc="    Alignment"):
        ast_info = sample['ast_info']
        if not ast_info:
            continue
        ast_pairs = get_ast_parent_pairs(ast_info)
        if not ast_pairs:
            continue

        for layer in range(num_layers):
            for head in KEY_HEADS:
                attn = get_attention_matrix(sample, layer, head)
                if attn is None:
                    continue
                result = alignment_score_for_head(attn, ast_pairs)
                if result:
                    key = (layer, head)
                    results_by_head[key]['scores'].append(result['score'])
                    results_by_head[key]['matrices'].append(attn)

    # Aggregate
    head_results = []
    for (layer, head), data in results_by_head.items():
        scores    = data['scores']
        variability = attention_variability(data['matrices'])
        head_results.append({
            'layer':      layer,
            'head':       head,
            'mean_score': float(np.mean(scores)),
            'std_score':  float(np.std(scores)),
            'variability': variability,
            'head_type':  'content-dependent' if variability > 0.25 else 'position-based',
            'n_samples':  len(scores),
        })

    head_results.sort(key=lambda x: -x['mean_score'])

    # Layer summary
    layer_scores = defaultdict(list)
    for r in head_results:
        layer_scores[r['layer']].append(r['mean_score'])
    layer_summary = {
        f'layer_{l}': {
            'mean': float(np.mean(v)),
            'max':  float(np.max(v)),
        }
        for l, v in layer_scores.items()
    }

    overall_scores = [r['mean_score'] for r in head_results]
    return {
        'all_heads':     head_results,
        'top_10':        head_results[:10],
        'layer_summary': layer_summary,
        'overall': {
            'mean_alignment':        float(np.mean(overall_scores)) if overall_scores else 0,
            'max_alignment':         float(np.max(overall_scores)) if overall_scores else 0,
            'content_dependent_n':   sum(1 for r in head_results if r['head_type'] == 'content-dependent'),
            'position_based_n':      sum(1 for r in head_results if r['head_type'] == 'position-based'),
        },
    }


# -----------------------------------------------------------------------
# Part B: Construct Attention Analysis
# -----------------------------------------------------------------------

def attention_entropy(attn_row: np.ndarray) -> float:
    """Shannon entropy of an attention distribution."""
    p = attn_row.astype(np.float32) + 1e-10
    p /= p.sum()
    return float(-np.sum(p * np.log(p)))


def run_part_b(samples: list, meta: dict) -> dict:
    """
    For each construct × each layer × key heads:
    compare attention entropy and max-attention between
    samples WITH vs WITHOUT the construct.
    """
    print("\n  Part B: Construct Attention Analysis...")

    num_layers = meta['num_layers']
    results = {}

    for construct in CONSTRUCTS:
        print(f"    {construct}...")
        construct_results = []

        for layer in range(num_layers):
            for head in KEY_HEADS:
                entropy_with    = []
                entropy_without = []
                maxattn_with    = []
                maxattn_without = []

                for sample in samples:
                    attn = get_attention_matrix(sample, layer, head)
                    if attn is None:
                        continue
                    avg_ent  = float(np.mean([attention_entropy(row) for row in attn]))
                    avg_max  = float(np.mean(np.max(attn, axis=1)))

                    if has_construct(sample, construct):
                        entropy_with.append(avg_ent)
                        maxattn_with.append(avg_max)
                    else:
                        entropy_without.append(avg_ent)
                        maxattn_without.append(avg_max)

                if len(entropy_with) < 3 or len(entropy_without) < 3:
                    continue

                # t-test
                t_ent, p_ent = stats.ttest_ind(entropy_with, entropy_without)
                t_max, p_max = stats.ttest_ind(maxattn_with, maxattn_without)

                # Cohen's d
                def cohen_d(a, b):
                    na, nb = len(a), len(b)
                    pooled = np.sqrt(((na-1)*np.var(a) + (nb-1)*np.var(b)) / (na+nb-2))
                    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0

                construct_results.append({
                    'layer':              layer,
                    'head':               head,
                    'n_with':             len(entropy_with),
                    'n_without':          len(entropy_without),
                    'mean_entropy_with':  float(np.mean(entropy_with)),
                    'mean_entropy_without': float(np.mean(entropy_without)),
                    'mean_max_attn_with':   float(np.mean(maxattn_with)),
                    'mean_max_attn_without': float(np.mean(maxattn_without)),
                    'ttest_entropy_p':    float(p_ent),
                    'ttest_maxattn_p':    float(p_max),
                    'cohens_d_entropy':   cohen_d(entropy_with, entropy_without),
                    'cohens_d_maxattn':   cohen_d(maxattn_with, maxattn_without),
                })

        # Sort by |Cohen's d| (effect size)
        construct_results.sort(key=lambda x: -abs(x.get('cohens_d_entropy', 0)))
        results[construct] = construct_results

    return results


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
    jsonl_key = f"stratified_2k_{args.task.replace('-', '_')}"

    if args.task == 'code-to-text':
        jsonl_path = DATA_DIR / "code-to-text/stratified_2k_code_to_text_with_asts.jsonl"
    else:
        jsonl_path = DATA_DIR / "code-to-code/stratified_2k_code_to_code_with_asts.jsonl"

    print("\n" + "="*70)
    print(f"SCRIPT 6: RQ2 ATTENTION ANALYSIS")
    print(f"  Model: {args.model}  Task: {args.task}")
    print("="*70)

    if not h5_path.exists():
        print(f"\n✗ HDF5 not found: {h5_path}. Run Script 4 first.")
        return

    meta = load_metadata(h5_path)
    print(f"\n  Model metadata: {meta['num_layers']} layers, "
          f"{meta['num_heads']} heads, hidden={meta['hidden_size']}")

    print("\n  Loading all samples into memory (streaming)...")
    samples = list(stream_samples(h5_path, jsonl_path))
    print(f"  Loaded {len(samples)} samples")

    # Run analyses
    part_a = run_part_a(samples, meta)
    part_b = run_part_b(samples, meta)

    # Print quick summary
    print(f"\n  [Part A Summary]")
    print(f"  Overall mean alignment: {part_a['overall']['mean_alignment']:.3f}")
    print(f"  Max alignment:          {part_a['overall']['max_alignment']:.3f}")
    print(f"  Position-based heads:   {part_a['overall']['position_based_n']}")
    print(f"  Content-dependent:      {part_a['overall']['content_dependent_n']}")

    if part_a['top_10']:
        best = part_a['top_10'][0]
        print(f"\n  Best head: Layer {best['layer']}, Head {best['head']} "
              f"→ {best['mean_score']:.3f} ({best['head_type']})")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"rq2_{task_key}_{args.model}.json"
    output = {
        'model':      args.model,
        'task':       args.task,
        'part_a_attention_ast_alignment': part_a,
        'part_b_construct_attention':     part_b,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✔ Saved: {output_path}  ({output_path.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
