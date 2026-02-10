#!/usr/bin/env python3
"""
Stratified 2k Sample Creator
Creates a 2,000-sample subset that mirrors the full dataset's
Go construct distribution.

Usage:
    python create_stratified_sample.py
"""

import json
import random
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

random.seed(42)
np.random.seed(42)

# Configuration
TARGET_SIZE = 2000
CODE_TO_TEXT_DIR = Path("data/code-to-text")
CODE_TO_CODE_DIR = Path("data/code-to-code")
FEATURES_DIR = Path("results/features")

# The 8 Go constructs
GO_CONSTRUCTS = [
    'goroutines', 'channels', 'defer', 'error_patterns',
    'select_statements', 'interfaces', 'type_assertions', 'context_usage'
]

def get_construct_profile(go_constructs: dict) -> tuple:
    """
    Create a construct profile for a sample.
    Returns a tuple of which constructs are present (for stratification).
    
    We bucket into 4 groups:
      - 'none':    no constructs
      - 'error':   only error_patterns (most common)
      - 'conc':    has goroutines or channels (concurrency)
      - 'mixed':   has defer/interfaces/context/etc.
    """
    if not go_constructs:
        return 'none'
    
    has_goroutines = len(go_constructs.get('goroutines', [])) > 0
    has_channels   = len(go_constructs.get('channels', [])) > 0
    has_error      = len(go_constructs.get('error_patterns', [])) > 0
    has_defer      = len(go_constructs.get('defer', [])) > 0
    has_iface      = len(go_constructs.get('interfaces', [])) > 0
    has_context    = len(go_constructs.get('context_usage', [])) > 0
    has_select     = len(go_constructs.get('select_statements', [])) > 0
    has_type_assert = len(go_constructs.get('type_assertions', [])) > 0
    
    if has_goroutines or has_channels or has_select:
        return 'concurrency'
    elif has_error and (has_defer or has_iface or has_context or has_type_assert):
        return 'error_plus'
    elif has_error:
        return 'error_only'
    else:
        return 'none'


def load_samples_with_profiles(ast_file: Path, code_label: str) -> list:
    """Load all samples and compute their construct profile."""
    print(f"  Loading {ast_file.name}...")
    
    samples = []
    with open(ast_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            go_constructs = data.get('go_constructs', {}).get(code_label, {})
            profile = get_construct_profile(go_constructs)
            samples.append({
                'line_index': i,
                'profile': profile,
                'construct_counts': {
                    c: len(go_constructs.get(c, [])) for c in GO_CONSTRUCTS
                }
            })
    
    print(f"  ✓ Loaded {len(samples)} samples")
    return samples


def stratified_sample(samples: list, target_size: int) -> list:
    """
    Select target_size samples while preserving construct profile distribution.
    """
    # Group by profile
    groups = defaultdict(list)
    for s in samples:
        groups[s['profile']].append(s)
    
    total = len(samples)
    
    print(f"\n  Profile distribution in full dataset:")
    for profile, group in sorted(groups.items()):
        pct = len(group) / total * 100
        target_n = round(len(group) / total * target_size)
        print(f"    {profile:15s}: {len(group):5d} ({pct:.1f}%) → {target_n} in sample")
    
    # Sample proportionally from each group
    selected = []
    for profile, group in groups.items():
        n = round(len(group) / total * target_size)
        n = min(n, len(group))  # Can't sample more than available
        selected.extend(random.sample(group, n))
    
    # If we're under target (due to rounding), top up from largest group
    largest_profile = max(groups, key=lambda k: len(groups[k]))
    already_selected_indices = {s['line_index'] for s in selected}
    remaining = [s for s in groups[largest_profile] 
                 if s['line_index'] not in already_selected_indices]
    
    shortfall = target_size - len(selected)
    if shortfall > 0 and remaining:
        selected.extend(random.sample(remaining, min(shortfall, len(remaining))))
    
    print(f"\n  ✓ Selected {len(selected)} samples (target: {target_size})")
    return selected


def write_stratified_jsonl(ast_file: Path, selected_indices: set, output_file: Path):
    """Write the selected samples to a new JSONL file."""
    written = 0
    with open(ast_file, 'r') as fin, open(output_file, 'w') as fout:
        for i, line in enumerate(fin):
            if i in selected_indices:
                fout.write(line)
                written += 1
    print(f"  ✓ Wrote {written} samples to {output_file.name}")


def write_stratified_hdf5(features_file: Path, selected_indices: set, 
                           output_file: Path, code_label: str):
    """Copy selected samples from HDF5 to a new HDF5 file."""
    print(f"  Copying HDF5 data for {len(selected_indices)} samples...")
    
    with h5py.File(features_file, 'r') as h5_in, \
         h5py.File(output_file, 'w') as h5_out:
        
        # Copy metadata
        meta = h5_out.create_group('metadata')
        for key, val in h5_in['metadata'].attrs.items():
            meta.attrs[key] = val
        meta.attrs['total_samples'] = len(selected_indices)
        meta.attrs['stratified_from'] = str(h5_in['metadata'].attrs.get('total_samples', '?'))
        
        # Copy selected samples
        new_idx = 0
        for old_idx in sorted(selected_indices):
            old_key = f'sample_{old_idx}'
            if old_key not in h5_in:
                continue
            
            # Copy under new sequential index
            h5_in.copy(h5_in[old_key], h5_out, name=f'sample_{new_idx}')
            new_idx += 1
            
            if new_idx % 200 == 0:
                print(f"    Copied {new_idx}/{len(selected_indices)}...")
    
    print(f"  ✓ Wrote HDF5 with {new_idx} samples to {output_file.name}")


def verify_distribution(samples: list, selected: list):
    """Print distribution comparison to confirm stratification worked."""
    full_counts = Counter(s['profile'] for s in samples)
    sel_counts  = Counter(s['profile'] for s in selected)
    total_full  = len(samples)
    total_sel   = len(selected)
    
    print(f"\n  {'Profile':<15} {'Full %':>8} {'Sample %':>10} {'Match?':>8}")
    print(f"  {'-'*45}")
    for profile in sorted(full_counts):
        full_pct = full_counts[profile] / total_full * 100
        sel_pct  = sel_counts.get(profile, 0) / total_sel * 100
        diff     = abs(full_pct - sel_pct)
        ok       = '✓' if diff < 2.0 else '⚠'
        print(f"  {profile:<15} {full_pct:>7.1f}% {sel_pct:>9.1f}% {ok:>8}")


def main():
    print("=" * 60)
    print("STRATIFIED 2K SAMPLE CREATOR")
    print("=" * 60)
    
    tasks = [
        {
            'name':        'code-to-text',
            'ast_file':    CODE_TO_TEXT_DIR / 'full_code_to_text_with_asts.jsonl',
            'out_ast':     CODE_TO_TEXT_DIR / 'stratified_2k_with_asts.jsonl',
            'feat_file':   FEATURES_DIR / 'code_to_text_unixcoder_optimized.h5',
            'out_feat':    FEATURES_DIR / 'code_to_text_unixcoder_strat2k.h5',
            'code_label':  'code',
        },
        {
            'name':        'code-to-code',
            'ast_file':    CODE_TO_CODE_DIR / 'full_code_to_code_with_asts.jsonl',
            'out_ast':     CODE_TO_CODE_DIR / 'stratified_2k_with_asts.jsonl',
            'feat_file':   FEATURES_DIR / 'code_to_code_unixcoder_optimized.h5',
            'out_feat':    FEATURES_DIR / 'code_to_code_unixcoder_strat2k.h5',
            'code_label':  'initial_segment',
        },
    ]
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task['name'].upper()}")
        print(f"{'='*60}")
        
        if not task['ast_file'].exists():
            print(f"  ✗ AST file not found: {task['ast_file']}")
            continue
        
        # Load samples + profiles
        samples = load_samples_with_profiles(task['ast_file'], task['code_label'])
        
        # Stratified selection
        selected = stratified_sample(samples, TARGET_SIZE)
        selected_indices = {s['line_index'] for s in selected}
        
        # Verify distribution matches
        print(f"\n  Distribution check:")
        verify_distribution(samples, selected)
        
        # Write stratified JSONL
        print(f"\n  Writing stratified JSONL...")
        write_stratified_jsonl(task['ast_file'], selected_indices, task['out_ast'])
        
        # Write stratified HDF5 (only if features exist)
        if task['feat_file'].exists():
            print(f"\n  Writing stratified HDF5...")
            write_stratified_hdf5(
                task['feat_file'], selected_indices, 
                task['out_feat'], task['code_label']
            )
        else:
            print(f"\n  ⚠ HDF5 not found at {task['feat_file']}, skipping feature copy")
        
    print(f"\n{'='*60}")
    print("✓ STRATIFIED SAMPLING COMPLETE")
    print(f"{'='*60}")
    print(f"""
Output files:
  data/code-to-text/stratified_2k_with_asts.jsonl
  data/code-to-code/stratified_2k_with_asts.jsonl
  results/features/code_to_text_unixcoder_strat2k.h5
  results/features/code_to_code_unixcoder_strat2k.h5

Run analysis with:
  python scripts/analyze_attention_ast_streaming.py \\
      --model unixcoder \\
      --task code-to-text \\
      --ast-file data/code-to-text/stratified_2k_with_asts.jsonl \\
      --features-file results/features/code_to_text_unixcoder_strat2k.h5
""")


if __name__ == '__main__':
    main()
