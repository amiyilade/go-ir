#!/usr/bin/env python3
"""
Script 2: Create Stratified 2k Sample
Selects 2,000 samples per task that mirror the full dataset distribution
across BOTH construct profile (4 levels) and code length (3 buckets).

Input:  data/full_code_to_text_constructs.jsonl
        data/full_code_to_code_constructs.jsonl
Output: data/stratified_2k_code_to_text.jsonl
        data/stratified_2k_code_to_code.jsonl

Usage:
    python scripts/create_stratified_sample.py
"""

import json
import random
from pathlib import Path
from collections import defaultdict, Counter

random.seed(42)

TARGET   = 2000
DATA_DIR = Path("data")

TASKS = [
    {
        "input":  DATA_DIR / "full_code_to_text_constructs.jsonl",
        "output": DATA_DIR / "stratified_2k_code_to_text.jsonl",
        "name":   "code-to-text",
    },
    {
        "input":  DATA_DIR / "full_code_to_code_constructs.jsonl",
        "output": DATA_DIR / "stratified_2k_code_to_code.jsonl",
        "name":   "code-to-code",
    },
]


def stratify(records: list, target: int) -> list:
    """
    Stratify by (construct_profile, length_bucket) — 12 possible cells.
    Each cell is sampled proportionally to its share of the full dataset.
    Returns exactly `target` records (adjusts for rounding via largest cell).
    """
    # Group into cells
    cells = defaultdict(list)
    for r in records:
        key = (r["construct_profile"], r["length_bucket"])
        cells[key].append(r)

    total = len(records)

    # Compute per-cell target counts
    cell_targets = {}
    for key, group in cells.items():
        cell_targets[key] = max(1, round(len(group) / total * target))

    # Sample from each cell
    selected = []
    for key, group in cells.items():
        n = min(cell_targets[key], len(group))
        selected.extend(random.sample(group, n))

    # Top up / trim to exactly TARGET
    if len(selected) < target:
        # Add from largest cell not yet fully used
        already = {r["original_index"] for r in selected}
        pool = [r for r in records if r["original_index"] not in already]
        random.shuffle(pool)
        selected.extend(pool[:target - len(selected)])
    elif len(selected) > target:
        random.shuffle(selected)
        selected = selected[:target]

    return selected


def verify(full: list, selected: list):
    """Print distribution comparison."""
    def dist(lst, key):
        c = Counter(r[key] for r in lst)
        t = len(lst)
        return {k: v / t * 100 for k, v in c.items()}

    print(f"\n  {'Cell':<30} {'Full %':>8} {'Sample %':>10} {'OK?':>5}")
    print(f"  {'-'*57}")

    full_profile = dist(full, "construct_profile")
    sel_profile  = dist(selected, "construct_profile")
    for k in sorted(full_profile):
        fp = full_profile[k]
        sp = sel_profile.get(k, 0)
        ok = "✓" if abs(fp - sp) < 3 else "⚠"
        print(f"  profile:{k:<21} {fp:>7.1f}% {sp:>9.1f}% {ok:>5}")

    full_bucket = dist(full, "length_bucket")
    sel_bucket  = dist(selected, "length_bucket")
    for k in ("short", "medium", "long"):
        fp = full_bucket.get(k, 0)
        sp = sel_bucket.get(k, 0)
        ok = "✓" if abs(fp - sp) < 3 else "⚠"
        print(f"  length:{k:<22} {fp:>7.1f}% {sp:>9.1f}% {ok:>5}")


def process_task(task: dict):
    if not task["input"].exists():
        print(f"  ✗ Not found: {task['input']}")
        return

    print(f"\n{'='*60}")
    print(f"TASK: {task['name'].upper()}")
    print(f"{'='*60}")

    with open(task["input"]) as f:
        records = [json.loads(l) for l in f]

    print(f"  Full dataset: {len(records)} samples")

    selected = stratify(records, TARGET)

    # Re-index sequentially; preserve original_index for traceability
    for new_idx, r in enumerate(sorted(selected, key=lambda x: x["original_index"])):
        r["new_index"] = new_idx

    selected_sorted = sorted(selected, key=lambda x: x["original_index"])

    verify(records, selected_sorted)

    with open(task["output"], "w") as f:
        for r in selected_sorted:
            f.write(json.dumps(r) + "\n")

    print(f"\n  ✓ Wrote {len(selected_sorted)} samples to {task['output'].name}")
    print(f"    Each record has: original_index, new_index, construct_profile, "
          f"length_bucket, go_constructs")


def main():
    print("=" * 60)
    print("SCRIPT 2: CREATE STRATIFIED 2K SAMPLE")
    print("=" * 60)

    for task in TASKS:
        process_task(task)

    print(f"\n{'='*60}")
    print("✓ DONE — next: python scripts/parse_asts.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
