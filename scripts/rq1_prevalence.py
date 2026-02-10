#!/usr/bin/env python3
"""
Script 5: RQ1 — Construct Prevalence
Analyses the distribution of Go-specific constructs across the full dataset.
No model required — runs on the construct-annotated full JSONL.

Input:  data/full_code_to_text_constructs.jsonl
        data/full_code_to_code_constructs.jsonl
Output: results/rq1_prevalence_code_to_text.json
        results/rq1_prevalence_code_to_code.json

Usage:
    python scripts/rq1_prevalence.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")

GO_CONSTRUCTS = [
    "goroutines", "channels", "defer", "error_patterns",
    "select_statements", "interfaces", "type_assertions", "context_usage",
]

TASKS = [
    {
        "name":   "code-to-text",
        "input":  DATA_DIR / "full_code_to_text_constructs.jsonl",
        "output": RESULTS_DIR / "rq1_prevalence_code_to_text.json",
    },
    {
        "name":   "code-to-code",
        "input":  DATA_DIR / "full_code_to_code_constructs.jsonl",
        "output": RESULTS_DIR / "rq1_prevalence_code_to_code.json",
    },
]


def analyse_task(task: dict) -> dict:
    if not task["input"].exists():
        print(f"  ✗ Not found: {task['input']}")
        return {}

    print(f"\n{'='*60}")
    print(f"TASK: {task['name'].upper()}")
    print(f"{'='*60}")

    records = []
    with open(task["input"]) as f:
        for line in f:
            records.append(json.loads(line))

    total = len(records)
    print(f"  Total samples: {total}")

    # ── per-construct stats ────────────────────────────────────────────────
    construct_stats = {}
    for c in GO_CONSTRUCTS:
        counts = [r["go_constructs"].get(c, 0) for r in records]
        positive = sum(1 for x in counts if x > 0)
        construct_stats[c] = {
            "samples_with":     positive,
            "prevalence_pct":   round(positive / total * 100, 2),
            "total_occurrences": int(sum(counts)),
            "mean_per_sample":  round(float(np.mean(counts)), 3),
            "max_per_sample":   int(max(counts)),
        }
        print(f"  {c:<22} {positive:5d} samples ({positive/total*100:.1f}%)  "
              f"total={sum(counts)}")

    # ── profile distribution ───────────────────────────────────────────────
    profile_dist = Counter(r["construct_profile"] for r in records)
    length_dist  = Counter(r["length_bucket"]     for r in records)

    # ── co-occurrence matrix ───────────────────────────────────────────────
    cooccurrence = defaultdict(lambda: defaultdict(int))
    for r in records:
        present = [c for c in GO_CONSTRUCTS if r["go_constructs"].get(c, 0) > 0]
        for i, ci in enumerate(present):
            for cj in present[i+1:]:
                cooccurrence[ci][cj] += 1
                cooccurrence[cj][ci] += 1

    # ── mean constructs per sample ─────────────────────────────────────────
    constructs_per_sample = [
        sum(1 for c in GO_CONSTRUCTS if r["go_constructs"].get(c, 0) > 0)
        for r in records
    ]

    result = {
        "task":              task["name"],
        "total_samples":     total,
        "construct_stats":   construct_stats,
        "profile_distribution": {k: v for k, v in profile_dist.items()},
        "length_distribution":  {k: v for k, v in length_dist.items()},
        "cooccurrence":      {k: dict(v) for k, v in cooccurrence.items()},
        "mean_constructs_per_sample": round(float(np.mean(constructs_per_sample)), 3),
        "samples_with_zero_constructs": int(sum(1 for x in constructs_per_sample if x == 0)),
    }

    task["output"].parent.mkdir(parents=True, exist_ok=True)
    with open(task["output"], "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  ✓ Saved to {task['output'].name}")
    return result


def main():
    print("=" * 60)
    print("SCRIPT 5: RQ1 — CONSTRUCT PREVALENCE")
    print("=" * 60)

    for task in TASKS:
        analyse_task(task)

    print(f"\n{'='*60}")
    print("✓ DONE — next: python scripts/rq2_attention.py --model unixcoder --task code-to-text")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
