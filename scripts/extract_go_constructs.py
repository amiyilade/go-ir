#!/usr/bin/env python3
"""
Extract Go Constructs
Extracts Go-specific constructs from dataset using tree-sitter.

Input:  data/full_code_to_text.jsonl, data/full_code_to_code.jsonl
Output: data/full_code_to_text_constructs.jsonl, data/full_code_to_code_constructs.jsonl

Disclaimer: ChatGPT and Copilot were used to edit and enhance this script for better readability, error handling, and user feedback.
The author (me) implemented the core logic.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List
from tqdm import tqdm
from tree_sitter import Language, Parser
import tree_sitter_go as tsgo

DATA_DIR = Path("data")

TASKS = [
    {
        "name":       "code-to-text",
        "input":      DATA_DIR / "full_code_to_text.jsonl",
        "output":     DATA_DIR / "full_code_to_text_constructs.jsonl",
        "code_field": "query",      
        "code_label": "code",        
    },
    {
        "name":       "code-to-code",
        "input":      DATA_DIR / "full_code_to_code.jsonl",
        "output":     DATA_DIR / "full_code_to_code_constructs.jsonl",
        "code_field": "query",       
        "code_label": "initial_segment",
    },
]

GO_CONSTRUCTS = [
    "goroutines", "channels", "defer", "error_patterns",
    "select_statements", "interfaces", "type_assertions", "context_usage",
]

class GoConstructExtractor:
    def __init__(self):
        self.parser = Parser(Language(tsgo.language()))

    def extract(self, code: str) -> Dict[str, int]:
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
        except Exception:
            return {c: 0 for c in GO_CONSTRUCTS}

        counts = defaultdict(int)
        self._walk(tree.root_node, code, counts)
        return {c: counts[c] for c in GO_CONSTRUCTS}

    def _walk(self, node, code: str, counts: Dict):
        t = node.type

        if t == "go_statement":
            counts["goroutines"] += 1
        elif t in ("channel_type", "send_statement", "receive_statement"):
            counts["channels"] += 1
        elif t == "defer_statement":
            counts["defer"] += 1
        elif t == "interface_type":
            counts["interfaces"] += 1
        elif t == "type_assertion_expression":
            counts["type_assertions"] += 1
        elif t == "select_statement":
            counts["select_statements"] += 1
        elif t == "if_statement":
            snippet = code[node.start_byte:node.end_byte]
            if "err" in snippet and "!= nil" in snippet:
                counts["error_patterns"] += 1
        elif t in ("qualified_type", "type_identifier"):
            text = code[node.start_byte:node.end_byte]
            if text in ("context.Context", "Context"):
                counts["context_usage"] += 1

        for child in node.children:
            self._walk(child, code, counts)



def code_length_bucket(code: str) -> str:
    n = len(code.split())
    if n < 50:
        return "short"
    elif n < 150:
        return "medium"
    else:
        return "long"


def construct_profile(counts: Dict[str, int]) -> str:
    has_conc  = counts.get("goroutines", 0) + counts.get("channels", 0) \
                + counts.get("select_statements", 0) > 0
    has_error = counts.get("error_patterns", 0) > 0
    has_other = sum(counts.get(c, 0) for c in
                    ("defer", "interfaces", "type_assertions", "context_usage")) > 0

    if has_conc:
        return "concurrency"
    elif has_error and has_other:
        return "error_plus"
    elif has_error:
        return "error_only"
    else:
        return "none"



def process_task(task: dict, extractor: GoConstructExtractor):
    if not task["input"].exists():
        print(f"  ✗ Input not found: {task['input']}")
        return

    print(f"\n{'='*60}")
    print(f"TASK: {task['name'].upper()}")
    print(f"{'='*60}")

    profile_counter = Counter()
    bucket_counter  = Counter()
    construct_total = Counter()

    with open(task["input"])  as fin, \
         open(task["output"], "w") as fout:

        lines = fin.readlines()
        for i, line in enumerate(tqdm(lines, desc="  Extracting")):
            record = json.loads(line)
            code   = record.get(task["code_field"], "")

            counts  = extractor.extract(code) if code else {c: 0 for c in GO_CONSTRUCTS}
            profile = construct_profile(counts)
            bucket  = code_length_bucket(code)

            out = {
                "original_index":  i,
                "task_type":       task["name"],
                "code_label":      task["code_label"],
                "code":            code,
                "go_constructs":   counts,
                "construct_profile": profile,
                "length_bucket":   bucket,
            }
            fout.write(json.dumps(out) + "\n")

            profile_counter[profile] += 1
            bucket_counter[bucket]   += 1
            for c, n in counts.items():
                if n > 0:
                    construct_total[c] += 1

    total = len(lines)
    print(f"\n  Construct profile distribution ({total} samples):")
    for p, n in profile_counter.most_common():
        print(f"    {p:<15} {n:5d}  ({n/total*100:.1f}%)")

    print(f"\n  Code length distribution:")
    for b in ("short", "medium", "long"):
        n = bucket_counter[b]
        print(f"    {b:<8} {n:5d}  ({n/total*100:.1f}%)")

    print(f"\n  Samples containing each construct:")
    for c in GO_CONSTRUCTS:
        n = construct_total[c]
        print(f"    {c:<20} {n:5d}  ({n/total*100:.1f}%)")

    print(f"\n  ✓ Saved to {task['output'].name}")


def main():
    print("=" * 60)
    print("EXTRACT GO CONSTRUCTS")
    print("=" * 60)

    extractor = GoConstructExtractor()
    print("✓ tree-sitter Go parser initialised")

    for task in TASKS:
        process_task(task, extractor)

    print(f"\n{'='*60}")
    print("✓ DONE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
