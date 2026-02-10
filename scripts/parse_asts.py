#!/usr/bin/env python3
"""
Script 3: Parse ASTs
Parses full AST trees for the 2k stratified sample only.

Input:  data/stratified_2k_code_to_text.jsonl
        data/stratified_2k_code_to_code.jsonl
Output: data/stratified_2k_code_to_text_with_asts.jsonl
        data/stratified_2k_code_to_code_with_asts.jsonl

Usage:
    python scripts/parse_asts.py
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from tree_sitter import Language, Parser
import tree_sitter_go as tsgo

DATA_DIR = Path("data")

TASKS = [
    {
        "name":       "code-to-text",
        "input":      DATA_DIR / "stratified_2k_code_to_text.jsonl",
        "output":     DATA_DIR / "stratified_2k_code_to_text_with_asts.jsonl",
        "code_field": "code",
    },
    {
        "name":       "code-to-code",
        "input":      DATA_DIR / "stratified_2k_code_to_code.jsonl",
        "output":     DATA_DIR / "stratified_2k_code_to_code_with_asts.jsonl",
        "code_field": "code",
    },
]

MAX_AST_DEPTH = 12   # Prevents huge serialised trees


class ASTParser:
    def __init__(self):
        self.parser = Parser(Language(tsgo.language()))

    def parse(self, code: str) -> Dict:
        """Return full AST info needed by RQ scripts."""
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
        except Exception as e:
            return {"error": str(e)}

        root = tree.root_node
        return {
            "root_type":   root.type,
            "node_count":  self._count(root),
            "depth":       self._depth(root),
            "leaf_nodes":  self._leaves(root),
            "ast_tree":    self._serialize(root),
        }

    # ── helpers ────────────────────────────────────────────────────────────

    def _count(self, node) -> int:
        return 1 + sum(self._count(c) for c in node.children)

    def _depth(self, node) -> int:
        if not node.children:
            return 1
        return 1 + max(self._depth(c) for c in node.children)

    def _leaves(self, node) -> List[Dict]:
        if not node.children:
            return [{
                "type":  node.type,
                "text":  node.text.decode("utf8", errors="replace") if node.text else "",
                "start": list(node.start_point),
                "end":   list(node.end_point),
            }]
        leaves = []
        for c in node.children:
            leaves.extend(self._leaves(c))
        return leaves

    def _serialize(self, node, depth=0) -> Dict:
        if depth >= MAX_AST_DEPTH:
            return {"type": node.type, "truncated": True}

        result: Dict[str, Any] = {
            "type":  node.type,
            "start": list(node.start_point),
            "end":   list(node.end_point),
        }
        if not node.children:
            result["text"] = node.text.decode("utf8", errors="replace") if node.text else ""
        else:
            result["children"] = [self._serialize(c, depth + 1) for c in node.children]
        return result


def process_task(task: dict, parser: ASTParser):
    if not task["input"].exists():
        print(f"  ✗ Not found: {task['input']}")
        return

    print(f"\n{'='*60}")
    print(f"TASK: {task['name'].upper()}")
    print(f"{'='*60}")

    errors = 0
    with open(task["input"])  as fin, \
         open(task["output"], "w") as fout:

        lines = fin.readlines()
        for line in tqdm(lines, desc="  Parsing ASTs"):
            record = json.loads(line)
            code   = record.get(task["code_field"], "")

            if code:
                ast_info = parser.parse(code)
            else:
                ast_info = {"error": "empty code"}
                errors  += 1

            record["ast_info"] = ast_info
            fout.write(json.dumps(record) + "\n")

    print(f"  ✓ Parsed {len(lines)} samples  ({errors} errors)")
    print(f"  ✓ Saved to {task['output'].name}")


def main():
    print("=" * 60)
    print("SCRIPT 3: PARSE ASTs")
    print("=" * 60)

    parser = ASTParser()
    print("✓ tree-sitter Go parser initialised")

    for task in TASKS:
        process_task(task, parser)

    print(f"\n{'='*60}")
    print("✓ DONE — next: python scripts/extract_features.py --model unixcoder --task code-to-text")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
