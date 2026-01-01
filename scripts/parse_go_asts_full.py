#!/usr/bin/env python3
"""
Script 3 (FULL): Parse ASTs and Extract Go Constructs - Full Dataset
Processes all 16,244 samples with batch processing for memory efficiency.
"""

import json
from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_go as tsgo
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import sys
import gc

# Configuration
CODE_TO_TEXT_DIR = Path("data/code-to-text")
CODE_TO_CODE_DIR = Path("data/code-to-code")

# Process in batches to manage memory
BATCH_SIZE = 500  # Process 500 samples at a time

# Go-specific constructs to track
GO_CONSTRUCTS = {
    'goroutines': 'go_statement',
    'channels': ['channel_type', 'send_statement', 'receive_statement'],
    'defer': 'defer_statement',
    'error_patterns': None,
    'interfaces': 'interface_type',
    'type_assertions': 'type_assertion_expression',
    'select_statements': 'select_statement',
    'context_usage': None,
}

class GoASTParser:
    """Parser for Go code using tree-sitter."""
    
    def __init__(self):
        """Initialize the parser."""
        print("Initializing Go AST parser...")
        self.GO_LANGUAGE = Language(tsgo.language())
        self.parser = Parser(self.GO_LANGUAGE)
        print("✓ Parser initialized")
    
    # [Include all the methods from parse_go_asts.py]
    # I'll keep the same implementation but won't repeat it here for brevity
    # The methods are: parse_code, extract_ast_structure, extract_go_constructs, etc.
    
    def parse_code(self, code: str) -> Any:
        """Parse Go code and return AST tree."""
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            return tree
        except Exception as e:
            return None
    
    def extract_ast_structure(self, tree) -> Dict:
        """Extract AST structure information."""
        if tree is None:
            return {}
        
        root = tree.root_node
        
        return {
            'root_type': root.type,
            'node_count': self._count_nodes(root),
            'depth': self._get_tree_depth(root),
            'leaf_nodes': self._get_leaf_nodes(root),
            'ast_tree': self._serialize_tree(root)
        }
    
    def _count_nodes(self, node) -> int:
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _get_tree_depth(self, node) -> int:
        if len(node.children) == 0:
            return 1
        return 1 + max(self._get_tree_depth(child) for child in node.children)
    
    def _get_leaf_nodes(self, node) -> List[Dict]:
        leaves = []
        
        def traverse(n):
            if len(n.children) == 0:
                leaves.append({
                    'type': n.type,
                    'text': n.text.decode('utf8') if n.text else '',
                    'start': n.start_point,
                    'end': n.end_point
                })
            else:
                for child in n.children:
                    traverse(child)
        
        traverse(node)
        return leaves
    
    def _serialize_tree(self, node, max_depth=10, current_depth=0) -> Dict:
        if current_depth >= max_depth:
            return {'type': node.type, 'truncated': True}
        
        result = {
            'type': node.type,
            'start': node.start_point,
            'end': node.end_point,
        }
        
        if len(node.children) == 0:
            result['text'] = node.text.decode('utf8') if node.text else ''
        else:
            result['children'] = [
                self._serialize_tree(child, max_depth, current_depth + 1)
                for child in node.children
            ]
        
        return result
    
    def extract_go_constructs(self, tree, code: str) -> Dict:
        """Extract Go-specific constructs from AST."""
        if tree is None:
            return {}
        
        constructs = defaultdict(list)
        root = tree.root_node
        
        # Extract different construct types
        self._find_goroutines(root, code, constructs)
        self._find_channels(root, code, constructs)
        self._find_defer(root, code, constructs)
        self._find_error_patterns(root, code, constructs)
        self._find_interfaces(root, code, constructs)
        self._find_type_assertions(root, code, constructs)
        self._find_select_statements(root, code, constructs)
        self._find_context_usage(root, code, constructs)
        
        result = dict(constructs)
        result['construct_counts'] = {k: len(v) for k, v in result.items()}
        
        return result
    
    def _find_by_type(self, node, node_type: str, results: List):
        if node.type == node_type:
            results.append(node)
        for child in node.children:
            self._find_by_type(child, node_type, results)
    
    def _find_goroutines(self, root, code: str, constructs: Dict):
        nodes = []
        self._find_by_type(root, 'go_statement', nodes)
        for node in nodes:
            constructs['goroutines'].append({
                'line': node.start_point[0] + 1,
                'code': code[node.start_byte:node.end_byte]
            })
    
    def _find_channels(self, root, code: str, constructs: Dict):
        chan_types = []
        send_stmts = []
        receive_ops = []
        
        self._find_by_type(root, 'channel_type', chan_types)
        self._find_by_type(root, 'send_statement', send_stmts)
        self._find_by_type(root, 'receive_statement', receive_ops)
        
        for node in chan_types + send_stmts + receive_ops:
            constructs['channels'].append({
                'type': node.type,
                'line': node.start_point[0] + 1,
                'code': code[node.start_byte:node.end_byte]
            })
    
    def _find_defer(self, root, code: str, constructs: Dict):
        nodes = []
        self._find_by_type(root, 'defer_statement', nodes)
        for node in nodes:
            constructs['defer'].append({
                'line': node.start_point[0] + 1,
                'code': code[node.start_byte:node.end_byte]
            })
    
    def _find_error_patterns(self, root, code: str, constructs: Dict):
        def check_error_pattern(node):
            if node.type == 'if_statement':
                condition_text = code[node.start_byte:node.end_byte]
                if 'err' in condition_text and '!= nil' in condition_text:
                    return True
            return False
        
        def traverse(node):
            if check_error_pattern(node):
                constructs['error_patterns'].append({
                    'type': 'if_err_nil',
                    'line': node.start_point[0] + 1,
                    'code': code[node.start_byte:node.end_byte][:100] + '...'
                })
            for child in node.children:
                traverse(child)
        
        traverse(root)
        
        error_returns = []
        self._find_by_type(root, 'type_identifier', error_returns)
        for node in error_returns:
            if code[node.start_byte:node.end_byte] == 'error':
                constructs['error_patterns'].append({
                    'type': 'error_return',
                    'line': node.start_point[0] + 1,
                    'code': 'error return type'
                })
    
    def _find_interfaces(self, root, code: str, constructs: Dict):
        nodes = []
        self._find_by_type(root, 'interface_type', nodes)
        for node in nodes:
            constructs['interfaces'].append({
                'line': node.start_point[0] + 1,
                'code': code[node.start_byte:node.end_byte][:100] + '...'
            })
    
    def _find_type_assertions(self, root, code: str, constructs: Dict):
        nodes = []
        self._find_by_type(root, 'type_assertion_expression', nodes)
        for node in nodes:
            constructs['type_assertions'].append({
                'line': node.start_point[0] + 1,
                'code': code[node.start_byte:node.end_byte]
            })
    
    def _find_select_statements(self, root, code: str, constructs: Dict):
        nodes = []
        self._find_by_type(root, 'select_statement', nodes)
        for node in nodes:
            constructs['select_statements'].append({
                'line': node.start_point[0] + 1,
                'code': code[node.start_byte:node.end_byte][:100] + '...'
            })
    
    def _find_context_usage(self, root, code: str, constructs: Dict):
        def check_context(node):
            if node.type in ['qualified_type', 'type_identifier']:
                text = code[node.start_byte:node.end_byte]
                if 'context.Context' in text or text == 'Context':
                    return True
            return False
        
        def traverse(node):
            if check_context(node):
                constructs['context_usage'].append({
                    'line': node.start_point[0] + 1,
                    'code': code[node.start_byte:node.end_byte]
                })
            for child in node.children:
                traverse(child)
        
        traverse(root)


def process_file_batched(input_file: Path, output_file: Path, parser: GoASTParser, task_type: str):
    """
    Process a JSONL file in batches for memory efficiency.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {input_file.name} (FULL DATASET, BATCHED)")
    print(f"Task: {task_type}")
    print(f"{'='*80}")
    
    # Count total samples first
    with open(input_file, 'r', encoding='utf-8') as f:
        total_samples = sum(1 for _ in f)
    
    print(f"  Total samples: {total_samples:,}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Number of batches: {(total_samples + BATCH_SIZE - 1) // BATCH_SIZE}")
    
    # Process in batches
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    construct_stats = Counter()
    
    # Open output file in append mode
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Process batches
        with open(input_file, 'r', encoding='utf-8') as in_f:
            batch = []
            batch_num = 0
            
            for i, line in enumerate(in_f):
                sample = json.loads(line)
                batch.append(sample)
                
                # Process batch when full
                if len(batch) >= BATCH_SIZE:
                    batch_num += 1
                    print(f"\n  Processing batch {batch_num} (samples {i+1-BATCH_SIZE} to {i+1})...")
                    
                    success, stats = process_batch(batch, parser, task_type, out_f)
                    success_count += success
                    construct_stats.update(stats)
                    
                    # Clear batch and force garbage collection
                    batch = []
                    gc.collect()
            
            # Process remaining samples
            if batch:
                batch_num += 1
                print(f"\n  Processing final batch {batch_num} ({len(batch)} samples)...")
                success, stats = process_batch(batch, parser, task_type, out_f)
                success_count += success
                construct_stats.update(stats)
    
    print(f"\n  ✓ Successfully parsed: {success_count}/{total_samples} code fields")
    
    # Print construct statistics
    print(f"\n  [Go Construct Statistics]")
    for construct, count in construct_stats.most_common():
        print(f"    {construct}: {count:,}")
    
    print(f"  ✓ Saved to: {output_file.name}")
    
    return total_samples, construct_stats


def process_batch(batch: List[Dict], parser: GoASTParser, task_type: str, out_file) -> Tuple[int, Counter]:
    """Process a batch of samples."""
    success_count = 0
    construct_stats = Counter()
    
    for sample in batch:
        # Determine which field(s) to parse
        fields_to_parse = []
        
        if task_type == 'code-to-text':
            fields_to_parse = [('query', 'code')]
        elif task_type == 'code-to-code':
            fields_to_parse = [('query', 'initial_segment'), ('target', 'completion')]
        
        # Parse each code field
        parsed_sample = sample.copy()
        parsed_sample['ast_info'] = {}
        parsed_sample['go_constructs'] = {}
        
        for field_name, ast_label in fields_to_parse:
            code = sample.get(field_name, '')
            
            if not code or not isinstance(code, str):
                continue
            
            # Parse AST
            tree = parser.parse_code(code)
            
            if tree:
                # Extract AST structure
                ast_info = parser.extract_ast_structure(tree)
                parsed_sample['ast_info'][ast_label] = ast_info
                
                # Extract Go constructs
                constructs = parser.extract_go_constructs(tree, code)
                parsed_sample['go_constructs'][ast_label] = constructs
                
                # Update statistics
                if ast_label in ['code', 'initial_segment']:
                    for construct, items in constructs.items():
                        if construct != 'construct_counts':
                            construct_stats[construct] += len(items)
                
                success_count += 1
        
        # Write to output file immediately
        json.dump(parsed_sample, out_file, ensure_ascii=False)
        out_file.write('\n')
    
    return success_count, construct_stats


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("Go AST PARSING & CONSTRUCT EXTRACTION (FULL DATASET)")
    print("=" * 80)
    print("\nProcessing ALL 16,244 samples with batch processing")
    print("Batch size: 500 samples per batch")
    print()
    
    # Initialize parser
    parser = GoASTParser()
    
    # Track overall statistics
    total_files = 0
    total_samples = 0
    all_construct_stats = Counter()
    
    # Files to process
    files_to_process = [
        {
            'name': 'Code-to-Text FULL',
            'input': CODE_TO_TEXT_DIR / "full_code_to_text.jsonl",
            'output': CODE_TO_TEXT_DIR / "full_code_to_text_with_asts.jsonl",
            'task_type': 'code-to-text'
        },
        {
            'name': 'Code-to-Code FULL',
            'input': CODE_TO_CODE_DIR / "full_code_to_code.jsonl",
            'output': CODE_TO_CODE_DIR / "full_code_to_code_with_asts.jsonl",
            'task_type': 'code-to-code'
        },
    ]
    
    # Process each file
    for file_info in files_to_process:
        if not file_info['input'].exists():
            print(f"\n⚠ Skipping {file_info['name']}: File not found")
            continue
        
        num_samples, construct_stats = process_file_batched(
            file_info['input'],
            file_info['output'],
            parser,
            file_info['task_type']
        )
        
        total_files += 1
        total_samples += num_samples
        all_construct_stats.update(construct_stats)
    
    # Final summary
    print("\n" + "=" * 80)
    print("✓ AST PARSING COMPLETE")
    print("=" * 80)
    
    print(f"\n[Overall Statistics]")
    print(f"  Files processed: {total_files}")
    print(f"  Total samples: {total_samples:,}")
    print()
    
    print(f"[Go Construct Distribution Across All Files]")
    for construct, count in all_construct_stats.most_common():
        print(f"  {construct}: {count:,}")
    print()
    
    print(f"[Output Files]")
    print(f"  Code-to-Text:")
    print(f"    {CODE_TO_TEXT_DIR}/full_code_to_text_with_asts.jsonl")
    print(f"  Code-to-Code:")
    print(f"    {CODE_TO_CODE_DIR}/full_code_to_code_with_asts.jsonl")
    print()
    
    print("Next steps:")
    print("  1. Run model feature extraction:")
    print("     python scripts/extract_model_outputs_full.py --model unixcoder")
    print("     python scripts/extract_model_outputs_full.py --model codebert")
    print()

if __name__ == "__main__":
    main()
