#!/usr/bin/env python3
"""
Automated converter: Updates analysis scripts from JSON to HDF5 format.
Run this script to automatically update all 4 analysis scripts.

Usage:
    python convert_to_hdf5.py /path/to/your/go-eval/scripts/
"""

import sys
import re
from pathlib import Path

# HDF5 load_data function template
HDF5_LOAD_DATA = '''    def load_data(self, features_file: Path, ast_file: Path, task_type: str) -> List[Dict]:
        """Load features from HDF5 and corresponding AST data."""
        print(f"\\nLoading data...")
        print(f"  Features: {features_file.name}")
        print(f"  AST data: {ast_file.name}")
        print(f"  Task type: {task_type}")
        
        # Load AST data (still JSON)
        with open(ast_file, 'r', encoding='utf-8') as f:
            ast_data = [json.loads(line) for line in f]
        
        # Determine code label
        code_label = 'code' if task_type == 'code-to-text' else 'initial_segment'
        
        # Load features from HDF5
        merged = []
        with h5py.File(features_file, 'r') as h5f:
            total_samples = h5f['metadata'].attrs['total_samples']
            
            for sample_id in range(total_samples):
                sample_key = f'sample_{sample_id}'
                if sample_key not in h5f or code_label not in h5f[sample_key]:
                    continue
                
                label_grp = h5f[sample_key][code_label]
                
                # Read attention (already numpy arrays!)
                attention_weights = {}
                attn_grp = label_grp['attention']
                for layer_key in attn_grp.keys():
                    attention_weights[layer_key] = {}
                    for head_key in attn_grp[layer_key].keys():
                        attention_weights[layer_key][head_key] = attn_grp[layer_key][head_key][:]
                
                # Read embeddings
                embeddings = {}
                emb_grp = label_grp['embeddings']
                for layer_key in emb_grp.keys():
                    embeddings[layer_key] = emb_grp[layer_key][:]
                
                # Merge with AST
                if sample_id < len(ast_data):
                    merged.append({
                        'sample_id': sample_id,
                        'task_type': task_type,
                        'code_label': code_label,
                        'features': {
                            'attention_weights': attention_weights,
                            'embeddings': embeddings
                        },
                        'ast': ast_data[sample_id].get('ast_info', {}).get(code_label),
                        'go_constructs': ast_data[sample_id].get('go_constructs', {}).get(code_label)
                    })
        
        print(f"  ✓ Loaded {len(merged)} samples")
        return merged'''

def update_script(script_path: Path):
    """Update a single analysis script to use HDF5."""
    print(f"\nUpdating {script_path.name}...")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # 1. Add h5py import
    if 'import h5py' not in content:
        content = content.replace(
            'import json\nimport numpy as np',
            'import json\nimport numpy as np\nimport h5py'
        )
        print("  ✓ Added h5py import")
    
    # 2. Add FEATURES_DIR
    if 'FEATURES_DIR' not in content:
        content = content.replace(
            'RESULTS_DIR = Path("results")',
            'RESULTS_DIR = Path("results")\nFEATURES_DIR = RESULTS_DIR / "features"'
        )
        print("  ✓ Added FEATURES_DIR")
    
    # 3. Replace load_data function
    # Find the load_data function and replace it
    pattern = r'    def load_data\(self, features_file.*?return merged'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, HDF5_LOAD_DATA, content, flags=re.DOTALL)
        print("  ✓ Updated load_data() function")
    
    # 4. Update feature file paths in main()
    content = content.replace(
        'RESULTS_DIR / "model_outputs/unixcoder/code_to_text_sample_unixcoder_features.jsonl"',
        'FEATURES_DIR / "sample_code_to_text_unixcoder.h5"'
    )
    content = content.replace(
        'RESULTS_DIR / "model_outputs/unixcoder/code_to_code_sample_unixcoder_features.jsonl"',
        'FEATURES_DIR / "sample_code_to_code_unixcoder.h5"'
    )
    print("  ✓ Updated file paths")
    
    # Save backup
    backup_path = script_path.with_suffix('.py.json_backup')
    with open(backup_path, 'w') as f:
        with open(script_path, 'r') as orig:
            f.write(orig.read())
    print(f"  ✓ Saved backup to {backup_path.name}")
    
    # Write updated content
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"  ✅ {script_path.name} updated successfully!")

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_hdf5.py /path/to/scripts/")
        print("\nThis will update:")
        print("  - analyze_attention_ast.py")
        print("  - structural_probing.py")
        print("  - tree_induction.py")
        print("  - go_constructs.py")
        sys.exit(1)
    
    scripts_dir = Path(sys.argv[1])
    
    if not scripts_dir.exists():
        print(f"Error: Directory not found: {scripts_dir}")
        sys.exit(1)
    
    scripts_to_update = [
        'analyze_attention_ast.py',
        'structural_probing.py',
        'tree_induction.py',
        'go_constructs.py'
    ]
    
    print("="*80)
    print("HDF5 CONVERSION TOOL")
    print("="*80)
    print(f"\nScripts directory: {scripts_dir}")
    print(f"\nWill update {len(scripts_to_update)} scripts:")
    for script in scripts_to_update:
        print(f"  - {script}")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    for script_name in scripts_to_update:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            print(f"\n⚠ Skipping {script_name} (not found)")
            continue
        
        try:
            update_script(script_path)
        except Exception as e:
            print(f"\n❌ Error updating {script_name}: {e}")
            continue
    
    print("\n" + "="*80)
    print("✅ CONVERSION COMPLETE!")
    print("="*80)
    print("\nBackup files created with .json_backup extension")
    print("If anything went wrong, restore from backups.")
    print("\nNext steps:")
    print("  1. Test extraction: python extract_features_hdf5.py --model unixcoder --task code-to-text --sample")
    print("  2. Test analysis: python analyze_attention_ast.py")
    print()

if __name__ == "__main__":
    main()
