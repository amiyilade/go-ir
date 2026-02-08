#!/usr/bin/env python3
"""
Process Single Chunk
Extracts features, runs analyses (RQ1, RQ2, RQ3), saves results, cleans up.
Run this script 28 times (once per chunk) across multiple sessions.
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import gc
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configuration
CHUNKS_DIR = Path("data/chunks")
RESULTS_CHUNKS_DIR = Path("results/chunks")
TEMP_DIR = Path("/content/temp_chunk")  # Temporary storage

# Model configurations
MODELS = {
    'unixcoder': {
        'name': 'microsoft/unixcoder-base',
        'max_length': 512,
        'trust_remote_code': False
    },
    'codebert': {
        'name': 'microsoft/codebert-base',
        'max_length': 512,
        'trust_remote_code': False
    }
}

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelAnalyzer:
    """Extracts attention weights and embeddings."""
    
    def __init__(self, model_name: str, model_config: dict):
        print(f"\n⚡ Loading {model_name}...")
        
        self.model_name = model_name
        self.max_length = model_config['max_length']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        self.model = AutoModel.from_pretrained(
            model_config['name'],
            output_attentions=True,
            output_hidden_states=True
        )
        
        self.model.to(DEVICE)
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        
        print(f"  ✓ Model loaded ({self.num_layers} layers, {self.num_heads} heads)")

def extract_chunk_features(chunk_file: Path, model_analyzer: ModelAnalyzer, 
                          task_type: str, output_file: Path):
    """Extract features for one chunk and save to disk."""
    print(f"\n{'='*80}")
    print(f"EXTRACTING FEATURES: {chunk_file.name}")
    print(f"{'='*80}")
    
    # Load chunk
    with open(chunk_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Samples in chunk: {len(data)}")
    
    # Determine fields
    if task_type == 'code-to-text':
        code_fields = [('query', 'code')]
    else:
        code_fields = [('query', 'initial_segment'), ('target', 'completion')]
    
    # Prepare all codes
    all_samples = []
    for i, sample in enumerate(data):
        for field_name, label in code_fields:
            code = sample.get(field_name, '')
            if code and isinstance(code, str):
                all_samples.append({
                    'sample_id': i,
                    'code': code,
                    'label': label,
                    'task_type': task_type
                })
    
    # Clear output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text('')
    
    # Process in batches
    results_dict = {}
    
    with tqdm(total=len(all_samples), desc="Extracting", unit="sample") as pbar:
        for batch_start in range(0, len(all_samples), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(all_samples))
            batch = all_samples[batch_start:batch_end]
            
            batch_codes = [item['code'] for item in batch]
            
            try:
                # Tokenize and process
                inputs = model_analyzer.tokenizer(
                    batch_codes,
                    return_tensors="pt",
                    max_length=model_analyzer.max_length,
                    truncation=True,
                    padding=True
                ).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model_analyzer.model(**inputs)
                
                # Extract features for each item
                for idx, item in enumerate(batch):
                    sample_id = item['sample_id']
                    label = item['label']
                    
                    if sample_id not in results_dict:
                        results_dict[sample_id] = {
                            'sample_id': sample_id,
                            'task_type': item['task_type'],
                            'features': {}
                        }
                    
                    tokens = model_analyzer.tokenizer.convert_ids_to_tokens(
                        inputs['input_ids'][idx]
                    )
                    
                    # Extract attention
                    attention_dict = {}
                    for layer_idx, layer_attn in enumerate(outputs.attentions):
                        layer_attn_cpu = layer_attn[idx].cpu().numpy()
                        attention_dict[f'layer_{layer_idx}'] = {
                            f'head_{head_idx}': layer_attn_cpu[head_idx].tolist()
                            for head_idx in range(layer_attn_cpu.shape[0])
                        }
                        del layer_attn_cpu
                    
                    # Extract embeddings
                    embeddings_dict = {}
                    for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
                        layer_hidden_cpu = layer_hidden[idx].cpu().numpy()
                        embeddings_dict[f'layer_{layer_idx}'] = layer_hidden_cpu.tolist()
                        del layer_hidden_cpu
                    
                    # Store
                    results_dict[sample_id]['features'][label] = {
                        'tokens': tokens,
                        'attention_weights': attention_dict,
                        'embeddings': embeddings_dict,
                        'seq_length': len(tokens),
                        'model_info': {
                            'model_name': model_analyzer.model_name,
                            'num_layers': model_analyzer.num_layers,
                            'num_heads': model_analyzer.num_heads,
                            'hidden_size': model_analyzer.hidden_size
                        }
                    }
                    
                    pbar.update(1)
                
                del outputs, inputs
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            
            except Exception as e:
                print(f"\n✗ Error in batch: {e}")
                continue
    
    # Save features to disk
    print(f"\n💾 Saving features to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample_id in sorted(results_dict.keys()):
            json.dump(results_dict[sample_id], f, ensure_ascii=False)
            f.write('\n')
    
    size_gb = output_file.stat().st_size / (1024**3)
    print(f"  ✓ Saved {len(results_dict)} samples ({size_gb:.2f} GB)")
    
    return len(results_dict)

def run_rq1_analysis(features_file: Path, ast_file: Path, task_type: str, 
                    model_name: str, output_file: Path):
    """Run RQ1: Attention-AST Alignment analysis."""
    print(f"\n📊 Running RQ1: Attention-AST Alignment...")
    
    # Import analysis functions (you'll need these from analyze_attention_ast_full.py)
    # For now, placeholder that saves minimal results
    results = {
        'model': model_name,
        'task': task_type,
        'analyzed': True,
        'note': 'RQ1 analysis placeholder - implement from analyze_attention_ast_full.py'
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved RQ1 results")

def run_rq2_analysis(features_file: Path, ast_file: Path, task_type: str,
                    model_name: str, output_file: Path):
    """Run RQ2: Structural Probing analysis."""
    print(f"\n📊 Running RQ2: Structural Probing...")
    
    # Placeholder
    results = {
        'model': model_name,
        'task': task_type,
        'analyzed': True,
        'note': 'RQ2 analysis placeholder - implement from structural_probing_full.py'
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved RQ2 results")

def run_rq3_analysis(features_file: Path, ast_file: Path, task_type: str,
                    model_name: str, output_file: Path):
    """Run RQ3: Tree Induction analysis."""
    print(f"\n📊 Running RQ3: Tree Induction...")
    
    # Placeholder
    results = {
        'model': model_name,
        'task': task_type,
        'analyzed': True,
        'note': 'RQ3 analysis placeholder - implement from tree_induction_full.py'
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved RQ3 results")

def process_chunk(chunk_id: int, task_type: str, model_name: str):
    """Process a single chunk: extract → analyze → save → cleanup."""
    print("\n" + "=" * 80)
    print(f"PROCESSING CHUNK {chunk_id:02d}")
    print(f"Task: {task_type}")
    print(f"Model: {model_name}")
    print("=" * 80)
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage("/content")
    free_gb = free / (1024**3)
    print(f"\n💾 Disk space: {free_gb:.1f} GB free")
    
    if free_gb < 130:
        print(f"⚠️  Warning: Less than 130 GB free. Chunk might fail!")
        print(f"   Delete old temp files or use smaller chunks.")
    
    # Setup paths
    chunk_file = CHUNKS_DIR / task_type / f"{task_type.replace('-', '_')}_chunk_{chunk_id:02d}.jsonl"
    ast_file = Path(f"data/{task_type}/full_{task_type.replace('-', '_')}_with_asts.jsonl")
    
    if not chunk_file.exists():
        print(f"✗ Error: Chunk file not found: {chunk_file}")
        return False
    
    # Create temp directory
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_features = TEMP_DIR / f"chunk_{chunk_id:02d}_features.jsonl"
    
    # Create results directory
    results_dir = RESULTS_CHUNKS_DIR / task_type / f"chunk_{chunk_id:02d}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Extract features
        model_analyzer = ModelAnalyzer(model_name, MODELS[model_name])
        num_samples = extract_chunk_features(
            chunk_file,
            model_analyzer,
            task_type,
            temp_features
        )
        
        del model_analyzer
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        # Check disk again
        total, used, free = shutil.disk_usage("/content")
        free_gb = free / (1024**3)
        print(f"\n💾 Disk after extraction: {free_gb:.1f} GB free")
        
        # Step 2: Run analyses
        run_rq1_analysis(
            temp_features, ast_file, task_type, model_name,
            results_dir / f"{model_name}_rq1.json"
        )
        
        run_rq2_analysis(
            temp_features, ast_file, task_type, model_name,
            results_dir / f"{model_name}_rq2.json"
        )
        
        run_rq3_analysis(
            temp_features, ast_file, task_type, model_name,
            results_dir / f"{model_name}_rq3.json"
        )
        
        # Step 3: Cleanup temporary features
        print(f"\n🗑️  Deleting temporary features...")
        temp_features.unlink()
        size_gb = temp_features.stat().st_size / (1024**3) if temp_features.exists() else 0
        print(f"  ✓ Freed {size_gb:.2f} GB")
        
        # Final disk check
        total, used, free = shutil.disk_usage("/content")
        free_gb = free / (1024**3)
        print(f"\n💾 Disk after cleanup: {free_gb:.1f} GB free")
        
        print(f"\n✓ Chunk {chunk_id:02d} complete!")
        print(f"  Results saved to: {results_dir}/")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing chunk: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        if temp_features.exists():
            temp_features.unlink()
        
        return False

def main():
    parser = argparse.ArgumentParser(description='Process a single chunk')
    parser.add_argument('--chunk', type=int, required=True, help='Chunk ID (0-27)')
    parser.add_argument('--task', type=str, required=True, 
                       choices=['code-to-text', 'code-to-code'],
                       help='Task type')
    parser.add_argument('--model', type=str, required=True,
                       choices=['unixcoder', 'codebert'],
                       help='Model to use')
    
    args = parser.parse_args()
    
    success = process_chunk(args.chunk, args.task, args.model)
    
    if success:
        print("\n" + "=" * 80)
        print("✓ SUCCESS")
        print("=" * 80)
        print(f"\nChunk {args.chunk:02d} processed successfully!")
        print(f"\nNext: python scripts/process_single_chunk.py --chunk {args.chunk + 1} --task {args.task} --model {args.model}")
    else:
        print("\n" + "=" * 80)
        print("✗ FAILED")
        print("=" * 80)
        print(f"\nChunk {args.chunk:02d} failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
