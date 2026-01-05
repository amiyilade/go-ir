#!/usr/bin/env python3
"""
Script 4 (EMERGENCY STREAMING FIX): Extract Model Outputs with Per-Batch Saving
Saves after EVERY batch to prevent RAM overflow. No accumulation!
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List
from tqdm import tqdm
import sys
import argparse
import gc

# Configuration
CODE_TO_TEXT_DIR = Path("data/code-to-text")
CODE_TO_CODE_DIR = Path("data/code-to-code")
RESULTS_DIR = Path("/content/results")
MODEL_OUTPUT_DIR = RESULTS_DIR / "model_outputs"

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

# Batch processing settings
BATCH_SIZE = 16  # Reduced back to 16 for safety
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelAnalyzer:
    """Extracts attention weights and embeddings from code models."""
    
    def __init__(self, model_name: str, model_config: Dict):
        """Initialize model and tokenizer."""
        print(f"\nInitializing {model_name}...")
        
        self.model_name = model_name
        self.max_length = model_config['max_length']
        self.trust_remote_code = model_config.get('trust_remote_code', False)
        
        print(f"  Loading tokenizer from {model_config['name']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['name'],
            trust_remote_code=self.trust_remote_code
        )
        
        print(f"  Loading model from {model_config['name']}...")
        self.model = AutoModel.from_pretrained(
            model_config['name'],
            output_attentions=True,
            output_hidden_states=True,
            trust_remote_code=self.trust_remote_code
        )
        
        self.model.to(DEVICE)
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        
        print(f"  ✓ Model loaded on {DEVICE}")
        print(f"    Layers: {self.num_layers}, Heads: {self.num_heads}, Hidden: {self.hidden_size}")

def aggressive_cleanup():
    """Aggressively free memory."""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def process_batch_and_save(batch_samples: List[Dict], model_analyzer: ModelAnalyzer, 
                           output_file: Path, mode: str = 'a'):
    """
    Process one batch and immediately save to disk.
    Returns number of samples successfully processed.
    """
    # Extract codes from batch
    batch_codes = []
    batch_metadata = []
    
    for sample in batch_samples:
        for field_name, label in sample['fields']:
            code = sample['data'].get(field_name, '')
            if code and isinstance(code, str):
                batch_codes.append(code)
                batch_metadata.append({
                    'sample_id': sample['sample_id'],
                    'label': label,
                    'task_type': sample['task_type']
                })
    
    if not batch_codes:
        return 0
    
    results_dict = {}
    
    try:
        # Tokenize batch
        inputs = model_analyzer.tokenizer(
            batch_codes,
            return_tensors="pt",
            max_length=model_analyzer.max_length,
            truncation=True,
            padding=True
        ).to(DEVICE)
        
        # Forward pass
        with torch.no_grad():
            outputs = model_analyzer.model(**inputs)
        
        # Process each item
        for idx, metadata in enumerate(batch_metadata):
            sample_id = metadata['sample_id']
            label = metadata['label']
            
            if sample_id not in results_dict:
                results_dict[sample_id] = {
                    'sample_id': sample_id,
                    'task_type': metadata['task_type'],
                    'features': {}
                }
            
            # Extract tokens
            tokens = model_analyzer.tokenizer.convert_ids_to_tokens(
                inputs['input_ids'][idx]
            )
            
            # Extract attention - MOVE TO CPU IMMEDIATELY
            attention_dict = {}
            for layer_idx, layer_attn in enumerate(outputs.attentions):
                layer_attn_cpu = layer_attn[idx].cpu().numpy()
                attention_dict[f'layer_{layer_idx}'] = {
                    f'head_{head_idx}': layer_attn_cpu[head_idx].tolist()
                    for head_idx in range(layer_attn_cpu.shape[0])
                }
                del layer_attn_cpu  # Explicit cleanup
            
            # Extract embeddings - MOVE TO CPU IMMEDIATELY
            embeddings_dict = {}
            for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
                layer_hidden_cpu = layer_hidden[idx].cpu().numpy()
                embeddings_dict[f'layer_{layer_idx}'] = layer_hidden_cpu.tolist()
                del layer_hidden_cpu  # Explicit cleanup
            
            # Store features
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
        
        # Delete outputs immediately
        del outputs
        del inputs
        aggressive_cleanup()
        
        # Save to disk IMMEDIATELY
        with open(output_file, mode, encoding='utf-8') as f:
            for sample_id in sorted(results_dict.keys()):
                json.dump(results_dict[sample_id], f, ensure_ascii=False)
                f.write('\n')
        
        # Free memory
        num_saved = len(results_dict)
        del results_dict
        del batch_codes
        del batch_metadata
        aggressive_cleanup()
        
        return num_saved
        
    except Exception as e:
        print(f"\n✗ Error processing batch: {e}")
        # Cleanup on error
        del results_dict
        aggressive_cleanup()
        return 0

def process_dataset_streaming(dataset_path: Path, model_analyzer: ModelAnalyzer, 
                              output_dir: Path, dataset_name: str, task_type: str):
    """
    Process dataset in TRUE STREAMING fashion.
    Reads data in batches, processes immediately, saves immediately.
    NEVER accumulates more than one batch in RAM!
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING: {dataset_name}")
    print(f"Task: {task_type}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Save mode: STREAMING (saves after EVERY batch!)")
    print(f"{'='*80}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear output file
    output_file = output_dir / f"{dataset_name}_{model_analyzer.model_name}_features.jsonl"
    output_file.write_text('')
    
    # Count total samples first (lightweight)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        total_samples = sum(1 for _ in f)
    
    print(f"Total samples: {total_samples}")
    
    # Determine fields to process
    if task_type == 'code-to-text':
        code_fields = [('query', 'code')]
    else:
        code_fields = [('query', 'initial_segment'), ('target', 'completion')]
    
    # Stream through data in batches
    total_processed = 0
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        with tqdm(total=total_samples, desc="Processing", unit="sample") as pbar:
            
            sample_id = 0
            batch_buffer = []
            
            for line in f:
                # Parse one sample at a time
                sample_data = json.loads(line)
                
                # Add to batch buffer
                batch_buffer.append({
                    'sample_id': sample_id,
                    'data': sample_data,
                    'fields': code_fields,
                    'task_type': task_type
                })
                
                sample_id += 1
                
                # Process when batch is full
                if len(batch_buffer) >= BATCH_SIZE:
                    num_saved = process_batch_and_save(
                        batch_buffer,
                        model_analyzer,
                        output_file,
                        mode='a'
                    )
                    
                    total_processed += num_saved
                    pbar.update(len(batch_buffer))
                    
                    # Show RAM usage
                    import psutil
                    ram_gb = psutil.virtual_memory().used / 1e9
                    pbar.set_postfix({'saved': total_processed, 'RAM': f'{ram_gb:.1f}GB'})
                    
                    # Clear batch buffer
                    batch_buffer.clear()
                    aggressive_cleanup()
            
            # Process remaining samples
            if batch_buffer:
                num_saved = process_batch_and_save(
                    batch_buffer,
                    model_analyzer,
                    output_file,
                    mode='a'
                )
                total_processed += num_saved
                pbar.update(len(batch_buffer))
    
    print(f"\n✓ Saved {total_processed} features to {output_file.name}")
    
    # Verify
    with open(output_file, 'r') as f:
        actual_count = sum(1 for _ in f)
    print(f"  Verification: {actual_count} samples in file")
    
    # Save summary
    summary = {
        'model': model_analyzer.model_name,
        'dataset': dataset_name,
        'task_type': task_type,
        'total_samples': total_samples,
        'successful_extractions': actual_count,
        'model_config': {
            'num_layers': model_analyzer.num_layers,
            'num_heads': model_analyzer.num_heads,
            'hidden_size': model_analyzer.hidden_size
        }
    }
    
    summary_file = output_dir / f"{dataset_name}_{model_analyzer.model_name}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary to {summary_file.name}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Extract model features (STREAMING)')
    parser.add_argument('--model', type=str, choices=['unixcoder', 'codebert', 'both'],
                       default='both', help='Which model to run')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("MODEL FEATURE EXTRACTION (EMERGENCY STREAMING FIX)")
    print("=" * 80)
    print(f"\n⚡ Batch size: {BATCH_SIZE}")
    print(f"⚡ Device: {DEVICE}")
    print(f"💾 Save mode: STREAMING (after EVERY batch!)")
    print(f"⚡ Expected RAM: ~40-50 GB (STABLE)")
    print(f"⚡ NO accumulation - processes one batch at a time!")
    
    models_to_run = ['unixcoder', 'codebert'] if args.model == 'both' else [args.model]
    
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = [
        {
            'name': 'code_to_text_full',
            'path': CODE_TO_TEXT_DIR / "full_code_to_text_with_asts.jsonl",
            'task_type': 'code-to-text'
        },
        {
            'name': 'code_to_code_full',
            'path': CODE_TO_CODE_DIR / "full_code_to_code_with_asts.jsonl",
            'task_type': 'code-to-code'
        }
    ]
    
    # Verify files exist
    missing = [d for d in datasets if not d['path'].exists()]
    if missing:
        print("\n✗ Missing dataset files:")
        for d in missing:
            print(f"  {d['path']}")
        print("\nRun parse_go_asts_full.py first.")
        sys.exit(1)
    
    for model_key in models_to_run:
        print("\n" + "=" * 80)
        print(f"MODEL: {model_key.upper()}")
        print("=" * 80)
        
        model_output_dir = MODEL_OUTPUT_DIR / model_key
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            analyzer = ModelAnalyzer(model_key, MODELS[model_key])
        except Exception as e:
            print(f"\n✗ Error loading {model_key}: {e}")
            continue
        
        for dataset in datasets:
            try:
                process_dataset_streaming(
                    dataset['path'],
                    analyzer,
                    model_output_dir,
                    dataset['name'],
                    dataset['task_type']
                )
            except Exception as e:
                print(f"\n✗ Error processing {dataset['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Cleanup
        del analyzer
        aggressive_cleanup()
        
        print(f"\n✓ Completed {model_key}")
    
    print("\n" + "=" * 80)
    print("✓ FEATURE EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved in: {MODEL_OUTPUT_DIR}/")
    print("\nExpected runtime with A100:")
    print("  • UniXcoder: ~4-5 hours")
    print("  • CodeBERT: ~4-5 hours")
    print("  • Total: ~8-10 hours")
    print("\n")

if __name__ == "__main__":
    main()
