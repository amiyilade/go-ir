#!/usr/bin/env python3
"""
Script 4 (FULL - FIXED): Extract Model Outputs WITH PROPER BATCHING
Extracts attention weights and contextual embeddings from UniXcoder and CodeBERT models.

FIXED: Now properly batches samples for 3-4x speedup!
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
import argparse
import gc

# Configuration
CODE_TO_TEXT_DIR = Path("data/code-to-text")
CODE_TO_CODE_DIR = Path("data/code-to-code")
RESULTS_DIR = Path("results")
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

# CRITICAL: Proper batch size
BATCH_SIZE = 4  # Process 4 samples at once
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

def process_dataset_batched(dataset_path: Path, model_analyzer: ModelAnalyzer, 
                            output_dir: Path, dataset_name: str, task_type: str):
    """Process dataset with PROPER BATCHING."""
    print(f"\n{'='*80}")
    print(f"PROCESSING: {dataset_name}")
    print(f"Task: {task_type}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"{'='*80}")
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Total samples: {len(data)}")
    num_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Number of batches: {num_batches}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare all samples first
    all_samples = []
    for i, sample in enumerate(data):
        if task_type == 'code-to-text':
            code_fields = [('query', 'code')]
        else:  # code-to-code
            code_fields = [('query', 'initial_segment'), ('target', 'completion')]
        
        for field_name, label in code_fields:
            code = sample.get(field_name, '')
            if code and isinstance(code, str):
                all_samples.append({
                    'sample_id': i,
                    'code': code,
                    'label': label,
                    'task_type': task_type
                })
    
    print(f"Total code snippets to process: {len(all_samples)}")
    
    # Process in batches
    results_dict = {}  # sample_id -> features
    
    for batch_start in tqdm(range(0, len(all_samples), BATCH_SIZE), 
                           desc="Processing batches", 
                           unit="batch"):
        batch_end = min(batch_start + BATCH_SIZE, len(all_samples))
        batch = all_samples[batch_start:batch_end]
        
        # Extract codes and metadata
        batch_codes = [item['code'] for item in batch]
        
        try:
            # Tokenize entire batch
            inputs = model_analyzer.tokenizer(
                batch_codes,
                return_tensors="pt",
                max_length=model_analyzer.max_length,
                truncation=True,
                padding=True
            ).to(DEVICE)
            
            # Forward pass for entire batch (THIS IS THE KEY!)
            with torch.no_grad():
                outputs = model_analyzer.model(**inputs)
            
            # Process each item in batch
            for idx, item in enumerate(batch):
                sample_id = item['sample_id']
                label = item['label']
                
                # Initialize result for this sample if needed
                if sample_id not in results_dict:
                    results_dict[sample_id] = {
                        'sample_id': sample_id,
                        'task_type': item['task_type'],
                        'features': {}
                    }
                
                # Extract tokens
                tokens = model_analyzer.tokenizer.convert_ids_to_tokens(
                    inputs['input_ids'][idx]
                )
                
                # Extract attention (batch_idx, layer, head, seq, seq)
                attention_dict = {}
                for layer_idx, layer_attn in enumerate(outputs.attentions):
                    layer_attn_item = layer_attn[idx].cpu().numpy()
                    attention_dict[f'layer_{layer_idx}'] = {
                        f'head_{head_idx}': layer_attn_item[head_idx].tolist()
                        for head_idx in range(layer_attn_item.shape[0])
                    }
                
                # Extract embeddings (batch_idx, layer, seq, hidden)
                embeddings_dict = {}
                for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
                    embeddings_dict[f'layer_{layer_idx}'] = \
                        layer_hidden[idx].cpu().numpy().tolist()
                
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
            
            # Clear GPU cache periodically
            if batch_start % (BATCH_SIZE * 10) == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
        
        except Exception as e:
            print(f"\n✗ Error in batch {batch_start}-{batch_end}: {e}")
            continue
    
    # Convert dict to list
    results = list(results_dict.values())
    
    # Save results
    output_file = output_dir / f"{dataset_name}_{model_analyzer.model_name}_features.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"\n✓ Saved {len(results)} samples to {output_file.name}")
    
    # Save summary
    summary = {
        'model': model_analyzer.model_name,
        'dataset': dataset_name,
        'task_type': task_type,
        'total_samples': len(data),
        'successful_extractions': len(results),
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
    parser = argparse.ArgumentParser(description='Extract model features (BATCHED)')
    parser.add_argument('--model', type=str, choices=['unixcoder', 'codebert', 'both'],
                       default='both', help='Which model to run')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("MODEL FEATURE EXTRACTION (FIXED - WITH PROPER BATCHING)")
    print("=" * 80)
    print(f"\n⚡ Batch size: {BATCH_SIZE} samples per batch")
    print(f"⚡ Device: {DEVICE}")
    print(f"⚡ Expected speedup: 3-4x faster than before!")
    
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
                process_dataset_batched(
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
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        print(f"\n✓ Completed {model_key}")
    
    print("\n" + "=" * 80)
    print("✓ FEATURE EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved in: {MODEL_OUTPUT_DIR}/")
    print("\nExpected runtime with batching:")
    print("  • UniXcoder: ~2.5-3 hours (was ~11 hours)")
    print("  • CodeBERT: ~2.5-3 hours (was ~11 hours)")
    print("\n")

if __name__ == "__main__":
    main()
