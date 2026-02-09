#!/usr/bin/env python3
"""
RESUME-CAPABLE HDF5 Feature Extraction
Checks which samples are already processed and skips them.

Usage:
    python extract_features_hdf5_RESUME.py --model unixcoder --task code-to-text
"""

import argparse
import h5py
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import gc

# Configuration
CODE_TO_TEXT_DIR = Path("data/code-to-text")
CODE_TO_CODE_DIR = Path("data/code-to-code")
RESULTS_DIR = Path("results/features")

MODELS = {
    'unixcoder': {
        'name': 'microsoft/unixcoder-base',
        'max_length': 512,
        'trust_remote_code': False
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16

class ModelAnalyzer:
    """Extracts features from code models."""
    
    def __init__(self, model_name: str, model_config: dict):
        print(f"\n🤖 Initializing {model_name}...")
        print(f"   Device: {DEVICE}")
        
        self.model_name = model_name
        self.max_length = model_config['max_length']
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['name'],
            trust_remote_code=model_config.get('trust_remote_code', False)
        )
        
        self.model = AutoModel.from_pretrained(
            model_config['name'],
            output_attentions=True,
            output_hidden_states=True,
            trust_remote_code=model_config.get('trust_remote_code', False)
        )
        
        self.model.to(DEVICE)
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        
        print(f"   ✓ Loaded: {self.num_layers} layers, {self.num_heads} heads, {self.hidden_size} dims")

def get_processed_samples(h5_file: Path) -> set:
    """Check which samples are already in the HDF5 file."""
    if not h5_file.exists():
        return set()
    
    processed = set()
    try:
        with h5py.File(h5_file, 'r') as f:
            for key in f.keys():
                if key.startswith('sample_'):
                    sample_id = int(key.split('_')[1])
                    processed.add(sample_id)
    except Exception as e:
        print(f"⚠️  Error reading existing file: {e}")
        return set()
    
    return processed

def extract_features_resume(dataset_path: Path, output_path: Path, 
                           model_analyzer: ModelAnalyzer, task_type: str):
    """Extract features with resume capability."""
    print(f"\n{'='*80}")
    print(f"EXTRACTING FEATURES TO HDF5 (RESUME MODE)")
    print(f"{'='*80}\n")
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Total samples in dataset: {len(data):,}")
    
    # Check what's already processed
    processed = get_processed_samples(output_path)
    print(f"Already processed: {len(processed):,} samples")
    print(f"Remaining: {len(data) - len(processed):,} samples")
    
    if len(processed) >= len(data):
        print("\n✓ All samples already processed!")
        return
    
    # Determine which code fields to process
    if task_type == 'code-to-text':
        code_fields = [('query', 'code')]
    else:
        code_fields = [('query', 'initial_segment'), ('target', 'completion')]
    
    # Prepare samples (skip already processed)
    all_samples = []
    for i, sample in enumerate(data):
        if i in processed:
            continue  # Skip already processed samples
        
        for field_name, label in code_fields:
            code = sample.get(field_name, '')
            if code and isinstance(code, str):
                all_samples.append({
                    'sample_id': i,
                    'code': code,
                    'label': label,
                    'task_type': task_type
                })
    
    print(f"Will process {len(all_samples):,} code snippets\n")
    
    if len(all_samples) == 0:
        print("✓ Nothing to process!")
        return
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open HDF5 file in append mode ('a') instead of write mode ('w')
    file_mode = 'a' if output_path.exists() else 'w'
    
    with h5py.File(output_path, file_mode) as h5f:
        # Create/update metadata
        if 'metadata' not in h5f:
            meta = h5f.create_group('metadata')
        else:
            meta = h5f['metadata']
        
        meta.attrs['model_name'] = model_analyzer.model_name
        meta.attrs['task_type'] = task_type
        meta.attrs['num_layers'] = model_analyzer.num_layers
        meta.attrs['num_heads'] = model_analyzer.num_heads
        meta.attrs['hidden_size'] = model_analyzer.hidden_size
        meta.attrs['total_samples'] = len(data)
        
        # Process in batches
        samples_written = 0
        
        with tqdm(total=len(all_samples), desc="Extracting", unit="sample") as pbar:
            for batch_start in range(0, len(all_samples), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(all_samples))
                batch = all_samples[batch_start:batch_end]
                
                batch_codes = [item['code'] for item in batch]
                
                try:
                    inputs = model_analyzer.tokenizer(
                        batch_codes,
                        return_tensors="pt",
                        max_length=model_analyzer.max_length,
                        truncation=True,
                        padding=True
                    ).to(DEVICE)
                    
                    with torch.no_grad():
                        outputs = model_analyzer.model(**inputs)
                    
                    for idx, item in enumerate(batch):
                        sample_id = item['sample_id']
                        label = item['label']
                        
                        sample_key = f'sample_{sample_id}'
                        if sample_key not in h5f:
                            sample_grp = h5f.create_group(sample_key)
                            sample_grp.attrs['task_type'] = task_type
                        else:
                            sample_grp = h5f[sample_key]
                        
                        label_grp = sample_grp.create_group(label)
                        
                        # Store tokens (Unicode-safe)
                        tokens = model_analyzer.tokenizer.convert_ids_to_tokens(
                            inputs['input_ids'][idx]
                        )
                        dt = h5py.special_dtype(vlen=str)
                        label_grp.create_dataset(
                            'tokens',
                            data=np.array(tokens, dtype=object),
                            dtype=dt,
                            compression='gzip',
                            compression_opts=4
                        )
                        label_grp.attrs['seq_length'] = len(tokens)
                        
                        # Store attention
                        attn_grp = label_grp.create_group('attention')
                        for layer_idx, layer_attn in enumerate(outputs.attentions):
                            layer_grp = attn_grp.create_group(f'layer_{layer_idx}')
                            sample_attn = layer_attn[idx].cpu().numpy()
                            
                            for head_idx in range(sample_attn.shape[0]):
                                layer_grp.create_dataset(
                                    f'head_{head_idx}',
                                    data=sample_attn[head_idx],
                                    dtype='float32',
                                    compression='gzip',
                                    compression_opts=4
                                )
                        
                        # Store embeddings
                        emb_grp = label_grp.create_group('embeddings')
                        for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
                            sample_emb = layer_hidden[idx].cpu().numpy()
                            emb_grp.create_dataset(
                                f'layer_{layer_idx}',
                                data=sample_emb,
                                dtype='float32',
                                compression='gzip',
                                compression_opts=4
                            )
                        
                        samples_written += 1
                        pbar.update(1)
                    
                    del outputs, inputs
                    gc.collect()
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    elif DEVICE == "mps":
                        torch.mps.empty_cache()
                
                except Exception as e:
                    print(f"\n✗ Error in batch {batch_start}: {e}")
                    continue
    
    file_size_gb = output_path.stat().st_size / (1024**3)
    total_now = len(processed) + samples_written
    print(f"\n✓ Extraction complete!")
    print(f"  New samples written: {samples_written:,}")
    print(f"  Total samples in file: {total_now:,} / {len(data):,}")
    print(f"  File size: {file_size_gb:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description='Resume HDF5 extraction')
    parser.add_argument('--model', type=str, choices=['unixcoder'], required=True)
    parser.add_argument('--task', type=str, choices=['code-to-text', 'code-to-code'], required=True)
    args = parser.parse_args()
    
    # Set paths
    if args.task == 'code-to-text':
        dataset_path = CODE_TO_TEXT_DIR / "full_code_to_text_with_asts.jsonl"
    else:
        dataset_path = CODE_TO_CODE_DIR / "full_code_to_code_with_asts.jsonl"
    
    output_filename = f"{args.task.replace('-', '_')}_{args.model}.h5"
    output_path = RESULTS_DIR / output_filename
    
    # Initialize model
    model_config = MODELS[args.model]
    analyzer = ModelAnalyzer(args.model, model_config)
    
    # Extract with resume
    extract_features_resume(dataset_path, output_path, analyzer, args.task)
    
    print("\n✓ DONE!")

if __name__ == "__main__":
    main()
