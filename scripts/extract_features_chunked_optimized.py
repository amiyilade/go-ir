#!/usr/bin/env python3
"""
Optimized Chunked HDF5 Feature Extraction
- Processes in 1k chunks
- Only saves key attention heads (0, 3, 5, 7, 11)
- Uses float16 for storage (50% smaller)
- Auto-merges chunks

Usage:
    python extract_features_chunked_optimized.py --model unixcoder --task code-to-text --chunk-size 1000
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
import sys

# Configuration
CODE_TO_TEXT_DIR = Path("data/code-to-text")
CODE_TO_CODE_DIR = Path("data/code-to-code")
RESULTS_DIR = Path("results/features")
CHUNKS_DIR = RESULTS_DIR / "chunks"

# KEY HEADS TO SAVE (instead of all 12 heads per layer)
KEY_HEADS = [0, 3, 5, 7, 11]  # 5 heads per layer × 12 layers = 60 total

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
        print(f"   ✓ Saving only {len(KEY_HEADS)} key heads per layer: {KEY_HEADS}")

def extract_chunk(data_chunk, chunk_id, chunk_start_idx, output_path, model_analyzer, task_type):
    """Extract features for one chunk with optimizations."""
    
    if task_type == 'code-to-text':
        code_fields = [('query', 'code')]
    else:
        code_fields = [('query', 'initial_segment'), ('target', 'completion')]
    
    all_samples = []
    for i, sample in enumerate(data_chunk):
        global_id = chunk_start_idx + i
        for field_name, label in code_fields:
            code = sample.get(field_name, '')
            if code and isinstance(code, str):
                all_samples.append({
                    'sample_id': global_id,  # Use global ID
                    'code': code,
                    'label': label,
                    'task_type': task_type
                })
    
    with h5py.File(output_path, 'w') as h5f:
        # Metadata
        meta = h5f.create_group('metadata')
        meta.attrs['model_name'] = model_analyzer.model_name
        meta.attrs['task_type'] = task_type
        meta.attrs['num_layers'] = model_analyzer.num_layers
        meta.attrs['num_heads'] = model_analyzer.num_heads
        meta.attrs['hidden_size'] = model_analyzer.hidden_size
        meta.attrs['chunk_id'] = chunk_id
        meta.attrs['chunk_start_idx'] = chunk_start_idx
        meta.attrs['total_samples'] = len(data_chunk)
        meta.attrs['key_heads'] = KEY_HEADS
        meta.attrs['dtype'] = 'float16'
        
        # Process batches
        samples_written = 0
        
        with tqdm(total=len(all_samples), desc=f"Chunk {chunk_id}", unit="sample") as pbar:
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
                        tokens = model_analyzer.tokenizer.convert_ids_to_tokens(inputs['input_ids'][idx])
                        dt = h5py.special_dtype(vlen=str)
                        label_grp.create_dataset(
                            'tokens',
                            data=np.array(tokens, dtype=object),
                            dtype=dt,
                            compression='gzip',
                            compression_opts=4
                        )
                        label_grp.attrs['seq_length'] = len(tokens)
                        
                        # Store ONLY KEY attention heads (not all 12)
                        attn_grp = label_grp.create_group('attention')
                        for layer_idx, layer_attn in enumerate(outputs.attentions):
                            layer_grp = attn_grp.create_group(f'layer_{layer_idx}')
                            sample_attn = layer_attn[idx].cpu().numpy()
                            
                            # Only save key heads
                            for head_idx in KEY_HEADS:
                                if head_idx < sample_attn.shape[0]:  # Safety check
                                    layer_grp.create_dataset(
                                        f'head_{head_idx}',
                                        data=sample_attn[head_idx].astype('float16'),  # Use float16
                                        dtype='float16',
                                        compression='gzip',
                                        compression_opts=4
                                    )
                        
                        # Store embeddings (float16)
                        emb_grp = label_grp.create_group('embeddings')
                        for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
                            sample_emb = layer_hidden[idx].cpu().numpy()
                            emb_grp.create_dataset(
                                f'layer_{layer_idx}',
                                data=sample_emb.astype('float16'),  # Use float16
                                dtype='float16',
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
    
    return samples_written

def merge_chunks(chunk_files, output_path, total_samples):
    """Merge all chunks into final HDF5 file."""
    print(f"\n{'='*80}")
    print("MERGING CHUNKS INTO FINAL FILE")
    print(f"{'='*80}")
    
    with h5py.File(output_path, 'w') as out_f:
        # Copy metadata from first chunk
        with h5py.File(chunk_files[0], 'r') as first_chunk:
            meta = out_f.create_group('metadata')
            for key, value in first_chunk['metadata'].attrs.items():
                if key not in ['chunk_id', 'chunk_start_idx']:  # Don't copy chunk-specific attrs
                    meta.attrs[key] = value
            meta.attrs['total_samples'] = total_samples
        
        # Copy all samples (they already have correct global IDs)
        total_copied = 0
        
        for chunk_file in tqdm(chunk_files, desc="Merging chunks", unit="chunk"):
            with h5py.File(chunk_file, 'r') as chunk_f:
                for sample_key in chunk_f.keys():
                    if sample_key == 'metadata':
                        continue
                    
                    # Copy entire sample group (already has correct ID)
                    chunk_f.copy(sample_key, out_f)
                    total_copied += 1
        
        print(f"\n✓ Merged {len(chunk_files)} chunks")
        print(f"  Total samples: {total_copied}")
    
    return total_copied

def main():
    parser = argparse.ArgumentParser(description='Optimized chunked HDF5 extraction')
    parser.add_argument('--model', type=str, choices=['unixcoder'], required=True)
    parser.add_argument('--task', type=str, choices=['code-to-text', 'code-to-code'], required=True)
    parser.add_argument('--chunk-size', type=int, default=1000, help='Samples per chunk')
    parser.add_argument('--start-chunk', type=int, default=0, help='Start from this chunk (for resuming)')
    args = parser.parse_args()
    
    print("="*80)
    print("OPTIMIZED CHUNKED HDF5 EXTRACTION")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Task: {args.task}")
    print(f"Chunk size: {args.chunk_size:,} samples")
    print(f"Device: {DEVICE}")
    print(f"\nOptimizations:")
    print(f"  ✓ Saving only key heads: {KEY_HEADS} (60 heads instead of 144)")
    print(f"  ✓ Using float16 (50% smaller than float32)")
    print(f"  ✓ Estimated size: ~7 MB per sample (vs ~34 MB original)")
    
    # Load dataset
    if args.task == 'code-to-text':
        dataset_path = CODE_TO_TEXT_DIR / "full_code_to_text_with_asts.jsonl"
    else:
        dataset_path = CODE_TO_CODE_DIR / "full_code_to_code_with_asts.jsonl"
    
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    total_samples = len(data)
    num_chunks = (total_samples + args.chunk_size - 1) // args.chunk_size
    
    print(f"\nTotal samples: {total_samples:,}")
    print(f"Number of chunks: {num_chunks}")
    print(f"Estimated final size: ~{(total_samples * 7) / 1024:.1f} GB")
    
    # Create chunks directory
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print(f"\n{'='*80}")
    analyzer = ModelAnalyzer(args.model, MODELS[args.model])
    print(f"{'='*80}")
    
    # Process each chunk
    chunk_files = []
    
    for chunk_id in range(args.start_chunk, num_chunks):
        start_idx = chunk_id * args.chunk_size
        end_idx = min(start_idx + args.chunk_size, total_samples)
        data_chunk = data[start_idx:end_idx]
        
        chunk_file = CHUNKS_DIR / f"{args.task}_{args.model}_chunk_{chunk_id:03d}.h5"
        chunk_files.append(chunk_file)
        
        if chunk_file.exists():
            print(f"\n✓ Chunk {chunk_id} already exists ({chunk_file.name})")
            size_mb = chunk_file.stat().st_size / (1024**2)
            print(f"  Size: {size_mb:.1f} MB")
            continue
        
        print(f"\n{'='*80}")
        print(f"CHUNK {chunk_id+1}/{num_chunks}: Samples {start_idx} to {end_idx-1}")
        print(f"{'='*80}")
        
        samples_written = extract_chunk(data_chunk, chunk_id, start_idx, chunk_file, analyzer, args.task)
        
        size_mb = chunk_file.stat().st_size / (1024**2)
        print(f"\n✓ Chunk {chunk_id} complete!")
        print(f"  Samples: {samples_written}")
        print(f"  Size: {size_mb:.1f} MB ({size_mb/len(data_chunk):.1f} MB per sample)")
        print(f"  Saved: {chunk_file.name}")
    
    # Collect all chunk files (including pre-existing ones)
    all_chunk_files = sorted(CHUNKS_DIR.glob(f"{args.task}_{args.model}_chunk_*.h5"))
    
    if len(all_chunk_files) != num_chunks:
        print(f"\n⚠️  Warning: Expected {num_chunks} chunks, found {len(all_chunk_files)}")
        print(f"   Missing chunks - run again to complete")
        return
    
    # Merge all chunks
    final_output = RESULTS_DIR / f"{args.task.replace('-', '_')}_{args.model}_optimized.h5"
    total_merged = merge_chunks(all_chunk_files, final_output, total_samples)
    
    final_size_gb = final_output.stat().st_size / (1024**3)
    print(f"\n{'='*80}")
    print("✓ EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nFinal file: {final_output.name}")
    print(f"  Total samples: {total_merged:,}")
    print(f"  File size: {final_size_gb:.2f} GB")
    print(f"  Avg per sample: {final_size_gb * 1024 / total_merged:.1f} MB")
    print(f"\nChunk files saved in: {CHUNKS_DIR}/")
    print(f"You can delete chunks after verifying the merged file works.")

if __name__ == "__main__":
    main()
