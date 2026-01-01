#!/usr/bin/env python3
"""
Script 4 (FULL): Extract Model Outputs with Multi-Model Support
Extracts attention weights and contextual embeddings from UniXcoder AND CodeBERT.
Processes full dataset with GPU acceleration and batch processing.
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel
from typing import Dict, List
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
        'trust_remote_code': False,
        'tokenizer_class': AutoTokenizer,
        'model_class': AutoModel
    },
    'codebert': {
        'name': 'microsoft/codebert-base',
        'max_length': 512,
        'trust_remote_code': False,
        'tokenizer_class': RobertaTokenizer,
        'model_class': RobertaModel
    }
}

# Batch processing
BATCH_SIZE = 8  # Smaller batch for GPU memory
FILE_BATCH_SIZE = 100  # Save results every 100 samples

class ModelAnalyzer:
    """Extracts attention weights and embeddings from code models."""
    
    def __init__(self, model_name: str, model_config: Dict, device: str = 'cuda'):
        """Initialize model and tokenizer."""
        print(f"\nInitializing {model_name}...")
        
        self.model_name = model_name
        self.max_length = model_config['max_length']
        self.trust_remote_code = model_config.get('trust_remote_code', False)
        self.device = device
        
        # Load tokenizer
        print(f"  Loading tokenizer from {model_config['name']}...")
        tokenizer_class = model_config['tokenizer_class']
        self.tokenizer = tokenizer_class.from_pretrained(
            model_config['name'],
            trust_remote_code=self.trust_remote_code
        )
        
        # Load model
        print(f"  Loading model from {model_config['name']}...")
        model_class = model_config['model_class']
        self.model = model_class.from_pretrained(
            model_config['name'],
            output_attentions=True,
            output_hidden_states=True,
            trust_remote_code=self.trust_remote_code
        )
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        # Get model info
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        
        print(f"  ✓ Model loaded on {device}")
        print(f"    Layers: {self.num_layers}")
        print(f"    Heads per layer: {self.num_heads}")
        print(f"    Hidden size: {self.hidden_size}")
    
    def extract_features(self, code: str, sample_id: int = -1) -> Dict:
        """Extract attention weights and embeddings for a code snippet."""
        try:
            # Tokenize
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract attention weights
            attentions = outputs.attentions
            attention_dict = {}
            
            for layer_idx, layer_attn in enumerate(attentions):
                layer_attn = layer_attn.squeeze(0).cpu().numpy()
                attention_dict[f'layer_{layer_idx}'] = {}
                for head_idx in range(layer_attn.shape[0]):
                    attention_dict[f'layer_{layer_idx}'][f'head_{head_idx}'] = layer_attn[head_idx].tolist()
            
            # Extract embeddings
            hidden_states = outputs.hidden_states
            embeddings_dict = {}
            
            for layer_idx, layer_hidden in enumerate(hidden_states):
                layer_hidden = layer_hidden.squeeze(0).cpu().numpy()
                embeddings_dict[f'layer_{layer_idx}'] = layer_hidden.tolist()
            
            # Token mapping
            token_to_word = self._create_token_to_word_mapping(tokens)
            
            return {
                'tokens': tokens,
                'attention_weights': attention_dict,
                'embeddings': embeddings_dict,
                'token_to_word_mapping': token_to_word,
                'seq_length': len(tokens),
                'model_info': {
                    'model_name': self.model_name,
                    'num_layers': self.num_layers,
                    'num_heads': self.num_heads,
                    'hidden_size': self.hidden_size
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract features for sample {sample_id}: {str(e)}") from e
    
    def _create_token_to_word_mapping(self, tokens: List[str]) -> Dict:
        """Create mapping from subword tokens to word indices."""
        word_to_tokens = []
        current_word_tokens = []
        
        for i, token in enumerate(tokens):
            # Skip special tokens
            if token in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]', '<unk>']:
                if current_word_tokens:
                    word_to_tokens.append(current_word_tokens)
                    current_word_tokens = []
                continue
            
            # Check if token starts a new word
            if token.startswith('Ġ') or (i == 0) or not token.startswith('##'):
                if current_word_tokens:
                    word_to_tokens.append(current_word_tokens)
                current_word_tokens = [i]
            else:
                current_word_tokens.append(i)
        
        if current_word_tokens:
            word_to_tokens.append(current_word_tokens)
        
        return {
            'word_to_tokens': word_to_tokens,
            'num_words': len(word_to_tokens)
        }


def process_dataset_batched(dataset_path: Path, model_analyzer: ModelAnalyzer, 
                           output_dir: Path, dataset_name: str, task_type: str):
    """Process dataset with batched file I/O."""
    print(f"\n{'='*80}")
    print(f"PROCESSING: {dataset_name}")
    print(f"Task: {task_type}")
    print(f"{'='*80}")
    
    # Count total samples
    with open(dataset_path, 'r', encoding='utf-8') as f:
        total_samples = sum(1 for _ in f)
    
    print(f"Total samples: {total_samples:,}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output files
    features_file = output_dir / f"{dataset_name}_{model_analyzer.model_name}_features.jsonl"
    summary_file = output_dir / f"{dataset_name}_{model_analyzer.model_name}_summary.json"
    
    # Process samples
    results = []
    processed = 0
    errors = 0
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        # Open output file
        with open(features_file, 'w', encoding='utf-8') as out_f:
            batch_samples = []
            
            for i, line in enumerate(tqdm(f, total=total_samples, desc=f"Extracting features")):
                sample = json.loads(line)
                
                # Determine which code field(s) to process
                code_fields = []
                if task_type == 'code-to-text':
                    code_fields = [('query', 'code')]
                elif task_type == 'code-to-code':
                    code_fields = [('query', 'initial_segment'), ('target', 'completion')]
                
                # Extract features for each code field
                sample_features = {
                    'sample_id': i,
                    'task_type': task_type,
                    'features': {}
                }
                
                has_features = False
                for field_name, label in code_fields:
                    code = sample.get(field_name, '')
                    
                    if not code or not isinstance(code, str):
                        continue
                    
                    try:
                        features = model_analyzer.extract_features(code, sample_id=i)
                        sample_features['features'][label] = features
                        has_features = True
                        
                    except Exception as e:
                        errors += 1
                        if errors <= 10:  # Only print first 10 errors
                            print(f"\n  ✗ Error sample {i}: {str(e)[:100]}")
                        continue
                
                if has_features:
                    # Write to file immediately
                    json.dump(sample_features, out_f, ensure_ascii=False)
                    out_f.write('\n')
                    processed += 1
                
                # Periodic cleanup
                if (i + 1) % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    print(f"\n✓ Successfully processed: {processed}/{total_samples} samples")
    print(f"  Errors: {errors}")
    
    # Save summary
    summary = {
        'model': model_analyzer.model_name,
        'dataset': dataset_name,
        'task_type': task_type,
        'total_samples': total_samples,
        'successful_extractions': processed,
        'errors': errors,
        'model_config': {
            'num_layers': model_analyzer.num_layers,
            'num_heads': model_analyzer.num_heads,
            'hidden_size': model_analyzer.hidden_size
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary to {summary_file.name}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Extract model features from full Go dataset'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['unixcoder', 'codebert', 'both'],
        default='both',
        help='Which model to run (default: both)'
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("MODEL FEATURE EXTRACTION (FULL DATASET)")
    print("=" * 80)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Determine models to run
    if args.model == 'both':
        models_to_run = ['unixcoder', 'codebert']
    else:
        models_to_run = [args.model]
    
    print(f"\nModels to process: {', '.join(models_to_run)}")
    print(f"Processing FULL dataset (16,244 samples)")
    
    # Create output directory
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define datasets
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
        },
    ]
    
    # Verify files
    missing_files = [d for d in datasets if not d['path'].exists()]
    if missing_files:
        print("\n✗ Error: Missing dataset files:")
        for d in missing_files:
            print(f"  {d['path']}")
        print("\nPlease run parse_go_asts_full.py first.")
        sys.exit(1)
    
    # Process each model
    for model_key in models_to_run:
        model_config = MODELS[model_key]
        
        print("\n" + "=" * 80)
        print(f"MODEL: {model_key.upper()}")
        print("=" * 80)
        
        try:
            analyzer = ModelAnalyzer(model_key, model_config, device=device)
        except Exception as e:
            print(f"\n✗ Error loading model {model_key}: {e}")
            continue
        
        # Process each dataset
        for dataset in datasets:
            print(f"\n[Processing {dataset['name']}]")
            
            try:
                process_dataset_batched(
                    dataset['path'],
                    analyzer,
                    MODEL_OUTPUT_DIR / model_key,
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n✓ Completed {model_key}")
    
    print("\n" + "=" * 80)
    print("✓ FEATURE EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nFeatures saved in: {MODEL_OUTPUT_DIR}/")
    print("\nNext steps:")
    print("  1. Run analysis scripts on full dataset")
    print("  2. Compare UniXcoder vs CodeBERT results")
    print()

if __name__ == "__main__":
    main()
