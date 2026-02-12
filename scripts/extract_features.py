#!/usr/bin/env python3
"""
Extract Model Features
Extracts all 13-layer embeddings + 5 key attention heads from UniXcoder or CodeBERT.
Saves to HDF5 with float16 + chunking for memory efficiency.

Input:  data/stratified_2k_{task}_with_asts.jsonl
Output: data/features/{task}_{model}.h5

Disclaimer: ChatGPT and Copilot were used to edit and enhance this script for better readability, error handling, and user feedback.
The author (me) implemented the core logic.
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DATA_DIR     = Path("data")
FEATURES_DIR = DATA_DIR / "features"

MODEL_CONFIGS = {
    "unixcoder": {
        "hf_name":    "microsoft/unixcoder-base",
        "max_length": 512,
    },
    "codebert": {
        "hf_name":    "microsoft/codebert-base",
        "max_length": 512,
    },
}

TASK_CONFIGS = {
    "code-to-text": {
        "jsonl": DATA_DIR / "stratified_2k_code_to_text_with_asts.jsonl",
        "code_field": "code",
    },
    "code-to-code": {
        "jsonl": DATA_DIR / "stratified_2k_code_to_code_with_asts.jsonl",
        "code_field": "code",
    },
}

KEY_HEADS   = [0, 3, 5, 7, 11]
CHUNK_SIZE  = 50    
DTYPE       = np.float16

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_key: str, device: str):
    cfg = MODEL_CONFIGS[model_key]
    print(f"  Loading {cfg['hf_name']} …")
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    model = AutoModel.from_pretrained(
        cfg["hf_name"],
        output_attentions=True,
        output_hidden_states=True,
    ).to(device).eval()
    print(f"  ✓ Model on {device}")
    print(f"    layers={model.config.num_hidden_layers}  "
          f"heads={model.config.num_attention_heads}  "
          f"hidden={model.config.hidden_size}")
    return tokenizer, model


@torch.no_grad()
def extract(code: str, tokenizer, model, cfg: dict, device: str):
    try:
        inputs = tokenizer(
            code,
            return_tensors="pt",
            max_length=cfg["max_length"],
            truncation=True,
            padding=False,
        ).to(device)

        out = model(**inputs)

        embeddings = np.stack(
            [h.squeeze(0).cpu().to(torch.float32).numpy()
             for h in out.hidden_states],
            axis=0,
        ).astype(DTYPE)  

        attentions = np.stack(
            [a.squeeze(0)[KEY_HEADS].cpu().to(torch.float32).numpy()
             for a in out.attentions],
            axis=0,
        ).astype(DTYPE)  

        seq_len = embeddings.shape[1]
        return embeddings, attentions, seq_len

    except Exception as e:
        return None, None, None


def create_hdf5(path: Path, num_samples: int, model, n_key_heads: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    num_layers  = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    f = h5py.File(path, "w")

    meta = f.create_group("metadata")
    meta.attrs["num_layers"]  = num_layers
    meta.attrs["hidden_size"] = hidden_size
    meta.attrs["num_heads"]   = model.config.num_attention_heads
    meta.attrs["key_heads"]   = KEY_HEADS
    meta.attrs["num_samples"] = num_samples

    return f


def write_sample(f: h5py.File, new_idx: int, orig_idx: int,
                 embeddings, attentions, seq_len: int):
    grp = f.create_group(f"sample_{new_idx}")
    grp.attrs["original_index"] = orig_idx
    grp.attrs["new_index"]      = new_idx
    grp.attrs["seq_len"]        = seq_len

    grp.create_dataset("embeddings", data=embeddings,
                       compression="gzip", compression_opts=4)
    grp.create_dataset("attentions", data=attentions,
                       compression="gzip", compression_opts=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--task",  required=True,
                        choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already written to HDF5")
    args = parser.parse_args()

    print("=" * 60)
    print(f"EXTRACT FEATURES")
    print(f"  model = {args.model}   task = {args.task}")
    print("=" * 60)

    task_cfg  = TASK_CONFIGS[args.task]
    model_cfg = MODEL_CONFIGS[args.model]
    out_path  = FEATURES_DIR / f"{args.task}_{args.model}.h5"

    if not task_cfg["jsonl"].exists():
        print(f"✗ JSONL not found: {task_cfg['jsonl']}")
        return

    with open(task_cfg["jsonl"]) as f:
        records = [json.loads(l) for l in f]
    print(f"  Loaded {len(records)} records")

    device = get_device()
    tokenizer, model = load_model(args.model, device)

    done = set()
    if args.resume and out_path.exists():
        with h5py.File(out_path, "r") as f_check:
            done = {int(k.split("_")[1]) for k in f_check.keys()
                    if k.startswith("sample_")}
        print(f"  Resume: {len(done)} samples already written")

    mode = "a" if done else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, mode) as hf:

        if not done:
            meta = hf.require_group("metadata")
            meta.attrs["num_layers"]  = model.config.num_hidden_layers
            meta.attrs["hidden_size"] = model.config.hidden_size
            meta.attrs["num_heads"]   = model.config.num_attention_heads
            meta.attrs["key_heads"]   = KEY_HEADS
            meta.attrs["num_samples"] = len(records)

        errors = 0
        t0 = time.time()

        for record in tqdm(records, desc="  Extracting"):
            new_idx  = record["new_index"]
            orig_idx = record["original_index"]

            if new_idx in done:
                continue

            code = record.get(task_cfg["code_field"], "")
            embeddings, attentions, seq_len = extract(
                code, tokenizer, model, model_cfg, device)

            if embeddings is None:
                errors += 1
                continue

            write_sample(hf, new_idx, orig_idx, embeddings, attentions, seq_len)

        elapsed = time.time() - t0
        n_written = len([k for k in hf.keys() if k.startswith("sample_")])

    print(f"\n  ✓ Written: {n_written} / {len(records)} samples  "
          f"({errors} errors)  [{elapsed:.0f}s]")
    print(f"  ✓ HDF5: {out_path}")

    size_mb = out_path.stat().st_size / 1e6
    print(f"  ✓ File size: {size_mb:.0f} MB")

    task_slug = args.task.replace("-", "_")

if __name__ == "__main__":
    main()
