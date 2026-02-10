#!/usr/bin/env python3
"""
utils/data_loader.py
Shared streaming loader for all RQ analysis scripts.

Loads ONE sample at a time from HDF5 + matching JSONL record.
Never loads the entire dataset into RAM.

Usage:
    from utils.data_loader import stream_samples, load_metadata

    for sample in stream_samples(h5_path, jsonl_path):
        emb  = sample["embeddings"]   # np.ndarray (n_layers+1, seq, hidden) float16
        attn = sample["attentions"]   # np.ndarray (n_layers, n_key_heads, seq, seq) float16
        rec  = sample["record"]       # dict from JSONL (includes ast_info, go_constructs, …)
        meta = sample["meta"]         # dict: new_index, original_index, seq_len
"""

import json
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

import h5py
import numpy as np


KEY_HEADS = [0, 3, 5, 7, 11]   # mirrors extract_features.py


def load_metadata(h5_path: Path) -> Dict[str, Any]:
    """Return model metadata stored in HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        m = f["metadata"]
        return {
            "num_layers":  int(m.attrs["num_layers"]),
            "hidden_size": int(m.attrs["hidden_size"]),
            "num_heads":   int(m.attrs["num_heads"]),
            "key_heads":   list(m.attrs["key_heads"]),
            "num_samples": int(m.attrs["num_samples"]),
        }


def stream_samples(
    h5_path:   Path,
    jsonl_path: Path,
    max_seq_len: Optional[int] = None,
    construct_filter: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yield one sample dict at a time.

    Parameters
    ----------
    h5_path        : path to features HDF5 (from extract_features.py)
    jsonl_path     : path to stratified JSONL with ASTs (same ordering)
    max_seq_len    : if set, skip samples longer than this (used by RQ2 probing)
    construct_filter: if set, only yield samples whose construct_profile equals this

    Yields
    ------
    {
        "embeddings":  np.ndarray  float32  (n_layers+1, seq_len, hidden_size)
        "attentions":  np.ndarray  float32  (n_layers,  n_key_heads, seq_len, seq_len)
        "record":      dict        full JSONL record (includes ast_info, go_constructs)
        "meta": {
            "new_index":      int
            "original_index": int
            "seq_len":        int
        }
    }
    """
    # Load JSONL into a new_index → record dict for O(1) lookup
    records: Dict[int, dict] = {}
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            records[r["new_index"]] = r

    with h5py.File(h5_path, "r") as hf:
        sample_keys = sorted(
            [k for k in hf.keys() if k.startswith("sample_")],
            key=lambda k: int(k.split("_")[1]),
        )

        for key in sample_keys:
            grp     = hf[key]
            new_idx = int(grp.attrs["new_index"])
            seq_len = int(grp.attrs["seq_len"])

            if max_seq_len is not None and seq_len > max_seq_len:
                continue

            record = records.get(new_idx)
            if record is None:
                continue

            if construct_filter is not None:
                if record.get("construct_profile") != construct_filter:
                    continue

            # Load arrays — convert float16 → float32 for computation
            embeddings = grp["embeddings"][()].astype(np.float32)
            attentions = grp["attentions"][()].astype(np.float32)

            yield {
                "embeddings": embeddings,
                "attentions": attentions,
                "record":     record,
                "meta": {
                    "new_index":      new_idx,
                    "original_index": int(grp.attrs["original_index"]),
                    "seq_len":        seq_len,
                },
            }


def get_embedding_layer(sample: Dict, layer_idx: int) -> np.ndarray:
    """Return embeddings for a specific layer. Shape: (seq_len, hidden_size)."""
    return sample["embeddings"][layer_idx]


def get_attention_head(sample: Dict, layer_idx: int, head_key_idx: int) -> np.ndarray:
    """
    Return attention matrix for a specific layer and KEY head index.
    head_key_idx is the index into KEY_HEADS list (0-4), not the raw head number.
    Shape: (seq_len, seq_len)
    """
    return sample["attentions"][layer_idx][head_key_idx]


def get_attention_head_by_id(sample: Dict, layer_idx: int, head_id: int) -> Optional[np.ndarray]:
    """
    Return attention matrix by actual head number (e.g. head 7).
    Returns None if head_id was not stored.
    """
    if head_id not in KEY_HEADS:
        return None
    key_idx = KEY_HEADS.index(head_id)
    return get_attention_head(sample, layer_idx, key_idx)


def count_available(h5_path: Path) -> int:
    """Return number of samples written to HDF5."""
    with h5py.File(h5_path, "r") as f:
        return sum(1 for k in f.keys() if k.startswith("sample_"))
