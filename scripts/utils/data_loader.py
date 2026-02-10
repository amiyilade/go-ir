"""
utils/data_loader.py
Shared streaming data loader for all RQ analysis scripts.

Yields one sample at a time to prevent memory issues on large datasets.
Handles alignment between HDF5 features (positional) and JSONL metadata.

Usage:
    from utils.data_loader import stream_samples, load_metadata

    for sample in stream_samples(h5_path, jsonl_path):
        emb    = sample['embeddings']['layer_7']        # np.float16 [seq_len, 768]
        attn   = sample['attention']['layer_7_head_7']  # np.float16 [seq_len, seq_len]
        ast    = sample['ast_info']                     # dict or None
        constr = sample['go_constructs']                # dict
        oi     = sample['original_index']               # int
"""

import json
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

import h5py
import numpy as np


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def load_metadata(h5_path: Path) -> dict:
    """Return metadata from HDF5, inferring missing keys from the data itself."""
    with h5py.File(h5_path, 'r') as h5:
        meta = json.loads(h5.attrs.get('metadata', '{}'))

        # If keys are missing (older file format), infer from first sample
        if 'num_layers' not in meta or 'hidden_size' not in meta:
            sample_keys = sorted(
                [k for k in h5.keys() if k.startswith('sample_')],
                key=lambda x: int(x.split('_')[1])
            )
            if sample_keys:
                grp = h5[sample_keys[0]]
                emb_grp = grp.get('embeddings', {})
                layer_keys = [k for k in emb_grp.keys() if k.startswith('layer_')]
                if layer_keys:
                    meta['num_layers'] = len(layer_keys)
                    meta['hidden_size'] = int(emb_grp[layer_keys[0]].shape[-1])
                attn_grp = grp.get('attention', {})
                if attn_grp:
                    heads = set()
                    for k in attn_grp.keys():
                        parts = k.split('_')  # layer_0_head_3
                        if len(parts) == 4:
                            heads.add(int(parts[3]))
                    meta.setdefault('num_heads', len(heads))
                meta.setdefault('num_samples', len(sample_keys))

        return meta


def stream_samples(
    h5_path: Path,
    jsonl_path: Path,
    max_samples: Optional[int] = None,
    embedding_layers: Optional[list] = None,
    attention_keys: Optional[list] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yield one sample at a time, lazily reading HDF5 + JSONL in lockstep.

    Parameters
    ----------
    h5_path : Path
        HDF5 file produced by Script 4.
    jsonl_path : Path
        The with_asts JSONL file that matches the HDF5 (same order).
    max_samples : int, optional
        Stop after this many samples (useful for debugging).
    embedding_layers : list of int, optional
        If given, only load these layer indices. Default = all.
    attention_keys : list of str, optional
        If given, only load these attention key names
        (e.g. ['layer_7_head_7']). Default = all.

    Yields
    ------
    dict with keys:
        'sample_idx'     : int   (position in the stratified 2k file)
        'original_index' : int   (position in the full 8k file)
        'embeddings'     : dict  {layer_key: np.ndarray float16}
        'attention'      : dict  {layer_head_key: np.ndarray float16}
        'ast_info'       : dict or None
        'go_constructs'  : dict or None
        'construct_profile' : str
        'length_bucket'     : str
        'query'             : str
        'target'            : str
    """
    h5_path    = Path(h5_path)
    jsonl_path = Path(jsonl_path)

    # Pre-index the JSONL so we can random-access if needed, but we'll
    # always iterate in order which is the common case.
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        jsonl_records = [json.loads(line) for line in f]

    with h5py.File(h5_path, 'r') as h5:
        total = sum(1 for k in h5.keys() if k.startswith('sample_'))
        limit = min(total, max_samples) if max_samples else total

        for i in range(limit):
            grp_key = f'sample_{i}'
            if grp_key not in h5:
                continue

            grp = h5[grp_key]
            original_index = int(grp.attrs.get('original_index', -1))

            # --- embeddings ---
            emb_grp = grp['embeddings']
            embeddings = {}
            for key in emb_grp.keys():
                if embedding_layers is not None:
                    layer_idx = int(key.split('_')[1])
                    if layer_idx not in embedding_layers:
                        continue
                embeddings[key] = emb_grp[key][:]   # loads into RAM, float16

            # --- attention ---
            attn_grp = grp['attention']
            attention = {}
            for key in attn_grp.keys():
                if attention_keys is not None and key not in attention_keys:
                    continue
                attention[key] = attn_grp[key][:]

            # --- JSONL metadata ---
            if i < len(jsonl_records):
                rec = jsonl_records[i]
                # Sanity check alignment
                if rec.get('original_index', original_index) != original_index:
                    raise ValueError(
                        f"sample_{i}: HDF5 original_index={original_index} "
                        f"but JSONL original_index={rec.get('original_index')}. "
                        "Ensure HDF5 and JSONL were produced from the same stratified file.")
            else:
                rec = {}

            yield {
                'sample_idx':        i,
                'original_index':    original_index,
                'embeddings':        embeddings,
                'attention':         attention,
                'ast_info':          rec.get('ast_info'),
                'go_constructs':     rec.get('go_constructs'),
                'construct_profile': rec.get('construct_profile', 'unknown'),
                'length_bucket':     rec.get('length_bucket', 'unknown'),
                'query':             rec.get('query', ''),
                'target':            rec.get('target', ''),
            }


def get_embedding(sample: dict, layer: int) -> Optional[np.ndarray]:
    """Convenience: get float32 embedding for a given layer."""
    arr = sample['embeddings'].get(f'layer_{layer}')
    if arr is None:
        return None
    return arr.astype(np.float32)


def get_attention_matrix(sample: dict, layer: int, head: int) -> Optional[np.ndarray]:
    """Convenience: get float32 attention matrix for a given layer+head."""
    arr = sample['attention'].get(f'layer_{layer}_head_{head}')
    if arr is None:
        return None
    return arr.astype(np.float32)


def has_construct(sample: dict, construct_name: str) -> bool:
    """Return True if the sample contains at least one occurrence of construct."""
    constructs = sample.get('go_constructs') or {}
    items = constructs.get(construct_name, [])
    return isinstance(items, list) and len(items) > 0


def count_construct(sample: dict, construct_name: str) -> int:
    """Return number of occurrences of construct in the sample."""
    constructs = sample.get('go_constructs') or {}
    items = constructs.get(construct_name, [])
    if isinstance(items, list):
        return len(items)
    return 0
