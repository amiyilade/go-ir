"""
utils/data_loader.py
Shared streaming data loader for all RQ analysis scripts.

Actual HDF5 structure (written by Script 4):
    /metadata              Dataset, JSON string
    /sample_0/
        embeddings         Dataset, float16, shape (13, seq_len, 768)
                           axis-0 = layer index 0..12
        attentions         Dataset, float16, shape (12, 5, seq_len, seq_len)
                           axis-0 = layer index 0..11
                           axis-1 = key-head slot: KEY_HEADS = [0, 3, 5, 7, 11]
    /sample_1/  ...

Usage:
    from utils.data_loader import stream_samples, load_metadata, get_embedding, get_attention_matrix

    meta = load_metadata(h5_path)
    for sample in stream_samples(h5_path, jsonl_path):
        emb  = get_embedding(sample, layer=7)           # float32 (seq_len, 768)
        attn = get_attention_matrix(sample, layer=7, head=7)  # float32 (seq_len, seq_len)
"""

import json
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List

import h5py
import numpy as np

# Key heads stored in the attention tensor (axis-1 order)
KEY_HEADS = [0, 3, 5, 7, 11]
_HEAD_SLOT = {h: i for i, h in enumerate(KEY_HEADS)}


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def load_metadata(h5_path: Path) -> dict:
    """Return the metadata dict from the HDF5 file.

    Handles both storage formats:
      - top-level Dataset  /metadata  (current format)
      - top-level attrs    h5.attrs['metadata']  (older format)
    Missing keys are inferred from the data arrays.
    """
    with h5py.File(h5_path, 'r') as h5:
        meta = {}

        # Try top-level dataset first (current format)
        if 'metadata' in h5 and isinstance(h5['metadata'], h5py.Dataset):
            try:
                raw = h5['metadata'][()]
                if isinstance(raw, bytes):
                    raw = raw.decode()
                elif isinstance(raw, np.ndarray):
                    item = raw.item()
                    raw = item.decode() if isinstance(item, bytes) else str(item)
                meta = json.loads(raw)
            except Exception:
                pass

        # Fall back to attrs
        if not meta:
            raw = h5.attrs.get('metadata', '{}')
            meta = json.loads(raw)

        # Infer missing structural keys from first sample
        sample_keys = sorted(
            [k for k in h5.keys() if k.startswith('sample_')],
            key=lambda x: int(x.split('_')[1])
        )
        if sample_keys:
            grp = h5[sample_keys[0]]
            if 'embeddings' in grp and isinstance(grp['embeddings'], h5py.Dataset):
                shape = grp['embeddings'].shape  # (num_layers, seq_len, hidden_size)
                meta.setdefault('num_layers',  int(shape[0]))
                meta.setdefault('hidden_size', int(shape[2]))
            meta.setdefault('num_heads',   12)
            meta.setdefault('key_heads',   KEY_HEADS)
            meta.setdefault('num_samples', len(sample_keys))

    return meta


def stream_samples(
    h5_path: Path,
    jsonl_path: Path,
    max_samples: Optional[int] = None,
    embedding_layers: Optional[List[int]] = None,
    attention_keys: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yield one sample at a time from the HDF5 + JSONL pair.

    Parameters
    ----------
    embedding_layers : list of int, optional
        Which layer indices to include. Default = all 13.
    attention_keys : list of str, optional
        Which 'layer_{L}_head_{H}' keys to include.
        Only stored heads [0,3,5,7,11] are available. Default = all.

    Yields  (dict)
    ------
        sample_idx, original_index
        embeddings  : {'layer_7': float16 (seq_len, 768), ...}
        attention   : {'layer_7_head_7': float16 (seq_len, seq_len), ...}
        ast_info, go_constructs, construct_profile, length_bucket, query, target
    """
    h5_path    = Path(h5_path)
    jsonl_path = Path(jsonl_path)

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
            original_index = int(grp.attrs.get('original_index', i))

            # embeddings: (num_layers, seq_len, hidden_size)
            emb_raw    = grp['embeddings'][:]
            num_layers = emb_raw.shape[0]
            embeddings = {}
            for l in range(num_layers):
                if embedding_layers is not None and l not in embedding_layers:
                    continue
                embeddings[f'layer_{l}'] = emb_raw[l]

            # attentions: (num_attn_layers, num_key_heads, seq_len, seq_len)
            attn_raw       = grp['attentions'][:]
            num_attn_layers = attn_raw.shape[0]
            attention = {}
            for l in range(num_attn_layers):
                for slot, head_idx in enumerate(KEY_HEADS):
                    key = f'layer_{l}_head_{head_idx}'
                    if attention_keys is not None and key not in attention_keys:
                        continue
                    attention[key] = attn_raw[l, slot]

            rec = jsonl_records[i] if i < len(jsonl_records) else {}

            # Unwrap nested code key ('code' for c2t, 'initial_segment' for c2c)
            raw_ast  = rec.get('ast_info')  or {}
            raw_cstr = rec.get('go_constructs') or {}
            code_key = next(
                (k for k in ('code', 'initial_segment') if k in raw_ast),
                None
            )
            ast_info     = raw_ast.get(code_key)  if code_key else raw_ast or None
            go_constructs = raw_cstr.get(code_key) if code_key and code_key in raw_cstr else raw_cstr or None

            yield {
                'sample_idx':        i,
                'original_index':    original_index,
                'embeddings':        embeddings,
                'attention':         attention,
                'ast_info':          ast_info,
                'go_constructs':     go_constructs,
                'construct_profile': rec.get('construct_profile', 'unknown'),
                'length_bucket':     rec.get('length_bucket', 'unknown'),
                'query':             rec.get('query', ''),
                'target':            rec.get('target', ''),
            }


# -----------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------

def get_embedding(sample: dict, layer: int) -> Optional[np.ndarray]:
    """float32 embedding for layer, shape (seq_len, hidden_size)."""
    arr = sample['embeddings'].get(f'layer_{layer}')
    return None if arr is None else arr.astype(np.float32)


def get_attention_matrix(sample: dict, layer: int, head: int) -> Optional[np.ndarray]:
    """float32 attention matrix for layer+head, shape (seq_len, seq_len).
    Only heads in KEY_HEADS = [0, 3, 5, 7, 11] are available.
    """
    arr = sample['attention'].get(f'layer_{layer}_head_{head}')
    return None if arr is None else arr.astype(np.float32)


def has_construct(sample: dict, construct_name: str) -> bool:
    """True if the sample contains at least one occurrence of construct."""
    val = (sample.get('go_constructs') or {}).get(construct_name, 0)
    if isinstance(val, int):
        return val > 0
    return len(val) > 0


def count_construct(sample: dict, construct_name: str) -> int:
    """Number of occurrences of construct in the sample."""
    items = (sample.get('go_constructs') or {}).get(construct_name, [])
    return len(items) if isinstance(items, list) else 0
