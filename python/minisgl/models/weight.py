from __future__ import annotations

import glob
from typing import Dict

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_ceil, download_hf_weight


def _shard_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size
    SPLIT_DIM_0_LIST = [
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".gate_proj",
        ".up_proj",
    ]
    SPLIT_DIM_1_LIST = [
        ".o_proj",
        ".down_proj",
    ]
    for key, value in state_dict.items():
        if any(key.count(sub) for sub in SPLIT_DIM_0_LIST):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(key.count(sub) for sub in SPLIT_DIM_1_LIST):
            shard_state_dict[key] = value.chunk(n, dim=1)[r]
        elif key.count("lm_head") or key.count("embed_tokens"):
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = div_ceil(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            shard_state_dict[key] = value
    return shard_state_dict


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Merge Q/K/V and Gate/Up projections into single tensors."""
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    original_keys = set(state_dict.keys())

    for key in list(state_dict.keys()):
        # Skip already processed keys
        if key in filtered_state_dict:
            continue

        # Merge QKV projections (weight and weight_scale)
        if ".q_proj." in key:
            k_key = key.replace(".q_proj.", ".k_proj.")
            v_key = key.replace(".q_proj.", ".v_proj.")

            if k_key in original_keys and v_key in original_keys:
                # Concatenate Q, K, V
                q_val = state_dict[key]
                k_val = state_dict[k_key]
                v_val = state_dict[v_key]
                new_key = key.replace(".q_proj.", ".qkv_proj.")
                filtered_state_dict[new_key] = torch.cat([q_val, k_val, v_val], dim=0)
            else:
                filtered_state_dict[key] = state_dict[key]

        # Merge Gate/Up projections (weight and weight_scale)
        elif ".gate_proj." in key:
            up_key = key.replace(".gate_proj.", ".up_proj.")

            if up_key in original_keys:
                gate_val = state_dict[key]
                up_val = state_dict[up_key]
                new_key = key.replace(".gate_proj.", ".gate_up_proj.")
                filtered_state_dict[new_key] = torch.cat([gate_val, up_val], dim=0)
            else:
                filtered_state_dict[key] = state_dict[key]

        # Skip K, V, Up (already merged)
        elif ".k_proj." in key or ".v_proj." in key or ".up_proj." in key:
            continue

        # Keep all other keys (including norms, embeddings, o_proj, down_proj, weight_scale)
        else:
            filtered_state_dict[key] = state_dict[key]

    return filtered_state_dict


def load_weight(model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    model_folder = download_hf_weight(model_path)
    files = glob.glob(f"{model_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    if get_tp_info().size > 1:
        state_dict = _shard_state_dict(state_dict)

    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    return _merge_state_dict(state_dict)
