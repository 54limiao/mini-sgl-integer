from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


def _fwht_torch(x: torch.Tensor) -> torch.Tensor:
    size = x.shape[-1]
    if size <= 0 or size & (size - 1):
        raise ValueError(f"FWHT last dim must be power-of-two, got {size}")

    out = x
    step = 1
    while step < size:
        paired = out.view(*out.shape[:-1], -1, step * 2)
        left = paired[..., :step].clone()
        right = paired[..., step : (step * 2)].clone()
        paired[..., :step] = left + right
        paired[..., step : (step * 2)] = left - right
        step <<= 1

    return out * (1.0 / math.sqrt(size))


def _fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    work = x.to(torch.float32) if x.dtype == torch.bfloat16 else x.clone()
    out = _fwht_torch(work)
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.apply_hadamard = False
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        if self.apply_hadamard:
            # R3-style path for float inference: apply head-wise Hadamard after RoPE.
            q_heads = query.view(-1, query.shape[-1] // self.head_size, self.head_size)
            k_heads = key.view(-1, key.shape[-1] // self.head_size, self.head_size)
            query.copy_(_fwht_last_dim(q_heads).view_as(query))
            key.copy_(_fwht_last_dim(k_heads).view_as(key))
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
    apply_hadamard: bool = False,
) -> RotaryEmbedding:
    if rope_scaling is None:
        rope = RotaryEmbedding(head_dim, rotary_dim, max_position, base)
        rope.apply_hadamard = apply_hadamard
        return rope
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            rope = RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)
            rope.apply_hadamard = apply_hadamard
            return rope

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
    apply_hadamard: bool = False,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map, apply_hadamard)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map, apply_hadamard)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
