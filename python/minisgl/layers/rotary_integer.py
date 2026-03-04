from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from minisgl.kernel.fixed_point import TRIG_Q15_SCALE, from_fixed, rope_fwht_int_q15, to_fixed
from minisgl.layers.base import StateLessOP


class RotaryEmbeddingInteger(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
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
        self._cos_sin_cache_q15 = torch.clamp(
            torch.round(self._cos_sin_cache * TRIG_Q15_SCALE),
            -(1 << 15),
            (1 << 15) - 1,
        ).to(torch.int32)
        assert self.head_size in [64, 128, 256, 512]

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply pseudo-quantized RoPE (inplace operation like original RotaryEmbedding).
        
        Args:
            positions: Position indices [batch_size]
            query: Query tensor [batch_size, num_qo_heads * head_size] (float, 2D flat)
            key: Key tensor [batch_size, num_kv_heads * head_size] (float, 2D flat)
            
        Returns:
            query: Rotated tensor [batch_size, num_qo_heads, head_size] (3D)
            key: Rotated tensor [batch_size, num_kv_heads * head_size] (2D flat, same stride as input)
        """
        input_dtype = query.dtype
        query_fixed = to_fixed(query)
        key_fixed = to_fixed(key)
        query_out_fixed, key_out_fixed = self.forward_q15(positions, query_fixed, key_fixed)

        query_out = from_fixed(query_out_fixed, input_dtype)
        key_out_float = from_fixed(key_out_fixed, input_dtype)
        key.copy_(key_out_float)

        return query_out, key

    def forward_q15(
        self,
        positions: torch.Tensor,
        query_q15: torch.Tensor,
        key_q15: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE + Hadamard with Q15.16 input/output.

        Args:
            positions: Position indices [batch_size]
            query_q15: Query [batch_size, num_qo_heads * head_size] in Q15.16 int32
            key_q15: Key [batch_size, num_kv_heads * head_size] in Q15.16 int32

        Returns:
            query_out_q15: [batch_size, num_qo_heads, head_size] in Q15.16 int32
            key_out_q15: [batch_size, num_kv_heads * head_size] in Q15.16 int32 when key input is 2D,
                otherwise [batch_size, num_kv_heads, head_size]
        """
        if query_q15.dtype != torch.int32:
            raise TypeError(f"query_q15 must be int32 Q15.16, got {query_q15.dtype}")
        if key_q15.dtype != torch.int32:
            raise TypeError(f"key_q15 must be int32 Q15.16, got {key_q15.dtype}")

        batch_size = query_q15.shape[0]
        if key_q15.shape[0] != batch_size:
            raise ValueError(
                f"query/key batch mismatch: {batch_size} vs {key_q15.shape[0]}"
            )

        query_was_2d = query_q15.ndim == 2
        key_was_2d = key_q15.ndim == 2

        if query_q15.ndim == 2:
            if query_q15.shape[1] % self.head_size != 0:
                raise ValueError(
                    f"query last dim must be divisible by head_size={self.head_size}, got {query_q15.shape[1]}"
                )
            num_qo_heads = query_q15.shape[1] // self.head_size
            query_3d = query_q15.reshape(batch_size, num_qo_heads, self.head_size)
        elif query_q15.ndim == 3:
            if query_q15.shape[2] != self.head_size:
                raise ValueError(
                    f"query last dim must equal head_size={self.head_size}, got {query_q15.shape[2]}"
                )
            query_3d = query_q15
        else:
            raise ValueError(f"query_q15 must be 2D or 3D, got shape {tuple(query_q15.shape)}")

        if key_q15.ndim == 2:
            if key_q15.shape[1] % self.head_size != 0:
                raise ValueError(
                    f"key last dim must be divisible by head_size={self.head_size}, got {key_q15.shape[1]}"
                )
            num_kv_heads = key_q15.shape[1] // self.head_size
            key_3d = key_q15.reshape(batch_size, num_kv_heads, self.head_size)
        elif key_q15.ndim == 3:
            if key_q15.shape[2] != self.head_size:
                raise ValueError(
                    f"key last dim must equal head_size={self.head_size}, got {key_q15.shape[2]}"
                )
            key_3d = key_q15
        else:
            raise ValueError(f"key_q15 must be 2D or 3D, got shape {tuple(key_q15.shape)}")

        if positions.device != query_3d.device:
            positions = positions.to(query_3d.device)
        if self._cos_sin_cache_q15.device != query_3d.device:
            self._cos_sin_cache_q15 = self._cos_sin_cache_q15.to(query_3d.device)

        query_out_q15, key_out_q15 = rope_fwht_int_q15(
            positions=positions,
            query_q15=query_3d,
            key_q15=key_3d,
            cos_sin_cache_q15=self._cos_sin_cache_q15,
        )

        if key_was_2d:
            key_out_q15 = key_out_q15.reshape(batch_size, -1)

        return query_out_q15, key_out_q15


def _get_rope_integer(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbeddingInteger:
    if rope_scaling is None:
        return RotaryEmbeddingInteger(head_dim, rotary_dim, max_position, base)
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

            return RotaryEmbeddingInteger(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


# Import shared state from rotary module
from .rotary import _ROPE_DEVICE, set_rope_device


@functools.cache
def get_rope_integer(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbeddingInteger:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope_integer(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope_integer(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope_integer", "RotaryEmbeddingInteger"]
