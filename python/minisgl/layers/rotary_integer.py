"""
Pseudo-quantized Rotary Embedding using Q15.16 fixed-point format.

This module provides a pseudo-quantized version of RoPE that:
1. Converts float inputs to Q15.16 fixed-point (int32)
2. Applies RoPE using flashinfer (still float internally)
3. Converts results back to Q15.16 fixed-point
4. Returns as float

This is for validating the integer-only RoPE inference path.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from minisgl.kernel.fixed_point import FIXED_POINT_SCALE, from_fixed, to_fixed
from minisgl.layers.base import StateLessOP


class RotaryEmbeddingInteger(StateLessOP):
    """
    Pseudo-quantized Rotary Embedding using Q15.16 fixed-point format.
    
    Input: float tensor -> quantize to Q15.16 -> apply RoPE -> dequantize to float
    """

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
        
        # Compute inverse frequency (in float)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        
        # Buffer for cos/sin cache (stored in float, used directly)
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
        # Save input dtype for later dequantization
        input_dtype = query.dtype
        
        # Get shapes
        batch_size = query.shape[0]
        num_qo_heads = query.shape[1] // self.head_size
        num_kv_heads = key.shape[1] // self.head_size
        
        # Step 1: Quantize inputs to Q15.16
        query_fixed = to_fixed(query)
        key_fixed = to_fixed(key)
        
        # Step 2: Reshape to 3D and convert to bfloat16 for RoPE computation
        # flashinfer requires fp16/bf16, not float32
        query_3d = (query_fixed.to(torch.float32) / FIXED_POINT_SCALE).view(batch_size, num_qo_heads, self.head_size).to(torch.bfloat16)
        key_3d = (key_fixed.to(torch.float32) / FIXED_POINT_SCALE).view(batch_size, num_kv_heads, self.head_size).to(torch.bfloat16)
        
        # Step 3: Apply RoPE using flashinfer (inplace)
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query_3d,
            key=key_3d,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        
        # Step 4: Quantize back to Q15.16 (simulate integer-only output)
        query_out_fixed = torch.clamp(
            torch.round(query_3d.to(torch.float32) * FIXED_POINT_SCALE), 
            -2**31, 2**31 - 1
        ).to(torch.int32)
        key_out_fixed = torch.clamp(
            torch.round(key_3d.to(torch.float32) * FIXED_POINT_SCALE), 
            -2**31, 2**31 - 1
        ).to(torch.int32)
        
        # Step 5: Dequantize back to original dtype
        # query stays 3D [batch, num_heads, head_dim] for attention computation
        query_out = from_fixed(query_out_fixed, input_dtype)
        
        # Step 6: For key, we need to preserve the original stride/layout
        # Write the rotated values back to the original key tensor (inplace)
        key_out_float = from_fixed(key_out_fixed.view(batch_size, -1), input_dtype)
        key.copy_(key_out_float)
        
        return query_out, key


def _get_rope_integer(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbeddingInteger:
    """Create a pseudo-quantized RotaryEmbeddingInteger instance."""
    if rope_scaling is None:
        return RotaryEmbeddingInteger(head_dim, rotary_dim, max_position, base)
    
    # Handle rope scaling (same as original)
    match rope_scaling["rope_type"]:
        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
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
    """Get cached pseudo-quantized RoPE instance."""
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
