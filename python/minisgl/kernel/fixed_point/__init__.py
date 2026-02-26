"""
Fixed-point arithmetic kernels for integer-only LLM inference.
Uses Q15.16 format (32-bit signed integer with 16 fractional bits).
"""

import torch
import numpy as np

FIXED_POINT_SCALE = 65536  # 2^16 for Q15.16


def to_fixed(x: torch.Tensor) -> torch.Tensor:
    """Convert float tensor to Q15.16 int32."""
    return torch.clamp(torch.round(x * FIXED_POINT_SCALE), -2**31, 2**31 - 1).to(torch.int32)


def from_fixed(x: torch.Tensor) -> torch.Tensor:
    """Convert Q15.16 int32 tensor to bfloat16."""
    return x.to(torch.float32).to(torch.bfloat16) / FIXED_POINT_SCALE


from .rmsnorm_int_only import rmsnorm_int_only, fused_add_rmsnorm_int_only

__all__ = [
    "FIXED_POINT_SCALE",
    "to_fixed",
    "from_fixed",
    "rmsnorm_int_only",
    "fused_add_rmsnorm_int_only",
]