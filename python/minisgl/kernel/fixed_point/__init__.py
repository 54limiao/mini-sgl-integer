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


def from_fixed(x: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Convert Q15.16 int32 tensor to float with specified dtype."""
    return (x.to(torch.float32) / FIXED_POINT_SCALE).to(dtype)


# V1: Original implementation (may overflow with large values)
from .rmsnorm_int_only import rmsnorm_int_only, fused_add_rmsnorm_int_only

# V2: Overflow-safe implementation
from .rmsnorm_int_only_v2 import rmsnorm_int_only_v2, fused_add_rmsnorm_int_only_v2

# Activation functions
from .silu_and_mul_int_only import silu_and_mul_int_only

__all__ = [
    "FIXED_POINT_SCALE",
    "to_fixed",
    "from_fixed",
    # V1
    "rmsnorm_int_only",
    "fused_add_rmsnorm_int_only",
    # V2
    "rmsnorm_int_only_v2",
    "fused_add_rmsnorm_int_only_v2",
    # Activation
    "silu_and_mul_int_only",
]