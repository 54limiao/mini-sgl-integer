"""
Integer-only activation functions using Q15.16 fixed-point format.

This module provides true integer-only implementations using Triton kernels.
"""

import torch

# Fixed-point constants
FIXED_POINT_SCALE = 65536  # 2^16 for Q15.16


def to_fixed(x: torch.Tensor) -> torch.Tensor:
    """Convert float tensor to Q15.16 int32."""
    return torch.clamp(torch.round(x * FIXED_POINT_SCALE), -2**31, 2**31 - 1).to(torch.int32)


def from_fixed(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert Q15.16 int32 tensor to float with specified dtype."""
    return (x.to(torch.float32) / FIXED_POINT_SCALE).to(dtype)


def silu_and_mul_fixed(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    SILU and multiply using pure integer arithmetic (Q15.16 fixed-point).
    
    This function uses the true integer-only Triton kernel.
    
    Args:
        x: Input tensor [batch, 2 * hidden_dim] (float)
        out: Optional output tensor
    
    Returns:
        Output tensor [batch, hidden_dim] (float)
    """
    from minisgl.kernel.fixed_point import silu_and_mul_int_only

    # Save input dtype
    input_dtype = x.dtype

    # Convert to Q15.16 fixed-point
    x_fixed = to_fixed(x)

    # Pure integer SILU and multiply
    result_fixed = silu_and_mul_int_only(x_fixed)

    # Convert back to original dtype
    return from_fixed(result_fixed, input_dtype)


__all__ = ["silu_and_mul_fixed", "FIXED_POINT_SCALE", "to_fixed", "from_fixed"]