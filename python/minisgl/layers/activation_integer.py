"""
Integer-only activation functions using Q15.16 fixed-point format.

This module provides true integer-only implementations using Triton kernels.
"""

import torch

from minisgl.kernel.fixed_point import FIXED_POINT_SCALE, assert_q15, from_fixed, to_fixed


def silu_and_mul_q15(
    x_q15: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """SILU-and-mul with native Q15.16 input/output (int32)."""
    from minisgl.kernel.fixed_point import silu_and_mul_int_only

    assert_q15(x_q15, "x_q15")
    result_q15 = silu_and_mul_int_only(x_q15)

    if out is not None:
        assert_q15(out, "out")
        out.copy_(result_q15)
        return out

    return result_q15


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
    input_dtype = x.dtype
    x_q15 = to_fixed(x)
    result_q15 = silu_and_mul_q15(x_q15)
    return from_fixed(result_q15, input_dtype)


__all__ = [
    "silu_and_mul_q15",
    "silu_and_mul_fixed",
    "FIXED_POINT_SCALE",
    "to_fixed",
    "from_fixed",
]