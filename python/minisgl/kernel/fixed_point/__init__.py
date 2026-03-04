"""
Fixed-point arithmetic kernels for integer-only LLM inference.
Uses Q15.16 format (32-bit signed integer with 16 fractional bits).
"""

from .q15_tensor import FIXED_POINT_SCALE, Q15Tensor, assert_q15, from_fixed, to_fixed


# V1: Original implementation (may overflow with large values)
from .rmsnorm_int_only import rmsnorm_int_only, fused_add_rmsnorm_int_only

# V2: Overflow-safe implementation
from .rmsnorm_int_only_v2 import rmsnorm_int_only_v2, fused_add_rmsnorm_int_only_v2

# Activation functions
from .silu_and_mul_int_only import silu_and_mul_int_only

__all__ = [
    "FIXED_POINT_SCALE",
    "Q15Tensor",
    "assert_q15",
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