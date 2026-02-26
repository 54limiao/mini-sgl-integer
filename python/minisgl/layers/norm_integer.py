"""
Integer-only RMSNorm layer that converts float input to fixed-point,
performs integer RMSNorm, and converts back to float.

This is useful for validating integer-only inference path while keeping
the rest of the model in floating point.
"""

from typing import Tuple

import torch

from .base import BaseOP

# Fixed-point constants
FIXED_POINT_SCALE = 65536  # 2^16 for Q15.16


def to_fixed(x: torch.Tensor) -> torch.Tensor:
    """Convert float tensor to Q15.16 int32."""
    return torch.clamp(torch.round(x * FIXED_POINT_SCALE), -2**31, 2**31 - 1).to(torch.int32)


def from_fixed(x: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert Q15.16 int32 tensor to float with specified dtype."""
    return (x.to(torch.float32) / FIXED_POINT_SCALE).to(dtype)


class RMSNormInteger(BaseOP):
    """RMSNorm using pure integer arithmetic (Q15.16 fixed-point)."""

    def __init__(self, size: int, eps: float) -> None:
        from minisgl.kernel.fixed_point import rmsnorm_int_only

        self.eps = eps
        self.weight = torch.empty(size)  # Float weight, converted at runtime
        self._rmsnorm_int = rmsnorm_int_only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input dtype
        input_dtype = x.dtype
        
        # Convert to fixed-point
        x_fixed = to_fixed(x)
        weight_fixed = to_fixed(self.weight)

        # Integer RMSNorm
        out_fixed = self._rmsnorm_int(x_fixed, weight_fixed, self.eps)

        # Convert back to float, preserving input dtype
        return from_fixed(out_fixed, input_dtype)


class RMSNormFusedInteger(BaseOP):
    """Fused add + RMSNorm using pure integer arithmetic (Q15.16 fixed-point)."""

    def __init__(self, size: int, eps: float) -> None:
        from minisgl.kernel.fixed_point import fused_add_rmsnorm_int_only

        self.eps = eps
        self.weight = torch.empty(size)  # Float weight, converted at runtime
        self._fused_add_rmsnorm_int = fused_add_rmsnorm_int_only

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Save input dtype
        input_dtype = x.dtype
        
        if residual is None:
            # No residual, just do RMSNorm
            return self._rmsnorm_only(x), x

        # Convert to fixed-point
        x_fixed = to_fixed(x)
        residual_fixed = to_fixed(residual)
        weight_fixed = to_fixed(self.weight)

        # Fused add + integer RMSNorm
        out_fixed, residual_out_fixed = self._fused_add_rmsnorm_int(
            residual_fixed, x_fixed, weight_fixed, self.eps
        )

        # Convert back to float, preserving input dtype
        return from_fixed(out_fixed, input_dtype), from_fixed(residual_out_fixed, input_dtype)

    def _rmsnorm_only(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback for no residual case."""
        from minisgl.kernel.fixed_point import rmsnorm_int_only

        input_dtype = x.dtype
        x_fixed = to_fixed(x)
        weight_fixed = to_fixed(self.weight)
        out_fixed = rmsnorm_int_only(x_fixed, weight_fixed, self.eps)
        return from_fixed(out_fixed, input_dtype)


__all__ = ["FIXED_POINT_SCALE", "to_fixed", "from_fixed", "RMSNormInteger", "RMSNormFusedInteger"]
