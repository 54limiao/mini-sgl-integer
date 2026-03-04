"""Integer RMSNorm layers with native Q15.16 APIs.

These layers now expose `forward_q15` methods that accept/return int32
Q15.16 tensors directly for fixed-point-only activation pipelines.
The existing `forward` methods are kept for float-compatibility bridges.
"""

from typing import Tuple

import torch

from minisgl.kernel.fixed_point import assert_q15, from_fixed, to_fixed

from .base import BaseOP


class RMSNormInteger(BaseOP):
    """RMSNorm using pure integer arithmetic (Q15.16 fixed-point) - V2 overflow-safe."""

    def __init__(self, size: int, eps: float) -> None:
        from minisgl.kernel.fixed_point import rmsnorm_int_only_v2

        self.eps = eps
        self.weight = torch.empty(size)  # Float weight, converted at runtime
        self._rmsnorm_int = rmsnorm_int_only_v2

    def forward_q15(self, x_q15: torch.Tensor) -> torch.Tensor:
        assert_q15(x_q15, "x_q15")
        weight_q15 = to_fixed(self.weight)
        return self._rmsnorm_int(x_q15, weight_q15, self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_q15 = to_fixed(x)
        out_q15 = self.forward_q15(x_q15)
        return from_fixed(out_q15, input_dtype)


class RMSNormFusedInteger(BaseOP):
    """Fused add + RMSNorm using pure integer arithmetic (Q15.16 fixed-point) - V2 overflow-safe."""

    def __init__(self, size: int, eps: float) -> None:
        from minisgl.kernel.fixed_point import fused_add_rmsnorm_int_only_v2

        self.eps = eps
        self.weight = torch.empty(size)  # Float weight, converted at runtime
        self._fused_add_rmsnorm_int = fused_add_rmsnorm_int_only_v2

    def forward_q15(
        self,
        x_q15: torch.Tensor,
        residual_q15: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert_q15(x_q15, "x_q15")

        if residual_q15 is None:
            return self._rmsnorm_only_q15(x_q15), x_q15

        assert_q15(residual_q15, "residual_q15")
        weight_q15 = to_fixed(self.weight)
        out_q15, residual_out_q15 = self._fused_add_rmsnorm_int(
            residual_q15,
            x_q15,
            weight_q15,
            self.eps,
        )
        return out_q15, residual_out_q15

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_dtype = x.dtype
        x_q15 = to_fixed(x)
        residual_q15 = to_fixed(residual) if residual is not None else None
        out_q15, residual_out_q15 = self.forward_q15(x_q15, residual_q15)
        return from_fixed(out_q15, input_dtype), from_fixed(residual_out_q15, input_dtype)

    def _rmsnorm_only_q15(self, x_q15: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel.fixed_point import rmsnorm_int_only_v2

        assert_q15(x_q15, "x_q15")
        weight_q15 = to_fixed(self.weight)
        return rmsnorm_int_only_v2(x_q15, weight_q15, self.eps)


__all__ = ["to_fixed", "from_fixed", "RMSNormInteger", "RMSNormFusedInteger"]
