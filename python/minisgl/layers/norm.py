from typing import Tuple

import torch
from minisgl.quant import get_quant_context

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        try:
            from flashinfer import rmsnorm
        except ImportError:
            rmsnorm = None
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rmsnorm is None:
            variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
            if get_quant_context().is_int_w8a8_static:
                y = x.to(torch.float32) * torch.rsqrt(variance + self.eps)
                return y * self.weight.to(torch.float32)
            y = x * torch.rsqrt(variance + self.eps).to(x.dtype)
            return y * self.weight.to(x.dtype)
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        if self.rmsnorm is None:
            x.copy_(self.forward(x))
            return
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        try:
            from flashinfer import fused_add_rmsnorm, rmsnorm
        except ImportError:
            fused_add_rmsnorm = None
            rmsnorm = None
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rmsnorm is None:
            if residual is None:
                residual = x
            else:
                residual = residual + x
            variance = residual.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
            if get_quant_context().is_int_w8a8_static:
                residual = residual.to(torch.float32)
                y = residual * torch.rsqrt(variance + self.eps)
                return y * self.weight.to(torch.float32), residual
            y = residual * torch.rsqrt(variance + self.eps).to(residual.dtype)
            return y * self.weight.to(residual.dtype), residual
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
