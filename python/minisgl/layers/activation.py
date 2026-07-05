from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    try:
        from flashinfer import silu_and_mul as flashinfer_silu_and_mul
    except ImportError:
        gate, up = x.chunk(2, dim=-1)
        y = F.silu(gate) * up
        if out is not None:
            out.copy_(y)
            return out
        return y
    return flashinfer_silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    try:
        from flashinfer import gelu_and_mul as flashinfer_gelu_and_mul
    except ImportError:
        gate, up = x.chunk(2, dim=-1)
        y = F.gelu(gate) * up
        if out is not None:
            out.copy_(y)
            return out
        return y
    return flashinfer_gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
