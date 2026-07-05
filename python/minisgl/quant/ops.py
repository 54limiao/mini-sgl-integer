from __future__ import annotations

import torch
import torch.nn.functional as F


def fast_hadamard(x: torch.Tensor, block_dim: int | None = None) -> torch.Tensor:
    shape = x.shape
    n = shape[-1] if block_dim is None else block_dim
    y = x.reshape(-1, n).to(torch.float32)
    step = 1
    while step < n:
        y = y.reshape(-1, n // (step * 2), step * 2)
        a = y[..., :step].clone()
        b = y[..., step:]
        y[..., :step] = a + b
        y[..., step:] = a - b
        y = y.reshape(-1, n)
        step *= 2
    return (y * (n**-0.5)).reshape(shape).to(x.dtype)


def silu_hadamard(gate_up: torch.Tensor, block_dim: int) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return fast_hadamard(F.silu(gate) * up, block_dim)
