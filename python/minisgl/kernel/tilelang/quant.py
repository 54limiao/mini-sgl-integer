import os

import torch

from .profile import profile_cuda
from .runtime import compile_tilelang, require_tilelang


def _bucket_rows(rows: int) -> int:
    if rows <= 16:
        return 16
    if rows <= 32:
        return 32
    if rows <= 64:
        return 64
    return ((rows + 127) // 128) * 128


def _quant_program(numel: int, scale_numel: int, inner_dim: int):
    _, T = require_tilelang()
    block = 256
    threads = 128

    @T.prim_func
    def kernel(
        X: T.Tensor((numel,), "bfloat16"),
        SCALE: T.Tensor((scale_numel,), "float32"),
        Y: T.Tensor((numel,), "int8"),
    ):
        with T.Kernel(T.ceildiv(numel, block), threads=threads) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < numel:
                    scale_idx = T.if_then_else(
                        scale_numel == 1,
                        0,
                        T.floormod(T.floordiv(idx, inner_dim), scale_numel),
                    )
                    value = T.cast(X[idx], "float32") / SCALE[scale_idx]
                    rounded = T.if_then_else(
                        value >= 0,
                        T.floor(value + 0.5),
                        T.ceil(value - 0.5),
                    )
                    clipped = T.min(T.max(rounded, -128), 127)
                    Y[idx] = T.cast(clipped, "int8")

    return kernel


def _compiled_quant(numel: int, scale_numel: int, inner_dim: int):
    key = ("static_quantize_i8", numel, scale_numel, inner_dim)
    return compile_tilelang(key, lambda: _quant_program(numel, scale_numel, inner_dim), (2,))


def static_quantize_i8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous() and os.environ.get("MINISGL_QUANT_REF") != "1":
        scale = scale.reshape(-1).contiguous().to(device=x.device, dtype=torch.float32)
        if scale.numel() == 1 or x.numel() % scale.numel() == 0:
            inner_dim = x.shape[-1] if x.ndim > 1 else 1
            head_dim = scale.numel() if x.ndim >= 3 and scale.numel() == x.shape[-2] else 1
            row_dim = x.numel() // (head_dim * inner_dim)
            bucket_rows = _bucket_rows(row_dim)
            if row_dim != bucket_rows:
                x_2d = x.reshape(row_dim, head_dim * inner_dim)
                x_bucket = torch.empty(
                    (bucket_rows, head_dim * inner_dim),
                    device=x.device,
                    dtype=x.dtype,
                )
                x_bucket[:row_dim] = x_2d
                x_bucket[row_dim:] = 0
                y = profile_cuda(
                    f"static_quant[{bucket_rows}x{head_dim * inner_dim}]",
                    _compiled_quant(bucket_rows * head_dim * inner_dim, scale.numel(), inner_dim),
                    x_bucket.reshape(-1),
                    scale,
                )
                return y[: x.numel()].reshape(x.shape)
            y = profile_cuda(
                f"static_quant[{row_dim}x{head_dim * inner_dim}]",
                _compiled_quant(x.numel(), scale.numel(), inner_dim),
                x.reshape(-1),
                scale,
            )
            return y.reshape(x.shape)
    scale = scale.to(device=x.device, dtype=torch.float32)
    while scale.ndim < x.ndim:
        scale = scale.unsqueeze(0)
    q = torch.round(x.to(torch.float32) / scale)
    return q.clamp(-128, 127).to(torch.int8)
