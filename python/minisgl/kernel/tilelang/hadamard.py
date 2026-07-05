import math

import torch

from .profile import profile_cuda
from .runtime import compile_tilelang, require_tilelang


def _tl_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise TypeError(f"Unsupported TileLang Hadamard input dtype: {dtype}")


def _bucket_rows(rows: int) -> int:
    if rows <= 16:
        return 16
    if rows <= 32:
        return 32
    if rows <= 64:
        return 64
    return ((rows + 127) // 128) * 128


def _quant_i8_expr(T, value, scale):
    q = T.if_then_else(
        value >= 0,
        T.floor(value / scale + 0.5),
        T.ceil(value / scale - 0.5),
    )
    return T.cast(T.min(T.max(q, -128), 127), "int8")


def _hadamard_quant_program(
    rows: int,
    cols: int,
    block_dim: int,
    scale_numel: int,
    x_dtype: str,
):
    _, T = require_tilelang()
    groups = cols // block_dim
    inv_sqrt_block = 1.0 / math.sqrt(block_dim)
    thread_elem = 8
    lanes = block_dim // thread_elem
    thread_round = int(math.log2(thread_elem))
    warp_round = int(math.log2(lanes))
    threads = lanes

    @T.macro
    def warp_hadamard_f32(local, buf):
        tx = T.get_thread_binding(0)
        for i in T.serial(warp_round):
            stride = 1 << i
            other_thread = tx ^ stride
            sign = (tx >> i) & 1
            for j in T.Pipelined(thread_elem, num_stages=1):
                buf[j] = T.tvm_warp_shuffle(
                    0xFFFFFFFF,
                    local[j],
                    other_thread % lanes,
                    lanes,
                    lanes,
                )
                local[j] = T.if_then_else(sign == 0, local[j] + buf[j], buf[j] - local[j])

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, cols), x_dtype),
        SCALE: T.Tensor((scale_numel,), "float32"),
        Y: T.Tensor((rows, cols), "int8"),
    ):
        with T.Kernel(rows, groups, threads=threads) as (r, g):
            lane = T.get_thread_binding(0)
            local = T.alloc_local((thread_elem,), "float32")
            other = T.alloc_local((thread_elem,), "float32")
            for i in T.serial(thread_elem):
                local[i] = T.cast(X[r, g * block_dim + lane * thread_elem + i], "float32")

            for round_idx in T.serial(thread_round):
                step = 1 << round_idx
                chunks = thread_elem // (step * 2)
                for base in T.serial(chunks):
                    for i in T.serial(step):
                        lo = base * step * 2 + i
                        hi = lo + step
                        a = local[lo]
                        b = local[hi]
                        local[lo] = a + b
                        local[hi] = a - b
            warp_hadamard_f32(local, other)

            for i in T.serial(thread_elem):
                scale_idx = T.if_then_else(scale_numel == 1, 0, T.floormod(r, scale_numel))
                Y[r, g * block_dim + lane * thread_elem + i] = _quant_i8_expr(
                    T,
                    local[i] * T.float32(inv_sqrt_block),
                    SCALE[scale_idx],
                )

    return kernel


def _compiled_hadamard_quant(
    rows: int,
    cols: int,
    block_dim: int,
    scale_numel: int,
    x_dtype: str,
):
    key = ("hadamard_quant_i8", rows, cols, block_dim, scale_numel, x_dtype)
    return compile_tilelang(
        key,
        lambda: _hadamard_quant_program(rows, cols, block_dim, scale_numel, x_dtype),
        (2,),
    )


def hadamard_quant_i8(x: torch.Tensor, scale: torch.Tensor, block_dim: int) -> torch.Tensor:
    if block_dim != 128:
        raise ValueError("TileLang Hadamard quant currently supports block_dim=128 only")
    if x.shape[-1] % block_dim != 0:
        raise ValueError(f"Hadamard block_dim {block_dim} must divide last dim {x.shape[-1]}")
    x_shape = x.shape
    x_2d = x.reshape(-1, x_shape[-1]).contiguous()
    scale = scale.reshape(-1).contiguous().to(device=x.device, dtype=torch.float32)
    rows = x_2d.shape[0]
    bucket_rows = _bucket_rows(rows)
    if rows != bucket_rows:
        x_bucket = torch.empty(
            (bucket_rows, x_2d.shape[1]),
            device=x_2d.device,
            dtype=x_2d.dtype,
        )
        x_bucket[:rows] = x_2d
        x_bucket[rows:] = 0
        x_2d = x_bucket
    kernel = _compiled_hadamard_quant(
        bucket_rows,
        x_2d.shape[1],
        block_dim,
        scale.numel(),
        _tl_dtype(x_2d.dtype),
    )
    y = profile_cuda(
        f"hadamard_quant[{bucket_rows}x{x_2d.shape[1]}]",
        kernel,
        x_2d,
        scale,
    )
    return y[:rows].reshape(x_shape)


def _silu_hadamard_quant_program(
    rows: int,
    cols: int,
    block_dim: int,
    x_dtype: str,
):
    _, T = require_tilelang()
    groups = cols // block_dim
    inv_sqrt_block = 1.0 / math.sqrt(block_dim)
    thread_elem = 8
    lanes = block_dim // thread_elem
    thread_round = int(math.log2(thread_elem))
    warp_round = int(math.log2(lanes))
    threads = lanes

    @T.macro
    def warp_hadamard_f32(local, buf):
        tx = T.get_thread_binding(0)
        for i in T.serial(warp_round):
            stride = 1 << i
            other_thread = tx ^ stride
            sign = (tx >> i) & 1
            for j in T.Pipelined(thread_elem, num_stages=1):
                buf[j] = T.tvm_warp_shuffle(
                    0xFFFFFFFF,
                    local[j],
                    other_thread % lanes,
                    lanes,
                    lanes,
                )
                local[j] = T.if_then_else(sign == 0, local[j] + buf[j], buf[j] - local[j])

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, cols * 2), x_dtype),
        SCALE: T.Tensor((1,), "float32"),
        Y: T.Tensor((rows, cols), "int8"),
    ):
        with T.Kernel(rows, groups, threads=threads) as (r, g):
            lane = T.get_thread_binding(0)
            local = T.alloc_local((thread_elem,), "float32")
            other = T.alloc_local((thread_elem,), "float32")
            for i in T.serial(thread_elem):
                c = g * block_dim + lane * thread_elem + i
                gate = T.cast(X[r, c], "float32")
                up = T.cast(X[r, cols + c], "float32")
                sig = T.sigmoid(gate)
                local[i] = T.if_then_else(
                    gate < T.float32(-7.0),
                    T.float32(0.0),
                    T.if_then_else(gate > T.float32(7.0), gate * up, gate * sig * up),
                )

            for round_idx in T.serial(thread_round):
                step = 1 << round_idx
                chunks = thread_elem // (step * 2)
                for base in T.serial(chunks):
                    for i in T.serial(step):
                        lo = base * step * 2 + i
                        hi = lo + step
                        a = local[lo]
                        b = local[hi]
                        local[lo] = a + b
                        local[hi] = a - b
            warp_hadamard_f32(local, other)

            for i in T.serial(thread_elem):
                Y[r, g * block_dim + lane * thread_elem + i] = _quant_i8_expr(
                    T,
                    local[i] * T.float32(inv_sqrt_block),
                    SCALE[0],
                )

    return kernel


def _compiled_silu_hadamard_quant(rows: int, cols: int, block_dim: int, x_dtype: str):
    key = ("silu_hadamard_quant_i8", rows, cols, block_dim, x_dtype)
    return compile_tilelang(
        key,
        lambda: _silu_hadamard_quant_program(rows, cols, block_dim, x_dtype),
        (2,),
    )


def silu_hadamard_quant_i8(
    gate_up: torch.Tensor,
    scale: torch.Tensor,
    block_dim: int,
) -> torch.Tensor:
    if block_dim != 128:
        raise ValueError("TileLang SiLU Hadamard quant currently supports block_dim=128 only")
    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(f"Expected gate_up last dim to be even, got {gate_up.shape[-1]}")
    cols = gate_up.shape[-1] // 2
    if cols % block_dim != 0:
        raise ValueError(f"Hadamard block_dim {block_dim} must divide hidden dim {cols}")
    x_shape = gate_up.shape
    x_2d = gate_up.reshape(-1, x_shape[-1]).contiguous()
    scale = scale.reshape(-1)[:1].contiguous().to(device=gate_up.device, dtype=torch.float32)
    rows = x_2d.shape[0]
    bucket_rows = _bucket_rows(rows)
    if rows != bucket_rows:
        x_bucket = torch.empty(
            (bucket_rows, x_2d.shape[1]),
            device=x_2d.device,
            dtype=x_2d.dtype,
        )
        x_bucket[:rows] = x_2d
        x_bucket[rows:] = 0
        x_2d = x_bucket
    kernel = _compiled_silu_hadamard_quant(
        bucket_rows,
        cols,
        block_dim,
        _tl_dtype(x_2d.dtype),
    )
    y = profile_cuda(
        f"silu_hadamard_quant[{bucket_rows}x{cols}]",
        kernel,
        x_2d,
        scale,
    )
    return y[:rows].reshape(*x_shape[:-1], cols)
