import torch

from .profile import profile_cuda
from .quant import static_quantize_i8
from .runtime import compile_tilelang, require_tilelang


def _bucket_rows(rows: int) -> int:
    if rows <= 16:
        return 16
    if rows <= 32:
        return 32
    if rows <= 64:
        return 64
    return ((rows + 127) // 128) * 128


def _default_block_m(rows: int) -> int:
    if rows <= 32:
        return rows
    if rows < 128:
        return 64
    return 128


def _linear_program(rows: int, in_features: int, out_features: int):
    _, T = require_tilelang()

    block_m = _default_block_m(rows)
    block_n = 128
    block_k = 128 if in_features % 128 == 0 else 64
    threads = 128
    policy = T.GemmWarpPolicy.FullCol if rows <= 32 else T.GemmWarpPolicy.Square

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, in_features), "int8"),
        W: T.Tensor((out_features, in_features), "int8"),
        X_SCALE: T.Tensor((1,), "float32"),
        W_SCALE: T.Tensor((out_features,), "float32"),
        Y: T.Tensor((rows, out_features), "bfloat16"),
    ):
        with T.Kernel(
            T.ceildiv(rows, block_m),
            T.ceildiv(out_features, block_n),
            threads=threads,
        ) as (bm, bn):
            x_shared = T.alloc_shared((block_m, block_k), "int8")
            w_shared = T.alloc_shared((block_n, block_k), "int8")
            scale_shared = T.alloc_shared((block_n,), "float32")
            acc = T.alloc_fragment((block_m, block_n), "int32")
            T.clear(acc)
            for n in T.Parallel(block_n):
                col = bn * block_n + n
                if col < out_features:
                    scale_shared[n] = X_SCALE[0] * W_SCALE[col]
            for bk in T.Pipelined(T.ceildiv(in_features, block_k), num_stages=2):
                T.copy(X[bm * block_m, bk * block_k], x_shared)
                T.copy(W[bn * block_n, bk * block_k], w_shared)
                T.gemm(x_shared, w_shared, acc, transpose_B=True, policy=policy)
            for m, n in T.Parallel(block_m, block_n):
                row = bm * block_m + m
                col = bn * block_n + n
                if row < rows and col < out_features:
                    Y[row, col] = T.cast(
                        T.cast(acc[m, n], "float32") * scale_shared[n],
                        "bfloat16",
                    )

    return kernel


def _compiled_linear(rows: int, in_features: int, out_features: int):
    key = ("linear_w8a8_static_bf16_semantics", rows, in_features, out_features)
    return compile_tilelang(key, lambda: _linear_program(rows, in_features, out_features), (4,))


def w8a8_static_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    x_shape = x.shape
    x_2d = x.reshape(-1, x_shape[-1])
    if x_2d.dtype == torch.int8:
        x_i8 = x_2d.contiguous()
    else:
        x_i8 = static_quantize_i8(x_2d, input_scale).contiguous()
    weight_i8 = weight.contiguous()
    input_scale = input_scale.reshape(-1)[:1].contiguous().to(device=x.device, dtype=torch.float32)
    weight_scale = weight_scale.reshape(-1).contiguous().to(device=x.device, dtype=torch.float32)
    rows = x_i8.shape[0]
    bucket_rows = _bucket_rows(rows)
    if rows != bucket_rows:
        x_bucket = torch.empty(
            (bucket_rows, x_i8.shape[1]),
            device=x_i8.device,
            dtype=x_i8.dtype,
        )
        x_bucket[:rows] = x_i8
        x_bucket[rows:] = 0
        x_i8 = x_bucket
    kernel = _compiled_linear(bucket_rows, x_i8.shape[1], weight_i8.shape[0])
    y = profile_cuda(
        f"linear[{bucket_rows}x{x_i8.shape[1]}->{weight_i8.shape[0]}]",
        kernel,
        x_i8,
        weight_i8,
        input_scale,
        weight_scale,
    )
    return y[:rows].reshape(*x_shape[:-1], weight_i8.shape[0])
