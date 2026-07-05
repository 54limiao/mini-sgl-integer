import math
import torch

from .profile import profile_cuda
from .runtime import compile_tilelang, require_tilelang


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _bucket_batch(batch: int) -> int:
    if batch <= 1:
        return 1
    if batch <= 2:
        return 2
    if batch <= 4:
        return 4
    return _align_up(batch, 8)


def _paged_decode_program(
    batch: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    cache_tokens: int,
    max_seqlen_k: int,
    block_n: int,
    block_h: int,
    num_stages: int,
    threads: int,
):
    _, T = require_tilelang()

    accum_dtype = T.float32
    q_group = q_heads // kv_heads
    valid_block_h = min(block_h, q_group)
    heads_per_kv_block = q_group // valid_block_h
    softmax_scale_log2 = (1.0 / math.sqrt(head_dim)) * 1.4426950408889634

    @T.prim_func
    def kernel(
        Q: T.Tensor((batch, q_heads, head_dim), "int8"),
        K: T.Tensor((cache_tokens, kv_heads, head_dim), "int8"),
        V: T.Tensor((cache_tokens, kv_heads, head_dim), "bfloat16"),
        PAGE_TABLE: T.Tensor((batch, max_seqlen_k), T.int32),
        SEQ_LENS: T.Tensor((batch,), T.int32),
        Q_SCALE: T.Tensor((q_heads,), "float32"),
        K_SCALE: T.Tensor((kv_heads,), "float32"),
        O: T.Tensor((batch, q_heads, head_dim), "bfloat16"),
    ):
        with T.Kernel(
            batch,
            T.ceildiv(q_heads, valid_block_h),
            threads=threads,
        ) as (bid, hid):
            q_shared = T.alloc_shared((block_h, head_dim), "int8")
            k_shared = T.alloc_shared((block_n, head_dim), "int8")
            v_shared = T.alloc_shared((block_n, head_dim), "bfloat16")
            acc_i32 = T.alloc_fragment((block_h, block_n), "int32")
            acc_s = T.alloc_fragment((block_h, block_n), accum_dtype)
            acc_s_cast = T.alloc_fragment((block_h, block_n), "bfloat16")
            acc_o = T.alloc_fragment((block_h, head_dim), accum_dtype)
            scores_max = T.alloc_fragment((block_h,), accum_dtype)
            scores_max_prev = T.alloc_fragment((block_h,), accum_dtype)
            scores_scale = T.alloc_fragment((block_h,), accum_dtype)
            scores_sum = T.alloc_fragment((block_h,), accum_dtype)
            logsum = T.alloc_fragment((block_h,), accum_dtype)

            cur_kv_head = hid // heads_per_kv_block
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for i, d in T.Parallel(block_h, head_dim):
                qh = hid * valid_block_h + i
                qh_safe = T.min(qh, q_heads - 1)
                q_shared[i, d] = T.if_then_else(
                    i < valid_block_h,
                    Q[bid, qh_safe, d],
                    T.cast(0, "int8"),
                )

            for block_idx in T.Pipelined(T.ceildiv(max_seqlen_k, block_n), num_stages=num_stages):
                if block_idx * block_n < SEQ_LENS[bid]:
                    for n, d in T.Parallel(block_n, head_dim):
                        pos = block_idx * block_n + n
                        token = PAGE_TABLE[bid, pos]
                        k_shared[n, d] = T.if_then_else(
                            pos < SEQ_LENS[bid],
                            K[token, cur_kv_head, d],
                            T.cast(0, "int8"),
                        )
                        v_shared[n, d] = T.if_then_else(
                            pos < SEQ_LENS[bid],
                            V[token, cur_kv_head, d],
                            T.cast(0, "bfloat16"),
                        )

                    T.clear(acc_i32)
                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_i32,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    for i, n in T.Parallel(block_h, block_n):
                        qh = hid * valid_block_h + i
                        qh_safe = T.min(qh, q_heads - 1)
                        pos = block_idx * block_n + n
                        dequant = Q_SCALE[qh_safe] * K_SCALE[cur_kv_head] * softmax_scale_log2
                        acc_s[i, n] = T.if_then_else(
                            (i < valid_block_h) & (pos < SEQ_LENS[bid]),
                            T.cast(acc_i32[i, n], accum_dtype) * dequant,
                            -T.infinity(accum_dtype),
                        )

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_h):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                    for i, n in T.Parallel(block_h, block_n):
                        acc_s[i, n] = T.exp2(acc_s[i, n] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_h):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, d in T.Parallel(block_h, head_dim):
                        acc_o[i, d] *= scores_scale[i]
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, d in T.Parallel(block_h, head_dim):
                qh = hid * valid_block_h + i
                if i < valid_block_h:
                    O[bid, qh, d] = T.cast(acc_o[i, d] / logsum[i], "bfloat16")

    return kernel


def _compiled_paged_decode(
    batch: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    cache_tokens: int,
    max_seqlen_k: int,
):
    block_n = 128
    block_h = 64
    num_stages = 3
    threads = 128
    max_seqlen_k = _align_up(max_seqlen_k, block_n)
    key = (
        "paged_gqa_decode_w8a8_bf16",
        batch,
        q_heads,
        kv_heads,
        head_dim,
        cache_tokens,
        max_seqlen_k,
        block_n,
        block_h,
        num_stages,
        threads,
    )
    return compile_tilelang(
        key,
        lambda: _paged_decode_program(
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            cache_tokens=cache_tokens,
            max_seqlen_k=max_seqlen_k,
            block_n=block_n,
            block_h=block_h,
            num_stages=num_stages,
            threads=threads,
        ),
        (7,),
    )


def paged_gqa_decode_w8a8_bf16(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
) -> torch.Tensor:
    actual_batch = q.shape[0]
    bucket_batch = _bucket_batch(actual_batch)
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_n = 128
    aligned_seqlen = _align_up(page_table.shape[1], block_n)
    if actual_batch != bucket_batch:
        q_bucket = torch.empty(
            (bucket_batch, q.shape[1], q.shape[2]),
            device=q.device,
            dtype=q.dtype,
        )
        q_bucket[:actual_batch] = q
        q_bucket[actual_batch:] = 0
        q = q_bucket

    if page_table.shape[1] != aligned_seqlen:
        padded_page_table = torch.empty(
            (bucket_batch, aligned_seqlen),
            device=page_table.device,
            dtype=page_table.dtype,
        )
        padded_page_table[:actual_batch, : page_table.shape[1]] = page_table
        padded_page_table[:actual_batch, page_table.shape[1] :] = 0
        if actual_batch != bucket_batch:
            padded_page_table[actual_batch:] = 0
        page_table = padded_page_table
    elif actual_batch != bucket_batch:
        padded_page_table = torch.empty(
            (bucket_batch, aligned_seqlen),
            device=page_table.device,
            dtype=page_table.dtype,
        )
        padded_page_table[:actual_batch] = page_table
        padded_page_table[actual_batch:] = 0
        page_table = padded_page_table
    else:
        page_table = page_table.contiguous()
    if actual_batch != bucket_batch:
        seq_lens_bucket = torch.empty(bucket_batch, device=seq_lens.device, dtype=seq_lens.dtype)
        seq_lens_bucket[:actual_batch] = seq_lens
        seq_lens_bucket[actual_batch:] = 0
        seq_lens = seq_lens_bucket
    else:
        seq_lens = seq_lens.contiguous()
    q_scale = q_scale.reshape(-1).contiguous().to(device=q.device, dtype=torch.float32)
    k_scale = k_scale.reshape(-1).contiguous().to(device=q.device, dtype=torch.float32)
    kernel = _compiled_paged_decode(
        batch=bucket_batch,
        q_heads=q.shape[1],
        kv_heads=k_cache.shape[1],
        head_dim=q.shape[2],
        cache_tokens=k_cache.shape[0],
        max_seqlen_k=aligned_seqlen,
    )
    out = profile_cuda(
        f"paged_gqa_decode[bs={bucket_batch},q={q.shape[1]},kv={k_cache.shape[1]},seq={aligned_seqlen}]",
        kernel,
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        q_scale,
        k_scale,
    )
    return out[:actual_batch]
