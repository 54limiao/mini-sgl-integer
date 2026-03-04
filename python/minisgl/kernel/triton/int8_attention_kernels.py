"""
Integer-8 Triton attention kernels for mini-sglang.

This file provides two paths:
1) Prefill: prototype-style integer-only attention (except final bf16 output writeback)
2) Decode: existing int8 kernel (kept for compatibility, not used by v2 backend)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

import torch as _torch

_is_hip = _torch.version.hip is not None
_is_cuda = _torch.cuda.is_available() and not _is_hip
CUDA_CAPABILITY = (
    _torch.cuda.get_device_capability() if (_is_cuda or _is_hip) else (0, 0)
)

FIXED_POINT_SCALE = 65536
LOG2E_Q15_16 = 94548


def _exp2_lut_i32(device: torch.device) -> torch.Tensor:
    lut = torch.exp2(torch.linspace(0, 1, 1024, dtype=torch.float32, device=device))
    return (lut * (1 << 29) + 0.5).to(torch.int32)


def quantize_per_tensor_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_abs_max = x.abs().amax().clamp(min=1e-5)
    scale = x_abs_max / 127.0
    x_int8 = (x.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


def quantize_per_tensor_int8_ms(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prototype-style int8 quantization returning (int8, multiplier, shift)."""
    max_abs = x.abs().amax().clamp(min=1e-8)
    largest_exp2 = torch.ceil(torch.log2(max_abs))
    shift = (31 - largest_exp2).to(torch.int32)
    multiplier_f = max_abs * torch.pow(
        torch.tensor(2.0, device=x.device, dtype=torch.float32), shift.to(torch.float32)
    )
    multiplier = torch.clamp(multiplier_f, max=torch.iinfo(torch.int32).max).to(torch.int32)
    q = ((x.float() / max_abs) * 127.0).round().clamp(-128, 127).to(torch.int8)
    return q, multiplier, (shift + 7).to(torch.int32)


@triton.jit
def _int8_prefill_kernel_integer(
    Q,
    K,
    V,
    Out,
    Q_mul,
    Q_shift,
    K_mul,
    K_shift,
    V_mul,
    V_shift,
    EXP2_LUT,
    B_Start_Loc,
    B_Seqlen,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kbs: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vbs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_obs: tl.constexpr,
    stride_oh: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    q_offset = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    if q_offset >= cur_batch_seq_len:
        return

    cur_batch_start = tl.load(B_Start_Loc + cur_batch)
    q_abs = cur_batch_start + q_offset

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < Lk

    off_q = q_abs * stride_qbs + cur_head * stride_qh + offs_d
    q_i8 = tl.load(Q + off_q, mask=mask_d, other=0).to(tl.int8)
    q_i32 = q_i8.to(tl.int32)

    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    q_mul = tl.load(Q_mul)
    q_shift = tl.load(Q_shift)
    k_mul = tl.load(K_mul)
    k_shift = tl.load(K_shift)
    v_mul = tl.load(V_mul)
    v_shift = tl.load(V_shift)

    combined_mul = ((q_mul.to(tl.int64) * k_mul.to(tl.int64)) >> 32).to(tl.int32)
    score_shift = (q_shift + k_shift - 32) - 16

    seq_end = cur_batch_seq_len if not IS_CAUSAL else (q_offset + 1)

    row_max = tl.full([1], -2147483648, dtype=tl.int32)
    for start_n in range(0, seq_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_mask = (start_n + offs_n) < seq_end
        k_i8 = tl.load(
            k_ptrs + (cur_batch_start + start_n) * stride_kbs,
            mask=n_mask[None, :] & mask_d[:, None],
            other=0,
        ).to(tl.int8)
        k_i32 = k_i8.to(tl.int32)
        qk_i32 = tl.sum(q_i32[None, :] * k_i32.T, axis=1)

        round_offset = tl.where(score_shift > 0, (1 << (score_shift - 1)), 0)
        qk_q15 = tl.where(
            score_shift >= 0,
            ((qk_i32.to(tl.int64) * combined_mul.to(tl.int64) + round_offset) >> score_shift).to(
                tl.int32
            ),
            ((qk_i32.to(tl.int64) * combined_mul.to(tl.int64)) << (-score_shift)).to(tl.int32),
        )
        qk_q15 = tl.where(n_mask, qk_q15, -2147483648)
        row_max = tl.maximum(row_max, tl.max(qk_q15, axis=0))

    sum_exp = tl.zeros([1], dtype=tl.int64)
    for start_n in range(0, seq_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_mask = (start_n + offs_n) < seq_end
        k_i8 = tl.load(
            k_ptrs + (cur_batch_start + start_n) * stride_kbs,
            mask=n_mask[None, :] & mask_d[:, None],
            other=0,
        ).to(tl.int8)
        k_i32 = k_i8.to(tl.int32)
        qk_i32 = tl.sum(q_i32[None, :] * k_i32.T, axis=1)
        round_offset = tl.where(score_shift > 0, (1 << (score_shift - 1)), 0)
        qk_q15 = tl.where(
            score_shift >= 0,
            ((qk_i32.to(tl.int64) * combined_mul.to(tl.int64) + round_offset) >> score_shift).to(
                tl.int32
            ),
            ((qk_i32.to(tl.int64) * combined_mul.to(tl.int64)) << (-score_shift)).to(tl.int32),
        )
        qk_diff = qk_q15 - row_max
        y = ((qk_diff.to(tl.int64) * 94548) >> 16).to(tl.int32)
        y_int = y >> 16
        y_frac = y & 0xFFFF
        lut_idx = (y_frac >> 6) & 0x3FF
        exp2_frac = tl.load(EXP2_LUT + lut_idx)
        shift_amt = 13 - y_int
        shift_amt = tl.maximum(0, tl.minimum(31, shift_amt))
        x_exp = exp2_frac >> shift_amt
        x_exp = tl.where(n_mask, x_exp, 0)
        sum_exp += tl.sum(x_exp.to(tl.int64), axis=0)

    inv = ((1 << 31) - 1) // sum_exp
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.int32)
    for start_n in range(0, seq_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_mask = (start_n + offs_n) < seq_end
        k_i8 = tl.load(
            k_ptrs + (cur_batch_start + start_n) * stride_kbs,
            mask=n_mask[None, :] & mask_d[:, None],
            other=0,
        ).to(tl.int8)
        k_i32 = k_i8.to(tl.int32)
        qk_i32 = tl.sum(q_i32[None, :] * k_i32.T, axis=1)
        round_offset = tl.where(score_shift > 0, (1 << (score_shift - 1)), 0)
        qk_q15 = tl.where(
            score_shift >= 0,
            ((qk_i32.to(tl.int64) * combined_mul.to(tl.int64) + round_offset) >> score_shift).to(
                tl.int32
            ),
            ((qk_i32.to(tl.int64) * combined_mul.to(tl.int64)) << (-score_shift)).to(tl.int32),
        )
        qk_diff = qk_q15 - row_max
        y = ((qk_diff.to(tl.int64) * 94548) >> 16).to(tl.int32)
        y_int = y >> 16
        y_frac = y & 0xFFFF
        lut_idx = (y_frac >> 6) & 0x3FF
        exp2_frac = tl.load(EXP2_LUT + lut_idx)
        shift_amt = 13 - y_int
        shift_amt = tl.maximum(0, tl.minimum(31, shift_amt))
        x_exp = exp2_frac >> shift_amt
        x_exp = tl.where(n_mask, x_exp, 0)
        attn_q0_15 = ((x_exp.to(tl.int64) * inv) >> 16).to(tl.int32)

        v_i8 = tl.load(
            v_ptrs + (cur_batch_start + start_n) * stride_vbs,
            mask=n_mask[:, None] & mask_d[None, :],
            other=0,
        ).to(tl.int8)
        v_i32 = v_i8.to(tl.int32)
        acc += tl.sum(attn_q0_15[:, None] * v_i32, axis=0)

    v_shift_eff = v_shift - 1
    round_v = tl.where(v_shift_eff > 0, (1 << (v_shift_eff - 1)), 0)
    out_q15 = tl.where(
        v_shift_eff >= 0,
        ((acc.to(tl.int64) * v_mul.to(tl.int64) + round_v) >> v_shift_eff).to(tl.int32),
        ((acc.to(tl.int64) * v_mul.to(tl.int64)) << (-v_shift_eff)).to(tl.int32),
    )
    out_f32 = out_q15.to(tl.float32) * (1.0 / 65536.0)

    off_o = q_abs * stride_obs + cur_head * stride_oh + offs_d
    tl.store(Out + off_o, out_f32.to(tl.bfloat16), mask=mask_d)


def int8_context_attention_fwd(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    o: torch.Tensor,
    q_mul: torch.Tensor,
    q_shift: torch.Tensor,
    k_mul: torch.Tensor,
    k_shift: torch.Tensor,
    v_mul: torch.Tensor,
    v_shift: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    is_causal: bool = True,
) -> None:
    """Prototype-style integer-only prefill attention (torch reference path).

    This function intentionally has no ``sm_scale``: attention scaling is
    expected to be fused into q_norm / q projection upstream.
    """

    # Normalize layouts to [tokens, heads, head_dim].
    if q_int8.ndim != 3:
        raise ValueError("q_int8 must be 3D [tokens, qo_heads, head_dim]")

    head_dim = q_int8.shape[-1]
    if k_int8.ndim == 2:
        if k_int8.shape[-1] % head_dim != 0:
            raise ValueError("k_int8 last dim must be divisible by q head_dim")
        k_int8 = k_int8.view(k_int8.shape[0], -1, head_dim)
    elif k_int8.ndim != 3:
        raise ValueError("k_int8 must be 2D or 3D")

    if v_int8.ndim == 2:
        if v_int8.shape[-1] % head_dim != 0:
            raise ValueError("v_int8 last dim must be divisible by q head_dim")
        v_int8 = v_int8.view(v_int8.shape[0], -1, head_dim)
    elif v_int8.ndim != 3:
        raise ValueError("v_int8 must be 2D or 3D")

    out = o
    if out.ndim == 2:
        if out.shape[-1] != q_int8.shape[1] * head_dim:
            raise ValueError("2D output last dim must equal qo_heads * head_dim")
        out = out.view(out.shape[0], q_int8.shape[1], head_dim)
    elif out.ndim != 3:
        raise ValueError("output tensor must be 2D or 3D")

    def _apply_mul_shift_torch(x: torch.Tensor, multiplier: int, shift: int) -> torch.Tensor:
        x64 = x.to(torch.int64)
        m64 = int(multiplier)
        s = int(shift)
        if s >= 0:
            round_offset = (1 << (s - 1)) if s > 0 else 0
            return ((x64 * m64 + round_offset) >> s)
        return ((x64 * m64) << (-s))

    def _exp_q15_16_torch(x_q15: torch.Tensor, exp2_lut: torch.Tensor) -> torch.Tensor:
        # x_q15 is expected <= 0
        y = ((x_q15.to(torch.int64) * 94548) >> 16).to(torch.int32)
        y_int = y >> 16
        y_frac = y & 0xFFFF
        lut_idx = ((y_frac >> 6) & 0x3FF).to(torch.int64)
        exp2_frac = exp2_lut[lut_idx]
        shift_amt = (13 - y_int).clamp(min=0, max=31)
        return (exp2_frac >> shift_amt).to(torch.int32)

    def _softmax_q15_16_torch(
        scores_q15: torch.Tensor,
        exp2_lut: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # scores_q15: [H, Lq, Lk] int32
        if mask is not None:
            scores_for_max = scores_q15.masked_fill(mask.unsqueeze(0), -(1 << 30))
        else:
            scores_for_max = scores_q15

        row_max = scores_for_max.max(dim=-1, keepdim=True).values
        x_diff = (scores_for_max.to(torch.int64) - row_max.to(torch.int64)).clamp(
            min=-(1 << 30), max=0
        ).to(torch.int32)
        x_exp = _exp_q15_16_torch(x_diff, exp2_lut)
        if mask is not None:
            x_exp = x_exp.masked_fill(mask.unsqueeze(0), 0)
        x_sum = x_exp.to(torch.int64).sum(dim=-1, keepdim=True).clamp(min=1)
        inv = ((1 << 31) - 1) // x_sum
        return ((x_exp.to(torch.int64) * inv) >> 16).to(torch.int32)  # Q0.15

    assert q_int8.is_cuda and k_int8.is_cuda and v_int8.is_cuda and out.is_cuda
    device = q_int8.device
    exp2_lut = _exp2_lut_i32(device)

    q_mul_i = int(q_mul.item())
    q_shift_i = int(q_shift.item())
    k_mul_i = int(k_mul.item())
    k_shift_i = int(k_shift.item())
    v_mul_i = int(v_mul.item())
    v_shift_i = int(v_shift.item())

    combined_mul = int((int(q_mul_i) * int(k_mul_i)) >> 32)
    combined_shift = int(q_shift_i + k_shift_i - 32)
    score_shift = int(combined_shift - 16)

    num_qo_heads = q_int8.shape[1]
    num_kv_heads = k_int8.shape[1]
    kv_group_num = num_qo_heads // num_kv_heads
    kv_head_map = torch.arange(num_qo_heads, device=device, dtype=torch.int64) // kv_group_num

    bs = int(b_seq_len.shape[0])
    for b in range(bs):
        start = int(b_start_loc[b].item())
        seqlen = int(b_seq_len[b].item())
        if seqlen <= 0:
            continue

        q_b = q_int8[start : start + seqlen]  # [L, Hq, D]
        k_b = k_int8[start : start + seqlen]  # [L, Hk, D]
        v_b = v_int8[start : start + seqlen]  # [L, Hk, D]

        k_b = k_b[:, kv_head_map, :]  # [L, Hq, D]
        v_b = v_b[:, kv_head_map, :]  # [L, Hq, D]

        q_h = q_b.permute(1, 0, 2).to(torch.int32)  # [Hq, L, D]
        k_h = k_b.permute(1, 0, 2).to(torch.int32)  # [Hq, L, D]
        v_h = v_b.permute(1, 0, 2).to(torch.int32)  # [Hq, L, D]

        # Integer-only qk: avoid CUDA int32 matmul (not implemented)
        qk_i32 = torch.empty((num_qo_heads, seqlen, seqlen), device=device, dtype=torch.int32)
        for t in range(seqlen):
            q_vec = q_h[:, t : t + 1, :]  # [Hq, 1, D]
            qk_i32[:, t, :] = (q_vec * k_h).sum(dim=-1, dtype=torch.int64).to(torch.int32)

        scores_q15 = _apply_mul_shift_torch(qk_i32, combined_mul, score_shift)

        causal_mask = None
        if is_causal:
            causal_mask = torch.triu(
                torch.ones((seqlen, seqlen), device=device, dtype=torch.bool),
                diagonal=1,
            )

        attn_q0_15 = _softmax_q15_16_torch(scores_q15, exp2_lut, mask=causal_mask)  # [Hq, L, L]
        # Integer-only av: avoid CUDA int32 matmul (not implemented)
        acc = torch.empty((num_qo_heads, seqlen, v_h.shape[-1]), device=device, dtype=torch.int64)
        v_h_64 = v_h.to(torch.int64)
        for t in range(seqlen):
            w = attn_q0_15[:, t, :].to(torch.int64)  # [Hq, L]
            acc[:, t, :] = (w[:, :, None] * v_h_64).sum(dim=1)

        out_q15 = _apply_mul_shift_torch(acc, v_mul_i, v_shift_i - 1)
        out_bf16 = (out_q15.to(torch.float32) / FIXED_POINT_SCALE).to(torch.bfloat16)

        out[start : start + seqlen].copy_(out_bf16.permute(1, 0, 2))


@triton.jit
def _int8_decode_kernel(
    Q,
    K_Buffer,
    V_Buffer,
    Out,
    Q_scale,
    K_scale,
    V_scale,
    kv_indptr,
    kv_indices,
    sm_scale,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kbs: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vbs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_obs: tl.constexpr,
    stride_oh: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    kv_start = tl.load(kv_indptr + cur_batch)
    kv_end = tl.load(kv_indptr + cur_batch + 1)
    seq_len = kv_end - kv_start

    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < Lk
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q_i8 = tl.load(Q + off_q, mask=mask_d, other=0)
    q_i32 = q_i8.to(tl.int32)

    qk_combined_scale = tl.load(Q_scale) * tl.load(K_scale) * sm_scale
    v_scale_val = tl.load(V_scale)

    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, seq_len, BLOCK_N):
        n_mask = (start_n + offs_n) < seq_len
        page_ids = tl.load(kv_indices + kv_start + start_n + offs_n, mask=n_mask, other=0)

        k_ptrs = K_Buffer + page_ids[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :]
        k_i8 = tl.load(k_ptrs, mask=n_mask[:, None] & mask_d[None, :], other=0)
        k_i32 = k_i8.to(tl.int32)

        qk_i32 = tl.sum(q_i32[None, :] * k_i32, axis=1)
        qk_f32 = qk_i32.to(tl.float32) * qk_combined_scale
        qk_f32 += tl.where(n_mask, 0.0, float("-inf"))

        m_ij = tl.max(qk_f32, 0)
        m_ij = tl.maximum(m_ij, m_i)
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk_f32 - m_ij)
        l_ij = tl.sum(p, 0)
        l_i_new = alpha * l_i + l_ij

        acc = acc * (alpha * l_i / l_i_new)

        v_ptrs = V_Buffer + page_ids[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
        v_i8 = tl.load(v_ptrs, mask=n_mask[:, None] & mask_d[None, :], other=0)
        v_f32 = v_i8.to(tl.float32) * v_scale_val

        p_norm = p / l_i_new
        acc += tl.sum(p_norm[:, None] * v_f32, axis=0)

        l_i = l_i_new
        m_i = m_ij

    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d
    tl.store(Out + off_o, acc.to(tl.bfloat16), mask=mask_d)


def int8_decode_attention_fwd(
    q_int8: torch.Tensor,
    k_cache_int8: torch.Tensor,
    v_cache_int8: torch.Tensor,
    o: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float = 1.0,
) -> None:
    bs = q_int8.shape[0]
    num_heads = q_int8.shape[1]
    Lk = q_int8.shape[-1]
    kv_group_num = q_int8.shape[1] // k_cache_int8.shape[-2]

    BLOCK_N = 64
    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    kv_indices_64 = kv_indices.to(torch.int64) if kv_indices.dtype != torch.int64 else kv_indices

    grid = (bs, num_heads)
    _int8_decode_kernel[grid](
        q_int8,
        k_cache_int8,
        v_cache_int8,
        o,
        q_scale,
        k_scale,
        v_scale,
        kv_indptr,
        kv_indices_64,
        sm_scale,
        stride_qbs=q_int8.stride(0),
        stride_qh=q_int8.stride(1),
        stride_kbs=k_cache_int8.stride(0),
        stride_kh=k_cache_int8.stride(1),
        stride_vbs=v_cache_int8.stride(0),
        stride_vh=v_cache_int8.stride(1),
        stride_obs=o.stride(0),
        stride_oh=o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=1,
        Lk=Lk,
    )


__all__ = [
    "quantize_per_tensor_int8",
    "quantize_per_tensor_int8_ms",
    "int8_context_attention_fwd",
    "int8_decode_attention_fwd",
]
