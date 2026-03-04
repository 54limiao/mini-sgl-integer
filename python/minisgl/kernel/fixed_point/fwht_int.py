from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1


@triton.jit
def _round_shift_right(x: tl.tensor, SHIFT: tl.constexpr):
    bias = 1 << (SHIFT - 1)
    return tl.where(x >= 0, (x + bias) >> SHIFT, (x - bias) >> SHIFT)


@triton.jit
def _fwht_stage_int_q15_kernel(
    src_ptr,
    dst_ptr,
    stride_src_b,
    stride_dst_b,
    n_cols,
    STEP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    src_row = src_ptr + row * stride_src_b
    dst_row = dst_ptr + row * stride_dst_b

    src_val = tl.load(src_row + offs, mask=mask, other=0).to(tl.int64)
    partner_offs = offs ^ STEP
    partner_val = tl.load(src_row + partner_offs, mask=mask, other=0).to(tl.int64)

    is_lower = (offs & STEP) == 0
    out = tl.where(is_lower, src_val + partner_val, partner_val - src_val)
    out = tl.minimum(tl.maximum(out, -2147483648), 2147483647)

    tl.store(dst_row + offs, out.to(tl.int32), mask=mask)


@triton.jit
def _fwht_scale_q15_kernel(
    src_ptr,
    dst_ptr,
    stride_src_b,
    stride_dst_b,
    n_cols,
    scale_q15,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    src_row = src_ptr + row * stride_src_b
    dst_row = dst_ptr + row * stride_dst_b

    val = tl.load(src_row + offs, mask=mask, other=0).to(tl.int64)
    scaled = _round_shift_right(val * scale_q15.to(tl.int64), SHIFT=15)
    scaled = tl.minimum(tl.maximum(scaled, -2147483648), 2147483647)

    tl.store(dst_row + offs, scaled.to(tl.int32), mask=mask)


def _round_shift_right_torch(x: torch.Tensor, shift: int) -> torch.Tensor:
    bias = 1 << (shift - 1)
    return torch.where(x >= 0, (x + bias) >> shift, (x - bias) >> shift)


def _fwht_int_q15_torch(x_q15: torch.Tensor, normalize: bool) -> torch.Tensor:
    head_dim = x_q15.shape[-1]
    flat = x_q15.reshape(-1, head_dim).to(torch.int64)

    idx = torch.arange(head_dim, device=x_q15.device)
    cur = flat
    step = 1
    while step < head_dim:
        partner_idx = idx ^ step
        partner = cur.index_select(1, partner_idx)
        lower = ((idx & step) == 0).unsqueeze(0)
        out = torch.where(lower, cur + partner, partner - cur)
        cur = torch.clamp(out, _I32_MIN, _I32_MAX)
        step <<= 1

    if normalize:
        scale_q15 = int(round((1.0 / math.sqrt(head_dim)) * (1 << 15)))
        cur = _round_shift_right_torch(cur * scale_q15, 15)
        cur = torch.clamp(cur, _I32_MIN, _I32_MAX)

    return cur.to(torch.int32).view_as(x_q15)


def fwht_int_q15(x_q15: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    if x_q15.dtype != torch.int32:
        raise TypeError(f"x_q15 must be int32 Q15.16, got {x_q15.dtype}")
    if x_q15.ndim < 2:
        raise ValueError(f"x_q15 must have at least 2 dims, got shape {tuple(x_q15.shape)}")

    head_dim = x_q15.shape[-1]
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError(f"head_dim must be power-of-two, got {head_dim}")

    if not x_q15.is_cuda:
        return _fwht_int_q15_torch(x_q15, normalize=normalize)

    x_flat = x_q15.reshape(-1, head_dim).contiguous().clone()
    rows = x_flat.shape[0]
    workspace = torch.empty_like(x_flat)

    block = head_dim
    src = x_flat
    dst = workspace
    step = 1
    while step < head_dim:
        _fwht_stage_int_q15_kernel[(rows,)](
            src,
            dst,
            src.stride(0),
            dst.stride(0),
            head_dim,
            STEP=step,
            BLOCK=block,
        )
        src, dst = dst, src
        step <<= 1

    if normalize:
        out_flat = torch.empty_like(src)
        scale_q15 = int(round((1.0 / math.sqrt(head_dim)) * (1 << 15)))
        _fwht_scale_q15_kernel[(rows,)](
            src,
            out_flat,
            src.stride(0),
            out_flat.stride(0),
            head_dim,
            scale_q15,
            BLOCK=block,
        )
        return out_flat.view_as(x_q15)

    return src.view_as(x_q15)


__all__ = ["fwht_int_q15"]
