from __future__ import annotations

import torch
import triton
import triton.language as tl

TRIG_Q15_SCALE = 1 << 15
_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1


@triton.jit
def _round_shift_right(x: tl.tensor, SHIFT: tl.constexpr):
    bias = 1 << (SHIFT - 1)
    return tl.where(x >= 0, (x + bias) >> SHIFT, (x - bias) >> SHIFT)


@triton.jit
def _rope_int_q15_kernel(
    positions_ptr,
    cos_sin_ptr,
    x_in_ptr,
    x_out_ptr,
    stride_cos_b,
    stride_x_b,
    stride_x_h,
    stride_x_d,
    num_heads,
    HALF_DIM: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if head_idx >= num_heads:
        return

    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < HALF_DIM

    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    cos_row = cos_sin_ptr + pos * stride_cos_b

    cos_val = tl.load(cos_row + offs, mask=mask, other=32768).to(tl.int64)
    sin_val = tl.load(cos_row + HALF_DIM + offs, mask=mask, other=0).to(tl.int64)

    x_base = x_in_ptr + token_idx * stride_x_b + head_idx * stride_x_h
    out_base = x_out_ptr + token_idx * stride_x_b + head_idx * stride_x_h

    x_lo = tl.load(x_base + offs * stride_x_d, mask=mask, other=0).to(tl.int64)
    x_hi = tl.load(x_base + (HALF_DIM + offs) * stride_x_d, mask=mask, other=0).to(tl.int64)

    y_lo = _round_shift_right(x_lo * cos_val - x_hi * sin_val, SHIFT=15)
    y_hi = _round_shift_right(x_lo * sin_val + x_hi * cos_val, SHIFT=15)

    y_lo = tl.minimum(tl.maximum(y_lo, -2147483648), 2147483647)
    y_hi = tl.minimum(tl.maximum(y_hi, -2147483648), 2147483647)

    tl.store(out_base + offs * stride_x_d, y_lo.to(tl.int32), mask=mask)
    tl.store(out_base + (HALF_DIM + offs) * stride_x_d, y_hi.to(tl.int32), mask=mask)


def _as_3d_q15_with_head_dim(
    x: torch.Tensor,
    head_dim: int,
    name: str,
) -> tuple[torch.Tensor, bool]:
    if x.dtype != torch.int32:
        raise TypeError(f"{name} must be int32 Q15.16, got {x.dtype}")
    if x.ndim == 3:
        if x.shape[-1] != head_dim:
            raise ValueError(f"{name} last dim must be head_dim={head_dim}, got {x.shape[-1]}")
        return x, False
    if x.ndim == 2:
        if x.shape[-1] % head_dim != 0:
            raise ValueError(
                f"{name} last dim must be divisible by head_dim={head_dim}, got {x.shape[-1]}"
            )
        return x.view(x.shape[0], -1, head_dim), True
    raise ValueError(f"{name} must be 2D or 3D, got shape {tuple(x.shape)}")


def _round_shift_right_torch(x: torch.Tensor, shift: int) -> torch.Tensor:
    bias = 1 << (shift - 1)
    return torch.where(x >= 0, (x + bias) >> shift, (x - bias) >> shift)


def _rope_int_q15_torch(
    positions: torch.Tensor,
    query_q15: torch.Tensor,
    key_q15: torch.Tensor,
    cos_sin_cache_q15: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_3d, query_was_2d = _as_3d_q15_with_head_dim(query_q15, head_dim, "query_q15")
    key_3d, key_was_2d = _as_3d_q15_with_head_dim(key_q15, head_dim, "key_q15")

    half_dim = head_dim // 2
    pos_idx = positions.to(torch.long)
    trig = cos_sin_cache_q15.index_select(0, pos_idx)

    cos_q15 = trig[:, :half_dim].to(torch.int64)
    sin_q15 = trig[:, half_dim:head_dim].to(torch.int64)

    def _apply(x_3d: torch.Tensor) -> torch.Tensor:
        x_lo = x_3d[..., :half_dim].to(torch.int64)
        x_hi = x_3d[..., half_dim:head_dim].to(torch.int64)

        cos = cos_q15[:, None, :]
        sin = sin_q15[:, None, :]

        y_lo = _round_shift_right_torch(x_lo * cos - x_hi * sin, 15)
        y_hi = _round_shift_right_torch(x_lo * sin + x_hi * cos, 15)

        y = torch.cat([y_lo, y_hi], dim=-1)
        return torch.clamp(y, _I32_MIN, _I32_MAX).to(torch.int32)

    q_out = _apply(query_3d)
    k_out = _apply(key_3d)

    if query_was_2d:
        q_out = q_out.view(query_q15.shape)
    if key_was_2d:
        k_out = k_out.view(key_q15.shape)
    return q_out, k_out


def rope_int_q15(
    positions: torch.Tensor,
    query_q15: torch.Tensor,
    key_q15: torch.Tensor,
    cos_sin_cache_q15: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_q15.dtype != torch.int32:
        raise TypeError(f"query_q15 must be int32 Q15.16, got {query_q15.dtype}")
    if key_q15.dtype != torch.int32:
        raise TypeError(f"key_q15 must be int32 Q15.16, got {key_q15.dtype}")
    if cos_sin_cache_q15.dtype != torch.int32:
        raise TypeError(
            f"cos_sin_cache_q15 must be int32 Q1.15 cache, got {cos_sin_cache_q15.dtype}"
        )

    if query_q15.ndim not in (2, 3):
        raise ValueError(f"query_q15 must be 2D or 3D, got shape {tuple(query_q15.shape)}")
    if key_q15.ndim not in (2, 3):
        raise ValueError(f"key_q15 must be 2D or 3D, got shape {tuple(key_q15.shape)}")

    head_dim = query_q15.shape[-1] if query_q15.ndim == 3 else key_q15.shape[-1]
    if query_q15.ndim == 2 and key_q15.ndim == 2:
        candidates = [
            d
            for d in (64, 128, 256, 512)
            if query_q15.shape[-1] % d == 0 and key_q15.shape[-1] % d == 0
        ]
        if not candidates:
            raise ValueError(
                "Cannot infer head_dim from 2D query/key shapes; expected common divisors in "
                f"[64, 128, 256, 512], got {query_q15.shape[-1]} and {key_q15.shape[-1]}"
            )
        if len(candidates) > 1 and query_q15.shape[-1] not in candidates:
            raise ValueError(
                "Ambiguous head_dim for 2D query/key inputs. Please pass 3D tensors with explicit "
                f"head_dim. Candidate head_dims: {candidates}"
            )
        head_dim = query_q15.shape[-1] if query_q15.shape[-1] in candidates else candidates[0]

    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    query_3d, query_was_2d = _as_3d_q15_with_head_dim(query_q15, head_dim, "query_q15")
    key_3d, key_was_2d = _as_3d_q15_with_head_dim(key_q15, head_dim, "key_q15")

    if query_3d.shape[0] != key_3d.shape[0]:
        raise ValueError(
            f"query and key batch mismatch: {query_3d.shape[0]} vs {key_3d.shape[0]}"
        )

    batch_size = query_3d.shape[0]
    if positions.numel() != batch_size:
        raise ValueError(
            f"positions size mismatch: expected {batch_size}, got {positions.numel()}"
        )

    if not query_3d.is_cuda or not key_3d.is_cuda or not cos_sin_cache_q15.is_cuda:
        q_out, k_out = _rope_int_q15_torch(
            positions=positions,
            query_q15=query_q15,
            key_q15=key_q15,
            cos_sin_cache_q15=cos_sin_cache_q15,
            head_dim=head_dim,
        )
        return q_out, k_out

    if positions.device != query_3d.device:
        positions = positions.to(query_3d.device)
    if key_3d.device != query_3d.device:
        key_3d = key_3d.to(query_3d.device)
    if cos_sin_cache_q15.device != query_3d.device:
        cos_sin_cache_q15 = cos_sin_cache_q15.to(query_3d.device)

    positions = positions.to(torch.int32).contiguous()
    query_3d = query_3d.contiguous()
    key_3d = key_3d.contiguous()
    cos_sin_cache_q15 = cos_sin_cache_q15.contiguous()

    half_dim = head_dim // 2
    block_half = triton.next_power_of_2(half_dim)

    query_out = torch.empty_like(query_3d)
    key_out = torch.empty_like(key_3d)

    grid_q = (batch_size, query_3d.shape[1])
    _rope_int_q15_kernel[grid_q](
        positions,
        cos_sin_cache_q15,
        query_3d,
        query_out,
        cos_sin_cache_q15.stride(0),
        query_3d.stride(0),
        query_3d.stride(1),
        query_3d.stride(2),
        query_3d.shape[1],
        HALF_DIM=half_dim,
        BLOCK_HALF=block_half,
    )

    grid_k = (batch_size, key_3d.shape[1])
    _rope_int_q15_kernel[grid_k](
        positions,
        cos_sin_cache_q15,
        key_3d,
        key_out,
        cos_sin_cache_q15.stride(0),
        key_3d.stride(0),
        key_3d.stride(1),
        key_3d.stride(2),
        key_3d.shape[1],
        HALF_DIM=half_dim,
        BLOCK_HALF=block_half,
    )

    if query_was_2d:
        query_out = query_out.view(query_q15.shape)
    if key_was_2d:
        key_out = key_out.view(key_q15.shape)

    return query_out, key_out


__all__ = ["TRIG_Q15_SCALE", "rope_int_q15"]
