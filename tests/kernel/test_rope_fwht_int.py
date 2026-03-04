from __future__ import annotations

import torch

from minisgl.kernel.fixed_point import to_fixed
from minisgl.kernel.fixed_point.fwht_int import fwht_int_q15
from minisgl.kernel.fixed_point.rope_fwht_int import rope_fwht_int_q15
from minisgl.kernel.fixed_point.rope_int import TRIG_Q15_SCALE, rope_int_q15
from minisgl.layers.rotary_integer import RotaryEmbeddingInteger

_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1


def _round_shift_right(x: torch.Tensor, shift: int) -> torch.Tensor:
    bias = 1 << (shift - 1)
    return torch.where(x >= 0, (x + bias) >> shift, (x - bias) >> shift)


def _rope_ref_q15(
    positions: torch.Tensor,
    query_q15: torch.Tensor,
    key_q15: torch.Tensor,
    cos_sin_cache_q15: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = query_q15.shape[-1]
    half_dim = head_dim // 2

    trig = cos_sin_cache_q15.index_select(0, positions.to(torch.long))
    cos_q15 = trig[:, :half_dim].to(torch.int64)[:, None, :]
    sin_q15 = trig[:, half_dim:].to(torch.int64)[:, None, :]

    def _apply(x_q15: torch.Tensor) -> torch.Tensor:
        x_lo = x_q15[..., :half_dim].to(torch.int64)
        x_hi = x_q15[..., half_dim:].to(torch.int64)

        y_lo = _round_shift_right(x_lo * cos_q15 - x_hi * sin_q15, 15)
        y_hi = _round_shift_right(x_lo * sin_q15 + x_hi * cos_q15, 15)
        y = torch.cat([y_lo, y_hi], dim=-1)
        return torch.clamp(y, _I32_MIN, _I32_MAX).to(torch.int32)

    return _apply(query_q15), _apply(key_q15)


def _fwht_ref_q15(x_q15: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    head_dim = x_q15.shape[-1]
    idx = torch.arange(head_dim, device=x_q15.device)

    out = x_q15.to(torch.int64)
    step = 1

    while step < head_dim:
        partner_idx = idx ^ step
        partner = out.index_select(-1, partner_idx)
        lower = ((idx & step) == 0).view(*([1] * (out.ndim - 1)), head_dim)
        out = torch.where(lower, out + partner, partner - out)
        out = torch.clamp(out, _I32_MIN, _I32_MAX)
        step <<= 1

    if normalize:
        scale_q15 = int(round((1.0 / (head_dim**0.5)) * (1 << 15)))
        out = _round_shift_right(out * scale_q15, 15)
        out = torch.clamp(out, _I32_MIN, _I32_MAX)

    return out.to(torch.int32)


class TestRopeFwhtInt:
    def test_rotary_forward_q15_packed_io_shapes(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        batch, num_qo_heads, num_kv_heads, head_dim = 2, 4, 2, 64
        rotary = RotaryEmbeddingInteger(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=64,
            base=10000,
        )

        q_q15 = to_fixed(torch.randn(batch, num_qo_heads * head_dim, device=device, dtype=torch.float32))
        k_q15 = to_fixed(torch.randn(batch, num_kv_heads * head_dim, device=device, dtype=torch.float32))
        positions = torch.randint(0, 64, (batch,), device=device, dtype=torch.int32)

        q_out_q15, k_out_q15 = rotary.forward_q15(positions, q_q15, k_q15)

        assert q_out_q15.dtype == torch.int32
        assert k_out_q15.dtype == torch.int32
        assert q_out_q15.shape == (batch, num_qo_heads, head_dim)
        assert k_out_q15.shape == (batch, num_kv_heads * head_dim)

    def test_rope_int_matches_integer_reference(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        batch, num_qo_heads, num_kv_heads, head_dim = 4, 6, 2, 64
        q = torch.randn(batch, num_qo_heads, head_dim, device=device, dtype=torch.float32) * 0.2
        k = torch.randn(batch, num_kv_heads, head_dim, device=device, dtype=torch.float32) * 0.2

        q_q15 = to_fixed(q)
        k_q15 = to_fixed(k)
        positions = torch.randint(0, 128, (batch,), device=device, dtype=torch.int32)

        rotary = RotaryEmbeddingInteger(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=128,
            base=10000,
        )
        cos_sin_cache_q15 = torch.clamp(
            torch.round(rotary._cos_sin_cache * TRIG_Q15_SCALE),
            -(1 << 15),
            (1 << 15) - 1,
        ).to(torch.int32)
        if cos_sin_cache_q15.device != q_q15.device:
            cos_sin_cache_q15 = cos_sin_cache_q15.to(q_q15.device)

        out_q, out_k = rope_int_q15(positions, q_q15, k_q15, cos_sin_cache_q15)
        ref_q, ref_k = _rope_ref_q15(positions, q_q15, k_q15, cos_sin_cache_q15)

        assert torch.equal(out_q, ref_q)
        assert torch.equal(out_k, ref_k)

    def test_fwht_int_matches_integer_reference(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        x_q15 = torch.randint(
            -(1 << 20),
            (1 << 20),
            (8, 128),
            device=device,
            dtype=torch.int32,
        )

        out = fwht_int_q15(x_q15, normalize=True)
        ref = _fwht_ref_q15(x_q15, normalize=True)

        assert torch.equal(out, ref)

    def test_rope_fwht_matches_composed_path(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        batch, num_qo_heads, num_kv_heads, head_dim = 3, 4, 2, 64
        q = torch.randn(batch, num_qo_heads, head_dim, device=device, dtype=torch.float32) * 0.15
        k = torch.randn(batch, num_kv_heads, head_dim, device=device, dtype=torch.float32) * 0.15

        q_q15 = to_fixed(q)
        k_q15 = to_fixed(k)
        positions = torch.randint(0, 64, (batch,), device=device, dtype=torch.int32)

        rotary = RotaryEmbeddingInteger(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=64,
            base=10000,
        )
        cos_sin_cache_q15 = torch.clamp(
            torch.round(rotary._cos_sin_cache * TRIG_Q15_SCALE),
            -(1 << 15),
            (1 << 15) - 1,
        ).to(torch.int32)
        if cos_sin_cache_q15.device != q_q15.device:
            cos_sin_cache_q15 = cos_sin_cache_q15.to(q_q15.device)

        fused_q, fused_k = rope_fwht_int_q15(positions, q_q15, k_q15, cos_sin_cache_q15)
        rope_q, rope_k = rope_int_q15(positions, q_q15, k_q15, cos_sin_cache_q15)
        composed_q = fwht_int_q15(rope_q, normalize=True)
        composed_k = fwht_int_q15(rope_k, normalize=True)

        assert torch.equal(fused_q, composed_q)
        assert torch.equal(fused_k, composed_k)
