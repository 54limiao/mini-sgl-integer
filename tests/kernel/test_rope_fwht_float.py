from __future__ import annotations

import torch

from minisgl.layers.rotary import _fwht_last_dim


def test_fwht_preserves_qk_dot_products_float32() -> None:
    torch.manual_seed(0)
    q = torch.randn(24, 8, 128, dtype=torch.float32)
    k = torch.randn(24, 8, 128, dtype=torch.float32)

    q_h = _fwht_last_dim(q)
    k_h = _fwht_last_dim(k)

    before = torch.einsum("thd,shd->ths", q, k)
    after = torch.einsum("thd,shd->ths", q_h, k_h)

    assert torch.allclose(before, after, atol=1e-4, rtol=1e-4)


def test_fwht_preserves_qk_dot_products_bfloat16_close() -> None:
    torch.manual_seed(0)
    q = torch.randn(24, 8, 128, dtype=torch.bfloat16)
    k = torch.randn(24, 8, 128, dtype=torch.bfloat16)

    q_h = _fwht_last_dim(q)
    k_h = _fwht_last_dim(k)

    before = torch.einsum("thd,shd->ths", q.float(), k.float())
    after = torch.einsum("thd,shd->ths", q_h.float(), k_h.float())

    diff = (before - after).abs()
    assert diff.max().item() < 1e-1
    assert diff.mean().item() < 3e-2
