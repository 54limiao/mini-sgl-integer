from __future__ import annotations

import torch

from .fwht_int import fwht_int_q15
from .rope_int import rope_int_q15


def rope_fwht_int_q15(
    positions: torch.Tensor,
    query_q15: torch.Tensor,
    key_q15: torch.Tensor,
    cos_sin_cache_q15: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rope_q15, k_rope_q15 = rope_int_q15(
        positions=positions,
        query_q15=query_q15,
        key_q15=key_q15,
        cos_sin_cache_q15=cos_sin_cache_q15,
    )
    q_out_q15 = fwht_int_q15(q_rope_q15, normalize=True)
    k_out_q15 = fwht_int_q15(k_rope_q15, normalize=True)
    return q_out_q15, k_out_q15


__all__ = ["rope_fwht_int_q15"]
