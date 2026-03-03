"""
Triton Attention Backend for mini-sglang.

Simplified implementation based on SGLang's Triton attention backend.
Supports both decode and prefill attention using Triton kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.kvcache import BaseKVCache
    from minisgl.models import ModelConfig


logger = init_logger(__name__)


@dataclass
class TritonCaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class TritonMetadata(BaseAttnMetadata):
    """Metadata for Triton attention backend."""
    kv_indptr: torch.Tensor          # KV cache cumulative lengths [batch_size + 1]
    kv_indices: torch.Tensor         # Physical KV page indices [total_kv_tokens]
    qo_indptr: Optional[torch.Tensor]  # Cumulative extend lengths [batch_size + 1] (prefill only)
    max_seqlen_q: int
    max_seqlen_k: int
    causal: bool = True

    def get_last_indices(self, bs: int) -> torch.Tensor:
        if self.qo_indptr is not None:
            # Prefill: output is packed [total_q_tokens, heads, dim].
            # Return the index of the *last* query token per sequence.
            return self.qo_indptr[1 : 1 + bs] - 1
        else:
            # Decode: output is [bs, heads, dim], one row per sequence.
            return torch.arange(bs, device=self.kv_indptr.device)


class TritonBackend(BaseAttnBackend):
    """Triton attention backend for mini-sglang."""

    def __init__(self, config: ModelConfig, kvcache: BaseKVCache) -> None:
        from minisgl.kernel.triton.attention_kernels import (
            decode_attention_fwd,
            prefill_attention_fwd,
        )

        self.config = config
        self.kvcache = kvcache
        self.device = kvcache.device

        # Head dimensions
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size)

        # Cache for indices
        self.kv_indptr_buffer = torch.zeros(
            1024, dtype=torch.int32, device=self.device
        )
        self.qo_indptr_buffer = torch.zeros(
            1024, dtype=torch.int32, device=self.device
        )

        # CUDA graph support
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.capture: TritonCaptureData | None = None
        self.max_seq_len = 0  # Added for CUDA graph capture

        # Attention scale
        self.scale = (self.config.head_dim ** -0.5)

        # Store kernel functions
        self.decode_attention_fwd = decode_attention_fwd
        self.prefill_attention_fwd = prefill_attention_fwd

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """
        Compute attention using Triton kernels.

        Process:
        1. Store K, V to KV cache
        2. Prepare indices
        3. Call appropriate kernel (decode or prefill)
        """
        metadata = batch.attn_metadata
        assert isinstance(metadata, TritonMetadata)

        # Step 1: Store K, V to cache
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        # Step 2: Get cached KV — reshape from [num_pages, page_size, num_kv_heads, head_dim]
        # to [num_pages*page_size, num_kv_heads, head_dim] so Triton strides are correct.
        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)
        k_cache = k_cache.view(-1, k_cache.shape[-2], k_cache.shape[-1])
        v_cache = v_cache.view(-1, v_cache.shape[-2], v_cache.shape[-1])

        # Step 3: Prepare output tensor
        output = torch.empty_like(q)

        # Step 4: Call appropriate kernel
        if batch.is_decode:
            # Decode: single token per sequence
            self.decode_attention_fwd(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                att_out=output,
                kv_indptr=metadata.kv_indptr,
                kv_indices=metadata.kv_indices,
                sm_scale=self.scale,
            )
        else:
            # Prefill: multiple tokens per sequence
            self.prefill_attention_fwd(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                output=output,
                qo_indptr=metadata.qo_indptr,
                kv_indptr=metadata.kv_indptr,
                kv_indices=metadata.kv_indices,
                sm_scale=self.scale,
                causal=metadata.causal,
            )

        return output

    def prepare_metadata(self, batch: Batch) -> None:
        """Prepare metadata for Triton attention backend."""
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        max_seqlen_k = max(seqlens_k)

        device = self.device
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        # Prepare KV indices
        kv_indptr = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
        kv_indptr = kv_indptr.to(device, non_blocking=True)

        page_table = get_global_ctx().page_table
        kv_indices = torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs])

        # Prepare QO indices (for prefill)
        qo_indptr = None
        causal = True

        if batch.is_prefill:
            if max_seqlen_q == 1:
                # All sequences have single token
                qo_indptr = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
            elif all(l == 0 for l in cached_lens):
                # Prefill with no cache hit
                qo_indptr = kv_indptr.clone()
            else:
                # Normal extend prefill with partial cache hit
                qo_indptr = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(dim=0)
                qo_indptr = qo_indptr.to(device, non_blocking=True)
        else:
            # Decode: qo_indptr is not needed
            qo_indptr = None

        batch.attn_metadata = TritonMetadata(
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=qo_indptr,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """Initialize CUDA graph capture (decode only for simplicity)."""
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = TritonCaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        # Save max_seq_len before reshaping the page table.
        self.max_seq_len = max_seq_len
        # Re-initialise cu_seqlens_k so that each sequence contributes max_seq_len
        # tokens during CUDA-graph capture (required by the decode kernel).
        capture.cu_seqlens_k.copy_(
            torch.arange(0, max_bs + 1, dtype=torch.int32, device=self.device) * max_seq_len
        )
        # Flatten the 2-D page table into a 1-D index buffer used as kv_indices.
        capture.page_table = capture.page_table.view(-1)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        """Prepare for CUDA graph capture."""
        bs = batch.size
        assert bs in self.capture_bs and self.capture
        assert batch.is_decode, "CUDA graph capture only supports decode mode"

        capture = self.capture
        # kv_indptr[i] = i * max_seq_len  (each sequence uses max_seq_len tokens during capture)
        kv_indptr = capture.cu_seqlens_k[: bs + 1]
        # Total KV tokens during capture = bs * max_seq_len
        total_tokens = int(capture.cu_seqlens_k[bs].item())
        # The flat page-table buffer serves as the kv_indices buffer (views into it are
        # updated in-place by prepare_for_replay before each graph replay).
        kv_indices = capture.page_table[:total_tokens]

        batch.attn_metadata = TritonMetadata(
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=None,
            max_seqlen_q=1,
            max_seqlen_k=self.max_seq_len,
            causal=True,
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        """Prepare for CUDA graph replay."""
        metadata = batch.attn_metadata
        bs = batch.padded_size
        assert isinstance(metadata, TritonMetadata)
        assert self.capture is not None and bs in self.capture_bs
        assert batch.is_decode, "CUDA graph replay only supports decode mode"

        # Copy the real kv_indptr values (cumulative sequence lengths) into the
        # capture buffer so the replayed kernel uses the correct lengths.
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.kv_indptr)
        # Copy the real page indices for all KV tokens into the flat index buffer.
        total_tokens = int(metadata.kv_indptr[bs].item())
        self.capture.page_table[:total_tokens].copy_(metadata.kv_indices)


__all__ = ["TritonBackend"]