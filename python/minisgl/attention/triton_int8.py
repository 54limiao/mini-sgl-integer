"""
Integer Triton attention backend for mini-sglang.

End-to-end flow
---------------
1. bf16 Q / K / V arrive from model layers.
2. Prefill path dynamically quantizes Q / K / V to int8 using prototype-style
   multiplier/shift parameters and runs integer-only attention compute.
3. Decode path uses standard bf16 Triton decode (memory-bound; avoids KV int8
   cache quantization overhead/OOM risk).
4. Output is written back as bf16.

This backend requires q-scale fusion (1/sqrt(head_dim) folded into q_norm) for
the integer prefill path, so ``sm_scale = 1.0``.
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
class Int8CaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class Int8Metadata(BaseAttnMetadata):
    """Metadata for the int8 Triton attention backend."""

    kv_indptr: torch.Tensor  # [batch_size + 1] cumulative KV lengths
    kv_indices: torch.Tensor  # [total_kv_tokens] physical page indices
    qo_indptr: Optional[torch.Tensor]  # [batch_size + 1] (prefill only)
    max_seqlen_q: int
    max_seqlen_k: int
    causal: bool = True

    def get_last_indices(self, bs: int) -> torch.Tensor:
        if self.qo_indptr is not None:
            return self.qo_indptr[1 : 1 + bs] - 1
        return torch.arange(bs, device=self.kv_indptr.device)


class Int8Backend(BaseAttnBackend):
    """Integer Triton attention backend.

    Heavy prefill matmuls (QK^T, attn·V) run in int8/int32 through the integer
    reference kernel path. Decode remains bf16 Triton for memory/perf stability.
    """

    def __init__(self, config: "ModelConfig", kvcache: "BaseKVCache") -> None:
        from minisgl.kernel.triton.int8_attention_kernels import (
            int8_context_attention_fwd,
            quantize_per_tensor_int8,
            quantize_per_tensor_int8_ms,
        )
        from minisgl.kernel.triton.attention_kernels import (
            decode_attention_fwd as _fp_decode_fwd,
        )

        self.config = config
        self.kvcache = kvcache
        self.device = kvcache.device

        # Head info
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(config.num_kv_heads, tp_size)

        # Backend capability flag consumed by integer attention layers.
        self.requires_q_scale_fusion = True

        # Decode path still calls fp Triton decode; with fusion active,
        # q is already scaled so sm_scale must remain 1.0.
        self.sm_scale = 1.0
        logger.info("Int8Backend: sm_scale=1.0 (backend requires q-scale fusion)")

        # CUDA graph support
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.capture: Int8CaptureData | None = None
        self.max_seq_len = 0

        # Store kernel functions
        self._quantize = quantize_per_tensor_int8
        self._quantize_ms = quantize_per_tensor_int8_ms
        self._prefill_fwd = int8_context_attention_fwd
        self._fp_decode_fwd = _fp_decode_fwd

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, Int8Metadata)

        # 1) Store K, V to cache (bf16)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        # 2) Prepare output tensor
        output = torch.empty_like(q)

        if batch.is_decode:
            self._forward_decode(q, layer_id, metadata, output)
        else:
            k_for_prefill = (
                k.view(-1, self.kv_head_local, self.config.head_dim) if k.ndim == 2 else k
            )
            v_for_prefill = (
                v.view(-1, self.kv_head_local, self.config.head_dim) if v.ndim == 2 else v
            )
            k_int8, k_mul, k_shift = self._quantize_ms(k_for_prefill)
            v_int8, v_mul, v_shift = self._quantize_ms(v_for_prefill)
            self._forward_prefill(
                q=q,
                k_int8=k_int8,
                v_int8=v_int8,
                k_mul=k_mul,
                k_shift=k_shift,
                v_mul=v_mul,
                v_shift=v_shift,
                metadata=metadata,
                output=output,
            )

        return output

    def _forward_decode(
        self,
        q: torch.Tensor,
        layer_id: int,
        metadata: Int8Metadata,
        output: torch.Tensor,
    ) -> None:
        # Decode is memory-bound; use standard bf16 Triton decode.
        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)
        k_cache = k_cache.view(-1, k_cache.shape[-2], k_cache.shape[-1])
        v_cache = v_cache.view(-1, v_cache.shape[-2], v_cache.shape[-1])

        self._fp_decode_fwd(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            att_out=output,
            kv_indptr=metadata.kv_indptr,
            kv_indices=metadata.kv_indices,
            sm_scale=self.sm_scale,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k_int8: torch.Tensor,
        v_int8: torch.Tensor,
        k_mul: torch.Tensor,
        k_shift: torch.Tensor,
        v_mul: torch.Tensor,
        v_shift: torch.Tensor,
        metadata: Int8Metadata,
        output: torch.Tensor,
    ) -> None:
        q_int8, q_mul, q_shift = self._quantize_ms(q)

        assert metadata.qo_indptr is not None
        bs = metadata.qo_indptr.shape[0] - 1
        b_start_loc = metadata.qo_indptr[:bs]
        b_seq_len = metadata.qo_indptr[1 : bs + 1] - metadata.qo_indptr[:bs]

        self._prefill_fwd(
            q_int8=q_int8,
            k_int8=k_int8,
            v_int8=v_int8,
            o=output,
            q_mul=q_mul,
            q_shift=q_shift,
            k_mul=k_mul,
            k_shift=k_shift,
            v_mul=v_mul,
            v_shift=v_shift,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            max_input_len=metadata.max_seqlen_q,
            is_causal=metadata.causal,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        max_seqlen_k = max(seqlens_k)

        device = self.device
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        kv_indptr = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
        kv_indptr = kv_indptr.to(device, non_blocking=True)

        page_table = get_global_ctx().page_table
        kv_indices = torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs])

        qo_indptr = None
        causal = True

        if batch.is_prefill:
            if max_seqlen_q == 1:
                qo_indptr = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
            elif all(l == 0 for l in cached_lens):
                qo_indptr = kv_indptr.clone()
            else:
                qo_indptr = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(dim=0)
                qo_indptr = qo_indptr.to(device, non_blocking=True)
        else:
            qo_indptr = None

        batch.attn_metadata = Int8Metadata(
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=qo_indptr,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialised."
        max_bs = max(bs_list)
        capture = Int8CaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        self.max_seq_len = max_seq_len
        capture.cu_seqlens_k.copy_(
            torch.arange(0, max_bs + 1, dtype=torch.int32, device=self.device) * max_seq_len
        )
        capture.page_table = capture.page_table.view(-1)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert bs in self.capture_bs and self.capture
        assert batch.is_decode, "CUDA graph capture only supports decode mode"

        capture = self.capture
        kv_indptr = capture.cu_seqlens_k[: bs + 1]
        total_tokens = int(capture.cu_seqlens_k[bs].item())
        kv_indices = capture.page_table[:total_tokens]

        batch.attn_metadata = Int8Metadata(
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=None,
            max_seqlen_q=1,
            max_seqlen_k=self.max_seq_len,
            causal=True,
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata = batch.attn_metadata
        bs = batch.padded_size
        assert isinstance(metadata, Int8Metadata)
        assert self.capture is not None and bs in self.capture_bs
        assert batch.is_decode

        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.kv_indptr)
        total_tokens = int(metadata.kv_indptr[bs].item())
        self.capture.page_table[:total_tokens].copy_(metadata.kv_indices)


__all__ = ["Int8Backend"]
