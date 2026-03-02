"""
Int8 Quantized Attention Backend using FlashInfer.

This implementation provides:
1. Per-token dynamic quantization of Q, K, V to int8
2. Dequantization before attention computation
3. Reuses FlashInfer backend for actual attention (supports Paged KV Cache)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch

from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
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
    """Metadata for Int8 attention (reuses FlashInfer's metadata structure)."""
    cu_seqlens_q_cpu: torch.Tensor  # on cpu
    cu_seqlens_k_cpu: torch.Tensor  # on cpu
    cu_seqlens_q_gpu: torch.Tensor  # on gpu
    indices: torch.Tensor  # on gpu
    last_page_len_cpu: torch.Tensor  # on cpu
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int  # currently only support page_size=1
    pos_encoding_mode: str
    seq_lens_cpu: torch.Tensor  # on cpu
    dtype: torch.dtype
    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized: bool = False

    def __post_init__(self) -> None:
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class Int8Backend(BaseAttnBackend):
    """Int8 quantized attention backend using FlashInfer."""

    def __init__(self, config: ModelConfig, kvcache: BaseKVCache) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = kvcache
        self.device = kvcache.device

        # Allocate workspace buffer for FlashInfer
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )

        # Create wrappers (same as FlashInferBackend)
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self._use_tensor_cores(),
            kv_layout="NHD",
            backend="fa2",
        )

        # Reuse int workspace buffer
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        # Head dimensions
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size)

        # Cache for ones tensor
        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)

        # CUDA graph support
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: Int8CaptureData | None = None

        # Attention scale
        self.scale = (self.config.head_dim ** -0.5)

    def _use_tensor_cores(self) -> bool:
        """Determine if tensor cores should be used."""
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        """Get cached ones tensor."""
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        next_len = max(bs, 1)
        while next_len < bs:
            next_len *= 2
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    @staticmethod
    def _initialize_metadata_once(metadata: Int8Metadata) -> None:
        """Initialize metadata for FlashInfer wrapper (same as FlashInferBackend)."""
        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )

    def _quantize_to_int8(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-tensor dynamic quantization to int8.

        Args:
            x: Input tensor [total_tokens, num_heads, head_dim]

        Returns:
            (quantized_tensor, scale)
            - quantized_tensor: int8 tensor [total_tokens, num_heads, head_dim]
            - scale: scalar scale factor
        """
        # Compute max absolute value across the entire tensor
        x_abs_max = torch.abs(x).amax()
        # Compute scale factor (scalar)
        scale = x_abs_max.clamp(min=1e-5) / 127.0
        # Quantize
        x_quantized = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
        return x_quantized, scale

    def _dequantize_from_int8(
        self, x_int8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Dequantize from int8.

        Args:
            x_int8: int8 tensor [total_tokens, num_heads, head_dim]
            scale: scalar scale factor
            dtype: output dtype

        Returns:
            Dequantized tensor
        """
        return (x_int8.to(torch.float32) * scale).to(dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """
        Compute attention with int8 quantization.

        Process:
        1. Quantize Q, K, V to int8 (per-token dynamic)
        2. Dequantize back to original dtype
        3. Use FlashInfer for actual attention computation
        4. Store dequantized K, V to KV cache

        Note: Currently this is for testing the quantization pipeline.
        Future optimization: keep K, V in int8 in cache and use int8 kernels.
        """
        metadata = batch.attn_metadata
        assert isinstance(metadata, Int8Metadata)

        # Step 1: Quantize Q, K, V to int8
        q_int8, q_scale = self._quantize_to_int8(q)
        k_int8, k_scale = self._quantize_to_int8(k)
        v_int8, v_scale = self._quantize_to_int8(v)

        # Step 2: Dequantize back to original dtype
        k_deq = self._dequantize_from_int8(k_int8, k_scale, k.dtype)
        v_deq = self._dequantize_from_int8(v_int8, v_scale, v.dtype)
        q_deq = self._dequantize_from_int8(q_int8, q_scale, q.dtype)

        # Step 3: Store K, V to cache
        self.kvcache.store_kv(k_deq, v_deq, batch.out_loc, layer_id)

        # Step 4: Compute attention using FlashInfer
        self._initialize_metadata_once(metadata)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        return metadata.wrapper.run(q=q_deq, paged_kv_cache=kv_cache)

    def prepare_metadata(self, batch: Batch) -> None:
        """Prepare metadata (same as FlashInferBackend)."""
        reqs = batch.padded_reqs
        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.device
        seq_len_cpu = torch.tensor(seqlens_k, **cpu_kwargs)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **cpu_kwargs)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(dim=0)

        page_table = get_global_ctx().page_table
        batch.attn_metadata = Int8Metadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            indices=torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs]),
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """Initialize CUDA graph capture (same as FlashInferBackend)."""
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = Int8CaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        capture.page_table = capture.page_table.view(-1)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        """Prepare for CUDA graph capture (same as FlashInferBackend)."""
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self._use_tensor_cores(),
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        self.graph_wrappers[bs]._backend = "fa2"
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, Int8Metadata)
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        """Prepare for CUDA graph replay (same as FlashInferBackend)."""
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, Int8Metadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)


__all__ = ["Int8Backend"]