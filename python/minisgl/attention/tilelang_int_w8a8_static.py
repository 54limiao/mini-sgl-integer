from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.kernel import store_cache
from minisgl.kernel.tilelang import (
    hadamard_quant_i8,
    paged_gqa_decode_w8a8_bf16,
    static_quantize_i8,
)
from minisgl.quant import get_quant_context
from minisgl.utils import div_even

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class TileLangIntMetadata(BaseAttnMetadata):
    last_indices: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seqlen_k: int

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


@dataclass
class TileLangIntCaptureData(BaseCaptureData):
    pass


class TileLangIntW8A8StaticBackend(BaseAttnBackend):
    """TileLang int8 attention backend with bf16 value/output semantics.

    Q/K are quantized at the attention boundary and the decode kernel computes
    QK with int8 GEMM before applying the static dequant scales. V remains in
    the existing bf16 KV cache for bf16 PV/output semantics.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        tp_size = get_tp_info().size
        self.qo_heads = div_even(config.num_qo_heads, tp_size)
        self.kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.head_dim = config.head_dim
        self.page_table_block = int(os.environ.get("MINISGL_INT_ATTN_BUCKET", "2048"))
        self.capture: TileLangIntCaptureData | None = None
        self.capture_bs: List[int] = []
        self.q_scale = torch.tensor([1.0 / 127.0], device=self.device, dtype=torch.float32)
        self.k_scale = torch.tensor([1.0 / 127.0], device=self.device, dtype=torch.float32)
        self.v_scale = torch.tensor([1.0 / 127.0], device=self.device, dtype=torch.float32)
        self.k_i8_cache = torch.empty(
            (config.num_layers, *self.kvcache.k_cache(0).shape),
            device=self.device,
            dtype=torch.int8,
        )
        self.k_i8_valid = torch.zeros(
            (config.num_layers, self.k_i8_cache.shape[1] * self.k_i8_cache.shape[2]),
            device=self.device,
            dtype=torch.bool,
        )
        self._store_dummy_v = torch.empty(
            self.k_i8_cache.shape[1] * self.k_i8_cache.shape[2],
            self.kv_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.int8,
        )
        artifact = get_quant_context().artifact
        self.q_scales = []
        self.k_scales = []
        self.v_scales = []
        if artifact is not None:
            for layer_id in range(config.num_layers):
                self.q_scales.append(
                    artifact.load_tensor(f"layers.{layer_id}.q_post_rope_i8.scale", self.device)
                    .to(torch.float32)
                    .view(1, -1, 1)
                )
                self.k_scales.append(
                    artifact.load_tensor(f"layers.{layer_id}.k_post_rope_i8.scale", self.device)
                    .to(torch.float32)
                    .view(1, -1, 1)
                )
                self.v_scales.append(
                    artifact.load_tensor(f"layers.{layer_id}.v_i8.scale", self.device)
                    .to(torch.float32)
                    .view(1, -1, 1)
                )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        v = v.view(-1, self.kv_heads, self.head_dim)
        q_scale = self.q_scales[layer_id] if self.q_scales else self.q_scale
        k_scale = self.k_scales[layer_id] if self.k_scales else self.k_scale
        v_scale = self.v_scales[layer_id] if self.v_scales else self.v_scale
        q_i8 = hadamard_quant_i8(q, q_scale, self.head_dim)
        k_i8 = hadamard_quant_i8(k.view(-1, self.kv_heads, self.head_dim), k_scale, self.head_dim)
        v_i8 = static_quantize_i8(v, v_scale)
        self._store_kv_direct(
            k_i8,
            k_i8.to(torch.bfloat16) * k_scale.to(torch.bfloat16),
            v_i8.to(torch.bfloat16) * v_scale.to(torch.bfloat16),
            batch.out_loc,
            layer_id,
        )
        if batch.is_decode:
            return self._run_tilelang_decode(q_i8, q_scale, k_scale, layer_id, batch)
        return self._run_reference_attention(q_i8, q_scale, k_scale, layer_id, batch)

    def _store_kv_direct(
        self,
        k_i8: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        k_cache = self.kvcache.k_cache(layer_id).view(-1, self.kv_heads, self.head_dim)
        v_cache = self.kvcache.v_cache(layer_id).view(-1, self.kv_heads, self.head_dim)
        store_cache(
            k_cache=self.k_i8_cache[layer_id].view(-1, self.kv_heads, self.head_dim),
            v_cache=self._store_dummy_v,
            indices=out_loc,
            k=k_i8.view(k_i8.shape[0], -1),
            v=k_i8.view(k_i8.shape[0], -1),
        )
        self.kvcache.store_kv(
            k.view(k.shape[0], -1),
            v.view(v.shape[0], -1),
            out_loc,
            layer_id,
        )
        if not torch.cuda.is_current_stream_capturing():
            indices = out_loc.long()
            self.k_i8_valid[layer_id, indices] = True

    def _run_tilelang_decode(
        self,
        q_i8: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, TileLangIntMetadata)
        if q_i8.shape[0] != batch.size:
            return self._run_reference_attention(q_i8, q_scale, k_scale, layer_id, batch)
        k_i8_cache = self.k_i8_cache[layer_id].view(-1, self.kv_heads, self.head_dim)
        v_cache = self.kvcache.v_cache(layer_id).view(-1, self.kv_heads, self.head_dim)
        return paged_gqa_decode_w8a8_bf16(
            q=q_i8.view(batch.size, self.qo_heads, self.head_dim),
            k_cache=k_i8_cache,
            v_cache=v_cache,
            page_table=metadata.page_table,
            seq_lens=metadata.seq_lens,
            q_scale=q_scale.reshape(-1),
            k_scale=k_scale.reshape(-1),
        ).view(batch.size, self.qo_heads, self.head_dim)

    def _run_reference_attention(
        self,
        q_i8: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        outputs = []
        metadata = batch.attn_metadata
        assert isinstance(metadata, TileLangIntMetadata)
        k_i8_cache = self.k_i8_cache[layer_id].view(-1, self.kv_heads, self.head_dim)
        v_cache = self.kvcache.v_cache(layer_id).view(-1, self.kv_heads, self.head_dim)
        cursor = 0
        group = self.qo_heads // self.kv_heads
        for row_id, req in enumerate(batch.reqs):
            q_req_i8 = q_i8[cursor : cursor + req.extend_len].to(torch.float32).permute(1, 0, 2)
            loc = metadata.page_table[row_id, : req.device_len]
            k_req_i8 = k_i8_cache[loc].to(torch.float32)
            v_req = v_cache[loc].to(torch.float32)
            k_req_i8 = k_req_i8.permute(1, 0, 2).repeat_interleave(group, dim=0)
            v_req = v_req.permute(1, 0, 2).repeat_interleave(group, dim=0)
            scale = (
                q_scale.reshape(-1).view(-1, 1, 1)
                * k_scale.reshape(-1).repeat_interleave(group).view(-1, 1, 1)
            )
            scores = torch.matmul(q_req_i8, k_req_i8.transpose(-1, -2)) * scale
            scores = scores / (self.head_dim**0.5)
            query_pos = torch.arange(
                req.cached_len, req.device_len, device=self.device, dtype=torch.int32
            )[:, None]
            key_pos = torch.arange(req.device_len, device=self.device, dtype=torch.int32)[None, :]
            scores = scores.masked_fill(key_pos.unsqueeze(0) > query_pos.unsqueeze(0), -torch.inf)
            out = torch.matmul(torch.softmax(scores, dim=-1), v_req)
            outputs.append(out.permute(1, 0, 2))
            cursor += req.extend_len
        return torch.cat(outputs, dim=0).to(torch.bfloat16)

    def prepare_metadata(self, batch: Batch) -> None:
        last_indices = []
        offset = 0
        reqs = batch.padded_reqs
        seqlens_k = [req.device_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        padded_max_seqlen_k = _align_up(max_seqlen_k, self.page_table_block)
        page_table = get_global_ctx().page_table
        compact_page_table = torch.stack(
            [page_table[req.table_idx, :padded_max_seqlen_k] for req in reqs]
        ).contiguous()
        seq_lens = torch.tensor(seqlens_k, device=self.device, dtype=torch.int32)
        for req in reqs:
            offset += req.extend_len
            last_indices.append(offset - 1)
        batch.attn_metadata = TileLangIntMetadata(
            last_indices=torch.tensor(last_indices, device=self.device, dtype=torch.int32),
            page_table=compact_page_table,
            seq_lens=seq_lens,
            max_seqlen_k=max_seqlen_k,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        self.capture = TileLangIntCaptureData.create(max_bs, max_seq_len, self.device)
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert bs in self.capture_bs and self.capture is not None
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, TileLangIntMetadata)
        capture = self.capture
        capture.seq_lens[:bs].copy_(metadata.seq_lens)
        capture.page_table[:bs, : metadata.page_table.shape[1]].copy_(metadata.page_table)
        capture.cu_seqlens_q[:bs].copy_(metadata.last_indices)
        batch.attn_metadata = TileLangIntMetadata(
            last_indices=capture.cu_seqlens_q[:bs],
            page_table=capture.page_table[:bs, : metadata.page_table.shape[1]],
            seq_lens=capture.seq_lens[:bs],
            max_seqlen_k=metadata.max_seqlen_k,
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, TileLangIntMetadata)
        assert self.capture is not None and bs in self.capture_bs
        capture = self.capture
        capture.seq_lens[:bs].copy_(metadata.seq_lens)
        capture.page_table[:bs, : metadata.page_table.shape[1]].copy_(metadata.page_table)
        capture.cu_seqlens_q[:bs].copy_(metadata.last_indices)
        batch.attn_metadata = TileLangIntMetadata(
            last_indices=capture.cu_seqlens_q[:bs],
            page_table=capture.page_table[:bs, : metadata.page_table.shape[1]],
            seq_lens=capture.seq_lens[:bs],
            max_seqlen_k=metadata.max_seqlen_k,
        )

def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
