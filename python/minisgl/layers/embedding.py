from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.quant import get_quant_context
from minisgl.utils import div_ceil, nvtx_annotate

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if get_quant_context().is_int_w8a8_static:
            if self.tp_size == 1:
                return self.weight[x.long()].contiguous()
            start, length = self.vocab_range
            local = x.long() - start
            mask = (local >= 0) & (local < length)
            y = self.weight.new_zeros(x.shape[0], self.weight.shape[1])
            if mask.any():
                y[mask] = self.weight[local[mask]]
            return self._comm.all_reduce(y)

        from minisgl.kernel import indexing

        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        quant_ctx = get_quant_context()
        self._quant_backend = quant_ctx.backend
        if quant_ctx.is_int_w8a8_static:
            self.i8_weight = torch.empty_like(self.weight, dtype=torch.int8)
            self.i8_weight_scale = torch.empty(self.num_embeddings_tp, dtype=torch.float32)
            self.input_scale = torch.empty(1, dtype=torch.float32)
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if self._quant_backend == "int-w8a8-static":
            base = f"{prefix}." if prefix else ""

            def pop_any(*names: str) -> torch.Tensor:
                for name in names:
                    value = state_dict.pop(name, None)
                    if value is not None:
                        return value
                raise KeyError(names[0])

            self.i8_weight = pop_any(f"{base}i8.weight", f"{base}i8_weight")
            self.i8_weight_scale = pop_any(f"{base}i8.weight_scale", f"{base}i8_weight_scale")
            self.input_scale = state_dict.pop(f"{base}input_scale")
            if not self.tied_embedding:
                self.weight = state_dict.pop(f"{base}weight")
            state_dict.pop(f"{base}i8.weight", None)
            state_dict.pop(f"{base}i8.weight_scale", None)
            state_dict.pop(f"{base}i8_weight", None)
            state_dict.pop(f"{base}i8_weight_scale", None)
            return
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            result = super().state_dict(prefix=prefix, result=result)
        elif result is None:
            result = {}
        if self._quant_backend == "int-w8a8-static":
            base = f"{prefix}." if prefix else ""
            result[f"{base}i8.weight"] = self.i8_weight
            result[f"{base}i8.weight_scale"] = self.i8_weight_scale
            result[f"{base}input_scale"] = self.input_scale
        return result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        module = self.tied_embedding or self
        if self._quant_backend == "int-w8a8-static":
            from minisgl.kernel.tilelang import w8a8_static_linear

            logits = w8a8_static_linear(
                x,
                self.i8_weight,
                self.input_scale,
                self.i8_weight_scale,
                self.input_scale,
            ).float()
            if self.bias is not None:
                logits = logits + self.bias.float()
        else:
            logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        if bs == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]
