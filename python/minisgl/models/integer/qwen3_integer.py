"""
Qwen3 model with integer-first operator pipeline.

This integer model does not load float fallback operators.
Attention currently remains a float island at runtime (Q15<->bf16 bridge),
while all other supported blocks run through integer modules.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.kernel.fixed_point import from_fixed, to_fixed
from minisgl.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from minisgl.layers.norm_integer import RMSNormFusedInteger
from minisgl.utils import nvtx_annotate

from ..base import BaseLLMModel
from ..utils_integer import GatedMLPInteger, RopeAttnInteger

if TYPE_CHECKING:
    from ..config import ModelConfig


# Environment variable to control integer mode
# Set MINISGL_INTEGER_MODE=1 to enable integer ops (RMSNorm/MLP/RoPE-Attn)
INTEGER_MODE_ENABLED = os.environ.get("MINISGL_INTEGER_MODE", "0")


def _str_to_bool(s: str | None) -> bool:
    """Convert string to boolean."""
    if s is None:
        return False
    return s.lower() in ("1", "true", "yes", "on")


_INTEGER_MODE_ENABLED = _str_to_bool(INTEGER_MODE_ENABLED)


class Qwen3DecoderLayerInteger(BaseOP):
    """Qwen3 decoder layer with integer-only components."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.mlp = GatedMLPInteger(config, layer_id=layer_id)
        self.self_attn = RopeAttnInteger(config, layer_id, has_qk_norm=True)

        self.input_layernorm = RMSNormFusedInteger(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFusedInteger(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward_q15(
        self,
        x_q15: torch.Tensor,
        residual_q15: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_q15.dtype != torch.int32:
            raise TypeError(f"x_q15 must be int32 Q15.16, got {x_q15.dtype}")
        if residual_q15 is not None and residual_q15.dtype != torch.int32:
            raise TypeError(f"residual_q15 must be int32 Q15.16, got {residual_q15.dtype}")

        x_q15, residual_q15 = self.input_layernorm.forward_q15(x_q15, residual_q15)

        x_q15 = self.self_attn.forward_q15(x_q15)

        x_q15, residual_q15 = self.post_attention_layernorm.forward_q15(x_q15, residual_q15)

        x_q15 = self.mlp.forward_q15(x_q15)

        return x_q15, residual_q15

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_q15 = to_fixed(x)
        residual_q15 = to_fixed(residual) if residual is not None else None
        out_q15, residual_out_q15 = self.forward_q15(x_q15, residual_q15)
        return from_fixed(out_q15, x.dtype), from_fixed(residual_out_q15, x.dtype)


class Qwen3ModelInteger(BaseOP):
    """Qwen3 model with integer-only components."""

    def __init__(self, config: ModelConfig):
        if not _INTEGER_MODE_ENABLED:
            raise RuntimeError(
                "Qwen3ForCausalLMInteger requires MINISGL_INTEGER_MODE=1. "
                "Float fallback ops have been removed from qwen3_integer.py."
            )

        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3DecoderLayerInteger(config, layer_id) for layer_id in range(config.num_layers)]
        )

        self.norm = RMSNormFusedInteger(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        x_q15 = to_fixed(x)
        residual_q15: torch.Tensor | None = None

        for layer in self.layers.op_list:
            x_q15, residual_q15 = layer.forward_q15(x_q15, residual_q15)

        x_q15, _ = self.norm.forward_q15(x_q15, residual_q15)
        return from_fixed(x_q15, x.dtype)


class Qwen3ForCausalLMInteger(BaseLLMModel):
    """Qwen3 for causal LM with integer-only components."""

    def __init__(self, config: ModelConfig):
        self.model = Qwen3ModelInteger(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3ForCausalLMInteger", "_INTEGER_MODE_ENABLED"]
