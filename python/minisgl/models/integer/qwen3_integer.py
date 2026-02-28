"""
Qwen3 model with integer-only RMSNorm.

This is a variant of Qwen3 that uses integer-only RMSNorm while keeping
the rest of the computation in floating point. Useful for validating the
integer inference path.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from minisgl.layers.norm_integer import RMSNormFusedInteger
from minisgl.utils import nvtx_annotate

from ..base import BaseLLMModel
from ..utils import GatedMLP as Qwen3MLP
from ..utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from ..config import ModelConfig


# Environment variable to control integer mode
# Set MINISGL_INTEGER_MODE=1 to enable integer RMSNorm
INTEGER_MODE_ENABLED = os.environ.get("MINISGL_INTEGER_MODE", "0")

# Max layers to apply integer mode (-1 for all)
MAX_INTEGER_LAYERS = int(os.environ.get("MINISGL_MAX_INT_LAYERS", "-1"))


class Qwen3DecoderLayerInteger(BaseOP):
    """Qwen3 decoder layer with integer-only RMSNorm."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)

        from minisgl.layers import RMSNormFused
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Check if we should apply integer/hybrid mode to this layer
        apply_to_this_layer = MAX_INTEGER_LAYERS < 0 or layer_id < MAX_INTEGER_LAYERS
        
        if apply_to_this_layer:
            if INTEGER_MODE_ENABLED:
                # Full integer mode: fixed-point residual + fixed-point RMSNorm
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
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)

        x = self.self_attn.forward(x)

        x, residual = self.post_attention_layernorm.forward(x, residual)

        x = self.mlp.forward(x)

        return x, residual


class Qwen3ModelInteger(BaseOP):
    """Qwen3 model with integer-only RMSNorm."""

    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3DecoderLayerInteger(config, layer_id) for layer_id in range(config.num_layers)]
        )

        # Use integer RMSNorm for final norm if enabled
        if INTEGER_MODE_ENABLED:
                self.norm = RMSNormFusedInteger(
                    size=config.hidden_size,
                    eps=config.rms_norm_eps,
                )
        else:
            from minisgl.layers import RMSNormFused
            self.norm = RMSNormFused(
                size=config.hidden_size,
                eps=config.rms_norm_eps,
            )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)

        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        
        return self.norm.forward(x, residual)[0]


class Qwen3ForCausalLMInteger(BaseLLMModel):
    """Qwen3 for causal LM with integer-only RMSNorm."""

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


__all__ = ["Qwen3ForCausalLMInteger", "INTEGER_MODE_ENABLED"]
