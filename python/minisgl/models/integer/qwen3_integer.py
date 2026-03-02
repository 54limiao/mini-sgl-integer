"""
Qwen3 model with integer-only RMSNorm and MLP.

This is a variant of Qwen3 that uses integer-only components while keeping
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
from ..utils_integer import GatedMLPInteger, RopeAttnInteger

if TYPE_CHECKING:
    from ..config import ModelConfig


# Environment variable to control integer mode
# Set MINISGL_INTEGER_MODE=1 to enable integer RMSNorm
INTEGER_MODE_ENABLED = os.environ.get("MINISGL_INTEGER_MODE", "0")

# Max layers to apply integer mode (-1 for all)
MAX_INTEGER_LAYERS = int(os.environ.get("MINISGL_MAX_INT_LAYERS", "-1"))

# Environment variable to control integer MLP
# Set MINISGL_INTEGER_MLP=1 to enable integer MLP
INTEGER_MLP_ENABLED = os.environ.get("MINISGL_INTEGER_MLP", "0")

# Environment variable to control integer RoPE
# Set MINISGL_INTEGER_ROPE=1 to enable pseudo-quantized RoPE
INTEGER_ROPE_ENABLED = os.environ.get("MINISGL_INTEGER_ROPE", "0")


def _str_to_bool(s: str | None) -> bool:
    """Convert string to boolean."""
    if s is None:
        return False
    return s.lower() in ("1", "true", "yes", "on")


_INTEGER_MODE_ENABLED = _str_to_bool(INTEGER_MODE_ENABLED)
_INTEGER_MLP_ENABLED = _str_to_bool(INTEGER_MLP_ENABLED)
_INTEGER_ROPE_ENABLED = _str_to_bool(INTEGER_ROPE_ENABLED)


class Qwen3DecoderLayerInteger(BaseOP):
    """Qwen3 decoder layer with integer-only components."""

    def __init__(self, config: ModelConfig, layer_id: int):
        # Check if we should apply integer/hybrid mode to this layer
        apply_to_this_layer = MAX_INTEGER_LAYERS < 0 or layer_id < MAX_INTEGER_LAYERS
        
        # Choose MLP implementation
        if apply_to_this_layer and _INTEGER_MLP_ENABLED:
            self.mlp = GatedMLPInteger(config)
        else:
            self.mlp = Qwen3MLP(config)
        
        # Choose Attention implementation
        if apply_to_this_layer and _INTEGER_ROPE_ENABLED:
            self.self_attn = RopeAttnInteger(config, layer_id, has_qk_norm=True)
        else:
            self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)

        # Layernorm
        from minisgl.layers import RMSNormFused
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Apply integer RMSNorm if enabled
        if apply_to_this_layer and _INTEGER_MODE_ENABLED:
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
    """Qwen3 model with integer-only components."""

    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3DecoderLayerInteger(config, layer_id) for layer_id in range(config.num_layers)]
        )

        # Final norm
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


__all__ = ["Qwen3ForCausalLMInteger", "_INTEGER_MODE_ENABLED", "_INTEGER_MLP_ENABLED", "_INTEGER_ROPE_ENABLED"]
