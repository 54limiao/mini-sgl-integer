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
INTEGER_MODE_ENABLED = os.environ.get("MINISGL_INTEGER_MODE", "0") == "1"

# Debug mode: print hidden states
DEBUG_HIDDEN_STATES = os.environ.get("MINISGL_DEBUG_HIDDEN", "0") == "1"


class Qwen3DecoderLayerInteger(BaseOP):
    """Qwen3 decoder layer with integer-only RMSNorm."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)

        # Use integer RMSNorm if enabled
        if INTEGER_MODE_ENABLED and layer_id < 7:
            self.input_layernorm = RMSNormFusedInteger(
                size=config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = RMSNormFusedInteger(
                size=config.hidden_size,
                eps=config.rms_norm_eps,
            )
        else:
            # Fallback to float RMSNorm
            from minisgl.layers import RMSNormFused
            self.input_layernorm = RMSNormFused(
                size=config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = RMSNormFused(
                size=config.hidden_size,
                eps=config.rms_norm_eps,
            )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if DEBUG_HIDDEN_STATES and self._layer_id == 0:
            print(f"[Layer {self._layer_id}] Before input_ln: x range=[{x.min().item():.4f}, {x.max().item():.4f}], mean={x.mean().item():.6f}")
            if residual is not None:
                print(f"[Layer {self._layer_id}] Before input_ln: residual range=[{residual.min().item():.4f}, {residual.max().item():.4f}]")
        
        x, residual = self.input_layernorm.forward(x, residual)
        
        if DEBUG_HIDDEN_STATES and self._layer_id == 0:
            print(f"[Layer {self._layer_id}] After input_ln: x range=[{x.min().item():.4f}, {x.max().item():.4f}], mean={x.mean().item():.6f}")
        
        x = self.self_attn.forward(x)
        
        if DEBUG_HIDDEN_STATES and self._layer_id == 0:
            print(f"[Layer {self._layer_id}] After attn: x range=[{x.min().item():.4f}, {x.max().item():.4f}], mean={x.mean().item():.6f}")
        
        x, residual = self.post_attention_layernorm.forward(x, residual)
        
        if DEBUG_HIDDEN_STATES and self._layer_id == 0:
            print(f"[Layer {self._layer_id}] After post_attn_ln: x range=[{x.min().item():.4f}, {x.max().item():.4f}], mean={x.mean().item():.6f}")
        
        x = self.mlp.forward(x)
        
        if DEBUG_HIDDEN_STATES and self._layer_id == 0:
            print(f"[Layer {self._layer_id}] After MLP: x range=[{x.min().item():.4f}, {x.max().item():.4f}], mean={x.mean().item():.6f}")
        
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
        if 0:
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
        
        if DEBUG_HIDDEN_STATES:
            print(f"[Embedding] x range=[{x.min().item():.4f}, {x.max().item():.4f}], mean={x.mean().item():.6f}")
        
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        
        if DEBUG_HIDDEN_STATES:
            print(f"[Final norm input] x range=[{x.min().item():.4f}, {x.max().item():.4f}]")
            if residual is not None:
                print(f"[Final norm input] residual range=[{residual.min().item():.4f}, {residual.max().item():.4f}]")
        
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
