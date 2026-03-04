"""
Integer-only MLP components using Q15.16 fixed-point format.

This module provides integer-first building blocks. For attention, qk_norm and
rotary run through Q15.16 paths, and only the backend attention op remains a
float bridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.kernel.fixed_point import from_fixed, to_fixed
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearRowParallel,
    StateLessOP,
)
from minisgl.layers.activation_integer import silu_and_mul_fixed, silu_and_mul_q15
from minisgl.layers.norm_integer import RMSNormInteger
from minisgl.layers.rotary_integer import get_rope_integer
from minisgl.models import ModelConfig
from minisgl.utils import div_even, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.models import RotaryConfig


class GatedMLPInteger(BaseOP):
    """GatedMLP with pseudo-quantized SILU activation."""

    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        # Use pseudo-quantized activation function
        self.act_fn = silu_and_mul_fixed
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP (Integer)")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)

    def forward_q15(self, x_q15: torch.Tensor) -> torch.Tensor:
        if x_q15.dtype != torch.int32:
            raise TypeError(f"x_q15 must be int32 Q15.16, got {x_q15.dtype}")

        gate_up_is_int8 = (
            getattr(self.gate_up_proj, "weight", None) is not None
            and self.gate_up_proj.weight.dtype == torch.int8
            and hasattr(self.gate_up_proj, "weight_scale")
        )
        down_proj_is_int8 = (
            getattr(self.down_proj, "weight", None) is not None
            and self.down_proj.weight.dtype == torch.int8
            and hasattr(self.down_proj, "weight_scale")
        )

        if not gate_up_is_int8 or not down_proj_is_int8:
            return to_fixed(self.forward(from_fixed(x_q15, torch.bfloat16)))

        gate_up_q15 = self.gate_up_proj.forward(x_q15)
        y_q15 = silu_and_mul_q15(gate_up_q15)
        return self.down_proj.forward(y_q15)


class AttentionLayerInteger(StateLessOP):
    """Attention layer with pseudo-quantized RoPE.

    The q_norm attention scale fusion is backend-driven. If the active
    attention backend sets ``requires_q_scale_fusion=True``, we fuse
    ``1/sqrt(head_dim)`` into q_norm once and then run with ``sm_scale=1.0``.
    """

    # Whether sm_scale has been absorbed into q_norm weight.
    attn_scale_fused: bool = False

    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: BaseOP | None = None,
        k_norm: BaseOP | None = None,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        # Use pseudo-quantized RoPE
        self.rotary = get_rope_integer(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

        # Track whether the scale has been fused (will be applied after weights
        # are loaded – see ``fuse_attn_scale_into_qnorm``).
        self.attn_scale_fused = False

    # ------------------------------------------------------------------
    # Public helper – call *after* model weights have been loaded.
    # ------------------------------------------------------------------
    def fuse_attn_scale_into_qnorm(self) -> None:
        """Multiply ``q_norm.weight`` by ``head_dim^(-0.5)`` in-place.

        After calling this the integer attention kernel can use ``sm_scale=1.0``
        because the attention scale is already part of the normalised Q.
        """
        if self.attn_scale_fused:
            return  # idempotent
        if self.q_norm is None:
            return  # nothing to fuse
        sm_scale = self.head_dim ** -0.5
        self.q_norm.weight.data.mul_(sm_scale)
        self.attn_scale_fused = True

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        if qkv.dtype == torch.int32:
            return self.forward_q15(qkv)

        out_q15 = self.forward_q15(to_fixed(qkv))
        return from_fixed(out_q15, qkv.dtype)

    def forward_q15(self, qkv_q15: torch.Tensor) -> torch.Tensor:
        if qkv_q15.dtype != torch.int32:
            raise TypeError(f"qkv_q15 must be int32 Q15.16, got {qkv_q15.dtype}")

        ctx = get_global_ctx()

        # Backend-driven policy: fuse q scale only when requested by backend.
        backend = ctx.attn_backend
        requires_q_scale_fusion = getattr(backend, "requires_q_scale_fusion", False)
        if not requires_q_scale_fusion and hasattr(backend, "prefill_backend") and hasattr(
            backend, "decode_backend"
        ):
            selected_backend = backend.prefill_backend if ctx.batch.is_prefill else backend.decode_backend
            requires_q_scale_fusion = getattr(selected_backend, "requires_q_scale_fusion", False)

        if requires_q_scale_fusion and not self.attn_scale_fused:
            self.fuse_attn_scale_into_qnorm()

        q_q15, k_q15, v_q15 = qkv_q15.split(
            [self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1
        )

        if self.q_norm is not None:
            q_q15 = self.q_norm.forward_q15(q_q15)

        if self.k_norm is not None:
            k_q15 = self.k_norm.forward_q15(k_q15)

        q_q15, k_q15 = self.rotary.forward_q15(ctx.batch.positions, q_q15, k_q15)

        q = from_fixed(q_q15, torch.bfloat16)
        k = from_fixed(k_q15, torch.bfloat16)
        v = from_fixed(v_q15, torch.bfloat16)

        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return to_fixed(o.view(-1, self.qo_attn_dim))


class RopeAttnInteger(BaseOP):
    """Attention with pseudo-quantized RoPE."""

    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
    ):
        head_dim = config.head_dim
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
        )
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            self.q_norm = RMSNormInteger(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNormInteger(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        # Use AttentionLayerInteger with pseudo-quantized RoPE
        self.attn = AttentionLayerInteger(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        o = self.attn.forward(qkv)
        return self.o_proj.forward(o)

    def forward_q15(self, x_q15: torch.Tensor) -> torch.Tensor:
        if x_q15.dtype != torch.int32:
            raise TypeError(f"x_q15 must be int32 Q15.16, got {x_q15.dtype}")

        qkv_is_int8 = (
            getattr(self.qkv_proj, "weight", None) is not None
            and self.qkv_proj.weight.dtype == torch.int8
            and hasattr(self.qkv_proj, "weight_scale")
        )
        o_proj_is_int8 = (
            getattr(self.o_proj, "weight", None) is not None
            and self.o_proj.weight.dtype == torch.int8
            and hasattr(self.o_proj, "weight_scale")
        )

        if not qkv_is_int8 or not o_proj_is_int8:
            x = from_fixed(x_q15, torch.bfloat16)
            out = self.forward(x)
            return to_fixed(out)

        qkv_q15 = self.qkv_proj.forward(x_q15)
        o_q15 = self.attn.forward_q15(qkv_q15)
        return self.o_proj.forward(o_q15)


__all__ = [
    "GatedMLPInteger",
    "RopeAttnInteger",
    "AttentionLayerInteger",
]