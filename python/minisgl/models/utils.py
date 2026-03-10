from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    RMSNorm,
    gelu_and_mul,
    silu_and_mul,
)
from minisgl.kernel.fixed_point.hadamard_triton import fwht
from minisgl.models import ModelConfig
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    pass


def _match_target(layer_name: str, target: str) -> bool:
    if target.startswith("re:"):
        return re.match(target[3:], layer_name) is not None
    return layer_name == target


def _is_hadamard_targeted(layer_name: str, targets: tuple[str, ...]) -> bool:
    return any(_match_target(layer_name, target) for target in targets)


def _fwht_torch(x: torch.Tensor) -> torch.Tensor:
    size = x.shape[-1]
    out = x.clone()
    step = 1
    inv_sqrt = 1.0 / (size**0.5)
    while step < size:
        left = out[..., :: 2 * step].clone()
        right = out[..., step :: 2 * step].clone()
        out[..., :: 2 * step] = left + right
        out[..., step :: 2 * step] = left - right
        step <<= 1
    return out * inv_sqrt


def apply_blockwise_fwht(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    hidden = x.shape[-1]
    if hidden % block_size != 0:
        raise ValueError(f"Last dim {hidden} must be divisible by hadamard block_size {block_size}")

    x_blocks = x.contiguous().view(-1, hidden // block_size, block_size)
    if x_blocks.is_cuda:
        orig_dtype = x_blocks.dtype
        fwht_input = x_blocks
        if fwht_input.dtype == torch.bfloat16:
            fwht_input = fwht_input.to(torch.float32)
        y_blocks = fwht(fwht_input, scale=1.0 / (block_size**0.5), inplace=False)
        if y_blocks.dtype != orig_dtype:
            y_blocks = y_blocks.to(orig_dtype)
    else:
        y_blocks = _fwht_torch(x_blocks)
    return y_blocks.view_as(x)


class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int | None = None):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
        act_fn = FN_MAP.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = act_fn
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

        self._down_proj_name = (
            f"model.layers.{layer_id}.mlp.down_proj" if layer_id is not None else "mlp.down_proj"
        )
        self._hadamard_block_size = config.hadamard_transform.block_size
        self._apply_hadamard = (
            config.hadamard_transform.enabled
            and self._hadamard_block_size > 0
            and _is_hadamard_targeted(self._down_proj_name, config.hadamard_transform.targets)
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        if self._apply_hadamard:
            y = apply_blockwise_fwht(y, self._hadamard_block_size)
        return self.down_proj.forward(y)


class MoEMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.gate = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate.forward(hidden_states)
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)

        return final_hidden_states


class RopeAttn(BaseOP):
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
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.attn = AttentionLayer(
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


__all__ = ["GatedMLP", "RopeAttn", "MoEMLP"]
