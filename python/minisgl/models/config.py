from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class HadamardTransformConfig:
    enabled: bool = False
    block_size: int = 0
    targets: Tuple[str, ...] = ()


def _parse_hadamard_transform_config(config: PretrainedConfig) -> HadamardTransformConfig:
    quantization_config = getattr(config, "quantization_config", None)
    if not isinstance(quantization_config, dict):
        return HadamardTransformConfig()

    transform_config = quantization_config.get("transform_config")
    if not isinstance(transform_config, dict):
        return HadamardTransformConfig()

    config_groups = transform_config.get("config_groups")
    if not isinstance(config_groups, dict):
        return HadamardTransformConfig()

    targets: list[str] = []
    block_size = 0
    for group in config_groups.values():
        if not isinstance(group, dict):
            continue
        if str(group.get("type", "")).lower() != "hadamard":
            continue

        candidate_size = group.get("head_dim", group.get("size", 0))
        if isinstance(candidate_size, int) and candidate_size > 0 and block_size <= 0:
            block_size = candidate_size

        apply_rules = group.get("apply", [])
        if not isinstance(apply_rules, list):
            continue

        for rule in apply_rules:
            if not isinstance(rule, dict):
                continue
            if rule.get("location") != "input":
                continue
            if bool(rule.get("inverse", False)):
                continue
            rule_targets = rule.get("targets", [])
            if not isinstance(rule_targets, list):
                continue
            for target in rule_targets:
                if isinstance(target, str):
                    targets.append(target)

    dedup_targets: list[str] = []
    seen: set[str] = set()
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        dedup_targets.append(target)

    enabled = bool(dedup_targets) and block_size > 0
    return HadamardTransformConfig(
        enabled=enabled,
        block_size=block_size,
        targets=tuple(dedup_targets),
    )


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    hadamard_transform: HadamardTransformConfig = HadamardTransformConfig()

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])
        hadamard_transform = _parse_hadamard_transform_config(config)

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                scaling=getattr(config, "rope_scaling", None),
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            hadamard_transform=hadamard_transform,
        )
