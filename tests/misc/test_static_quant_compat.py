from __future__ import annotations

import torch

import minisgl.layers.quantization.w8a8_int8 as qmod
from minisgl.layers.base import BaseOP
from minisgl.distributed import DistributedInfo
from minisgl.models.config import HadamardTransformConfig, ModelConfig, RotaryConfig
from minisgl.models.utils import GatedMLP
from minisgl.layers.quantization.w8a8_int8 import (
    W8A8Int8LinearMethod,
    per_tensor_quant_int8_static,
)
from minisgl.models.weight import _merge_state_dict


class DummyLinear(BaseOP):
    def __init__(self):
        self.weight = torch.zeros((4, 3), dtype=torch.int8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_load_state_dict_accepts_input_scale() -> None:
    layer = DummyLinear()
    state = {
        "weight": torch.randint(-8, 8, (4, 3), dtype=torch.int8),
        "weight_scale": torch.ones((4, 1), dtype=torch.float32) * 0.1,
        "input_scale": torch.tensor([0.2], dtype=torch.float32),
    }
    layer.load_state_dict(dict(state))

    assert hasattr(layer, "weight_scale")
    assert hasattr(layer, "input_scale")


def test_merge_state_dict_fuses_input_scale_by_max() -> None:
    merged = _merge_state_dict(
        {
            "model.layers.0.self_attn.q_proj.input_scale": torch.tensor(
                [0.10], dtype=torch.float32
            ),
            "model.layers.0.self_attn.k_proj.input_scale": torch.tensor(
                [0.30], dtype=torch.float32
            ),
            "model.layers.0.self_attn.v_proj.input_scale": torch.tensor(
                [0.20], dtype=torch.float32
            ),
            "model.layers.0.mlp.gate_proj.input_scale": torch.tensor([0.05], dtype=torch.float32),
            "model.layers.0.mlp.up_proj.input_scale": torch.tensor([0.08], dtype=torch.float32),
        }
    )

    assert torch.allclose(
        merged["model.layers.0.self_attn.qkv_proj.input_scale"], torch.tensor([0.30])
    )
    assert torch.allclose(
        merged["model.layers.0.mlp.gate_up_proj.input_scale"], torch.tensor([0.08])
    )


def test_w8a8_static_input_scale_fallback_path() -> None:
    old_has_int8_kernel = qmod._HAS_INT8_KERNEL
    qmod._HAS_INT8_KERNEL = False
    try:
        method = W8A8Int8LinearMethod()
        layer = DummyLinear()
        layer.weight = torch.tensor(
            [[1, -2, 3], [4, 0, -1], [2, 2, 2], [-3, 1, 1]], dtype=torch.int8
        )
        layer.weight_scale = torch.tensor([[0.1], [0.05], [0.2], [0.1]], dtype=torch.float32)
        layer.input_scale = torch.tensor([0.25], dtype=torch.float32)

        x = torch.tensor([[0.6, -0.1, 0.3], [-0.2, 0.4, 1.2]], dtype=torch.float32)
        output = method.apply_weights(layer, x)

        x_q = torch.clamp(torch.round(x / 0.25), -128, 127)
        x_dq = x_q * 0.25
        w_dq = layer.weight.to(torch.float32) * layer.weight_scale
        expected = x_dq @ w_dq.t()

        assert torch.allclose(output, expected, atol=1e-5)
    finally:
        qmod._HAS_INT8_KERNEL = old_has_int8_kernel


def test_static_quant_scales_are_contiguous() -> None:
    x = torch.randn((2, 3, 16), dtype=torch.float32)
    input_scale = torch.tensor([0.125], dtype=torch.float32)
    _, scales = per_tensor_quant_int8_static(x, input_scale)
    assert scales.is_contiguous()


def test_float_mode_gated_mlp_applies_hadamard_for_r4_target(monkeypatch) -> None:
    monkeypatch.setattr(
        "minisgl.layers.linear.get_tp_info", lambda: DistributedInfo(rank=0, size=1)
    )

    cfg = ModelConfig(
        num_layers=1,
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim=16,
        hidden_size=32,
        vocab_size=100,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(
            head_dim=16, rotary_dim=16, max_position=128, base=10000.0, scaling=None
        ),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
        hadamard_transform=HadamardTransformConfig(
            enabled=True,
            block_size=16,
            targets=("model.layers.0.mlp.down_proj",),
        ),
    )

    mlp = GatedMLP(cfg, layer_id=0)
    mlp.gate_up_proj = DummyLinear()  # type: ignore[assignment]
    mlp.down_proj = DummyLinear()  # type: ignore[assignment]
    mlp.act_fn = lambda x: x  # type: ignore[assignment]

    called = {"value": False}

    def fake_fwht(x: torch.Tensor, block_size: int) -> torch.Tensor:
        called["value"] = True
        assert block_size == 16
        return x

    monkeypatch.setattr("minisgl.models.utils.apply_blockwise_fwht", fake_fwht)

    x = torch.randn(2, 32)
    _ = mlp.forward(x)

    assert called["value"]
