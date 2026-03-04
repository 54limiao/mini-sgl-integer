from __future__ import annotations

import torch

import minisgl.layers.quantization.w8a8_int8 as qmod
from minisgl.layers.base import BaseOP
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
            "model.layers.0.self_attn.q_proj.input_scale": torch.tensor([0.10], dtype=torch.float32),
            "model.layers.0.self_attn.k_proj.input_scale": torch.tensor([0.30], dtype=torch.float32),
            "model.layers.0.self_attn.v_proj.input_scale": torch.tensor([0.20], dtype=torch.float32),
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
        layer.weight_scale = torch.tensor(
            [[0.1], [0.05], [0.2], [0.1]], dtype=torch.float32
        )
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
