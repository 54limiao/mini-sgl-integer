"""Regression tests for Q15 activation flow utilities and layer APIs."""

from __future__ import annotations

import torch

from minisgl.kernel.fixed_point import Q15Tensor, from_fixed, to_fixed
from minisgl.layers.activation_integer import silu_and_mul_fixed, silu_and_mul_q15
from minisgl.layers.norm_integer import RMSNormFusedInteger, RMSNormInteger
from minisgl.layers.quantization.w8a8_int8 import W8A8Int8LinearMethod


class _FakeQuantLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: str):
        super().__init__()
        self.weight = torch.randint(
            -127,
            127,
            (out_features, in_features),
            device=device,
            dtype=torch.int8,
        )
        self.weight_scale = torch.rand((out_features, 1), device=device, dtype=torch.float32) * 0.02
        self.bias = None


class TestQ15LayerFlow:
    def test_q15_tensor_roundtrip(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(8, 32, device=device, dtype=torch.float32)
        q = Q15Tensor.from_float(x)
        x_recon = q.to_float(torch.float32)
        diff = (x - x_recon).abs().max().item()
        assert diff < 2e-4, f"Q15 roundtrip error too large: {diff}"

    def test_activation_q15_matches_float_wrapper(self):
        if not torch.cuda.is_available():
            return

        x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
        x_q15 = to_fixed(x)

        out_q15 = silu_and_mul_q15(x_q15)
        out_q15_fp = from_fixed(out_q15, torch.float32)
        out_float = silu_and_mul_fixed(x).to(torch.float32)

        diff = (out_q15_fp - out_float).abs()
        assert diff.max().item() < 0.02, f"Activation Q15 mismatch too large: {diff.max().item()}"

    def test_rmsnorm_q15_api_consistency(self):
        if not torch.cuda.is_available():
            return

        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        x_q15 = to_fixed(x)

        norm = RMSNormInteger(size=64, eps=1e-6)
        norm.weight = torch.ones(64, device="cuda", dtype=torch.bfloat16)

        out_q15 = norm.forward_q15(x_q15)
        out_fp = norm.forward(x).to(torch.float32)
        out_q15_fp = from_fixed(out_q15, torch.float32)

        diff = (out_q15_fp - out_fp).abs().max().item()
        assert diff < 0.02, f"RMSNorm Q15 API mismatch too large: {diff}"

    def test_fused_rmsnorm_q15_api_shapes(self):
        if not torch.cuda.is_available():
            return

        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)

        fused = RMSNormFusedInteger(size=64, eps=1e-6)
        fused.weight = torch.ones(64, device="cuda", dtype=torch.bfloat16)

        out_q15, res_q15 = fused.forward_q15(to_fixed(x), to_fixed(residual))
        assert out_q15.dtype == torch.int32
        assert res_q15.dtype == torch.int32
        assert out_q15.shape == x.shape
        assert res_q15.shape == residual.shape

    def test_w8a8_linear_q15_io(self):
        if not torch.cuda.is_available():
            return

        layer = _FakeQuantLinear(64, 32, "cuda")
        method = W8A8Int8LinearMethod()

        x_fp = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
        out_fp = method.apply_weights(layer, x_fp).to(torch.float32)

        x_q15 = to_fixed(x_fp)
        out_q15 = method.apply_weights_q15(layer, x_q15)
        out_q15_fp = from_fixed(out_q15, torch.float32)

        cos_num = (out_fp.flatten() * out_q15_fp.flatten()).sum()
        cos_den = out_fp.norm() * out_q15_fp.norm() + 1e-12
        cos = (cos_num / cos_den).item()

        assert out_q15.dtype == torch.int32
        assert out_q15.shape == out_fp.shape
        assert cos > 0.999, f"Q15 linear cosine too low: {cos}"
