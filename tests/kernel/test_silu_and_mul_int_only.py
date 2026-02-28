"""
Unit tests for silu_and_mul_int_only Triton kernel.
"""

import torch
import numpy as np

from minisgl.kernel.fixed_point import (
    silu_and_mul_int_only,
    to_fixed,
    from_fixed,
)


class TestSiluAndMulIntOnly:
    """Test SILU and multiply integer-only Triton kernel."""

    def _silu_and_mul_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Reference SILU and multiply implementation in float."""
        batch, two_hidden = x.shape
        hidden = two_hidden // 2
        gate = x[:, :hidden]
        up = x[:, hidden:]
        silu = gate / (1.0 + torch.exp(-gate))
        return silu * up

    def test_silu_and_mul_int_only_cpu(self):
        """Test SILU and multiply integer-only kernel on CPU."""
        device = 'cpu'
        batch, hidden = 4, 256

        torch.manual_seed(42)
        x = torch.randn(batch, hidden * 2, device=device, dtype=torch.float32) * 0.5

        # Float reference
        out_ref = self._silu_and_mul_reference(x)

        # Q15.16 version using Triton kernel
        x_fixed = to_fixed(x)
        out_fixed = silu_and_mul_int_only(x_fixed)
        out_computed = from_fixed(out_fixed, dtype=torch.float32)

        # Compare
        diff = (out_ref - out_computed).abs()
        print(f"SILU and multiply (CPU) max diff: {diff.max().item():.6f}")
        print(f"SILU and multiply (CPU) mean diff: {diff.mean().item():.6f}")

        assert diff.max().item() < 0.01, f"SILU and multiply error too large: {diff.max().item()}"

    def test_silu_and_mul_int_only_cuda(self):
        """Test SILU and multiply integer-only kernel on CUDA."""
        if not torch.cuda.is_available():
            print("CUDA not available, skipping CUDA test")
            return

        device = 'cuda'
        batch, hidden = 4, 256

        torch.manual_seed(42)
        x = torch.randn(batch, hidden * 2, device=device, dtype=torch.float32) * 0.5

        # Float reference
        out_ref = self._silu_and_mul_reference(x)

        # Q15.16 version using Triton kernel
        x_fixed = to_fixed(x)
        out_fixed = silu_and_mul_int_only(x_fixed)
        out_computed = from_fixed(out_fixed, dtype=torch.float32)

        # Compare
        diff = (out_ref - out_computed).abs()
        print(f"SILU and multiply (CUDA) max diff: {diff.max().item():.6f}")
        print(f"SILU and multiply (CUDA) mean diff: {diff.mean().item():.6f}")

        assert diff.max().item() < 0.01, f"SILU and multiply error too large: {diff.max().item()}"

    def test_silu_and_mul_int_only_large_values(self):
        """Test SILU and multiply with larger values."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch, hidden = 4, 256

        torch.manual_seed(42)
        x = torch.randn(batch, hidden * 2, device=device, dtype=torch.float32) * 2.0

        # Float reference
        out_ref = self._silu_and_mul_reference(x)

        # Q15.16 version using Triton kernel
        x_fixed = to_fixed(x)
        out_fixed = silu_and_mul_int_only(x_fixed)
        out_computed = from_fixed(out_fixed, dtype=torch.float32)

        # Compare
        diff = (out_ref - out_computed).abs()
        print(f"SILU and multiply large values max diff: {diff.max().item():.6f}")
        print(f"SILU and multiply large values mean diff: {diff.mean().item():.6f}")

        assert diff.max().item() < 0.01, f"SILU and multiply large values error too large: {diff.max().item()}"

    def test_silu_and_mul_int_only_edge_cases(self):
        """Test SILU and multiply with edge cases."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch, hidden = 4, 256

        # Test with zeros
        x_zeros = torch.zeros(batch, hidden * 2, device=device, dtype=torch.float32)
        x_fixed = to_fixed(x_zeros)
        out_fixed = silu_and_mul_int_only(x_fixed)
        out_computed = from_fixed(out_fixed, dtype=torch.float32)
        assert out_computed.abs().max().item() < 1e-6, "Zeros should produce zeros"

        # Test with negative values
        x_neg = -torch.randn(batch, hidden * 2, device=device, dtype=torch.float32) * 0.5
        out_ref = self._silu_and_mul_reference(x_neg)
        x_fixed = to_fixed(x_neg)
        out_fixed = silu_and_mul_int_only(x_fixed)
        out_computed = from_fixed(out_fixed, dtype=torch.float32)
        diff = (out_ref - out_computed).abs()
        print(f"SILU and multiply negative values max diff: {diff.max().item():.6f}")
        assert diff.max().item() < 0.01, f"Negative values error too large: {diff.max().item()}"


if __name__ == "__main__":
    test = TestSiluAndMulIntOnly()
    # Skip CPU tests since Triton kernel requires CUDA
    # test.test_silu_and_mul_int_only_cpu()
    test.test_silu_and_mul_int_only_cuda()
    test.test_silu_and_mul_int_only_large_values()
    test.test_silu_and_mul_int_only_edge_cases()
    print("✓ All SILU and multiply integer-only tests passed")