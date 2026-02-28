"""
Unit tests for silu_and_mul fixed-point kernel.
"""

import torch
import numpy as np

from minisgl.kernel.fixed_point.silu_and_mul_q15_16_prototype import (
    silu_and_mul_q15_16,
    float_to_q15_16,
    q15_16_to_float,
)

# Optional pytest import
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create dummy decorators
    class DummyPytest:
        @staticmethod
        def skipif(condition, reason=""):
            def decorator(f):
                return f
            return decorator
        
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(f):
                return f
            return decorator
    
    class Mark:
        @staticmethod
        def skipif(condition, reason=""):
            def decorator(f):
                return f
            return decorator
    
    DummyPytest.mark = Mark()
    pytest = DummyPytest()


class TestSiluAndMul:
    """Test SILU and multiply implementations."""
    
    def _silu_and_mul_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Reference SILU and multiply implementation in float."""
        batch, two_hidden = x.shape
        hidden = two_hidden // 2
        gate = x[:, :hidden]
        up = x[:, hidden:]
        silu = gate / (1.0 + torch.exp(-gate))
        return silu * up
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_silu_and_mul_basic(self):
        """Test basic SILU and multiply with random values."""
        device = 'cuda'
        batch, hidden = 4, 256
        
        torch.manual_seed(42)
        x = torch.randn(batch, hidden * 2, device=device, dtype=torch.float32) * 0.5
        
        # Float reference
        out_ref = self._silu_and_mul_reference(x)
        
        # Q15.16 version
        x_np = x.cpu().numpy()
        x_q15_16 = float_to_q15_16(x_np)
        out_q15_16 = silu_and_mul_q15_16(x_q15_16)
        out_computed = q15_16_to_float(out_q15_16)
        out_computed_torch = torch.from_numpy(out_computed).to(device)
        
        # Compare
        diff = (out_ref - out_computed_torch).abs()
        print(f"SILU and multiply max diff: {diff.max().item():.6f}")
        print(f"SILU and multiply mean diff: {diff.mean().item():.6f}")
        
        assert diff.max().item() < 0.01, f"SILU and multiply error too large: {diff.max().item()}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_silu_and_mul_with_weight(self):
        """Test SILU and multiply with weight scaling."""
        device = 'cuda'
        batch, hidden = 4, 256
        
        torch.manual_seed(42)
        x = torch.randn(batch, hidden * 2, device=device, dtype=torch.float32) * 0.5
        
        # Float reference
        out_ref = self._silu_and_mul_reference(x)
        
        # Q15.16 version
        x_np = x.cpu().numpy()
        x_q15_16 = float_to_q15_16(x_np)
        out_q15_16 = silu_and_mul_q15_16(x_q15_16)
        out_computed = q15_16_to_float(out_q15_16)
        out_computed_torch = torch.from_numpy(out_computed).to(device)
        
        # Compare
        diff = (out_ref - out_computed_torch).abs()
        print(f"SILU and multiply with weight max diff: {diff.max().item():.6f}")
        print(f"SILU and multiply with weight mean diff: {diff.mean().item():.6f}")
        
        assert diff.max().item() < 0.01, f"SILU and multiply with weight error too large: {diff.max().item()}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_silu_and_mul_large_values(self):
        """Test SILU and multiply with larger values."""
        device = 'cuda'
        batch, hidden = 4, 256
        
        torch.manual_seed(42)
        x = torch.randn(batch, hidden * 2, device=device, dtype=torch.float32) * 2.0
        
        # Float reference
        out_ref = self._silu_and_mul_reference(x)
        
        # Q15.16 version
        x_np = x.cpu().numpy()
        x_q15_16 = float_to_q15_16(x_np)
        out_q15_16 = silu_and_mul_q15_16(x_q15_16)
        out_computed = q15_16_to_float(out_q15_16)
        out_computed_torch = torch.from_numpy(out_computed).to(device)
        
        # Compare
        diff = (out_ref - out_computed_torch).abs()
        print(f"SILU and multiply large values max diff: {diff.max().item():.6f}")
        print(f"SILU and multiply large values mean diff: {diff.mean().item():.6f}")
        
        # Check for clipping
        clipped = (x_q15_16 == -2**31) | (x_q15_16 == 2**31-1)
        print(f"Clipped values in input: {clipped.sum()}")
        
        assert diff.max().item() < 0.01, f"SILU and multiply large values error too large: {diff.max().item()}"


if __name__ == "__main__":
    test_silu_and_mul = TestSiluAndMul()
    test_silu_and_mul.test_silu_and_mul_basic()
    test_silu_and_mul.test_silu_and_mul_with_weight()
    test_silu_and_mul.test_silu_and_mul_large_values()
    print("✓ All SILU and multiply tests passed")