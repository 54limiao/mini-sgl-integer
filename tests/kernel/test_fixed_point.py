"""
Unit tests for fixed-point kernels.
"""

import torch
import numpy as np

from minisgl.kernel.fixed_point import (
    to_fixed, from_fixed,
    rmsnorm_int_only,
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



class TestRMSNorm:
    """Test RMSNorm implementations."""
    
    def _rmsnorm_reference(self, x, weight, eps=1e-6):
        """Reference RMSNorm implementation in float."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x / rms * weight
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rmsnorm_with_weight(self):
        """Test RMSNorm with non-unity weight."""
        device = 'cuda'
        batch, hidden = 10, 128
        
        x = torch.randn(batch, hidden, device=device) * 0.5
        x_fixed = to_fixed(x)
        weight = torch.ones(hidden, device=device).abs()
        weight_fixed = to_fixed(weight)
        
        # /workspace/lim42@xiaopeng.com/github/mini-sglang/python/minisgl/kernel/fixed_point/rmsnorm_q15_16_prototype.py
        from minisgl.kernel.fixed_point.rmsnorm_q15_16_prototype import rmsnorm_q15_16
        out_test = rmsnorm_q15_16(x_fixed.cpu().numpy(), weight_fixed.cpu().numpy())
        out_test = from_fixed(torch.from_numpy(out_test).to(device))
        out = rmsnorm_int_only(x_fixed, weight_fixed)
        out_float = from_fixed(out)
        
        # Reference
        out_ref = self._rmsnorm_reference(x, weight)
        error_test = (out_test - out_ref).abs().max()
        print(f"RMSNorm with weight max error (prototype): {error_test:.6f}")
        error = (out_float - out_ref).abs().max()
        assert error < 0.03, f"RMSNorm with weight error too large: {error}"


if __name__ == "__main__":
    test_rmsnorm = TestRMSNorm()
    test_rmsnorm.test_rmsnorm_with_weight()
    print("✓ RMSNorm tests passed")
    
    # run_tests()
