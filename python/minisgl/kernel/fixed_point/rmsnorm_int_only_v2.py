"""
Pure integer RMSNorm implementation using Q15.16 fixed-point format - V2 (Overflow-safe).

This version avoids overflow by:
1. Finding max absolute value in the row
2. Normalizing input to int16 range
3. Computing sum of squares using normalized values (won't overflow)
4. Computing inv_sqrt and final output

Format:
- Input/Output: Q15.16 (int32)
- Normalized input: Q0.15 (int16)
- 1/sqrt: Pure integer using De Bruijn CLZ + LUT
"""

import torch
import triton
import triton.language as tl
import numpy as np

# ============================================================================
# LUT Generation (lazy initialization for multi-process support)
# ============================================================================

_INV_SQRT_LUT_SIZE = 1024
_INV_SQRT_LUT: torch.Tensor | None = None
_DEBRUIJN_TABLE: torch.Tensor | None = None
_DEBRUIJN_MULT = 0x06EB14F9


def _get_lut_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """Get or create LUT tables (lazy initialization for multi-process support)."""
    global _INV_SQRT_LUT, _DEBRUIJN_TABLE
    
    if _INV_SQRT_LUT is None:
        # Precompute LUT for 1/sqrt(x) where x in [0, 4)
        # Output format: Q0.16
        _INV_SQRT_LUT = torch.zeros(_INV_SQRT_LUT_SIZE, dtype=torch.int32)
        lut_values = np.linspace(0.0, 4.0, _INV_SQRT_LUT_SIZE)
        _INV_SQRT_LUT[1:] = torch.tensor(
            np.clip((1.0 / np.sqrt(lut_values[1:])) * (1 << 16) + 0.5, 0, (1 << 16) - 1),
            dtype=torch.int32
        )
        
        # De Bruijn table for CLZ (Count Leading Zeros) - 32-bit version
        _DEBRUIJN_TABLE = torch.tensor([
            0, 1, 16, 2, 29, 17, 3, 22, 30, 20, 18, 11, 13, 4, 7, 23,
            31, 15, 28, 21, 19, 10, 12, 6, 14, 27, 9, 5, 26, 8, 25, 24,
        ], dtype=torch.int32)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _INV_SQRT_LUT = _INV_SQRT_LUT.cuda()
            _DEBRUIJN_TABLE = _DEBRUIJN_TABLE.cuda()
    
    return _INV_SQRT_LUT, _DEBRUIJN_TABLE


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def clz_debruijn(x: tl.int32, debruijn_ptr) -> tl.int32:
    """Count leading zeros using De Bruijn method."""
    # Handle zero
    is_zero = x == 0
    x_safe = tl.where(is_zero, 1, x)
    
    # Bit smear to get largest power of 2 <= x
    largest_pow2 = x_safe
    largest_pow2 = largest_pow2 | (largest_pow2 >> 1)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 2)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 4)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 8)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 16)
    largest_pow2 = largest_pow2 - (largest_pow2 >> 1)
    
    # De Bruijn lookup
    debruijn_mult = 0x06EB14F9
    idx = ((largest_pow2.to(tl.int64) * debruijn_mult) >> 27) & 0x1F
    
    # Load from De Bruijn table
    result = tl.load(debruijn_ptr + idx.to(tl.int32))
    
    return tl.where(is_zero, 32, result)


@triton.jit
def inverse_sqrt_q0_31(x_q0_31: tl.int32, lut_ptr, debruijn_ptr) -> tl.int32:
    """
    Compute 1/sqrt(x) for Q0.31 input.
    Output is in Q0.16 format.
    """
    # Handle zero
    is_zero = x_q0_31 <= 0
    x_safe = tl.where(is_zero, 1, x_q0_31)
    
    # Get bit position of leading 1
    max_pos = clz_debruijn(x_safe, debruijn_ptr)
    
    # Round up to even position
    even_pos = ((max_pos + 1) >> 1) << 1
    
    # Normalize for LUT lookup
    x_norm = (x_safe << (30 - even_pos)) >> 21
    x_norm = tl.maximum(x_norm, 0)
    x_norm = tl.minimum(x_norm, 1023)
    
    # LUT lookup
    inv_sqrt_m = tl.load(lut_ptr + x_norm.to(tl.int32))
    
    # Adjust output
    even_pos_half = even_pos >> 1
    result = inv_sqrt_m << (16 - even_pos_half)
    
    return tl.where(is_zero, 0, result)


@triton.jit
def rmsnorm_int_only_v2_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    lut_ptr,
    debruijn_ptr,
    stride_xb, stride_xh,
    stride_wb,
    stride_outb, stride_outh,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Overflow-safe RMSNorm kernel using Q15.16 format.
    
    Algorithm:
    1. Find max(|x|) in the row
    2. Normalize to int16 range
    3. Compute sum of squares (won't overflow)
    4. Compute inv_sqrt
    5. Apply normalization to output
    """
    row_idx = tl.program_id(0)
    
    x_row = x_ptr + row_idx * stride_xb
    out_row = out_ptr + row_idx * stride_outb
    
    # Step 1: Find max absolute value
    max_val = tl.full((), 0, dtype=tl.int32)
    
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x_val = tl.load(x_row + col_offsets * stride_xh, mask=mask, other=0)
        x_abs = tl.where(x_val < 0, -x_val, x_val)
        max_val = tl.maximum(max_val, tl.max(x_abs, axis=0))
    
    # Handle zero row
    is_zero_row = max_val == 0
    
    # Get bit position of max value (only if not zero)
    max_pos = tl.where(is_zero_row, 0, clz_debruijn(max_val, debruijn_ptr))
    
    # Step 2 & 3: Normalize and compute sum of squares
    # x_norm = x << (30 - max_pos) >> 15  (to get int16 range)
    sum_sq = tl.full((), 0, dtype=tl.int64)
    
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x_val = tl.load(x_row + col_offsets * stride_xh, mask=mask, other=0)
        
        # Normalize to int16: x << (30 - max_pos) >> 15
        # shift = 30 - max_pos
        # x_norm = tl.where(
        #     shift >= 0,
        #     (x_val.to(tl.int64) << shift) >> 15,
        #     x_val.to(tl.int64) >> (15 - shift)
        # )
        x_norm = x_val << (30 - max_pos)
        x_norm = x_norm >> 16
        
        # Compute square (int32, won't overflow for int16 values)
        x_sq = x_norm * x_norm
        sum_sq += tl.sum(x_sq.to(tl.int64), axis=0)
    
    # Step 4: Compute mean and inv_sqrt
    # mean_sq in Q0.31 format: (sum_sq // n_cols) << 1
    # Use int64 for the division to avoid overflow before the shift
    mean_sq = ((sum_sq // n_cols) >> 1).to(tl.int32)
    mean_sq_eps = mean_sq + eps
    
    # Compute inverse sqrt (only if not zero row)
    inv_sqrt = inverse_sqrt_q0_31(mean_sq_eps, lut_ptr, debruijn_ptr)
    
    # Step 5: Compute output
    # output = x * inv_sqrt >> max_pos
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x_val = tl.load(x_row + col_offsets * stride_xh, mask=mask, other=0)
        w_val = tl.load(w_ptr + col_offsets * stride_wb, mask=mask, other=0)
        
        # output = x * inv_sqrt * weight >> max_pos
        out_val = (x_val.to(tl.int64) * inv_sqrt.to(tl.int64) * w_val.to(tl.int64)) >> (max_pos + 18)
        
        # Clamp to int32 range
        out_val = tl.maximum(out_val, -2147483648)
        out_val = tl.minimum(out_val, 2147483647)
        
        # Handle zero row
        out_val = tl.where(is_zero_row, 0, out_val)
        
        tl.store(out_row + col_offsets * stride_outh, out_val.to(tl.int32), mask=mask)


@triton.jit
def fused_add_rmsnorm_int_only_v2_kernel(
    residual_ptr,
    x_ptr,
    w_ptr,
    out_ptr,
    residual_out_ptr,
    lut_ptr,
    debruijn_ptr,
    stride_b, stride_h,
    stride_wb,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused add + RMSNorm kernel for fixed-point (pure integer) - V2 overflow-safe.
    """
    row_idx = tl.program_id(0)
    
    residual_row = residual_ptr + row_idx * stride_b
    x_row = x_ptr + row_idx * stride_b
    out_row = out_ptr + row_idx * stride_b
    residual_out_row = residual_out_ptr + row_idx * stride_b
    
    # First: residual = residual + x (element-wise, in int)
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        res_val = tl.load(residual_row + col_offsets * stride_h, mask=mask, other=0)
        x_val = tl.load(x_row + col_offsets * stride_h, mask=mask, other=0)
        
        # Add with saturation
        sum_val = res_val + x_val
        sum_val = tl.maximum(sum_val, -2147483648)
        sum_val = tl.minimum(sum_val, 2147483647)
        
        tl.store(residual_out_row + col_offsets * stride_h, sum_val, mask=mask)
    
    # Compute RMSNorm on residual_out - V2 algorithm
    # Step 1: Find max absolute value
    max_val = tl.full((), 0, dtype=tl.int32)
    
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        val = tl.load(residual_out_row + col_offsets * stride_h, mask=mask, other=0)
        val_abs = tl.where(val < 0, -val, val)
        max_val = tl.maximum(max_val, tl.max(val_abs, axis=0))
    
    is_zero_row = max_val == 0
    max_pos = tl.where(is_zero_row, 0, clz_debruijn(max_val, debruijn_ptr))
    
    # Step 2 & 3: Normalize and compute sum of squares
    sum_sq = tl.full((), 0, dtype=tl.int64)
    
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        val = tl.load(residual_out_row + col_offsets * stride_h, mask=mask, other=0)
        
        val_norm = val << (30 - max_pos)
        val_norm = val_norm >> 16
        
        val_sq = val_norm * val_norm
        sum_sq += tl.sum(val_sq.to(tl.int64), axis=0)
    
    # Step 4: Compute mean and inv_sqrt
    mean_sq = ((sum_sq // n_cols) >> 1).to(tl.int32)
    mean_sq_eps = mean_sq + eps
    
    inv_sqrt = inverse_sqrt_q0_31(mean_sq_eps, lut_ptr, debruijn_ptr)
    
    # Step 5: Compute output
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        val = tl.load(residual_out_row + col_offsets * stride_h, mask=mask, other=0)
        w_val = tl.load(w_ptr + col_offsets * stride_wb, mask=mask, other=0)
        
        out_val = (val.to(tl.int64) * inv_sqrt.to(tl.int64) * w_val.to(tl.int64)) >> (max_pos + 18)
        
        out_val = tl.maximum(out_val, -2147483648)
        out_val = tl.minimum(out_val, 2147483647)
        
        out_val = tl.where(is_zero_row, 0, out_val)
        
        tl.store(out_row + col_offsets * stride_h, out_val.to(tl.int32), mask=mask)


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def rmsnorm_int_only_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Overflow-safe RMSNorm (Q15.16 format).
    
    Args:
        x: Input tensor [batch, hidden_dim] (int32, Q15.16)
        weight: Weight tensor [hidden_dim] (int32, Q15.16)
        eps: Epsilon for numerical stability
    
    Returns:
        Output tensor [batch, hidden_dim] (int32, Q15.16)
    """
    assert x.dtype == torch.int32, f"Expected int32 input, got {x.dtype}"
    assert weight.dtype == torch.int32, f"Expected int32 weight, got {weight.dtype}"
    
    batch, hidden_dim = x.shape
    device = x.device
    
    # Get LUT tables (lazy initialization)
    inv_sqrt_lut, debruijn_table = _get_lut_tables()
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Convert eps to Q0.31 (eps = 1 for simplicity in integer)
    eps_int = 1
    
    # Launch kernel
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    
    grid = (batch,)
    rmsnorm_int_only_v2_kernel[grid](
        x, weight, out,
        inv_sqrt_lut, debruijn_table,
        x.stride(0), x.stride(1),
        weight.stride(0),
        out.stride(0), out.stride(1),
        hidden_dim,
        eps=eps_int,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def fused_add_rmsnorm_int_only_v2(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused add + RMSNorm for fixed-point (pure integer) - V2 overflow-safe.
    
    Args:
        residual: Residual tensor [batch, hidden_dim] (int32, Q15.16)
        x: Input tensor [batch, hidden_dim] (int32, Q15.16)
        weight: Weight tensor [hidden_dim] (int32, Q15.16)
        eps: Epsilon for numerical stability
    
    Returns:
        (output, residual_out) tuple of int32 tensors (Q15.16)
    """
    assert residual.dtype == torch.int32, f"Expected int32 residual, got {residual.dtype}"
    assert x.dtype == torch.int32, f"Expected int32 x, got {x.dtype}"
    assert weight.dtype == torch.int32, f"Expected int32 weight, got {weight.dtype}"
    
    batch, hidden_dim = x.shape
    device = x.device
    
    # Get LUT tables (lazy initialization)
    inv_sqrt_lut, debruijn_table = _get_lut_tables()
    
    # Allocate outputs
    out = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    
    # Convert eps
    eps_int = 1
    
    # Launch kernel
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    
    grid = (batch,)
    fused_add_rmsnorm_int_only_v2_kernel[grid](
        residual, x, weight, out, residual_out,
        inv_sqrt_lut, debruijn_table,
        residual.stride(0), residual.stride(1),
        weight.stride(0),
        hidden_dim,
        eps=eps_int,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out, residual_out


# ============================================================================
# Test functions
# ============================================================================

def float_to_q15_16(x: torch.Tensor) -> torch.Tensor:
    """Convert float to Q15.16 int32."""
    return torch.clamp(torch.round(x * 65536), -2**31, 2**31 - 1).to(torch.int32)


def q15_16_to_float(x: torch.Tensor) -> torch.Tensor:
    """Convert Q15.16 int32 to float."""
    return x.float() / 65536


def test_triton_kernel():
    """Test Triton kernel against numpy reference."""
    import numpy as np
    
    # Import numpy prototype
    import sys
    sys.path.insert(0, '/workspace/lim42@xiaopeng.com/github/mini-sglang')
    from python.minisgl.kernel.fixed_point.rmsnorm_q15_16_v2_prototype import (
        rmsnorm_q15_16_v2 as rmsnorm_numpy,
        float_to_q15_16 as np_float_to_q15_16,
        q15_16_to_float as np_q15_16_to_float,
    )
    
    batch, hidden = 4, 4096
    
    # Test with random values
    x_float = torch.randn(batch, hidden, dtype=torch.float32) * 100.0
    weight_float = torch.ones(hidden, dtype=torch.float32)
    
    # Convert to Q15.16
    x_q = float_to_q15_16(x_float.cuda())
    weight_q = float_to_q15_16(weight_float.cuda())
    
    # Reference in float
    rms = torch.sqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + 1e-6)
    out_ref_float = x_float / rms * weight_float
    
    # Test Triton V2
    out_triton = rmsnorm_int_only_v2(x_q, weight_q)
    out_triton_float = q15_16_to_float(out_triton)
    
    # Test numpy V2
    x_np = np_float_to_q15_16(x_float.cpu().numpy())
    weight_np = np_float_to_q15_16(weight_float.cpu().numpy())
    out_numpy = rmsnorm_numpy(x_np, weight_np)
    out_numpy_float = np_q15_16_to_float(out_numpy)
    
    # Compare
    diff_triton = torch.abs(out_ref_float.cuda() - out_triton_float)
    diff_numpy = np.abs(out_ref_float.cpu().numpy() - out_numpy_float)
    
    print(f"Reference output range: [{out_ref_float.min():.4f}, {out_ref_float.max():.4f}]")
    print(f"Triton output range: [{out_triton_float.min():.4f}, {out_triton_float.max():.4f}]")
    print(f"Numpy output range: [{out_numpy_float.min():.4f}, {out_numpy_float.max():.4f}]")
    print(f"Max diff (Triton vs float): {diff_triton.max():.6f}")
    print(f"Max diff (Numpy vs float): {diff_numpy.max():.6f}")
    print(f"Max diff (Triton vs Numpy): {np.abs(out_triton_float.cpu().numpy() - out_numpy_float).max():.6f}")


def test_simple_case():
    """Test with simple known values."""
    # x = [1, 2, 4] -> mean(x^2) = 7, sqrt = 2.646, 1/sqrt = 0.378
    # output = [0.378, 0.756, 1.512]
    
    x_float = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
    weight_float = torch.ones(3, dtype=torch.float32)
    
    x_q = float_to_q15_16(x_float.cuda())
    weight_q = float_to_q15_16(weight_float.cuda())
    
    # Reference
    rms = torch.sqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + 1e-6)
    out_ref_float = x_float / rms * weight_float
    print(f"Reference: {out_ref_float}")
    
    # Triton
    out_triton = rmsnorm_int_only_v2(x_q, weight_q)
    out_triton_float = q15_16_to_float(out_triton)
    print(f"Triton: {out_triton_float}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Triton V2 RMSNorm Kernel")
    print("=" * 60)
    
    print("\n1. Simple case test:")
    test_simple_case()
    
    print("\n2. Random values test:")
    test_triton_kernel()
