"""
Pure integer RMSNorm implementation using Q15.16 fixed-point format.

This implementation uses pure integer arithmetic except for the initial
LUT generation. All computation is done in fixed-point:
- Input/Output: Q15.16 (int32)
- Internal squares: Q30.32 (int64)
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
        # Output format: Q0.24 for better precision during intermediate computation
        _INV_SQRT_LUT = torch.zeros(_INV_SQRT_LUT_SIZE, dtype=torch.int32)
        # Avoid divide by zero by starting from index 1
        lut_values = np.linspace(0.0, 4.0, _INV_SQRT_LUT_SIZE)
        _INV_SQRT_LUT[1:] = torch.tensor(
            np.clip((1.0 / np.sqrt(lut_values[1:])) * (1 << 24) + 0.5, 0, (1 << 24) - 1),
            dtype=torch.int32
        )
        
        # De Bruijn table for CLZ (Count Leading Zeros) - 32-bit version
        _DEBRUIJN_TABLE = torch.tensor([
            0, 0, 16, 2, 28, 16, 2, 22, 30, 20, 18, 10, 12, 4, 6, 22,
            30, 14, 28, 20, 18, 10, 12, 6, 14, 26, 8, 4, 26, 8, 24, 24,
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
def inverse_sqrt_q15_16_kernel(
    x_ptr,
    out_ptr,
    lut_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Pure integer inverse sqrt kernel for Q15.16 format.
    
    Args:
        x_ptr: Input array in Q30.32 format (int64)
        out_ptr: Output array in Q15.16 format (int32)
        lut_ptr: LUT for 1/sqrt lookup (Q0.24 format)
        n_elements: Number of elements
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input (Q30.32 format, stored as int64)
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    
    # Handle zero input
    is_zero = x <= 0
    x_safe = tl.where(is_zero, 1, x)
    
    # Find largest power of 2 <= x using bit smearing
    # This works for values up to 64-bit
    x_smear = x_safe
    x_smear = x_smear | (x_smear >> 1)
    x_smear = x_smear | (x_smear >> 2)
    x_smear = x_smear | (x_smear >> 4)
    x_smear = x_smear | (x_smear >> 8)
    x_smear = x_smear | (x_smear >> 16)
    # For 64-bit, need one more shift
    x_smear = x_smear | (x_smear >> 32)
    largest_pow2 = x_smear - (x_smear >> 1)
    
    # De Bruijn lookup to find bit position
    # Note: This only works for 32-bit values, but our LUT approach
    # handles the normalization differently
    idx = ((largest_pow2 * _DEBRUIJN_MULT) >> 27) & 0x1F
    
    # Load from De Bruijn table
    # We need to handle this carefully in Triton
    # For now, use a simpler approach: compute bit position from largest_pow2
    
    # Alternative: Use tl.ctlz if available (Triton 2.1+)
    # bit_pos = 63 - tl.ctlz(x_safe)
    
    # For compatibility, use iterative approach or tl.math.log2
    # Convert to float temporarily for bit position calculation
    x_float = x_safe.to(tl.float32)
    # log2(x) gives us the bit position
    log2_x = tl.math.log2(x_float)
    bit_pos = log2_x.to(tl.int32)
    
    # Normalize x to [0, 1024) range for LUT lookup
    # We want the top 10 bits after the leading 1
    shift_for_lut = bit_pos - 10
    x_norm = tl.where(
        shift_for_lut >= 0,
        (x_safe >> shift_for_lut) & 0x3FF,
        (x_safe << (-shift_for_lut)) & 0x3FF,
    )
    x_norm = tl.maximum(x_norm, 0)
    x_norm = tl.minimum(x_norm, 1023)
    
    # LUT lookup for m^-0.5 (Q0.24 format)
    inv_sqrt_m = tl.load(lut_ptr + x_norm, mask=mask, other=0)
    
    # Compute n = bit_pos / 2
    n = bit_pos >> 1
    
    # Compute 2^-n * m^-0.5
    # inv_sqrt_m is Q0.24, we want Q15.16 output
    # Result = inv_sqrt_m * 2^(16-n) >> 24 if n <= 16
    #        = inv_sqrt_m >> (n-8) if n > 16
    # Because: inv_sqrt_m * 2^-n in Q0.24 -> shift right by n
    # Then convert Q0.24 to Q15.16: shift left by 16, shift right by 24
    # Net: shift left by (16 - 24) = -8, then shift right by n
    # So: shift = -8 - n = -(8 + n)
    # Wait, let me recalculate:
    # inv_sqrt_m is Q0.24 (value in [0, 16) since 1/sqrt(0.25) = 2, but we clamp)
    # We want Q15.16 output
    # inv_sqrt = inv_sqrt_m * 2^-n
    # In fixed-point: inv_sqrt_q15_16 = (inv_sqrt_m * 2^-n) * 2^16
    #                                  = inv_sqrt_m * 2^(16-n)
    # inv_sqrt_m is already scaled by 2^24
    # So: result = inv_sqrt_m * 2^(16-n) / 2^24 = inv_sqrt_m >> (24 - 16 + n) = inv_sqrt_m >> (8 + n)
    shift = 8 + n
    inv_sqrt = inv_sqrt_m >> shift
    
    # Handle zero input
    inv_sqrt = tl.where(is_zero, 0, inv_sqrt)
    
    # Clamp to int32 range
    inv_sqrt = tl.maximum(inv_sqrt, -2147483648)
    inv_sqrt = tl.minimum(inv_sqrt, 2147483647)
    
    tl.store(out_ptr + offsets, inv_sqrt, mask=mask)


@triton.jit
def rmsnorm_int_only_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    lut_ptr,
    debruijn_ptr,
    stride_xb, stride_xh,
    stride_wb,
    stride_outb, stride_outh,
    n_cols,
    eps_q15_16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Pure integer RMSNorm kernel using Q15.16 format.
    
    Format:
    - Input/Output: Q15.16 (int32)
    - Internal squares: Q30.32 (int64)
    - 1/sqrt: Pure integer using LUT and De Bruijn
    """
    row_idx = tl.program_id(0)
    
    x_row = x_ptr + row_idx * stride_xb
    out_row = out_ptr + row_idx * stride_outb
    
    # Step 1: Compute sum of squares in int64 (exact)
    sum_sq = tl.full((), 0, dtype=tl.int64)
    
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x_val = tl.load(x_row + col_offsets * stride_xh, mask=mask, other=0)
        x_sq = x_val.to(tl.int64) * x_val.to(tl.int64)
        sum_sq += tl.sum(x_sq, axis=0)
    
    # Step 2: Compute mean and convert to Q15.16 (following prototype)
    mean_sq = (sum_sq >> 16) // n_cols
    
    # Step 3: Add epsilon
    mean_sq_eps = mean_sq + eps_q15_16
    
    # Step 4: Compute 1/sqrt using De Bruijn (following prototype exactly)
    is_zero = mean_sq_eps <= 0
    x_safe = tl.where(is_zero, 1, mean_sq_eps)
    
    # De Bruijn method to find leading bit position
    largest_pow2 = x_safe.to(tl.int64)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 1)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 2)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 4)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 8)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 16)
    largest_pow2 = largest_pow2 - (largest_pow2 >> 1)
    
    # De Bruijn lookup
    debruijn_mult = 0x06EB14F9
    idx = ((largest_pow2 * debruijn_mult) >> 27) & 0x1F
    largest_bit_pos = tl.load(debruijn_ptr + idx.to(tl.int32))
    
    # Normalize to LUT range
    x_norm = tl.where(
        largest_bit_pos < 8,
        (x_safe << (8 - largest_bit_pos)) & 0x3FF,
        (x_safe >> (largest_bit_pos - 8)) & 0x3FF,
    )
    x_norm = tl.maximum(x_norm, 0)
    x_norm = tl.minimum(x_norm, 1023)
    
    # LUT lookup
    inv_sqrt_m = tl.load(lut_ptr + x_norm.to(tl.int32))
    
    # Following prototype: n = largest_bit_pos >> 1
    n = largest_bit_pos >> 1
    inv_sqrt = inv_sqrt_m >> n
    
    inv_sqrt = tl.where(is_zero, 0, inv_sqrt)
    inv_sqrt = tl.maximum(inv_sqrt, 0)
    inv_sqrt = tl.minimum(inv_sqrt, 2147483647)
    
    # Step 5: Apply normalization and weight
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x_val = tl.load(x_row + col_offsets * stride_xh, mask=mask, other=0)
        w_val = tl.load(w_ptr + col_offsets * stride_wb, mask=mask, other=0)
        
        # Following prototype: x * inv_sqrt * weight >> 32
        out_val = x_val.to(tl.int64) * inv_sqrt.to(tl.int64) * w_val.to(tl.int64)
        out_val = out_val >> 32
        
        # Clamp to int32 range
        out_val = tl.maximum(out_val, -2147483648)
        out_val = tl.minimum(out_val, 2147483647)
        
        tl.store(out_row + col_offsets * stride_outh, out_val.to(tl.int32), mask=mask)


@triton.jit
def fused_add_rmsnorm_int_only_kernel(
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
    eps_q15_16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused add + RMSNorm kernel for fixed-point (pure integer).
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
        
        res_val_i32 = tl.load(residual_row + col_offsets * stride_h, mask=mask, other=0)
        x_val_i32 = tl.load(x_row + col_offsets * stride_h, mask=mask, other=0)
        
        # Add with saturation
        sum_val = res_val_i32 + x_val_i32
        sum_val = tl.maximum(sum_val, -2147483648)
        sum_val = tl.minimum(sum_val, 2147483647)
        
        tl.store(residual_out_row + col_offsets * stride_h, sum_val, mask=mask)
    
    # Compute RMSNorm on residual_out
    sum_sq = tl.full((), 0, dtype=tl.int64)
    
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        val = tl.load(residual_out_row + col_offsets * stride_h, mask=mask, other=0)
        val_sq = val.to(tl.int64) * val.to(tl.int64)
        sum_sq += tl.sum(val_sq, axis=0)
    # Convert to Q15.16 and add epsilon
    mean_sq = (sum_sq >> 16) // n_cols
    mean_sq_eps = mean_sq + eps_q15_16
    
    # Compute 1/sqrt using De Bruijn method
    is_zero = mean_sq_eps <= 0
    x_safe = tl.where(is_zero, 1, mean_sq_eps)
    
    # De Bruijn method
    largest_pow2 = x_safe.to(tl.int64)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 1)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 2)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 4)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 8)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 16)
    largest_pow2 = largest_pow2 - (largest_pow2 >> 1)
    
    debruijn_mult = 0x06EB14F9
    idx = ((largest_pow2 * debruijn_mult) >> 27) & 0x1F
    largest_bit_pos = tl.load(debruijn_ptr + idx.to(tl.int32))
    
    x_norm = tl.where(
        largest_bit_pos < 8,
        (x_safe << (8 - largest_bit_pos)) & 0x3FF,
        (x_safe >> (largest_bit_pos - 8)) & 0x3FF,
    )
    x_norm = tl.maximum(x_norm, 0)
    x_norm = tl.minimum(x_norm, 1023)
    
    inv_sqrt_m = tl.load(lut_ptr + x_norm.to(tl.int32))
    n = largest_bit_pos >> 1
    inv_sqrt = inv_sqrt_m >> n
    inv_sqrt = tl.where(is_zero, 0, inv_sqrt)
    inv_sqrt = tl.maximum(inv_sqrt, 0)
    inv_sqrt = tl.minimum(inv_sqrt, 2147483647)
    
    # Apply normalization
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        val = tl.load(residual_out_row + col_offsets * stride_h, mask=mask, other=0)
        w_val = tl.load(w_ptr + col_offsets * stride_wb, mask=mask, other=0)
        
        # x * inv_sqrt * weight: >> 32
        out_val = val.to(tl.int64) * inv_sqrt.to(tl.int64) * w_val.to(tl.int64)
        out_val = out_val >> 32
        
        out_val = tl.maximum(out_val, -2147483648)
        out_val = tl.minimum(out_val, 2147483647)
        
        tl.store(out_row + col_offsets * stride_h, out_val.to(tl.int32), mask=mask)


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def rmsnorm_int_only(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Pure integer RMSNorm (Q15.16 format).
    
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
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Convert eps to Q15.16
    eps_q15_16 = int(eps * (1 << 16))
    
    # Launch kernel
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    
    grid = (batch,)
    rmsnorm_int_only_kernel[grid](
        x, weight, out,
        _INV_SQRT_LUT, _DEBRUIJN_TABLE,
        x.stride(0), x.stride(1),
        weight.stride(0),
        out.stride(0), out.stride(1),
        hidden_dim,
        eps_q15_16=eps_q15_16,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def fused_add_rmsnorm_int_only(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused add + RMSNorm for fixed-point (pure integer).
    
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
    
    # Allocate outputs
    out = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    
    # Convert eps to Q15.16
    eps_q15_16 = int(eps * (1 << 16))
    
    # Launch kernel
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    
    grid = (batch,)
    fused_add_rmsnorm_int_only_kernel[grid](
        residual, x, weight, out, residual_out,
        _INV_SQRT_LUT, _DEBRUIJN_TABLE,
        residual.stride(0), residual.stride(1),
        weight.stride(0),
        hidden_dim,
        eps_q15_16=eps_q15_16,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out, residual_out