"""
Pure integer SILU and multiply implementation using Q15.16 fixed-point format.

This implementation uses pure integer arithmetic for SILU and multiply:
- Input/Output: Q15.16 (int32)
- Internal computation: Q15.16 and Q30.32 (int64)
- exp() computation: Pure integer using exp2 LUT

SILU function: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
"""

import torch
import triton
import triton.language as tl
import numpy as np

# ============================================================================
# LUT Generation (done once at import time)
# ============================================================================

# exp2 LUT: exp2(x) for x in [0, 1) with Q0.29 format
# exp2(x) returns value in [1, 2], scaled by 2^29
_EXP2_LUT_SIZE = 1024
_EXP2_LUT: torch.Tensor | None = None

# LOG2E in Q15.16 format: log2(e) = 1.4426950408889634
# 1.4426950408889634 * 65536 = 94548.999999...
_LOG2E_Q15_16 = 94549

# Constants
FIXED_POINT_SCALE = 65536  # 2^16 for Q15.16
ONE_Q15_16 = 65536  # 1.0 in Q15.16


def _get_exp2_lut() -> torch.Tensor:
    """Get or create exp2 LUT (lazy initialization for multi-process support)."""
    global _EXP2_LUT

    if _EXP2_LUT is None:
        # Precompute exp2 LUT
        _EXP2_LUT = torch.zeros(_EXP2_LUT_SIZE, dtype=torch.int32)
        exp2_float = np.exp2(np.linspace(0.0, 1.0, _EXP2_LUT_SIZE)).astype(np.float32)
        _EXP2_LUT[:] = torch.tensor(
            np.clip(exp2_float * (1 << 29) + 0.5, 0, (1 << 32) - 1),
            dtype=torch.int32
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            _EXP2_LUT = _EXP2_LUT.cuda()

    return _EXP2_LUT


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def exp_q15_16_kernel(
    x_ptr,
    out_ptr,
    exp2_lut_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute exp(x) for x <= 0 in Q15.16 fixed-point format.
    
    exp(x) = 2^(log2(e) * x) = 2^(y_k + y_frac) where:
    - y_k = integer part of log2(e) * x (in Q16.0)
    - y_frac = fractional part in [0, 1) (Q0.16)
    - 2^(y_k + y_frac) = 2^y_k * 2^y_frac = 2^y_k * exp2(y_frac)
    
    Args:
        x_ptr: Input pointer (Q15.16 int32)
        out_ptr: Output pointer (Q15.16 int32, value in [0, 1])
        exp2_lut_ptr: exp2 LUT pointer (Q0.29 int32)
        n_elements: Number of elements
        BLOCK_SIZE: Block size
    """
    # Grid/Block setup
    pid = tl.program_id(0)
    
    # Process elements in blocks
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load x (Q15.16)
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.int64)
        
        # Compute y = x * log2(e) in Q30.32
        # log2(e) is Q15.16, x is Q15.16, so result is Q30.32
        y = (x * _LOG2E_Q15_16) >> 16  # Q30.32 -> Q15.16
        y_int = y >> 16  # Integer part (Q16.0)
        y_frac = y & 0xFFFF  # Fractional part (Q0.16)
        
        # LUT index: y_frac >> 6 (use top 10 bits for 1024-entry LUT)
        lut_idx = (y_frac >> 6) & 0x3FF
        
        # Lookup exp2(y_frac) in Q0.29
        exp2_frac = tl.load(exp2_lut_ptr + lut_idx.to(tl.int32))
        
        # Final result: exp2_frac >> (13 - y_int)
        # exp2(y_frac) in [1, 2), Q0.29, need to scale by 2^y_int
        shift = 13 - y_int
        exp_x = tl.where(shift >= 0, exp2_frac >> shift, exp2_frac << (-shift))
        
        # Clamp to int32 range and output (Q15.16, value in [0, 65536])
        exp_x = tl.minimum(exp_x, 65536)
        tl.store(out_ptr + offsets, exp_x.to(tl.int32), mask=mask)


@triton.jit
def silu_and_mul_int_only_kernel(
    x_ptr,
    out_ptr,
    exp2_lut_ptr,
    LOG2E_Q15_16,
    stride_xb,
    stride_xh,
    stride_outb,
    stride_outh,
    hidden,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SILU and multiply using pure integer arithmetic in Q15.16 format.

    Input x is [batch, 2*hidden]:
    - First half: gate (Q15.16)
    - Second half: up (Q15.16)

    Output: silu(gate) * up (Q15.16)

    SILU(x) = x * sigmoid(x)
    sigmoid(x) = exp(x) / (1 + exp(x))

    For numerical stability:
    - x <= 0: sigmoid(x) = exp(x) / (1 + exp(x))
    - x > 0: sigmoid(x) = 1 - sigmoid(-x)

    Args:
        x_ptr: Input pointer [batch, 2*hidden] (Q15.16 int32)
        out_ptr: Output pointer [batch, hidden] (Q15.16 int32)
        exp2_lut_ptr: exp2 LUT pointer (Q0.29 int32)
        LOG2E_Q15_16: log2(e) in Q15.16 format
        stride_xb: Batch stride for input
        stride_xh: Hidden stride for input
        stride_outb: Batch stride for output
        stride_outh: Hidden stride for output
        hidden: Hidden dimension
        BLOCK_SIZE: Block size for processing
    """
    batch_idx = tl.program_id(0)

    # Process this batch
    row_start_x = batch_idx * stride_xb
    row_start_out = batch_idx * stride_outb

    # Process hidden dimension in blocks
    for h_start in range(0, hidden, BLOCK_SIZE):
        h_offsets = h_start + tl.arange(0, BLOCK_SIZE)
        h_mask = h_offsets < hidden

        # Load gate (first half of input)
        gate_offsets = row_start_x + h_offsets * stride_xh
        gate = tl.load(x_ptr + gate_offsets, mask=h_mask, other=0).to(tl.int64)

        # Load up (second half of input)
        up_offsets = gate_offsets + hidden * stride_xh
        up = tl.load(x_ptr + up_offsets, mask=h_mask, other=0).to(tl.int64)

        # === Compute sigmoid(gate) ===

        # For x <= 0: sigmoid(x) = exp(x) / (1 + exp(x))
        # For x > 0: sigmoid(x) = 1 - sigmoid(-x)

        # Check sign of gate
        is_positive = gate > 0

        # Compute exp(|gate|) for all using vectorized operations
        gate_abs = tl.where(is_positive, -gate, gate)

        # Compute y = g * log2(e) in Q30.32
        y = (gate_abs * LOG2E_Q15_16) >> 16  # Q30.32 -> Q15.16
        y_int = y >> 16  # Integer part (Q16.0)
        y_frac = y & 0xFFFF  # Fractional part (Q0.16)

        # LUT index: y_frac >> 6 (use top 10 bits for 1024-entry LUT)
        lut_idx = (y_frac >> 6) & 0x3FF

        # Lookup exp2(y_frac) in Q0.29
        exp2_frac = tl.load(exp2_lut_ptr + lut_idx, mask=h_mask, other=0)

        # Final result: exp2_frac >> (13 - y_int)
        shift = 13 - y_int
        exp_g = tl.where(shift >= 0, exp2_frac >> shift, exp2_frac << (-shift))

        # Clamp to [0, 65536]
        exp_g = tl.minimum(exp_g, 65536)

        # sigmoid for negative branch (gate <= 0)
        # sigmoid = exp_g / (1 + exp_g)
        denom = 65536 + exp_g.to(tl.int64)  # 1 + exp_g in Q15.16
        sigmoid_neg = (exp_g.to(tl.int64) * 65536) // denom

        # For positive branch: sigmoid = 1 - sigmoid(-gate)
        # We have sigmoid(-gate) in sigmoid_neg, so:
        sigmoid = tl.where(is_positive, 65536 - sigmoid_neg, sigmoid_neg)

        # === Compute SILU(gate) = gate * sigmoid ===
        silu_gate = (gate * sigmoid.to(tl.int64)) >> 16  # Q30.32 -> Q15.16

        # === Multiply: silu_gate * up ===
        # Q15.16 * Q15.16 = Q30.32, divide by 65536 to get Q15.16
        output = (silu_gate * up) >> 16

        # Clamp to int32 range
        output = tl.maximum(output, -2147483648)
        output = tl.minimum(output, 2147483647)

        # Store output
        out_offsets = row_start_out + h_offsets * stride_outh
        tl.store(out_ptr + out_offsets, output.to(tl.int32), mask=h_mask)


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def silu_and_mul_int_only(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Pure integer SILU and multiply (Q15.16 format).
    
    Args:
        x: Input tensor [batch, 2*hidden_dim] (int32, Q15.16)
    
    Returns:
        Output tensor [batch, hidden_dim] (int32, Q15.16)
    """
    assert x.dtype == torch.int32, f"Expected int32 input, got {x.dtype}"
    
    batch, two_hidden = x.shape
    hidden = two_hidden // 2
    device = x.device
    
    # Allocate output
    out = torch.empty((batch, hidden), dtype=torch.int32, device=device)
    
    # Launch kernel
    BLOCK_SIZE = triton.next_power_of_2(hidden)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    
    # Get exp2 LUT
    exp2_lut = _get_exp2_lut()
    
    grid = (batch,)
    silu_and_mul_int_only_kernel[grid](
        x,
        out,
        exp2_lut,
        _LOG2E_Q15_16,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        hidden,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


__all__ = ["silu_and_mul_int_only"]