"""
Prototype for Q15.16 fixed-point RMSNorm - V2 (Overflow-safe version).

This version avoids overflow by:
1. Finding max absolute value in the row
2. Normalizing input to [-1, 1] range using max (int16 format)
3. Computing sum of squares using normalized values (won't overflow)
4. Denormalizing the sum of squares back to original scale (int64)
5. Computing inv_sqrt and final output

Format:
- Input/Output: Q15.16 (int32, range: [-32768, 32767.9999])
- Normalized input: Q0.15 (int16, range: [-1, 1))
- Sum of squares: int64 (full precision)
"""

import numpy as np

FIXED_POINT_SCALE = 65536  # 2^16 for Q15.16
FIXED_POINT_SCALE_INT16 = 32768  # 2^15 for Q0.15 (int16)

# ============================================================================
# LUT Generation
# ============================================================================

# Precompute LUT for 1/sqrt(x) where x in [0, 4)
# Output format: Q0.24 for better precision
_INV_SQRT_LUT_SIZE = 1024
_INV_SQRT_LUT = np.zeros(_INV_SQRT_LUT_SIZE, dtype=np.int32)
lut_values = np.linspace(0.0, 4.0, _INV_SQRT_LUT_SIZE)
_INV_SQRT_LUT[1:] = np.clip((1.0 / np.sqrt(lut_values[1:])) * (1 << 16) + 0.5, 0, (1 << 16) - 1).astype(np.int32)

# De Bruijn table for CLZ (Count Leading Zeros) - 32-bit version
_DEBRUIJN_TABLE = np.array([
     0,  1, 16,  2, 29, 17,  3, 22, 30, 20, 18, 11, 13,  4,  7, 23,
    31, 15, 28, 21, 19, 10, 12,  6, 14, 27,  9,  5, 26,  8, 25, 24,
], dtype=np.int32)
_DEBRUIJN_MULT = 0x06EB14F9


# ============================================================================
# Helper functions
# ============================================================================

def float_to_q15_16(x: np.ndarray) -> np.ndarray:
    """Convert float to Q15.16 int32."""
    return np.clip(np.round(x * FIXED_POINT_SCALE), -2**31, 2**31 - 1).astype(np.int32)


def q15_16_to_float(x: np.ndarray) -> np.ndarray:
    """Convert Q15.16 int32 to float."""
    return x.astype(np.float32) / FIXED_POINT_SCALE

def clz_debruijn(x: np.ndarray) -> np.ndarray:
    """Count leading zeros using De Bruijn method."""
    assert x.dtype == np.int32 
    largest_pow2 = x
    largest_pow2 = largest_pow2 | (largest_pow2 >> 1)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 2)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 4)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 8)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 16)
    largest_pow2 = largest_pow2 - (largest_pow2 >> 1)
    
    idx = ((largest_pow2 * _DEBRUIJN_MULT) >> 27) & 0x1F
    return _DEBRUIJN_TABLE[idx]

def inverse_sqrt_q0_31(x_q0_31: np.ndarray) -> np.ndarray:
    max_pos = clz_debruijn(x_q0_31)
    even_pos = (max_pos + 1) >> 1 << 1
    x_norm = x_q0_31 << (30 - even_pos)
    x_norm = x_norm >> 21
    inv_sqrt_m = _INV_SQRT_LUT[x_norm]
    even_pos = even_pos >> 1
    return inv_sqrt_m << (16 - even_pos)

def rmsnorm_q15_16_v2(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """
    RMSNorm for Q15.16 format (overflow-safe version).
    
    Args:
        x: Input array [batch, hidden] in Q15.16 (int32)
        weight: Weight array [hidden] in Q15.16 (int32)
        eps: Epsilon for numerical stability (float)
    
    Returns:
        Output array [batch, hidden] in Q15.16 (int32)
    """
    batch, hidden = x.shape
    output = np.zeros_like(x)
    eps = 1
    
    for b in range(batch):
        x_row = x[b]
        
        # Step 1: Find max absolute value
        max_val = np.max(np.abs(x_row))
        max_pos = clz_debruijn(max_val)
        
        # Step 2: Normalize to int16 (Q0.15)
        x_norm = x_row << (30 - max_pos)
        x_norm = x_norm >> 16
        
        # Step 3: Compute sum of squares using normalized values
        mean_sq = np.mean(x_norm * x_norm).astype(np.int32) >> 1
        inv_sqrt = inverse_sqrt_q0_31(mean_sq + eps)
        
        # Step 5: Compute output
        output[b] = (x_row.astype(np.int64) * inv_sqrt.astype(np.int64) >> (max_pos + 2)).astype(np.int32)
    
    return output


# ============================================================================
# Test functions
# ============================================================================

def test_overflow_case():
    """Test with large values that would overflow in original version."""
    batch, hidden = 1, 4096
    
    # Large values that would cause overflow in original Q15.16
    x_float = np.random.randn(batch, hidden).astype(np.float32) * 100.0  # Large scale
    weight_float = np.ones(hidden, dtype=np.float32)
    
    x_q15_16 = float_to_q15_16(x_float)
    weight_q15_16 = float_to_q15_16(weight_float)
    
    print(f"Input range: [{x_float.min():.2f}, {x_float.max():.2f}]")
    print(f"Weight: {weight_float[0]:.1f}")
    
    # Reference in float
    rms = np.sqrt(np.mean(x_float ** 2, axis=-1, keepdims=True) + 1e-6)
    out_ref = (x_float / rms * weight_float)
    print(f"Reference output range: [{out_ref.min():.2f}, {out_ref.max():.2f}]")
    
    # Test V2 implementation
    out_v2 = rmsnorm_q15_16_v2(x_q15_16, weight_q15_16)
    out_v2_float = q15_16_to_float(out_v2)
    
    diff = np.abs(out_ref - out_v2_float)
    print(f"Max diff: {diff.max():.6f}")

    print(f"Output range (V2): [{out_v2_float.min():.2f}, {out_v2_float.max():.2f}]")
    print(f"First 5 values of output (V2): {out_v2_float[0, :5]}")
    print(f"First 5 values of reference: {out_ref[0, :5]}")


def test_normal_case():
    """Test with normal values."""
    batch, hidden = 9, 4096
    
    # x_float = np.random.randn(batch, hidden).astype(np.float32) * 0.5
    # x_float = np.array([[1.0, 2.0, 4.0]], dtype=np.float32).repeat(batch, axis=0)
    import glob, torch
    debug_files = glob.glob('/tmp/fused_norm_debug_layer447_*.pt')
    if not debug_files:
        print("Warning: No debug file found for layer 447, skipping test")
        return
    
    data = torch.load(debug_files[-1])
    x_float = data['x'].float().cpu().numpy().astype(np.float32)
    print(f"shape of x: {x_float.shape}, range: [{x_float.min():.4f}, {x_float.max():.4f}]")
    weight_float = np.ones(hidden, dtype=np.float32)
    
    x_q15_16 = float_to_q15_16(x_float)
    weight_q15_16 = float_to_q15_16(weight_float)
    
    # Reference
    rms = np.sqrt(np.mean(x_float ** 2, axis=-1, keepdims=True) + 1e-6)
    out_ref = x_float / rms * weight_float
    
    # Test V2
    out_v2 = rmsnorm_q15_16_v2(x_q15_16, weight_q15_16)
    out_v2_float = q15_16_to_float(out_v2)
    
    diff = np.abs(out_ref - out_v2_float)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Max diff pos: {np.unravel_index(np.argmax(diff), diff.shape)}")
    print(f"diff values at max diff pos: ref={out_ref[np.unravel_index(np.argmax(diff), diff.shape)]}, v2={out_v2_float[np.unravel_index(np.argmax(diff), diff.shape)]}")

    print(f"first row of output (float): {out_v2_float[0, :5]}")
    print(f"first row of reference (float): {out_ref[0, :5]}")
    cos_sim = np.sum(out_v2_float * out_ref) / (np.linalg.norm(out_v2_float) * np.linalg.norm(out_ref))
    print(f"Cosine similarity between V2 output and reference: {cos_sim:.6f}")


def test_residual_case():
    """Test with normal values."""
    batch, hidden = 9, 4096
    
    # x_float = np.random.randn(batch, hidden).astype(np.float32) * 0.5
    # x_float = np.array([[1.0, 2.0, 4.0]], dtype=np.float32).repeat(batch, axis=0)
    import glob, torch
    debug_files = glob.glob('/tmp/fused_norm_debug_layer447_*.pt')
    if not debug_files:
        print("Warning: No debug file found for layer 447, skipping test")
        return
    
    data = torch.load(debug_files[-1])
    x_float = (data['x'] + data['residual']).float().cpu().numpy().astype(np.float32)
    # x_float = x_float[3:4, :]
    print(f"shape of x: {x_float.shape}, range: [{x_float.min():.4f}, {x_float.max():.4f}]")
    weight_float = np.ones(hidden, dtype=np.float32)
    
    x_q15_16 = float_to_q15_16(x_float)
    weight_q15_16 = float_to_q15_16(weight_float)
    
    # Reference
    rms = np.sqrt(np.mean(x_float ** 2, axis=-1, keepdims=True) + 1e-6)
    out_ref = x_float / rms * weight_float
    
    # Test V2
    out_v2 = rmsnorm_q15_16_v2(x_q15_16, weight_q15_16)
    out_v2_float = q15_16_to_float(out_v2)
    
    diff = np.abs(out_ref - out_v2_float)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Max diff pos: {np.unravel_index(np.argmax(diff), diff.shape)}")
    print(f"diff values at max diff pos: ref={out_ref[np.unravel_index(np.argmax(diff), diff.shape)]}, v2={out_v2_float[np.unravel_index(np.argmax(diff), diff.shape)]}")

    print(f"first row of output (float): {out_v2_float[0, :5]}")
    print(f"first row of reference (float): {out_ref[0, :5]}")
    cos_sim = np.sum(out_v2_float * out_ref) / (np.linalg.norm(out_v2_float) * np.linalg.norm(out_ref))
    print(f"Cosine similarity between V2 output and reference: {cos_sim:.6f}")

def test_inverse_sqrt():
    """Test inverse_sqrt_q0_31 function."""
    test_values = np.array([0.02734375, 0.1, 0.25, 0.5, 0.999], dtype=np.float32)
    test_q0_31 = float_to_q15_16(test_values) << 15  # Convert to Q0.31
    inv_sqrt = inverse_sqrt_q0_31(test_q0_31)
    inv_sqrt_float = q15_16_to_float(inv_sqrt)  # Convert back to float
    print("Input:", test_values)
    print("Expected 1/sqrt:", 1.0 / np.sqrt(test_values))
    print("Computed 1/sqrt:", inv_sqrt_float)


if __name__ == "__main__":
    print("=" * 60)
    print("Q15.16 Fixed-Point RMSNorm Prototype V2 (Overflow-safe)")
    print("=" * 60)
    
    # print("\n1. Testing overflow case:")
    # test_overflow_case()
    
    # print("\n2. Testing normal case:")
    test_normal_case()
    test_residual_case()

    # print("\n3. Testing inverse_sqrt function:")
    # test_inverse_sqrt()
