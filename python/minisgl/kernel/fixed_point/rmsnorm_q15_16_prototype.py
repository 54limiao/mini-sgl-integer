"""
Prototype for Q15.16 fixed-point RMSNorm.

This is a reference NumPy implementation for testing and validation.
Once verified, it will be ported to Triton.

Format:
- Input/Output: Q15.16 (int32, range: [-32768, 32767.9999])
- Internal squares: Q30.32 (int64)
- 1/sqrt computation: Pure integer using LUT
"""

import numpy as np

FIXED_POINT_SCALE = 65536  # 2^16 for Q15.16

# ============================================================================
# Helper functions
# ============================================================================

def float_to_q15_16(x: np.ndarray) -> np.ndarray:
    """Convert float to Q15.16 int32."""
    return np.clip(np.round(x * FIXED_POINT_SCALE), -2**31, 2**31 - 1).astype(np.int32)


def q15_16_to_float(x: np.ndarray) -> np.ndarray:
    """Convert Q15.16 int32 to float."""
    return x.astype(np.float32) / FIXED_POINT_SCALE

def inverse_sqrt_q15_16(x_q15_16: np.ndarray) -> np.ndarray:
    """
    Compute 1/sqrt(x) in pure integer for Q15.16 format.
    
    Args:
        x_q30_32: Input in Q30.32 format (int64 or int32)
                  This is the mean of squares from RMSNorm computation
    
    Returns:
        inv_sqrt in Q15.16 format (int32)
    
    Formula:
        x^-0.5 = (2^2n * m)^-0.5 = 2^-n * m^-0.5
        where 1 <= m < 4, n = floor(log2(x)) / 2
    
    Steps:
        1. Find leading bit position of x
        2. Normalize x to [1, 4) range to get mantissa m
        3. LUT lookup for m^-0.5
        4. Compute 2^-n and combine with LUT result
        5. Return in Q15.16 format
    """
    # Hint: x is in Q30.32 format, so x = x_int / 2^32
    # The leading bit position tells us the exponent
    INV_SQERT_LUT = 1.0 / np.sqrt(np.linspace(0.0, 4.0, 1024).astype(np.float32))
    INV_SQERT_LUT = np.clip(INV_SQERT_LUT * (1 << 24) + 0.5, 0, 1 << 24).astype(np.int32)

    DEBRUIJN_TABLE = np.array([ 0,  0, 16,  2, 28, 16,  2, 22, 30, 20, 18, 10, 12,  4,  6, 22, 30, 14, 28, 20, 18, 10, 12,  6, 14, 26,  8,  4, 26,  8, 24, 24])
    largest_pow2 = x_q15_16
    largest_pow2 = largest_pow2 | (largest_pow2 >> 1)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 2)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 4)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 8)
    largest_pow2 = largest_pow2 | (largest_pow2 >> 16)
    largest_pow2 = largest_pow2 - (largest_pow2 >> 1)
    debruijn_mult = 0x06EB14F9
    idx = ((largest_pow2 * debruijn_mult) >> 27) & 0x1F
    largest_bit_pos = DEBRUIJN_TABLE[idx]
    x_norm = np.where(
        largest_bit_pos < 8,
        x_q15_16 << (8 - largest_bit_pos),
        x_q15_16 >> (largest_bit_pos - 8),
    )
    x_inv_sqrt = INV_SQERT_LUT[x_norm]
    
    largest_bit_pos = largest_bit_pos >> 1
    return x_inv_sqrt >> largest_bit_pos

    
    raise NotImplementedError("Please implement this function")

def rmsnorm_q15_16(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 2.0**-16
) -> np.ndarray:
    """
    RMSNorm for Q15.16 format (pure integer).
    
    Args:
        x: Input array [batch, hidden] in Q15.16 (int32)
        weight: Weight array [hidden] in Q15.16 (int32)
        eps: Epsilon for numerical stability (float)
    
    Returns:
        Output array [batch, hidden] in Q15.16 (int32)
    
    Formula:
        rms = sqrt(mean(x^2) + eps)
        output = x / rms * weight
    
    In fixed-point:
        1. Compute x^2 in Q30.32 (int64)
        2. Compute mean (integer division)
        3. Add eps in Q30.32 format
        4. Compute 1/sqrt(mean_sq_eps) using inverse_sqrt_q15_16
        5. Multiply: x * inv_sqrt * weight with proper scaling
    """
    batch, hidden = x.shape
    output = np.zeros_like(x)
    eps_q15_16 = float_to_q15_16(eps)
    
    for b in range(batch):
        # Step 1: Compute sum of squares (Q30.32)
        # Step 2: Compute mean and convert to Q30.32
        # Step 3: Add epsilon
        # Step 4: Compute inv_sqrt
        # Step 5: Apply normalization and weight

        x_row = x[b]
        sum_of_squares = np.sum(x_row.astype(np.int64) * x_row.astype(np.int64))
        mean_of_squares = eps_q15_16 + (sum_of_squares >> 16) // hidden
        inv_sqrt = inverse_sqrt_q15_16(mean_of_squares)  # Q15.16
        # Now compute output = x * inv_sqrt * weight
        # x is Q15.16, inv_sqrt is Q15.16, weight is Q15.16
        # So we need to do (x * inv_sqrt * weight) >> 24 to keep it in Q15.16
        output[b] = ((x_row.astype(np.int64) * inv_sqrt.astype(np.int64) * weight.astype(np.int64)) >> 32).astype(np.int32)
        
        # raise NotImplementedError("Please implement this function")
    
    return output


# ============================================================================
# Test functions
# ============================================================================

def test_inverse_sqrt():
    """Test inverse_sqrt_q15_16 function."""
    # Test data
    test_values = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=np.float32)
    
    # Convert to Q30.32 (for mean of squares)
    # Note: For RMSNorm, x^2 can be up to (32767)^2 ≈ 1e9
    # x_q15_16 = x_float * 2^32
    test_q15_16 = float_to_q15_16(test_values)  # This is wrong format, fix it
    
    print("Test values:", test_values)
    print("Expected 1/sqrt:", 1.0 / np.sqrt(test_values))
    
    result = inverse_sqrt_q15_16(test_q15_16)
    print("Computed 1/sqrt:", q15_16_to_float(result))


def test_rmsnorm():
    """Test rmsnorm_q15_16 function."""
    # Generate test data
    np.random.seed(42)
    batch, hidden = 4, 64
    
    x_float = np.random.randn(batch, hidden).astype(np.float32) * 0.5
    weight_float = np.ones(hidden, dtype=np.float32)
    
    # Convert to Q15.16
    x_q15_16 = float_to_q15_16(x_float)
    weight_q15_16 = float_to_q15_16(weight_float)
    
    # Reference implementation in float
    def rmsnorm_ref(x, w, eps=2.0**-16):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return x / rms * w
    
    out_ref = rmsnorm_ref(x_float, weight_float)

    out_q15_16 = rmsnorm_q15_16(x_q15_16, weight_q15_16)
    out_computed = q15_16_to_float(out_q15_16)
    
    print("Reference output (first row, first 5):", out_ref[0, :5])
    print("Computed output (first row, first 5):", out_computed[0, :5])


if __name__ == "__main__":
    print("=" * 60)
    print("Q15.16 Fixed-Point RMSNorm Prototype")
    print("=" * 60)
    
    print("\n1. Testing inverse_sqrt:")
    test_inverse_sqrt()
    
    print("\n2. Testing RMSNorm:")
    test_rmsnorm()
