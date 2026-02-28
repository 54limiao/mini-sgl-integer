"""
Prototype for Q15.16 fixed-point SILU and multiply.

This is a reference NumPy implementation for testing and validation.
Once verified, it will be ported to Triton.

Format:
- Input/Output: Q15.16 (int32, range: [-32768, 32767.9999])
- Internal: Q15.16 for intermediate computation

SILU function: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
"""

import numpy as np

FIXED_POINT_SCALE = 65536  # 2^16 for Q15.16


def float_to_q15_16(x: np.ndarray) -> np.ndarray:
    """Convert float to Q15.16 int32."""
    return np.clip(np.round(x * FIXED_POINT_SCALE), -2**31, 2**31 - 1).astype(np.int32)


def q15_16_to_float(x: np.ndarray) -> np.ndarray:
    """Convert Q15.16 int32 to float."""
    return x.astype(np.float32) / FIXED_POINT_SCALE

exp2_lut = np.exp2(np.linspace(0, 1, 1024).astype(np.float32))
exp2_lut_i32 = np.int32(exp2_lut * (1 << 29) + 0.5)

def exp_q15_16(x: np.ndarray) -> np.ndarray:
    assert np.all(x <= 0), "Input should be non-positive for exp(-x) in softmax"
    # e**x = 2**(log2e*x) = 2**(x_k + x_fract) where x_k is integer part, x_fract is LOOKUP_BIT fractional part
    LOG2E_Q15_16 = np.int64(int(1.4426950408889634 * (1 << 16) + 0.5))  # 94548
    y = ((x.astype(np.int64) * LOG2E_Q15_16) >> 16).astype(np.int32)
    y_int = y >> 16
    y_frac = y & 0xFFFF
    lut_idx = (y_frac >> 6) & 0x3FF
    exp2_frac = exp2_lut_i32[lut_idx]
    x_exp = exp2_frac >> (13-y_int)  # Shift according to integer part
    return x_exp

def sigmoid_q15_16(x: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid in Q15.16 fixed-point format using exp_q15_16.

    Strategy: sigmoid(x) = exp(x) / (1 + exp(x))
    - For x <= 0: exp(x) <= 1, compute directly via exp_q15_16(x)
      sigmoid(x) = exp_x / (1 + exp_x)  [both in Q15.16]
    - For x > 0: sigmoid(x) = 1 - sigmoid(-x), avoid exp overflow
    """
    x = np.asarray(x, dtype=np.int32)
    result = np.zeros_like(x, dtype=np.int32)

    ONE_Q15_16 = np.int32(FIXED_POINT_SCALE)  # 1.0 in Q15.16 = 65536

    neg_mask = x <= 0
    pos_mask = ~neg_mask

    # --- Negative branch: x <= 0 ---
    if np.any(neg_mask):
        x_neg = x[neg_mask]
        exp_x = exp_q15_16(x_neg)  # exp(x) in Q15.16, value in (0, 1]
        # sigmoid(x) = exp_x / (1 + exp_x)
        # denom = 1 + exp_x in Q15.16
        denom = ONE_Q15_16.astype(np.int64) + exp_x.astype(np.int64)
        # sigmoid = exp_x / denom, scaled to Q15.16
        # = (exp_x * 2^16) / denom
        sig = (exp_x.astype(np.int64) * FIXED_POINT_SCALE) // denom
        result[neg_mask] = sig.astype(np.int32)

    # --- Positive branch: x > 0 ---
    # sigmoid(x) = 1 - sigmoid(-x)
    if np.any(pos_mask):
        x_pos = x[pos_mask]
        sig_neg = sigmoid_q15_16(-x_pos)  # sigmoid(-x), x > 0 so -x < 0
        result[pos_mask] = (ONE_Q15_16 - sig_neg).astype(np.int32)

    return result


def silu_q15_16(x: np.ndarray) -> np.ndarray:
    """
    Compute SILU in Q15.16 fixed-point format.
    
    SILU(x) = x * sigmoid(x)
    
    In fixed-point:
    - x is Q15.16
    - sigmoid(x) is computed in float then converted to Q15.16
    - result = (x * sigmoid(x)) / FIXED_POINT_SCALE (to keep Q15.16)
    """
    sigmoid_x = sigmoid_q15_16(x)
    # x * sigmoid_x in Q30.32, then divide by FIXED_POINT_SCALE to get Q15.16
    result = (x.astype(np.int64) * sigmoid_x.astype(np.int64)) // FIXED_POINT_SCALE
    return result.astype(np.int32)


def silu_and_mul_q15_16(x: np.ndarray) -> np.ndarray:
    """
    SILU and multiply for Q15.16 format (pure integer).
    
    The input tensor is [batch, 2*hidden], where:
    - First half: gate
    - Second half: up
    
    Output: silu(gate) * up
    
    Args:
        x: Input array [batch, 2*hidden] in Q15.16 (int32)
    
    Returns:
        Output array [batch, hidden] in Q15.16 (int32)
    
    Formula:
        silu(gate) = gate * sigmoid(gate)
        output = silu(gate) * up
    """
    batch, two_hidden = x.shape
    hidden = two_hidden // 2
    
    output = np.zeros((batch, hidden), dtype=np.int32)
    
    for b in range(batch):
        # Split into gate and up
        gate = x[b, :hidden]
        up = x[b, hidden:]
        
        # Compute silu(gate)
        silu_gate = silu_q15_16(gate)
        
        # Multiply: silu_gate * up in Q30.32, divide by FIXED_POINT_SCALE to get Q15.16
        output[b] = (silu_gate.astype(np.int64) * up.astype(np.int64)) // FIXED_POINT_SCALE
    
    return output

def test_exp():
    """Test exp function."""
    x = np.array([-1.0, -0.5, 0.0], dtype=np.float32)
    x_q15_16 = float_to_q15_16(x)
    exp_q = exp_q15_16(x_q15_16)
    exp_float = np.exp(x)
    exp_q_float = q15_16_to_float(exp_q)
    
    print("Input:", x)
    print("Exp (float):", exp_float)
    print("Exp (Q15.16):", exp_q_float)
    print("Max diff:", np.max(np.abs(exp_float - exp_q_float)))

def test_sigmoid():
    """Test sigmoid function."""
    x = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], dtype=np.float32)
    x_q15_16 = float_to_q15_16(x)
    sigmoid_q = sigmoid_q15_16(x_q15_16)
    sigmoid_float = 1 / (1 + np.exp(-x))
    sigmoid_q_float = q15_16_to_float(sigmoid_q)
    
    print("Input:", x)
    print("Sigmoid (float):", sigmoid_float)
    print("Sigmoid (Q15.16):", sigmoid_q_float)
    print("Max diff:", np.max(np.abs(sigmoid_float - sigmoid_q_float)))


def test_silu():
    """Test SILU function."""
    import torch
    
    print("=" * 60)
    print("Testing SILU function")
    print("=" * 60)
    
    # Test values
    test_values = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    
    # Float reference
    def silu_float(x):
        return x / (1.0 + np.exp(-x))
    
    ref = silu_float(test_values)
    
    # Q15.16 version
    test_q15_16 = float_to_q15_16(test_values)
    silu_q = silu_q15_16(test_q15_16)
    result = q15_16_to_float(silu_q)
    
    print("Test values:", test_values)
    print("Float SILU:", ref)
    print("Q15.16 SILU:", result)
    print("Max diff:", np.max(np.abs(ref - result)))


def test_silu_and_mul():
    """Test SILU and multiply."""
    import torch
    
    print("\n" + "=" * 60)
    print("Testing SILU and multiply")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    batch, hidden = 4, 256
    
    x_float = np.random.randn(batch, hidden * 2).astype(np.float32) * 0.5
    
    # Reference implementation
    def silu_and_mul_ref(x):
        gate = x[:, :hidden]
        up = x[:, hidden:]
        silu = gate / (1.0 + np.exp(-gate))
        return silu * up
    
    out_ref = silu_and_mul_ref(x_float)
    
    # Q15.16 version
    x_q15_16 = float_to_q15_16(x_float)
    out_q15_16 = silu_and_mul_q15_16(x_q15_16)
    out_computed = q15_16_to_float(out_q15_16)
    
    print("Reference output (first row, first 5):", out_ref[0, :5])
    print("Computed output (first row, first 5):", out_computed[0, :5])
    print("Max diff:", np.max(np.abs(out_ref - out_computed)))
    print("Mean diff:", np.mean(np.abs(out_ref - out_computed)))


if __name__ == "__main__":
    test_exp()
    test_sigmoid()
    test_silu()
    test_silu_and_mul()