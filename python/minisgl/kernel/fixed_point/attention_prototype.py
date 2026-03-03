"""
Eager attention prototype (NumPy, float32).

  output = softmax(q @ k.T * scale) @ v

Shapes:
  q, k, v : [seq_len, head_dim]
  output  : [seq_len, head_dim]
"""

import numpy as np

FIXED_POINT_SCALE = 65536  # 2^16 for Q15.16

def float_to_q15_16(x: np.ndarray) -> np.ndarray:
    """Convert float to Q15.16 int32."""
    return np.clip(np.round(x * FIXED_POINT_SCALE), -2**31, 2**31 - 1).astype(np.int32)

def q15_16_to_float(x: np.ndarray) -> np.ndarray:
    """Convert Q15.16 int32 to float."""
    return x.astype(np.float32) / FIXED_POINT_SCALE

def cos_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def mse_error(a, b):
    return np.mean((a - b) ** 2)

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

def softmax_q15_16(x: np.ndarray) -> np.ndarray:
    """
    Compute row-wise softmax in Q15.16 fixed-point format.

    Strategy (per row):
    1. Find row max for numerical stability.
    2. Compute exp(x - max) using exp_q15_16.   result in Q0.16
    3. Sum the row exponentials.
    4. Compute 1/sum scaled to Q1.31: (2^31 - 1) // sum.
    5. Multiply each exp by (1/sum) and shift right 15 → Q15.16 output.
    """
    was_1d = x.ndim == 1
    if was_1d:
        x = x[np.newaxis, :]

    out = np.empty_like(x)
    for i in range(x.shape[0]):
        row   = x[i]
        x_max = np.max(row)
        x_diff = row - x_max
        x_exp  = exp_q15_16(x_diff)
        x_sum  = np.sum(x_exp)
        inv    = ((1 << 31) - 1) // x_sum
        out[i] = ((x_exp.astype(np.int64) * inv) >> 15).astype(np.int32)

    return out[0] if was_1d else out


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax.  x: [seq_len, seq_len]"""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def attention(
    q: np.ndarray,   # [seq_len, head_dim]
    k: np.ndarray,   # [seq_len, head_dim]
    v: np.ndarray,   # [seq_len, head_dim]
    scale: float | None = None,
) -> np.ndarray:     # [seq_len, head_dim]
    if scale is None:
        scale = q.shape[-1] ** -0.5
    scores = q @ k.T * scale   # [seq_len, seq_len]
    attn   = softmax(scores)   # [seq_len, seq_len]
    return attn @ v            # [seq_len, head_dim]

def fp32_to_multiplier_shift(v: float) -> tuple[int, int]:
    """
    Encode a positive float as (multiplier, shift) such that
      multiplier / 2^shift  ≈  v
    with multiplier fitting in int32 (≈ 2^30 .. 2^31).
    Apply as: round(x * v) ≈ (x * multiplier + (1 << (shift-1))) >> shift
    """
    import math
    assert v > 0
    e = math.floor(math.log2(v + 1e-38))
    shift = 30 - e                              # multiplier lives in [2^30, 2^31)
    multiplier = min(round(v * (1 << shift)), (1 << 31) - 1)
    return int(multiplier), int(shift)


def apply_mul_shift(x: np.ndarray, multiplier: int, shift: int) -> np.ndarray:
    """Integer multiply-shift: round(x * multiplier / 2^shift)."""
    round_offset = np.int64(1) << max(shift - 1, 0)
    return ((x.astype(np.int64) * multiplier + round_offset) >> shift)


def quant_int8_per_tensor(x: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Per-tensor symmetric int8 quantization of a Q15.16 int32 matrix.

    Returns:
      x_int8          : [rows, cols]  int8
      multiplier, shift: int32 scale encoding,  actual_float = x_int8 * multiplier / 2^shift
    """
    max_abs_float = float(np.max(np.abs(x))) / FIXED_POINT_SCALE   # de-Q15.16
    scale_fp = max(max_abs_float / 127.0, 1e-38)
    multiplier, shift = fp32_to_multiplier_shift(scale_fp)
    # x_int8 = round(x_q15_16 / (scale_fp * 2^16))
    denom = float(multiplier) / (1 << shift) * FIXED_POINT_SCALE   # ≈ scale_fp * 2^16
    x_int8 = np.clip(np.round(x.astype(np.float64) / denom), -127, 127).astype(np.int8)
    return x_int8, multiplier, shift


def attention_q15_16_int8(
    q_int8: np.ndarray,       # [seq_len, head_dim]  int8
    k_int8: np.ndarray,       # [seq_len, head_dim]  int8
    v_int8: np.ndarray,       # [seq_len, head_dim]  int8
    mul_q: int, shift_q: int, # Q scale: actual = mul_q / 2^shift_q
    mul_k: int, shift_k: int, # K scale
    mul_v: int, shift_v: int, # V scale
    sm_scale: float | None = None,
) -> np.ndarray:              # [seq_len, head_dim]  Q15.16 int32
    """
    Fully-integer attention — no fp32 after quantization.

    Scales are represented as (multiplier, shift): value = mul / 2^shift.
    All rescaling uses:  round(x * mul / 2^shift)  =  (x*mul + 2^(shift-1)) >> shift

    Step 1 – QK^T → Q15.16 scores (pure integer):
      dot[i,j]           = sum_d q_int8[i,d] * k_int8[j,d]          (int32)
      combined scale     = scale_q * scale_k * sm_scale
                         → (mul_c, shift_c) via fp32_to_multiplier_shift
      We want scores_q15_16 = dot * combined_scale, expressed in Q15.16
      ⟹ effective factor  = combined_scale * 2^16  → (mul_c16, shift_c16)
      scores_q15_16[i,j] = (dot * mul_c16 + round) >> shift_c16      (int32)

    Step 2 – Row-wise softmax in Q15.16 (unchanged).

    Step 3 – Weighted sum of V → Q15.16 (pure integer):
      acc[i,d]           = sum_j attn_q15_16[i,j] * v_int8[j,d]      (int64)
      out = acc * scale_v   (since attn is Q0.16 and v_int8 * scale_v = v_float)
          = (acc * mul_v + round) >> shift_v                          (int32 Q15.16)
    """
    if sm_scale is None:
        sm_scale = q_int8.shape[-1] ** -0.5

    # ------------------------------------------------------------------ #
    # 1. QK^T  →  Q15.16 scores  (all integer)
    # ------------------------------------------------------------------ #
    dot = q_int8.astype(np.int32) @ k_int8.astype(np.int32).T         # [S, S] int32

    # combined scale = scale_q * scale_k * sm_scale, then × 2^16 for Q15.16 output
    combined = (float(mul_q) / (1 << shift_q)) * (float(mul_k) / (1 << shift_k)) * sm_scale
    mul_c16, shift_c16 = fp32_to_multiplier_shift(combined * FIXED_POINT_SCALE)
    scores_q15_16 = apply_mul_shift(dot, mul_c16, shift_c16).astype(np.int32)

    # ------------------------------------------------------------------ #
    # 2. Softmax in Q15.16
    # ------------------------------------------------------------------ #
    attn_q15_16 = softmax_q15_16(scores_q15_16)                        # [S, S] Q15.16

    # ------------------------------------------------------------------ #
    # 3. Weighted sum of V  →  Q15.16 output  (all integer)
    #
    #   attn_q15_16[i,j] = attn[i,j] * 2^16
    #   v_int8[j,d]      = v_float[j,d] / scale_v  = v_float * 2^shift_v / mul_v
    #   acc              = attn_q15_16 @ v_int8
    #                    = output_float * 2^16 * 2^shift_v / mul_v
    #   out_q15_16       = (acc * mul_v + round) >> shift_v
    # ------------------------------------------------------------------ #
    acc = attn_q15_16.astype(np.int64) @ v_int8.astype(np.int64)       # [S, D] int64
    out_q15_16 = apply_mul_shift(acc, mul_v, shift_v).astype(np.int32)
    return out_q15_16






def test_softmax_q15_16():
    x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    softmax_fp = softmax(x)
    softmax_fixed = softmax_q15_16(float_to_q15_16(x))
    print("Softmax (float32):", softmax_fp)
    print("Softmax (Q15.16):", q15_16_to_float(softmax_fixed))

    print(f"first 5 elements of float32 softmax: {softmax_fp.flatten()[:5]}")
    print(f"first 5 elements of q15_16: {q15_16_to_float(softmax_fixed).flatten()[:5]}")
    print("Cosine similarity:", cos_similarity(softmax_fp.flatten(), q15_16_to_float(softmax_fixed).flatten()))
    print("MSE error:", mse_error(softmax_fp.flatten(), q15_16_to_float(softmax_fixed).flatten()))


def test_attention_q15_16_int8():
    print("\n" + "=" * 60)
    print("Test: attention_q15_16_int8  (seq_len=8, head_dim=32)")
    print("=" * 60)
    np.random.seed(42)
    seq_len, head_dim = 8, 32

    q_f = np.random.randn(seq_len, head_dim).astype(np.float32)
    k_f = np.random.randn(seq_len, head_dim).astype(np.float32)
    v_f = np.random.randn(seq_len, head_dim).astype(np.float32)

    # float32 reference
    out_ref = attention(q_f, k_f, v_f)

    # quantize float → Q15.16, then per-tensor int8
    q_int8, mq, shq = quant_int8_per_tensor(float_to_q15_16(q_f))
    k_int8, mk, shk = quant_int8_per_tensor(float_to_q15_16(k_f))
    v_int8, mv, shv = quant_int8_per_tensor(float_to_q15_16(v_f))
    print(f"  Q scale: {mq} >> {shq}  ≈  {mq / (1 << shq):.6f}")
    print(f"  K scale: {mk} >> {shk}  ≈  {mk / (1 << shk):.6f}")
    print(f"  V scale: {mv} >> {shv}  ≈  {mv / (1 << shv):.6f}")

    out_q = attention_q15_16_int8(q_int8, k_int8, v_int8, mq, shq, mk, shk, mv, shv)
    out_q_f = q15_16_to_float(out_q)

    cos = cos_similarity(out_ref.flatten(), out_q_f.flatten())
    mse = mse_error(out_ref.flatten(), out_q_f.flatten())
    max_err = np.max(np.abs(out_ref - out_q_f))

    print(f"  float32 out[0,:4]: {out_ref[0,:4]}")
    print(f"  int8    out[0,:4]: {out_q_f[0,:4]}")
    print(f"  Cosine similarity: {cos:.6f}")
    print(f"  MSE error:         {mse:.6f}")
    print(f"  Max abs error:     {max_err:.6f}")
    

if __name__ == "__main__":
    np.random.seed(0)
    seq_len, head_dim = 8, 32
    q = np.random.randn(seq_len, head_dim).astype(np.float32)
    k = np.random.randn(seq_len, head_dim).astype(np.float32)
    v = np.random.randn(seq_len, head_dim).astype(np.float32)
    out = attention(q, k, v)
    print("output shape:", out.shape)
    print("output[0]:", out[0])

    print("\nTesting softmax_q15_16...")
    test_softmax_q15_16()

    test_attention_q15_16_int8()

