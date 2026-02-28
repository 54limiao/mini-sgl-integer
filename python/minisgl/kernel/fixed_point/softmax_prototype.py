import numpy as np

def cos_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def mse_error(a, b):
    return np.mean((a - b) ** 2)

def quantize_f32i16(x, scale=None):
    if scale is None:
        max = np.max(np.abs(x))
    else:
        max = scale
        x = np.clip(x, -max, max)
    largest_exp2 = np.ceil(np.log2(max + 1e-20))
    shift = (31 - largest_exp2).astype(np.int32)
    multiplier = np.minimum(max * 2.0**shift, np.iinfo(np.int32).max).astype(np.int32)
    quantized = np.round(x / max * np.iinfo(np.int16).max).astype(np.int16)
    return quantized, multiplier, shift + 15

def quantize_i10(x, multiplier, shift):
    x = x.astype(np.int64)
    round_offset = (1 << (shift - 1))
    out = (x * multiplier + round_offset) >> shift
    # out = x * multiplier >> shift
    out = np.clip(out, -512, 511)
    return out.astype(np.int16)

def quantize_i16(x, multiplier, shift):
    x = x.astype(np.int64)
    round_offset = (1 << (shift - 1))
    out = (x * multiplier + round_offset) >> shift
    # out = x * multiplier >> shift
    out = np.clip(out, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return out.astype(np.int16)

def quantize_i32(x, multiplier, shift):
    x = x.astype(np.int64)
    round_offset = (1 << (shift - 1))
    out = (x * multiplier + round_offset) >> shift
    # out = x * multiplier >> shift
    out = np.clip(out, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    return out.astype(np.int32)

def dequantize(x, multiplier, shift):
    scale = multiplier / 2.0**shift
    return x.astype(np.float32) * scale

def precompute_softmax(multiplier, shift, lut_length):
    # precompute the softmax lookup table and data
    log2e = 1.442695040888963387
    multiplier = np.int32(log2e * multiplier / 2.0 + 0.5)
    shift -= 1
    exp2_lut = np.exp2(np.linspace(0, 1, lut_length).astype(np.float32))
    # we can only use 10bit for lut index, and 10 bit for lut value :(
    exp2_lut_i32 = np.int32(np.clip((exp2_lut * (1 << 9) + 0.5).astype(np.int64), 0, 1023))
    return multiplier, shift, exp2_lut_i32

def softmax_fixed_point_test(x, multiplier, shift):
    LOOKUP_BIT_LENGTH = 10
    SUMMATION_SHIFT = 8
    DIVITON_SHIFT = 0
    multiplier, shift, exp2_lut_i32 = precompute_softmax(multiplier, shift, 2 ** LOOKUP_BIT_LENGTH)

    x_max = np.max(x)
    diff = np.clip(x.astype(np.int32) - x_max, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    x_int32 = quantize_i32(diff, multiplier, shift - LOOKUP_BIT_LENGTH)
    # e**x = 2**(log2e*x) = 2**(x_k + x_fract) where x_k is integer part, x_fract is LOOKUP_BIT fractional part
    x_k = x_int32 >> LOOKUP_BIT_LENGTH
    x_fract = x_int32 & ((1 << LOOKUP_BIT_LENGTH) - 1)
    x_exp = exp2_lut_i32[x_fract]
    # softmax is neg exp calculation, so x_k is always negative, and use 10 as summation shift to avoid overflow
    x_exp = x_exp << SUMMATION_SHIFT
    x_exp_i32 = x_exp >> (-x_k)
    # div shift to improve one by sum precision
    x_exp_sum = np.sum(x_exp_i32) >> DIVITON_SHIFT
    x_one_by_exp_sum = ((1 << 31) - 1) // (x_exp_sum)

    # return dequantize(x_exp, x_one_by_exp_sum,  31 - x_k + DIVITON_SHIFT)
    # return dequantize(x_exp_i32, x_one_by_exp_sum, 31 + DIVITON_SHIFT)
    # return quantize_i16(x_exp, x_one_by_exp_sum, 31 - 15 + DIVITON_SHIFT - x_k) / 32768.0
    return quantize_i16(x_exp_i32, x_one_by_exp_sum, 31 - 15 + DIVITON_SHIFT) / 32768.0

def softmax_fixed_point_test2(x, multiplier, shift):
    # precompute exp lut
    exp_lut = np.exp(np.linspace(-8, 0, 1024).astype(np.float32))
    exp_lut_i32 = np.clip(exp_lut * (1 << 10) + 0.5, 0, 1023).astype(np.int32)

    x_max = np.max(x)
    # numpy do not support saturate directly, so we use clip
    diff = np.clip(x.astype(np.int32) - x_max, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    # quantize x - x_max to [-8, 0]
    x_i32 = quantize_i32(diff, multiplier, shift - 7) + 1023
    x_i32 = np.clip(x_i32, 0, 1023)
    x_lut_0 = exp_lut_i32[x_i32]
    sum_exp = np.sum(x_lut_0.astype(np.int32))
    x_one_by_exp_sum = ((1 << 31) - 1) // sum_exp

    # return dequantize(x_lut_0, x_one_by_exp_sum, 31)
    return quantize_i16(x_lut_0, x_one_by_exp_sum, 31 - 15) / 32768.0

if __name__ == "__main__":
    # set seed
    np.random.seed(0)
    np.set_printoptions(precision=8, suppress=True, linewidth=120)

    # 示例输入数据
    # input_data = np.random.uniform(0, 20, size=128).astype(np.float32)
    input_data = np.random.randn(100).astype(np.float32) * 10 + 10
    # input_data = np.array([-15.9, -30, -3.4, -4.5, 0.0], dtype=np.float32)
    print(f"{'='*20} Input Data {'='*20}")
    print(f"Input Data (first 5):       {input_data[:5]}")

    quantized_data, multiplier, shift = quantize_f32i16(input_data)
    print(f"Quantized Data (first 5):   {quantized_data[:5]}")
    print(f"Multiplier: {multiplier}, Shift: {shift}")

    dequantized_data = dequantize(quantized_data, multiplier, shift)
    print(f"Dequantized Data (first 5): {dequantized_data[:5]}")
    
    # 计算softmax
    exp_data = np.exp(dequantized_data - np.max(dequantized_data))
    softmax_f32 = exp_data / np.sum(exp_data)
    
    print(f"\n{'='*20} Softmax Results {'='*20}")
    print(f"Softmax F32 (first 5):         {softmax_f32[:5]}")

    softmax_fixed_test = softmax_fixed_point_test(quantized_data, multiplier, shift)
    print("-" * 60)
    print(f"Softmax Fixed Test1 (first 5): {softmax_fixed_test[:5]}")
    print(f"Cos Sim (F32 vs Test1):        {cos_similarity(softmax_f32, softmax_fixed_test):.8f}")
    print(f"MSE Error (F32 vs Test1):      {mse_error(softmax_f32, softmax_fixed_test):.8e}")

    softmax_fixed_test2 = softmax_fixed_point_test2(quantized_data, multiplier, shift)
    print("-" * 60)
    print(f"Softmax Fixed Test2 (first 5): {softmax_fixed_test2[:5]}")
    print(f"Cos Sim (F32 vs Test2):        {cos_similarity(softmax_f32, softmax_fixed_test2):.8f}")
    print(f"MSE Error (F32 vs Test2):      {mse_error(softmax_f32, softmax_fixed_test2):.8e}")
    print("=" * 60)


# ==================== Softmax Results ====================
# Softmax F32 (first 5):         [0.0034427  0.         0.00000134 0.40521291 0.00969581]
# ------------------------------------------------------------
# Softmax Fixed Test1 (first 5): [0.00344652 0.         0.00000134 0.40522767 0.00969152]
# Cos Sim (F32 vs Test1):        1.00000000
# MSE Error (F32 vs Test1):      5.54595989e-12
# ------------------------------------------------------------
# Softmax Fixed Test2 (first 5): [0.00317628 0.         0.         0.40603486 0.00952885]
# Cos Sim (F32 vs Test2):        0.99999950
# MSE Error (F32 vs Test2):      1.28280958e-08
# ============================================================