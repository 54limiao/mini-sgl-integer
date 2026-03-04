"""
W8A8 Int8 Quantization for Mini-SGLang

Adapted from SGLang's compressed_tensors_w8a8_int8.py
Simplified and self-contained for mini-sglang.

Key features:
- Weight: int8, per-channel static quantization
- Activation: int8, per-token dynamic quantization (or static per-tensor when input_scale exists)
- Computation: int8_scaled_mm kernel
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.nn import Parameter

from minisgl.kernel.fixed_point import FIXED_POINT_SCALE, assert_q15, from_fixed, to_fixed

# sgl_kernel import is deferred to avoid CUDA initialization at module load time
_HAS_INT8_KERNEL = None


def _has_int8_kernel():
    global _HAS_INT8_KERNEL
    if _HAS_INT8_KERNEL is None:
        try:
            from sgl_kernel import int8_scaled_mm

            _HAS_INT8_KERNEL = True
        except ImportError:
            _HAS_INT8_KERNEL = False
    return _HAS_INT8_KERNEL


def _get_int8_scaled_mm():
    from sgl_kernel import int8_scaled_mm

    return int8_scaled_mm


def per_token_quant_int8(x: torch.Tensor):
    """
    Per-token quantization to int8.

    Args:
        x: Input tensor

    Returns:
        x_q: Quantized int8 tensor
        scales: Per-token scales (float32)
    """
    # Reshape to 2D if needed
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])

    # Compute per-token max - ensure float32 for scale
    x_abs_max = x_2d.abs().amax(dim=1, keepdim=True).to(torch.float32)
    x_scale = x_abs_max / 127.0
    x_scale = torch.clamp(x_scale, min=1e-10)

    # Quantize
    x_q = torch.round(x_2d / x_scale).to(torch.int8)
    x_q = torch.clamp(x_q, -128, 127)

    # Reshape back
    x_q = x_q.view(orig_shape)
    scales = x_scale.view(orig_shape[:-1] + (1,))

    return x_q, scales


def per_tensor_quant_int8_static(x: torch.Tensor, input_scale: torch.Tensor):
    """Per-tensor static quantization to int8 using checkpoint-provided input_scale."""
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])

    scale = input_scale.to(device=x.device, dtype=torch.float32)
    if scale.numel() > 1:
        scale = scale.max()
    scale = torch.clamp(scale, min=1e-10)

    x_q = torch.round(x_2d / scale).to(torch.int32)
    x_q = torch.clamp(x_q, -128, 127).to(torch.int8)

    scales = scale.view(1, 1).repeat(x_2d.shape[0], 1)

    x_q = x_q.view(orig_shape)
    scales = scales.view(orig_shape[:-1] + (1,))
    return x_q, scales


def _get_static_input_scale(layer: torch.nn.Module) -> Optional[torch.Tensor]:
    input_scale = getattr(layer, "input_scale", None)
    if input_scale is None or not isinstance(input_scale, torch.Tensor):
        return None
    if input_scale.numel() == 0:
        return None
    return input_scale


def per_token_quant_int8_from_q15(x_q15: torch.Tensor):
    """Per-token quantization from Q15.16 int32 to int8.

    Args:
        x_q15: Input tensor in Q15.16 int32.

    Returns:
        x_q: Quantized int8 tensor.
        scales: Per-token scales in float32 (real-value scale for int8 dequant).
    """
    assert_q15(x_q15, "x_q15")

    orig_shape = x_q15.shape
    x_2d = x_q15.view(-1, x_q15.shape[-1])

    x_abs_max_q15 = x_2d.abs().amax(dim=1, keepdim=True).to(torch.int32)
    x_abs_max_q15 = torch.clamp(x_abs_max_q15, min=1)

    x_q = torch.round(
        (x_2d.to(torch.float32) * 127.0) / x_abs_max_q15.to(torch.float32)
    ).to(torch.int32)
    x_q = torch.clamp(x_q, -128, 127).to(torch.int8)

    scales = x_abs_max_q15.to(torch.float32) / (127.0 * float(FIXED_POINT_SCALE))

    x_q = x_q.view(orig_shape)
    scales = scales.view(orig_shape[:-1] + (1,))
    return x_q, scales


class W8A8Int8LinearMethod:
    """
    Linear method for W8A8 INT8 quantization.

    - Weight: static, per-channel, symmetric
    - Activation: dynamic per-token (default) or static per-tensor with input_scale
    """

    def __init__(self):
        pass

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
    ):
        """
        Create int8 weights and their scales for a linear layer.

        Args:
            layer: The torch.nn.Module to add parameters to
            output_partition_sizes: List of output sizes for each partition
            input_size_per_partition: Input size for this partition
            params_dtype: Dtype for parameters (unused for int8 weights)
            weight_loader: Function to load weights
        """
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # Int8 weight tensor [out_features, in_features]
        weight = Parameter(
            torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)

        # Weight scale [out_features, 1] for per-channel quantization
        weight_scale = Parameter(
            torch.empty((output_size_per_partition, 1), dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Process weights after loading.
        Weight stays as [out_features, in_features], will be transposed in apply_weights.
        """
        # No processing needed - weight will be transposed on-the-fly in apply_weights
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply W8A8 quantized linear transformation.

        Args:
            layer: Module with weight and weight_scale
            x: Input tensor (bfloat16/float16)
            bias: Optional bias

        Returns:
            Output tensor
        """
        input_scale = _get_static_input_scale(layer)
        if input_scale is not None:
            x_q, x_scale = per_tensor_quant_int8_static(x, input_scale)
        else:
            x_q, x_scale = per_token_quant_int8(x)

        # Reshape for matmul
        x_q_2d = x_q.view(-1, x_q.shape[-1]).contiguous()
        x_scale_2d = x_scale.view(-1, 1).contiguous()

        # Get weight and scale
        # Weight is stored as [out_features, in_features]
        # Transpose to [in_features, out_features] for column-major format
        weight = layer.weight
        weight_scale = layer.weight_scale

        if _has_int8_kernel():
            # Use optimized int8 kernel
            # Ensure scales are float32 as required by the kernel
            output = _get_int8_scaled_mm()(
                x_q_2d,
                weight.t(),
                x_scale_2d,  # Already float32 from per_token_quant_int8
                weight_scale.to(torch.float32),  # Ensure float32
                out_dtype=x.dtype,
                bias=bias,
            )
        else:
            weight_float = weight.to(x.dtype) * weight_scale.to(x.dtype)
            output = torch.matmul(x_q_2d.to(x.dtype) * x_scale_2d.to(x.dtype), weight_float.t())
            if bias is not None:
                output = output + bias

        return output

    def apply_weights_q15(
        self,
        layer: torch.nn.Module,
        x_q15: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply W8A8 quantized linear with Q15.16 int32 input/output."""
        assert_q15(x_q15, "x_q15")

        input_scale = _get_static_input_scale(layer)
        if input_scale is not None:
            x_fp32 = from_fixed(x_q15, dtype=torch.float32)
            x_q_int8, x_scale = per_tensor_quant_int8_static(x_fp32, input_scale)
        else:
            x_q_int8, x_scale = per_token_quant_int8_from_q15(x_q15)
        x_q_2d = x_q_int8.view(-1, x_q_int8.shape[-1]).contiguous()
        x_scale_2d = x_scale.view(-1, 1).contiguous()

        weight = layer.weight
        weight_scale = layer.weight_scale

        bias_fp32 = None
        if bias is not None:
            if bias.dtype == torch.int32:
                bias_fp32 = from_fixed(bias, dtype=torch.float32)
            else:
                bias_fp32 = bias.to(torch.float32)

        if _has_int8_kernel():
            bias_kernel = bias_fp32.to(torch.bfloat16) if bias_fp32 is not None else None
            output_fp32 = _get_int8_scaled_mm()(
                x_q_2d,
                weight.t(),
                x_scale_2d,
                weight_scale.to(torch.float32),
                out_dtype=torch.bfloat16,
                bias=bias_kernel,
            ).to(torch.float32)
        else:
            weight_float = weight.to(torch.float32) * weight_scale.to(torch.float32)
            output_fp32 = torch.matmul(x_q_2d.to(torch.float32) * x_scale_2d, weight_float.t())
            if bias_fp32 is not None:
                output_fp32 = output_fp32 + bias_fp32

        return to_fixed(output_fp32)


class W8A8Int8MoEMethod:
    """
    MoE method for W8A8 INT8 quantization.
    Placeholder for MoE support.
    """

    def __init__(self):
        raise NotImplementedError(
            "W8A8Int8MoEMethod is not yet implemented. "
            "Use dense models only for now."
        )
