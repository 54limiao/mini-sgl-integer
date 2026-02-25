"""
W8A8 Int8 Quantization for Mini-SGLang

Adapted from SGLang's compressed_tensors_w8a8_int8.py
Simplified and self-contained for mini-sglang.

Key features:
- Weight: int8, per-channel static quantization
- Activation: int8, per-token dynamic quantization
- Computation: int8_scaled_mm kernel
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.nn import Parameter

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


class W8A8Int8LinearMethod:
    """
    Linear method for W8A8 INT8 quantization.

    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
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
        # Dynamic per-token quantization of activation
        x_q, x_scale = per_token_quant_int8(x)

        # Reshape for matmul
        x_q_2d = x_q.view(-1, x_q.shape[-1])
        x_scale_2d = x_scale.view(-1, 1)

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
            output = torch.matmul(x_q_2d.to(x.dtype) * x_scale_2d.to(x.dtype), weight_float)
            if bias is not None:
                output = output + bias

        return output


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
