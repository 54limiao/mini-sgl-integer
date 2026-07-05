from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Tuple

import torch

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, num_splits, *config)
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if os.environ.get("MINISGL_INDEX_REF") == "1":
        if vocab_range is None:
            return weights[indices.long()].contiguous()
        start, length = vocab_range
        local = indices.long() - start
        mask = (local >= 0) & (local < length)
        out = weights.new_zeros(indices.shape[0], weights.shape[1])
        if mask.any():
            out[mask] = weights[local[mask]]
        return out

    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    element_size = weights.shape[1] * weights.element_size()
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1
    module = _jit_index_module(element_size, num_splits=num_splits)
    module.launch(weights, indices, output, vocab_range)
    return output
