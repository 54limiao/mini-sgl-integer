"""
Triton attention kernels for mini-sglang.

Thin wrappers around SGLang's Triton attention ops that adapt the interface
to the signatures expected by minisgl's TritonBackend.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Local Triton kernel imports (no sglang runtime dependency)
# ---------------------------------------------------------------------------
from .decode_attention import decode_attention_fwd as _sgl_decode_attention_fwd
from .extend_attention import extend_attention_fwd_unified

# Default maximum KV splits for decode attention.
# Each sequence's KV tokens are split into at most this many chunks processed
# in parallel; increasing it can improve GPU utilisation for long contexts.
_DEFAULT_MAX_KV_SPLITS: int = 8


def decode_attention_fwd(
    q: torch.Tensor,          # [batch, num_qo_heads, head_dim]
    k_cache: torch.Tensor,    # [num_pages, num_kv_heads, head_dim]
    v_cache: torch.Tensor,    # [num_pages, num_kv_heads, v_head_dim]
    att_out: torch.Tensor,    # [batch, num_qo_heads, v_head_dim]  (written in-place)
    kv_indptr: torch.Tensor,  # [batch + 1]  (int32, cumulative KV lengths)
    kv_indices: torch.Tensor, # [total_kv_tokens]  (int32 or int64, page indices)
    sm_scale: float,
    logit_cap: float = 0.0,
    max_kv_splits: int = _DEFAULT_MAX_KV_SPLITS,
) -> None:
    """
    Decode-phase (single new token per sequence) paged attention.

    Allocates the required intermediate split buffers internally and calls
    SGLang's two-stage flash-decoding Triton kernel.

    Args:
        q:           Query tensor, shape [batch, num_qo_heads, head_dim].
        k_cache:     Paged key buffer, shape [num_pages, num_kv_heads, head_dim].
        v_cache:     Paged value buffer, shape [num_pages, num_kv_heads, v_head_dim].
        att_out:     Pre-allocated output tensor; written in-place.
        kv_indptr:   KV token offset pointer, shape [batch + 1].
        kv_indices:  Physical page indices for every KV token, shape [total_kv_tokens].
        sm_scale:    Softmax temperature (typically 1 / sqrt(head_dim)).
        logit_cap:   Optional logit capping value (0.0 = disabled).
        max_kv_splits: Maximum number of KV splits (parallelism over the KV axis).
    """
    bs = q.shape[0]
    num_heads = q.shape[1]
    v_head_dim = v_cache.shape[-1]

    # SGLang kernels expect int64 indices to avoid 32-bit overflow on large caches.
    kv_indices_64 = kv_indices.to(torch.int64) if kv_indices.dtype != torch.int64 else kv_indices

    # Intermediate split buffers required by SGLang's two-stage kernel.
    # Shape: [batch, num_heads, max_kv_splits, v_head_dim]
    attn_logits = torch.empty(
        (bs, num_heads, max_kv_splits, v_head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    # Shape: [batch, num_heads, max_kv_splits]
    attn_lse = torch.empty(
        (bs, num_heads, max_kv_splits),
        dtype=torch.float32,
        device=q.device,
    )

    # Assign every sequence the maximum number of splits (static scheduling).
    # For a more adaptive schedule, compute ceil(seq_len / split_tile_size).
    num_kv_splits = torch.full(
        (bs,), max_kv_splits, dtype=torch.int32, device=q.device
    )

    _sgl_decode_attention_fwd(
        q=q,
        k_buffer=k_cache,
        v_buffer=v_cache,
        o=att_out,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices_64,
        attn_logits=attn_logits,
        attn_lse=attn_lse,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
    )


def prefill_attention_fwd(
    q: torch.Tensor,           # [total_q_tokens, num_qo_heads, head_dim]
    k_cache: torch.Tensor,     # [num_pages, num_kv_heads, head_dim]
    v_cache: torch.Tensor,     # [num_pages, num_kv_heads, v_head_dim]
    output: torch.Tensor,      # [total_q_tokens, num_qo_heads, v_head_dim]  (written in-place)
    qo_indptr: torch.Tensor,   # [batch + 1]  cumulative extend lengths
    kv_indptr: torch.Tensor,   # [batch + 1]  cumulative total KV lengths (prefix + extend)
    kv_indices: torch.Tensor,  # [total_kv_tokens]  unified page indices (prefix + extend)
    sm_scale: float,
    causal: bool = True,
    logit_cap: float = 0.0,
) -> None:
    """
    Prefill/extend-phase paged attention (supports RadixCache prefix reuse).

    Calls SGLang's unified single-stage extend Triton kernel which handles both
    the prefix (cached KV, no causal mask) and the extend (new KV, causal mask)
    regions in one pass.

    Args:
        q:           Packed query tensor for the *extend* tokens only,
                     shape [total_q_tokens, num_qo_heads, head_dim].
        k_cache:     Paged key buffer, shape [num_pages, num_kv_heads, head_dim].
        v_cache:     Paged value buffer, shape [num_pages, num_kv_heads, v_head_dim].
        output:      Pre-allocated output tensor; written in-place.
        qo_indptr:   Cumulative extend (query/output) lengths, shape [batch + 1].
        kv_indptr:   Cumulative total KV lengths (prefix + extend), shape [batch + 1].
        kv_indices:  Physical page indices for all KV tokens (prefix first, then
                     extend), length = sum(device_len_i).
        sm_scale:    Softmax temperature.
        causal:      Whether to apply causal masking within the extend region.
        logit_cap:   Optional logit capping value (0.0 = disabled).
    """
    bs = qo_indptr.shape[0] - 1

    # SGLang kernels expect int64 indices.
    kv_indices_64 = kv_indices.to(torch.int64) if kv_indices.dtype != torch.int64 else kv_indices

    # Ensure q and o are contiguous (Triton kernels require this).
    q_cont = q.contiguous()
    output_cont = output if output.is_contiguous() else output.contiguous()

    # Derive per-sequence prefix lengths.
    # prefix_len[i] = device_len[i] - extend_len[i] = cached_len[i]
    extend_lens = qo_indptr[1 : bs + 1] - qo_indptr[:bs]  # [batch]
    total_kv_lens = kv_indptr[1 : bs + 1] - kv_indptr[:bs]  # [batch]
    prefix_lens = (total_kv_lens - extend_lens).to(torch.int32)  # [batch]

    max_len_extend = int(extend_lens.max().item())
    if max_len_extend == 0:
        # Nothing to compute.
        return

    extend_attention_fwd_unified(
        q=q_cont,
        o=output_cont,
        k_buffer=k_cache,
        v_buffer=v_cache,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices_64,
        prefix_lens=prefix_lens,
        max_len_extend=max_len_extend,
        sm_scale=sm_scale,
        is_causal=causal,
        logit_cap=logit_cap,
    )

    # Copy back if output was not contiguous.
    if not output.is_contiguous():
        output.copy_(output_cont)


__all__ = ["decode_attention_fwd", "prefill_attention_fwd"]
