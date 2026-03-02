"""
Unit tests for Int8 quantized attention backend.
"""

import pytest
import torch
import numpy as np
from typing import List

from minisgl.core import Batch, Req, SamplingParams, Context, set_global_ctx
from minisgl.attention import Int8Backend, FlashInferBackend, create_attention_backend
from minisgl.models import ModelConfig
from minisgl.kvcache import RadixCacheManager
from minisgl.distributed import TPInfo


class MockKVCache:
    """Mock KV cache for testing."""
    def __init__(self, device, dtype, num_layers, max_seq_len):
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.k_cache = {}
        self.v_cache = {}

    def k_cache(self, layer_id):
        if layer_id not in self.k_cache:
            self.k_cache[layer_id] = torch.zeros(
                self.max_seq_len, self.config.num_kv_heads, self.config.head_dim,
                dtype=self.dtype, device=self.device
            )
        return self.k_cache[layer_id]

    def v_cache(self, layer_id):
        if layer_id not in self.v_cache:
            self.v_cache[layer_id] = torch.zeros(
                self.max_seq_len, self.config.num_kv_heads, self.config.head_dim,
                dtype=self.dtype, device=self.device
            )
        return self.v_cache[layer_id]

    def store_kv(self, k, v, out_loc, layer_id):
        if layer_id not in self.k_cache:
            self.k_cache[layer_id] = torch.zeros(
                self.max_seq_len, k.shape[1], k.shape[2],
                dtype=self.dtype, device=self.device
            )
        if layer_id not in self.v_cache:
            self.v_cache[layer_id] = torch.zeros(
                self.max_seq_len, v.shape[1], v.shape[2],
                dtype=self.dtype, device=self.device
            )
        num_tokens = k.shape[0]
        self.k_cache[layer_id][out_loc:out_loc+num_tokens] = k
        self.v_cache[layer_id][out_loc:out_loc+num_tokens] = v


def create_mock_batch(
    device,
    batch_size: int,
    seq_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_prefill: bool = True,
) -> Batch:
    """Create a mock batch for testing."""
    reqs = []
    for i in range(batch_size):
        input_ids = torch.randint(0, 32000, (seq_len,), device="cpu")
        req = Req(
            input_ids=input_ids,
            table_idx=i,
            cached_len=0 if is_prefill else seq_len - 1,
            output_len=100,
            uid=i,
            sampling_params=SamplingParams(),
            cache_handle=None,
        )
        if not is_prefill:
            req.device_len = seq_len
            req.cached_len = seq_len - 1
        reqs.append(req)

    batch = Batch(reqs=reqs, phase="prefill" if is_prefill else "decode")
    batch.input_ids = torch.cat([req.input_ids for req in reqs]).to(device)
    batch.positions = torch.cat([
        torch.arange(req.cached_len, req.device_len, device=device)
        for req in reqs
    ])
    batch.out_loc = torch.tensor([i * seq_len for i in range(batch_size)], device=device)
    batch.padded_reqs = reqs

    return batch


def create_model_config(
    num_qo_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    num_layers: int = 32,
) -> ModelConfig:
    """Create a mock model config."""
    from dataclasses import dataclass
    @dataclass
    class MockModelConfig:
        num_qo_heads: int
        num_kv_heads: int
        head_dim: int
        num_layers: int
        num_hidden_layers: int
        hidden_size: int

    hidden_size = num_qo_heads * head_dim
    return MockModelConfig(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
    )


class TestInt8Attention:
    """Test Int8 quantized attention backend."""

    @pytest.fixture
    def setup_backend(self):
        """Setup test environment."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = "cuda"
        dtype = torch.float16
        num_layers = 1
        max_seq_len = 2048

        # Mock TP info
        from minisgl.distributed import _TP_INFO
        original_tp_info = _TP_INFO.__dict__.copy()
        _TP_INFO.__dict__.clear()
        _TP_INFO.__dict__.update({
            'size': 1,
            'rank': 0,
            'local_rank': 0,
        })

        # Setup global context
        ctx = Context(page_size=1)
        page_table = torch.arange(0, max_seq_len * 10, dtype=torch.int32, device=device).view(10, max_seq_len)
        ctx.page_table = page_table
        set_global_ctx(ctx)

        yield device, dtype, num_layers, max_seq_len

        # Restore TP info
        _TP_INFO.__dict__.clear()
        _TP_INFO.__dict__.update(original_tp_info)

    def test_quantize_dequantize(self, setup_backend):
        """Test per-token dynamic quantization and dequantization."""
        device, dtype, num_layers, max_seq_len = setup_backend
        config = create_model_config(num_qo_heads=8, num_kv_heads=8, head_dim=128)

        kvcache = MockKVCache(device, dtype, num_layers, max_seq_len)
        kvcache.config = config

        backend = Int8Backend(config, kvcache)

        # Test data
        torch.manual_seed(42)
        x = torch.randn(64, 8, 128, device=device, dtype=dtype) * 0.5

        # Quantize and dequantize
        x_int8, scale = backend._quantize_to_int8(x)
        x_deq = backend._dequantize_from_int8(x_int8, scale, dtype)

        # Check quantization bounds
        assert x_int8.dtype == torch.int8
        assert x_int8.min() >= -128 and x_int8.max() <= 127

        # Check dequantization accuracy
        rel_error = (x - x_deq).abs() / (x.abs() + 1e-6)
        max_rel_error = rel_error.max().item()

        print(f"Quantization max relative error: {max_rel_error:.6f}")
        print(f"Quantization scale range: [{scale.min().item():.6f}, {scale.max().item():.6f}]")

        # The error should be reasonably small (int8 quantization)
        assert max_rel_error < 0.01, f"Quantization error too large: {max_rel_error}"

    def test_quantize_edge_cases(self, setup_backend):
        """Test quantization with edge cases."""
        device, dtype, num_layers, max_seq_len = setup_backend
        config = create_model_config(num_qo_heads=8, num_kv_heads=8, head_dim=128)

        kvcache = MockKVCache(device, dtype, num_layers, max_seq_len)
        kvcache.config = config

        backend = Int8Backend(config, kvcache)

        # Test with zeros
        x_zeros = torch.zeros(64, 8, 128, device=device, dtype=dtype)
        x_int8, scale = backend._quantize_to_int8(x_zeros)
        x_deq = backend._dequantize_from_int8(x_int8, scale, dtype)
        assert x_deq.abs().max().item() < 1e-5, "Zeros should produce near-zero output"

        # Test with large values
        x_large = torch.randn(64, 8, 128, device=device, dtype=dtype) * 10.0
        x_int8, scale = backend._quantize_to_int8(x_large)
        x_deq = backend._dequantize_from_int8(x_int8, scale, dtype)
        rel_error = (x_large - x_deq).abs() / (x_large.abs() + 1e-6)
        max_rel_error = rel_error.max().item()
        print(f"Large values max relative error: {max_rel_error:.6f}")
        assert max_rel_error < 0.02, f"Large values error too large: {max_rel_error}"

        # Test with negative values
        x_neg = -torch.abs(torch.randn(64, 8, 128, device=device, dtype=dtype))
        x_int8, scale = backend._quantize_to_int8(x_neg)
        x_deq = backend._dequantize_from_int8(x_int8, scale, dtype)
        rel_error = (x_neg - x_deq).abs() / (x_neg.abs() + 1e-6)
        max_rel_error = rel_error.max().item()
        print(f"Negative values max relative error: {max_rel_error:.6f}")
        assert max_rel_error < 0.01, f"Negative values error too large: {max_rel_error}"

    def test_forward_prefill(self, setup_backend):
        """Test Int8 attention forward pass for prefill."""
        device, dtype, num_layers, max_seq_len = setup_backend
        config = create_model_config(num_qo_heads=8, num_kv_heads=8, head_dim=128)

        kvcache = MockKVCache(device, dtype, num_layers, max_seq_len)
        kvcache.config = config

        backend = Int8Backend(config, kvcache)
        ref_backend = FlashInferBackend(config, kvcache)

        # Create batch
        batch_size = 4
        seq_len = 32
        batch = create_mock_batch(device, batch_size, seq_len, 8, 8, 128, is_prefill=True)

        # Prepare metadata
        backend.prepare_metadata(batch)
        ref_backend.prepare_metadata(batch)

        # Create Q, K, V
        torch.manual_seed(42)
        q = torch.randn(batch_size * seq_len, 8, 128, device=device, dtype=dtype) * 0.1
        k = torch.randn(batch_size * seq_len, 8, 128, device=device, dtype=dtype) * 0.1
        v = torch.randn(batch_size * seq_len, 8, 128, device=device, dtype=dtype) * 0.1

        # Reset cache for each backend
        kvcache_int8 = MockKVCache(device, dtype, num_layers, max_seq_len)
        kvcache_int8.config = config
        kvcache_float = MockKVCache(device, dtype, num_layers, max_seq_len)
        kvcache_float.config = config

        backend.kvcache = kvcache_int8
        ref_backend.kvcache = kvcache_float

        # Run forward
        out_int8 = backend.forward(q.clone(), k.clone(), v.clone(), layer_id=0, batch=batch)
        out_float = ref_backend.forward(q.clone(), k.clone(), v.clone(), layer_id=0, batch=batch)

        # Compare outputs
        rel_error = (out_int8 - out_float).abs() / (out_float.abs() + 1e-6)
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()

        print(f"Prefill - Int8 vs Float: max rel error = {max_rel_error:.6f}, mean rel error = {mean_rel_error:.6f}")

        # Int8 quantization should have reasonable accuracy
        assert max_rel_error < 0.05, f"Prefill max error too large: {max_rel_error}"

    def test_forward_decode(self, setup_backend):
        """Test Int8 attention forward pass for decode."""
        device, dtype, num_layers, max_seq_len = setup_backend
        config = create_model_config(num_qo_heads=8, num_kv_heads=8, head_dim=128)

        kvcache = MockKVCache(device, dtype, num_layers, max_seq_len)
        kvcache.config = config

        backend = Int8Backend(config, kvcache)
        ref_backend = FlashInferBackend(config, kvcache)

        # Create batch
        batch_size = 4
        seq_len = 128  # cached length
        batch = create_mock_batch(device, batch_size, seq_len, 8, 8, 128, is_prefill=False)

        # Prepare metadata
        backend.prepare_metadata(batch)
        ref_backend.prepare_metadata(batch)

        # Create Q, K, V (single token per request for decode)
        torch.manual_seed(42)
        q = torch.randn(batch_size, 8, 128, device=device, dtype=dtype) * 0.1
        k = torch.randn(batch_size, 8, 128, device=device, dtype=dtype) * 0.1
        v = torch.randn(batch_size, 8, 128, device=device, dtype=dtype) * 0.1

        # Reset cache for each backend
        kvcache_int8 = MockKVCache(device, dtype, num_layers, max_seq_len)
        kvcache_int8.config = config
        kvcache_float = MockKVCache(device, dtype, num_layers, max_seq_len)
        kvcache_float.config = config

        backend.kvcache = kvcache_int8
        ref_backend.kvcache = kvcache_float

        # Run forward
        out_int8 = backend.forward(q.clone(), k.clone(), v.clone(), layer_id=0, batch=batch)
        out_float = ref_backend.forward(q.clone(), k.clone(), v.clone(), layer_id=0, batch=batch)

        # Compare outputs
        rel_error = (out_int8 - out_float).abs() / (out_float.abs() + 1e-6)
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()

        print(f"Decode - Int8 vs Float: max rel error = {max_rel_error:.6f}, mean rel error = {mean_rel_error:.6f}")

        # Int8 quantization should have reasonable accuracy
        assert max_rel_error < 0.05, f"Decode max error too large: {max_rel_error}"

    def test_different_batch_sizes(self, setup_backend):
        """Test with different batch sizes."""
        device, dtype, num_layers, max_seq_len = setup_backend
        config = create_model_config(num_qo_heads=8, num_kv_heads=8, head_dim=128)

        for batch_size in [1, 2, 8, 16]:
            kvcache = MockKVCache(device, dtype, num_layers, max_seq_len)
            kvcache.config = config

            backend = Int8Backend(config, kvcache)
            batch = create_mock_batch(device, batch_size, 32, 8, 8, 128, is_prefill=True)

            backend.prepare_metadata(batch)

            torch.manual_seed(42)
            q = torch.randn(batch_size * 32, 8, 128, device=device, dtype=dtype) * 0.1
            k = torch.randn(batch_size * 32, 8, 128, device=device, dtype=dtype) * 0.1
            v = torch.randn(batch_size * 32, 8, 128, device=device, dtype=dtype) * 0.1

            output = backend.forward(q, k, v, layer_id=0, batch=batch)

            # Check output shape
            assert output.shape == (batch_size * 32, 8, 128), f"Output shape mismatch for batch_size={batch_size}"

            print(f"Batch size {batch_size}: OK")

    def test_gqa_configuration(self, setup_backend):
        """Test with Grouped Query Attention (GQA)."""
        device, dtype, num_layers, max_seq_len = setup_backend

        # Test different GQA ratios
        for num_qo_heads, num_kv_heads in [(32, 8), (16, 4), (8, 2)]:
            config = create_model_config(num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, head_dim=128)

            kvcache = MockKVCache(device, dtype, num_layers, max_seq_len)
            kvcache.config = config

            backend = Int8Backend(config, kvcache)
            batch = create_mock_batch(device, 4, 32, num_qo_heads, num_kv_heads, 128, is_prefill=True)

            backend.prepare_metadata(batch)

            torch.manual_seed(42)
            q = torch.randn(128, num_qo_heads, 128, device=device, dtype=dtype) * 0.1
            k = torch.randn(128, num_kv_heads, 128, device=device, dtype=dtype) * 0.1
            v = torch.randn(128, num_kv_heads, 128, device=device, dtype=dtype) * 0.1

            output = backend.forward(q, k, v, layer_id=0, batch=batch)

            # Check output shape
            assert output.shape == (128, num_qo_heads, 128), f"Output shape mismatch for QO={num_qo_heads}, KV={num_kv_heads}"

            print(f"GQA QO={num_qo_heads}, KV={num_kv_heads}: OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])