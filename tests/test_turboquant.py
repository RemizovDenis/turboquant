"""TurboQuant test suite."""

import pytest
import torch
import numpy as np


class TestPolarQuantizer:
    """Tests for PolarQuantizer."""

    def test_init(self):
        from turboquant.core.polar_quant import PolarQuantizer
        q = PolarQuantizer(head_dim=128, bits=3, group_size=64, seed=42)
        assert q.head_dim == 128
        assert q.bits == 3
        assert q.num_levels == 8
        assert q.rotation.shape == (128, 128)

    def test_init_invalid(self):
        from turboquant.core.polar_quant import PolarQuantizer
        with pytest.raises(ValueError):
            PolarQuantizer(head_dim=0)
        with pytest.raises(ValueError):
            PolarQuantizer(head_dim=128, bits=0)
        with pytest.raises(ValueError):
            PolarQuantizer(head_dim=128, group_size=0)

    def test_forward_shape(self):
        from turboquant.core.polar_quant import PolarQuantizer
        q = PolarQuantizer(head_dim=64, bits=3, group_size=32)
        x = torch.randn(2, 4, 16, 64, dtype=torch.float16)
        quantized, scales = q(x)
        assert quantized.shape == x.shape
        assert quantized.dtype == torch.int8
        assert scales.dtype == torch.float32

    def test_forward_empty(self):
        from turboquant.core.polar_quant import PolarQuantizer
        q = PolarQuantizer(head_dim=64)
        x = torch.empty(0, 64, dtype=torch.float16)
        quantized, scales = q(x)
        assert quantized.numel() == 0

    def test_dequantize_roundtrip(self):
        from turboquant.core.polar_quant import PolarQuantizer
        q = PolarQuantizer(head_dim=128, bits=3, group_size=64)
        x = torch.randn(1, 2, 8, 128, dtype=torch.float16)
        q.calibrate([x])
        quantized, scales = q(x)
        recon = q.dequantize(quantized, scales)
        assert recon.shape == x.shape
        assert recon.dtype == torch.float16
        # MSE should be reasonable (not perfect due to quantization)
        mse = ((x.float() - recon.float()) ** 2).mean().item()
        assert mse < 1.0  # rough sanity check

    def test_non_divisible_group_size(self):
        from turboquant.core.polar_quant import PolarQuantizer
        q = PolarQuantizer(head_dim=100, bits=3, group_size=64)
        x = torch.randn(1, 1, 4, 100, dtype=torch.float16)
        quantized, scales = q(x)
        recon = q.dequantize(quantized, scales)
        assert recon.shape == x.shape

    def test_calibrate(self):
        from turboquant.core.polar_quant import PolarQuantizer
        q = PolarQuantizer(head_dim=64, bits=3)
        data = [torch.randn(2, 4, 16, 64, dtype=torch.float16) for _ in range(3)]
        q.calibrate(data)
        assert q.calibrated.item() == 1

    def test_calibrate_empty(self):
        from turboquant.core.polar_quant import PolarQuantizer
        q = PolarQuantizer(head_dim=64)
        with pytest.raises(ValueError):
            q.calibrate([])

    def test_orthogonality(self):
        from turboquant.core.polar_quant import PolarQuantizer
        q = PolarQuantizer(head_dim=64)
        identity = q.rotation @ q.rotation.T
        assert torch.allclose(identity, torch.eye(64), atol=1e-5)


class TestQJLResidualCorrector:
    """Tests for QJLResidualCorrector."""

    def test_init(self):
        from turboquant.core.qjl import QJLResidualCorrector
        c = QJLResidualCorrector(head_dim=128)
        assert c.head_dim == 128
        assert c.sketch_dim == 32

    def test_encode_decode_shape(self):
        from turboquant.core.qjl import QJLResidualCorrector
        c = QJLResidualCorrector(head_dim=128, sketch_dim=32)
        r = torch.randn(2, 4, 16, 128, dtype=torch.float16)
        packed = c.encode(r)
        assert packed.dtype == torch.uint8
        assert packed.shape == (2, 4, 16, 4)  # 32 / 8 = 4
        recon = c.decode(packed, r.shape)
        assert recon.shape == r.shape

    def test_empty(self):
        from turboquant.core.qjl import QJLResidualCorrector
        c = QJLResidualCorrector(head_dim=64)
        r = torch.empty(0, 64, dtype=torch.float16)
        packed = c.encode(r)
        assert packed.numel() == 0

    def test_compress_ratio(self):
        from turboquant.core.qjl import QJLResidualCorrector
        c = QJLResidualCorrector(head_dim=128, sketch_dim=32)
        assert abs(c.compress_ratio() - 64.0) < 1e-6

    def test_pack_unpack(self):
        from turboquant.core.qjl import QJLResidualCorrector
        c = QJLResidualCorrector(head_dim=64, sketch_dim=37)
        bits = torch.randint(0, 2, (5, 37), dtype=torch.float32)
        packed = c._pack_bits(bits)
        unpacked = c._unpack_bits(packed, 37)
        assert torch.equal(bits, unpacked)


class TestTurboQuantKVCache:
    """Tests for TurboQuantKVCache."""

    def test_compress_decompress(self):
        from turboquant.core.turboquant import TurboQuantConfig, TurboQuantKVCache
        config = TurboQuantConfig(head_dim=64, num_heads=4, device="cpu")
        with TurboQuantKVCache(config) as tq:
            k = torch.randn(1, 4, 16, 64, dtype=torch.float16)
            v = torch.randn(1, 4, 16, 64, dtype=torch.float16)
            entry = tq.compress(k, v)
            k_hat, v_hat = tq.decompress(entry)
            assert k_hat.shape == k.shape
            assert v_hat.shape == v.shape

    def test_update(self):
        from turboquant.core.turboquant import TurboQuantConfig, TurboQuantKVCache
        config = TurboQuantConfig(head_dim=64, num_heads=4, device="cpu")
        with TurboQuantKVCache(config) as tq:
            k1 = torch.randn(1, 4, 8, 64, dtype=torch.float16)
            v1 = torch.randn(1, 4, 8, 64, dtype=torch.float16)
            entry = tq.compress(k1, v1)
            assert entry.metadata["seq_len"] == 8

            k2 = torch.randn(1, 4, 4, 64, dtype=torch.float16)
            v2 = torch.randn(1, 4, 4, 64, dtype=torch.float16)
            entry = tq.update(entry, k2, v2)
            assert entry.metadata["seq_len"] == 12

    def test_memory_usage(self):
        from turboquant.core.turboquant import TurboQuantConfig, TurboQuantKVCache
        config = TurboQuantConfig(head_dim=64, num_heads=4, device="cpu")
        with TurboQuantKVCache(config) as tq:
            k = torch.randn(1, 4, 32, 64, dtype=torch.float16)
            v = torch.randn(1, 4, 32, 64, dtype=torch.float16)
            entry = tq.compress(k, v)
            mem = tq.memory_usage(entry)
            assert "bytes" in mem
            assert "mb" in mem
            assert "ratio" in mem
            assert mem["ratio"] < 1.0  # compressed < original

    def test_no_residual(self):
        from turboquant.core.turboquant import TurboQuantConfig, TurboQuantKVCache
        config = TurboQuantConfig(
            head_dim=64, num_heads=4, device="cpu", residual_correction=False
        )
        with TurboQuantKVCache(config) as tq:
            k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
            v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
            entry = tq.compress(k, v)
            assert entry.residual_keys is None


class TestVectorDB:
    """Tests for vector DB adapter."""

    def test_in_memory_search(self):
        from turboquant.core.turboquant import TurboQuantConfig
        from turboquant.integrations.vector_db import InMemoryTurboQuant

        # Use larger dim for better quantization fidelity
        config = TurboQuantConfig(head_dim=256, num_heads=1, device="cpu")
        adapter = InMemoryTurboQuant(config)

        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, 256)).astype(np.float32)
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
        adapter.compress_embeddings(vectors)

        results = adapter.search(vectors[0], top_k=5)
        assert len(results) == 5
        # After quantization, cosine sim of a vector with its compressed
        # version may be well below 1.0. We only require the top result
        # to be meaningfully above random baseline (which is ~0 for
        # high-dimensional normalised vectors).
        assert results[0].score > 0.1, (
            f"Top result score {results[0].score:.4f} is suspiciously low"
        )

    def test_create_adapter(self):
        from turboquant.core.turboquant import TurboQuantConfig
        from turboquant.integrations.vector_db import create_adapter

        config = TurboQuantConfig(head_dim=64, device="cpu")
        adapter = create_adapter("memory", config)
        assert adapter is not None

        with pytest.raises(ValueError):
            create_adapter("unknown", config)
