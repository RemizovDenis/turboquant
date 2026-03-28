"""Tests for cross-layer KV sharing."""

from __future__ import annotations

import torch

from turboquant.core.cross_layer_kv import CrossLayerConfig, CrossLayerKVCache
from turboquant.core.turboquant import TurboQuantConfig, TurboQuantKVCache


def test_cross_layer_anchor_and_delta_roundtrip() -> None:
    base = TurboQuantKVCache(
        TurboQuantConfig(
            head_dim=8,
            num_heads=2,
            bits=3,
            residual_correction=False,
            device="cpu",
        )
    )
    cross = CrossLayerKVCache(CrossLayerConfig(num_layers=4, anchor_stride=2, device="cpu"), base)

    keys0 = torch.randn(1, 2, 16, 8, dtype=torch.float16)
    vals0 = torch.randn_like(keys0)
    e0 = cross.compress(0, keys0, vals0)
    assert e0.is_anchor

    keys1 = keys0 + 0.01 * torch.randn_like(keys0)
    vals1 = vals0 + 0.01 * torch.randn_like(vals0)
    e1 = cross.compress(1, keys1, vals1)
    rk, rv = cross.decompress(e1)
    assert rk.shape == keys1.shape
    assert rv.shape == vals1.shape

    report = cross.memory_report()
    assert "total_compression_ratio" in report
    assert report["total_compression_ratio"] >= 0.0
