"""Tests for adaptive per-token bitwidth quantization."""

from __future__ import annotations

import torch

from turboquant.core.adaptive_bitwidth import AdaptiveBitwidthQuantizer, AdaptiveBitwidthConfig


def test_assign_bitwidths_and_compress_roundtrip() -> None:
    cfg = AdaptiveBitwidthConfig(
        head_dim=8,
        num_heads=2,
        vocab_size=128,
        device="cpu",
        target_avg_bits=2.5,
    )
    quant = AdaptiveBitwidthQuantizer(cfg)
    token_ids = torch.randint(0, 128, (16,), dtype=torch.long)
    entropy = torch.rand(16, dtype=torch.float32)

    assignment = quant.assign_bitwidths(token_ids, entropy, seq_len=16)
    assert assignment.bits_per_token.dtype == torch.uint8
    assert assignment.avg_bits <= 4.0

    keys = torch.randn(1, 2, 16, 8, dtype=torch.float16)
    values = torch.randn_like(keys)
    compressed = quant.compress(keys, values, token_ids=token_ids, attention_entropy=entropy)
    rk, rv = quant.decompress(compressed, original_shape=tuple(keys.shape))
    assert rk.shape == keys.shape
    assert rv.shape == values.shape

    stats = quant.memory_savings_vs_uniform(compressed)
    assert "avg_bits" in stats
    assert "additional_savings_percent" in stats
