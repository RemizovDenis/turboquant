"""Tests for semantic KV eviction."""

from __future__ import annotations

import torch

from turboquant.core.semantic_eviction import SemanticEvictionConfig, SemanticKVEviction


def test_semantic_evict_keeps_sink_and_recent() -> None:
    cfg = SemanticEvictionConfig(
        eviction_target_len=12,
        sink_token_count=2,
        recent_token_count=3,
        history_window=4,
        device="cpu",
        online_update_freq=1,
    )
    ev = SemanticKVEviction(config=cfg, head_dim=8, num_heads=2)
    keys = torch.randn(1, 2, 20, 8, dtype=torch.float16)
    values = torch.randn_like(keys)

    kept_k, kept_v, result = ev.evict(keys, values, layer_id=0)
    assert kept_k.shape == kept_v.shape
    assert kept_k.shape[2] <= cfg.eviction_target_len
    assert int(result.kept_indices.min().item()) == 0
    assert 1 in result.kept_indices.tolist()


def test_semantic_history_and_online_update() -> None:
    cfg = SemanticEvictionConfig(
        eviction_target_len=8, history_window=4, device="cpu", online_update_freq=1
    )
    ev = SemanticKVEviction(config=cfg, head_dim=8, num_heads=2)
    attn = torch.rand(1, 2, 10, 10, dtype=torch.float32)
    ev.update_attention_history(layer_id=1, attention_weights=attn)

    keys = torch.randn(1, 2, 10, 8, dtype=torch.float16)
    values = torch.randn_like(keys)
    kept_k, kept_v, _ = ev.evict(keys, values, layer_id=1)
    future = torch.rand(1, 2, kept_k.shape[2], kept_k.shape[2], dtype=torch.float32)
    loss = ev.online_update(kept_k, kept_v, future_attention=future, layer_id=1)
    assert loss >= 0.0
