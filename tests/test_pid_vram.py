"""Tests for PID-based VRAM controller."""

from __future__ import annotations

import torch

from turboquant.core.moe_expert_cache import DynamicExpertCache, ExpertCacheConfig
from turboquant.core.pid_vram import PIDConfig, VRAM_PID_Controller


def _make_cache() -> DynamicExpertCache:
    cfg = ExpertCacheConfig(
        num_experts=4,
        top_k_experts=1,
        num_layers=1,
        gpu_cache_size=4,
        device="cpu",
    )
    cache = DynamicExpertCache(cfg)
    weights = {
        "gate": torch.randn(16, 16),
        "up": torch.randn(16, 16),
        "down": torch.randn(16, 16),
    }
    for expert_id in range(4):
        cache.register_expert(expert_id=expert_id, layer_id=0, weights=weights)
        cache.get_expert(expert_id=expert_id, layer_id=0)
    return cache


def test_pid_step_cpu_and_stats() -> None:
    cache = _make_cache()
    pid = VRAM_PID_Controller(
        PIDConfig(min_cache_size=1, max_cache_size=4), cache, initial_cache_size=4
    )
    new_size, state = pid.step()
    assert 1 <= new_size <= 4
    assert state.target_utilization == pid.config.target_vram_utilization
    stats = pid.stats()
    assert "avg_utilization" in stats
    assert "total_evictions" in stats


def test_pid_emergency_evict_reduces_cache() -> None:
    cache = _make_cache()
    pid = VRAM_PID_Controller(
        PIDConfig(min_cache_size=1, max_cache_size=4), cache, initial_cache_size=4
    )
    pid.measure_vram = lambda: (0.99, 100.0, 100.0)  # type: ignore[method-assign]
    new_size = pid.emergency_evict()
    assert new_size <= 2
    assert len(cache._gpu_experts) <= new_size
