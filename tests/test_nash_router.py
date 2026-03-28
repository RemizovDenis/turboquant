"""Unit tests for game-theoretic Nash router."""

from __future__ import annotations

import torch

from turboquant.core.nash_router import GameTheoreticRouter, NashRouterConfig


def test_nash_router_forward_shapes() -> None:
    config = NashRouterConfig(num_experts=8, top_k=2, nash_iterations=3)
    router = GameTheoreticRouter(config)

    logits = torch.randn(256, 8)
    locations = torch.zeros(8, dtype=torch.bool)
    locations[:4] = True
    output = router(logits, training=False, expert_locations_mask=locations)

    assert output.expert_indices.shape == (256, 2)
    assert output.combine_weights.shape == (256, 2)
    assert output.dispatch_mask.shape == (256, 8)
    assert output.dropped_tokens >= 0
    assert int(output.expert_indices.max().item()) < 8


def test_nash_router_stats_and_overhead() -> None:
    config = NashRouterConfig(num_experts=8, top_k=2, nash_iterations=2)
    router = GameTheoreticRouter(config)
    logits = torch.randn(64, 8)
    locations = torch.tensor([True, True, True, True, False, False, False, False])

    _ = router(logits, training=False, expert_locations_mask=locations)
    stats = router.get_nash_stats()
    assert "nash_convergence_rate" in stats
    assert "avg_iterations" in stats
    assert 0.0 <= stats["gpu_expert_preference"] <= 1.0

    overhead = router.overhead_ms(num_tokens=64, num_experts=8, n_warmup=1, n_iters=3)
    assert overhead >= 0.0
