"""Unit tests for Markov trajectory prefetch predictor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from turboquant.core.markov_prefetch import MarkovPrefetchConfig, MarkovTrajectoryPredictor
from turboquant.core.moe_expert_cache import DynamicExpertCache, ExpertCacheConfig


def _build_cache() -> DynamicExpertCache:
    cfg = ExpertCacheConfig(
        num_experts=4,
        top_k_experts=2,
        num_layers=4,
        gpu_cache_size=2,
        device="cpu",
    )
    cache = DynamicExpertCache(cfg)
    weights = {"w": torch.randn(16, 16)}
    for layer in range(cfg.num_layers):
        for expert in range(cfg.num_experts):
            cache.register_expert(expert, layer, weights)
    return cache


def test_markov_predict_and_update() -> None:
    cache = _build_cache()
    cfg = MarkovPrefetchConfig(
        num_layers=4,
        num_experts=4,
        top_k_experts=2,
        lookahead_steps=2,
        min_prefetch_prob=0.05,
        prefetch_threshold=0.0,
        device="cpu",
    )
    predictor = MarkovTrajectoryPredictor(cfg, cache)

    preds = predictor.predict(current_layer=0, active_experts=[1, 2], lookahead_steps=2)
    assert preds
    predictor.start_prefetch(preds)
    predictor.on_layer_complete(0, [1, 2])
    predictor.on_layer_complete(1, [2, 3])

    loaded = predictor.wait_for_layer(layer_id=1, timeout_ms=10.0)
    assert isinstance(loaded, list)

    stats = predictor.stats()
    assert stats.total_predictions >= 0
    assert 0.0 <= stats.accuracy_at_k <= 1.0
    assert "MarkovTrajectoryPredictor" in repr(predictor)


def test_markov_save_load_roundtrip() -> None:
    cache = _build_cache()
    cfg = MarkovPrefetchConfig(num_layers=4, num_experts=4, top_k_experts=2, device="cpu")
    predictor = MarkovTrajectoryPredictor(cfg, cache)
    predictor.on_layer_complete(0, [0, 1])
    predictor.on_layer_complete(1, [1, 2])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "markov"
        predictor.save(str(path))
        reloaded = MarkovTrajectoryPredictor(cfg, cache)
        reloaded.load(str(path))
        assert reloaded.matrix_entropy() >= 0.0
