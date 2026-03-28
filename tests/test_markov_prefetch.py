"""Unit tests for Markov trajectory prefetch predictor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from turboquant.core.markov_prefetch import (
    MarkovPrefetchConfig,
    MarkovTrajectoryPredictor,
    PrefetchPrediction,
)
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


def test_markov_dynamic_topk_and_batched_prefetch() -> None:
    cache = _build_cache()
    cfg = MarkovPrefetchConfig(
        num_layers=4,
        num_experts=4,
        top_k_experts=2,
        lookahead_steps=2,
        min_prefetch_prob=0.0,
        prefetch_threshold=0.0,
        uncertainty_topk_boost=2,
        per_source_topk=1,
        max_prefetch_per_layer=4,
        max_pending_prefetches=32,
        wait_timeout_ms=5.0,
        device="cpu",
    )
    predictor = MarkovTrajectoryPredictor(cfg, cache)

    preds = predictor.predict(current_layer=0, active_experts=[0, 1], lookahead_steps=2)
    assert preds
    assert all(pred.horizon >= 1 for pred in preds.values())
    assert any(len(pred.expert_ids) >= cfg.top_k_experts for pred in preds.values())

    predictor.start_prefetch(preds)
    assert any(pred.prefetch_started for pred in preds.values())

    loaded = predictor.wait_for_layer(layer_id=1, timeout_ms=20.0)
    assert isinstance(loaded, list)
    predictor.on_layer_complete(0, [0, 1])
    predictor.on_layer_complete(1, [1, 2])


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


def test_prefetch_gate_accepts_high_probability_when_confidence_low() -> None:
    cache = _build_cache()
    cfg = MarkovPrefetchConfig(
        num_layers=4,
        num_experts=4,
        top_k_experts=2,
        min_prefetch_prob=0.1,
        prefetch_threshold=0.9,
        max_prefetch_per_layer=2,
        max_pending_prefetches=8,
        device="cpu",
    )
    predictor = MarkovTrajectoryPredictor(cfg, cache)

    low_conf_high_prob = PrefetchPrediction(
        layer_id=1,
        expert_ids=[2],
        probabilities=[0.95],
        horizon=1,
        confidence=0.1,
        prefetch_started=False,
        estimated_load_ms=1.0,
    )
    predictor.start_prefetch({1: low_conf_high_prob})
    loaded = predictor.wait_for_layer(layer_id=1, timeout_ms=20.0)
    assert 2 in loaded

    low_conf_low_prob = PrefetchPrediction(
        layer_id=2,
        expert_ids=[3],
        probabilities=[0.3],
        horizon=1,
        confidence=0.1,
        prefetch_started=False,
        estimated_load_ms=1.0,
    )
    predictor.start_prefetch({2: low_conf_low_prob})
    loaded_low = predictor.wait_for_layer(layer_id=2, timeout_ms=20.0)
    assert 3 not in loaded_low
