# Copyright (c) 2026 Denis Remizov. Licensed under BUSL-1.1.
# See LICENSE file for details.

from __future__ import annotations

from turboquant.benchmarks.field_report import (
    needle_recall_percent,
    render_field_markdown,
    spike_rate_percent,
    summarize_field_results,
)


def test_needle_recall_percent_filters_non_ok_rows() -> None:
    needle = {
        "1024": {"status": "ok", "recall_percent": 100.0},
        "2048": {"status": "ok", "recall_percent": 0.0},
        "4096": {"status": "skipped_context_limit", "recall_percent": None},
    }
    assert needle_recall_percent(needle) == 50.0


def test_spike_rate_percent_with_threshold() -> None:
    series = [10.0, 30.0, 70.0, 90.0]
    assert spike_rate_percent(series, threshold_ms=50.0) == 50.0


def test_summarize_field_results_aggregates_portfolio() -> None:
    results = {
        "timestamp": "2026-04-04T15:00:00",
        "models": {
            "mistral:latest": {
                "baseline": {
                    "runs": [{"latency_ms": 40.0}, {"latency_ms": 80.0}],
                    "tokens_per_second_avg": 20.0,
                    "latency_avg_ms": 60.0,
                },
                "turboquant": {
                    "runs": [{"latency_ms": 30.0}, {"latency_ms": 70.0}],
                    "tokens_per_second_avg": 25.0,
                    "latency_avg_ms": 50.0,
                },
                "needle_baseline": {"1024": {"status": "ok", "recall_percent": 100.0}},
                "needle_turboquant": {"1024": {"status": "ok", "recall_percent": 100.0}},
                "speedup_x": 1.2,
                "memory_saved_mb": 256.0,
                "kv_compression_ratio": 3.0,
            },
            "llama3.1:latest": {
                "baseline": {
                    "runs": [{"latency_ms": 100.0}],
                    "tokens_per_second_avg": 12.0,
                    "latency_avg_ms": 100.0,
                },
                "turboquant": {
                    "runs": [{"latency_ms": 80.0}],
                    "tokens_per_second_avg": 15.0,
                    "latency_avg_ms": 80.0,
                },
                "needle_baseline": {"1024": {"status": "ok", "recall_percent": 100.0}},
                "needle_turboquant": {"1024": {"status": "ok", "recall_percent": 100.0}},
                "speedup_x": 1.25,
                "memory_saved_mb": 128.0,
                "kv_compression_ratio": 2.5,
            },
        },
    }

    summary = summarize_field_results(results)
    assert summary["portfolio"]["model_count"] == 2
    assert abs(summary["portfolio"]["avg_speedup_x"] - 1.225) < 1e-9
    assert summary["portfolio"]["total_memory_saved_mb"] == 384.0
    assert summary["portfolio"]["best_kv_compression_ratio_x"] == 3.0


def test_render_field_markdown_includes_kpi_table() -> None:
    results = {"timestamp": "2026-04-04T15:00:00", "models": {}}
    summary = {
        "models": [
            {
                "model": "mistral:latest",
                "speedup_x": 1.3,
                "kv_compression_ratio_x": 2.2,
                "memory_saved_mb": 64.0,
                "turboquant_recall_avg_percent": 100.0,
                "turboquant_spike_rate_percent_gt50ms": 12.5,
            }
        ],
        "portfolio": {
            "model_count": 1,
            "avg_speedup_x": 1.3,
            "total_memory_saved_mb": 64.0,
            "best_kv_compression_ratio_x": 2.2,
        },
    }

    md = render_field_markdown(results, summary)
    assert "# Local Field Benchmark (TurboQuant / Ollama)" in md
    assert "| Model | Speedup x |" in md
    assert "mistral:latest" in md
    assert "1.300" in md
