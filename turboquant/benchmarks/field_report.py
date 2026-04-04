# Copyright (c) 2026 Denis Remizov. Licensed under BUSL-1.1.
# See LICENSE file for details.

"""Local field benchmark reporting utilities.

The module is intentionally dependency-light so it can run on local Apple Silicon
without cloud services.
"""

from __future__ import annotations

import math
import statistics
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _to_float(value: Any) -> float | None:
    if _is_finite_number(value):
        return float(value)
    return None


def _numeric_series(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for item in values:
        v = _to_float(item)
        if v is not None:
            out.append(v)
    return out


def _fmt(value: float | None, digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def needle_recall_percent(needle: dict[str, Any]) -> float | None:
    recalls: list[float] = []
    for row in needle.values():
        if not isinstance(row, dict):
            continue
        if row.get("status") != "ok":
            continue
        recall = _to_float(row.get("recall_percent"))
        if recall is not None:
            recalls.append(recall)
    if not recalls:
        return None
    return statistics.fmean(recalls)


def spike_rate_percent(latency_values_ms: list[float], threshold_ms: float = 50.0) -> float | None:
    values = [v for v in latency_values_ms if math.isfinite(v)]
    if not values:
        return None
    spikes = sum(1 for v in values if v > threshold_ms)
    return spikes * 100.0 / len(values)


def _extract_latency_series(metrics: dict[str, Any]) -> list[float]:
    runs = metrics.get("runs")
    if not isinstance(runs, list):
        return []
    out: list[float] = []
    for row in runs:
        if not isinstance(row, dict):
            continue
        val = _to_float(row.get("latency_ms"))
        if val is not None:
            out.append(val)
    return out


def _summarize_model(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    baseline = _as_dict(payload.get("baseline"))
    turbo = _as_dict(payload.get("turboquant"))

    baseline_lat = _extract_latency_series(baseline)
    turbo_lat = _extract_latency_series(turbo)

    speedup = _to_float(payload.get("speedup_x"))
    if speedup is None:
        b_avg = _to_float(baseline.get("latency_avg_ms"))
        t_avg = _to_float(turbo.get("latency_avg_ms"))
        if b_avg is not None and t_avg is not None and t_avg > 0:
            speedup = b_avg / t_avg

    kv_ratio = _to_float(payload.get("kv_compression_ratio"))
    memory_saved_mb = _to_float(payload.get("memory_saved_mb"))
    baseline_recall = needle_recall_percent(_as_dict(payload.get("needle_baseline")))
    turbo_recall = needle_recall_percent(_as_dict(payload.get("needle_turboquant")))

    return {
        "model": name,
        "speedup_x": speedup,
        "kv_compression_ratio_x": kv_ratio,
        "memory_saved_mb": memory_saved_mb,
        "baseline_recall_avg_percent": baseline_recall,
        "turboquant_recall_avg_percent": turbo_recall,
        "baseline_spike_rate_percent_gt50ms": spike_rate_percent(baseline_lat),
        "turboquant_spike_rate_percent_gt50ms": spike_rate_percent(turbo_lat),
        "baseline_tokens_per_second_avg": _to_float(baseline.get("tokens_per_second_avg")),
        "turboquant_tokens_per_second_avg": _to_float(turbo.get("tokens_per_second_avg")),
    }


def summarize_field_results(results: dict[str, Any]) -> dict[str, Any]:
    models_blob = results.get("models")
    if not isinstance(models_blob, dict):
        return {
            "models": [],
            "portfolio": {
                "model_count": 0,
                "avg_speedup_x": None,
                "total_memory_saved_mb": None,
                "best_kv_compression_ratio_x": None,
            },
        }

    model_rows: list[dict[str, Any]] = []
    for model_name, payload in models_blob.items():
        if not isinstance(model_name, str):
            continue
        if isinstance(payload, dict):
            model_rows.append(_summarize_model(model_name, payload))

    speedups = _numeric_series([row.get("speedup_x") for row in model_rows])
    memory_saved = _numeric_series([row.get("memory_saved_mb") for row in model_rows])
    kv_ratios = _numeric_series([row.get("kv_compression_ratio_x") for row in model_rows])

    portfolio: dict[str, Any] = {
        "model_count": len(model_rows),
        "avg_speedup_x": statistics.fmean(speedups) if speedups else None,
        "total_memory_saved_mb": sum(memory_saved) if memory_saved else None,
        "best_kv_compression_ratio_x": max(kv_ratios) if kv_ratios else None,
    }

    return {"models": model_rows, "portfolio": portfolio}


def render_field_markdown(results: dict[str, Any], summary: dict[str, Any]) -> str:
    ts = str(results.get("timestamp", "unknown"))
    lines = [
        "# Local Field Benchmark (TurboQuant / Ollama)",
        "",
        f"Generated: `{ts}`",
        "",
        "## Portfolio Summary",
        "",
    ]

    portfolio = _as_dict(summary.get("portfolio"))
    avg_speedup = _to_float(portfolio.get("avg_speedup_x"))
    total_mem = _to_float(portfolio.get("total_memory_saved_mb"))
    best_kv = _to_float(portfolio.get("best_kv_compression_ratio_x"))

    lines.append(f"- Models tested: `{int(portfolio.get('model_count', 0))}`")
    lines.append(
        f"- Avg speedup: `{avg_speedup:.3f}x`"
        if avg_speedup is not None
        else "- Avg speedup: `n/a`"
    )
    lines.append(
        f"- Total memory saved: `{total_mem / 1024.0:.3f} GB`"
        if total_mem is not None
        else "- Total memory saved: `n/a`"
    )
    lines.append(
        f"- Best KV compression ratio: `{best_kv:.3f}x`"
        if best_kv is not None
        else "- Best KV compression ratio: `n/a`"
    )
    lines += ["", "## Per-model KPIs", ""]
    lines += [
        "| Model | Speedup x | KV ratio x | Memory saved MB | Recall avg % (TQ) | Spike rate %>50ms (TQ) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    model_rows = summary.get("models")
    if isinstance(model_rows, list) and model_rows:
        for row in model_rows:
            if not isinstance(row, dict):
                continue
            speedup = _to_float(row.get("speedup_x"))
            kv_ratio = _to_float(row.get("kv_compression_ratio_x"))
            mem = _to_float(row.get("memory_saved_mb"))
            recall = _to_float(row.get("turboquant_recall_avg_percent"))
            spikes = _to_float(row.get("turboquant_spike_rate_percent_gt50ms"))
            lines.append(
                "| "
                f"{row.get('model', 'unknown')} | "
                f"{_fmt(speedup, 3)} | "
                f"{_fmt(kv_ratio, 3)} | "
                f"{_fmt(mem, 1)} | "
                f"{_fmt(recall, 1)} | "
                f"{_fmt(spikes, 2)} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")

    lines += [
        "",
        "## Notes",
        "",
        "- This report is generated locally (no cloud runners required).",
        "- Results depend on model availability in local Ollama and host memory pressure.",
    ]
    return "\n".join(lines) + "\n"
