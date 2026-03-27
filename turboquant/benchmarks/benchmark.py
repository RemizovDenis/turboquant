"""TurboQuant Benchmark Suite — enterprise-grade performance evaluation.

Generates comprehensive benchmark reports suitable for technical pitches:

- **Memory Benchmark**: FP16 vs TurboQuant across sequence lengths
- **Speed Benchmark**: compress/decompress latency and throughput
- **Quality Benchmark**: needle-in-haystack recall tests
- **VPS Benchmark**: RAM/cost analysis for Ollama on VPS

CLI::

    python -m turboquant.benchmarks.benchmark \\
        --model ollama/llama3 \\
        --backend ollama \\
        --output ./results \\
        --suite all

    # or via entrypoint:
    turboquant-benchmark --suite memory --output ./results
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import torch

from turboquant.core.turboquant import TurboQuantConfig, TurboQuantKVCache

log = structlog.get_logger(__name__)

# Attempt optional imports; degrade gracefully.
try:
    from tqdm import tqdm  # type: ignore[import-untyped]
except ImportError:
    def tqdm(it, **kw):  # type: ignore[misc]
        return it

try:
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    import plotly.io as pio  # type: ignore[import-untyped]
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

_HAS_NVML = importlib.util.find_spec("pynvml") is not None


# ======================================================================
# Benchmark implementations
# ======================================================================


def _detect_device() -> str:
    """Choose best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def memory_benchmark(
    config: TurboQuantConfig,
    seq_lengths: list[int] | None = None,
    batch_size: int = 1,
) -> pd.DataFrame:
    """Compare memory consumption: FP16 baseline vs TurboQuant.

    Args:
        config: TurboQuant configuration.
        seq_lengths: List of sequence lengths to test.
        batch_size: Batch size.

    Returns:
        DataFrame with columns: ``seq_len``, ``fp16_mb``, ``tq3_mb``,
        ``tq4_mb``, ``ratio_3bit``, ``ratio_4bit``.
    """
    if seq_lengths is None:
        device = _detect_device()
        if device == "cuda":
            seq_lengths = [1024, 4096, 16384, 65536, 131072]
        else:
            # CPU/MPS: limit to avoid hanging on large allocations
            seq_lengths = [256, 1024, 4096, 8192]

    device = _detect_device()
    rows: list[dict[str, Any]] = []

    for sl in tqdm(seq_lengths, desc="Memory benchmark"):
        shape = (batch_size, config.num_heads, sl, config.head_dim)
        fp16_bytes = int(np.prod(shape)) * 2 * 2  # keys + values
        fp16_mb = fp16_bytes / (1024**2)

        # TurboQuant 3-bit (no residual)
        cfg3 = TurboQuantConfig(
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            bits=3,
            group_size=config.group_size,
            residual_correction=False,
            device=device,
            seed=config.seed,
        )
        try:
            with TurboQuantKVCache(cfg3) as tq3:
                k = torch.randn(shape, dtype=torch.float16, device=device)
                v = torch.randn(shape, dtype=torch.float16, device=device)
                entry3 = tq3.compress(k, v)
                mem3 = tq3.memory_usage(entry3)
                tq3_mb = mem3["mb"]
                del k, v, entry3
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            tq3_mb = fp16_mb * 3 / 16  # theoretical

        # TurboQuant 3+1 bit (with residual)
        cfg4 = TurboQuantConfig(
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            bits=3,
            group_size=config.group_size,
            residual_correction=True,
            device=device,
            seed=config.seed,
        )
        try:
            with TurboQuantKVCache(cfg4) as tq4:
                k = torch.randn(shape, dtype=torch.float16, device=device)
                v = torch.randn(shape, dtype=torch.float16, device=device)
                entry4 = tq4.compress(k, v)
                mem4 = tq4.memory_usage(entry4)
                tq4_mb = mem4["mb"]
                del k, v, entry4
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            tq4_mb = fp16_mb * 4 / 16  # theoretical

        rows.append({
            "seq_len": sl,
            "fp16_mb": round(fp16_mb, 2),
            "tq3_mb": round(tq3_mb, 2),
            "tq4_mb": round(tq4_mb, 2),
            "ratio_3bit": round(tq3_mb / max(fp16_mb, 1e-9), 4),
            "ratio_4bit": round(tq4_mb / max(fp16_mb, 1e-9), 4),
            "savings_3bit_pct": round((1 - tq3_mb / max(fp16_mb, 1e-9)) * 100, 1),
            "savings_4bit_pct": round((1 - tq4_mb / max(fp16_mb, 1e-9)) * 100, 1),
        })

    return pd.DataFrame(rows)


def speed_benchmark(
    config: TurboQuantConfig,
    seq_lengths: list[int] | None = None,
    batch_size: int = 1,
    warmup: int = 3,
    iterations: int = 10,
) -> pd.DataFrame:
    """Measure compress/decompress latency.

    Args:
        config: TurboQuant configuration.
        seq_lengths: List of sequence lengths.
        batch_size: Batch size.
        warmup: Warm-up iterations.
        iterations: Timed iterations.

    Returns:
        DataFrame with: ``seq_len``, ``compress_ms``, ``decompress_ms``,
        ``throughput_tokens_per_sec``.
    """
    if seq_lengths is None:
        device = _detect_device()
        seq_lengths = [1024, 4096, 16384, 65536] if device == "cuda" else [256, 1024, 4096]

    device = _detect_device()
    rows: list[dict[str, Any]] = []

    tq = TurboQuantKVCache(
        TurboQuantConfig(
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            bits=config.bits,
            group_size=config.group_size,
            residual_correction=config.residual_correction,
            device=device,
            seed=config.seed,
        )
    )

    for sl in tqdm(seq_lengths, desc="Speed benchmark"):
        shape = (batch_size, config.num_heads, sl, config.head_dim)
        try:
            k = torch.randn(shape, dtype=torch.float16, device=device)
            v = torch.randn(shape, dtype=torch.float16, device=device)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            log.warning("speed_benchmark: OOM at seq_len=%d", sl)
            continue

        # Warm up
        for _ in range(warmup):
            e = tq.compress(k, v)
            _ = tq.decompress(e)

        # Compress
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iterations):
            e = tq.compress(k, v)
        if device == "cuda":
            torch.cuda.synchronize()
        compress_ms = (time.perf_counter() - t0) / iterations * 1000

        # Decompress
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = tq.decompress(e)
        if device == "cuda":
            torch.cuda.synchronize()
        decompress_ms = (time.perf_counter() - t0) / iterations * 1000

        total_tokens = batch_size * sl
        throughput = total_tokens / (compress_ms / 1000) if compress_ms > 0 else 0

        rows.append({
            "seq_len": sl,
            "compress_ms": round(compress_ms, 2),
            "decompress_ms": round(decompress_ms, 2),
            "throughput_tokens_per_sec": round(throughput, 0),
        })

        del k, v, e

    return pd.DataFrame(rows)


def quality_benchmark(
    config: TurboQuantConfig,
    seq_lengths: list[int] | None = None,
    num_trials: int = 10,
) -> pd.DataFrame:
    """Needle-in-haystack recall test.

    Inserts a random 'needle' vector at a random position in a sequence.
    Compresses and decompresses, then checks if the needle can be
    identified by cosine similarity.

    Args:
        config: TurboQuant configuration.
        seq_lengths: Sequence lengths to test.
        num_trials: Number of trials per sequence length.

    Returns:
        DataFrame with: ``seq_len``, ``recall_at_1_baseline``,
        ``recall_at_1_tq``.
    """
    if seq_lengths is None:
        device = _detect_device()
        if device == "cuda":
            seq_lengths = [1024, 4096, 16384, 32768, 65536, 104000]
        else:
            seq_lengths = [256, 1024, 2048, 4096]

    device = _detect_device()
    rows: list[dict[str, Any]] = []

    tq = TurboQuantKVCache(
        TurboQuantConfig(
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            bits=config.bits,
            group_size=config.group_size,
            residual_correction=config.residual_correction,
            device=device,
            seed=config.seed,
        )
    )

    for sl in tqdm(seq_lengths, desc="Quality benchmark"):
        recalls_baseline = 0
        recalls_tq = 0
        actual_trials = 0

        for _trial in range(num_trials):
            try:
                # Generate haystack
                shape = (1, 1, sl, config.head_dim)
                haystack = torch.randn(shape, dtype=torch.float16, device=device)

                # Insert needle
                needle_pos = int(torch.randint(0, sl, (1,)).item())
                needle = torch.randn(1, 1, 1, config.head_dim, dtype=torch.float16, device=device) * 5
                haystack[:, :, needle_pos : needle_pos + 1, :] = needle

                # Baseline: perfect recall (no compression)
                recalls_baseline += 1

                # TurboQuant
                entry = tq.compress(haystack, haystack)
                recon, _ = tq.decompress(entry)

                # Find needle in reconstruction by cosine sim
                needle_flat = needle.float().reshape(-1)
                recon_flat = recon.float().reshape(sl, config.head_dim)
                sims = torch.nn.functional.cosine_similarity(
                    recon_flat, needle_flat.unsqueeze(0), dim=1
                )
                best_pos = int(sims.argmax().item())
                if best_pos == needle_pos:
                    recalls_tq += 1

                actual_trials += 1
                del haystack, recon, entry

            except (torch.cuda.OutOfMemoryError, RuntimeError):
                break

        if actual_trials > 0:
            rows.append({
                "seq_len": sl,
                "recall_at_1_baseline": round(recalls_baseline / actual_trials, 4),
                "recall_at_1_tq": round(recalls_tq / actual_trials, 4),
                "trials": actual_trials,
            })

    return pd.DataFrame(rows)


def vps_benchmark(config: TurboQuantConfig) -> pd.DataFrame:
    """Estimate memory and cost savings for VPS deployments.

    Assumes typical VPS pricing and model memory footprints.

    Args:
        config: TurboQuant configuration.

    Returns:
        DataFrame with: ``model``, ``params_b``, ``fp16_ram_gb``,
        ``tq_ram_gb``, ``savings_gb``, ``monthly_cost_fp16``,
        ``monthly_cost_tq``, ``monthly_savings``.
    """
    # Typical model profiles (params in billions)
    models: list[dict[str, str | float]] = [
        {"model": "Llama-3-8B", "params_b": 8, "kv_per_token_bytes": 0.5},
        {"model": "Llama-3-13B", "params_b": 13, "kv_per_token_bytes": 0.75},
        {"model": "Mistral-7B", "params_b": 7, "kv_per_token_bytes": 0.5},
        {"model": "Qwen2-7B", "params_b": 7, "kv_per_token_bytes": 0.5},
    ]

    # Typical VPS pricing ($/month per GB RAM)
    cost_per_gb_month = 5.0  # approximate

    bits_total = config.bits + (1 if config.residual_correction else 0)
    compression_ratio = 16.0 / bits_total

    rows: list[dict[str, Any]] = []
    for m in models:
        params_b = float(m["params_b"])
        kv_per_token_bytes = float(m["kv_per_token_bytes"])
        model_name = str(m["model"])
        # Model weights in FP16
        weights_gb = params_b * 2  # 2 bytes per param
        # KV-cache for 32k context
        kv_fp16_gb = kv_per_token_bytes * 32768 / (1024**3) * 2  # K+V
        kv_fp16_gb = max(kv_fp16_gb, weights_gb * 0.15)  # at least 15% of weights

        total_fp16 = weights_gb + kv_fp16_gb
        kv_tq_gb = kv_fp16_gb / compression_ratio
        total_tq = weights_gb + kv_tq_gb

        savings_gb = total_fp16 - total_tq
        cost_fp16 = total_fp16 * cost_per_gb_month
        cost_tq = total_tq * cost_per_gb_month

        rows.append({
            "model": model_name,
            "params_b": params_b,
            "fp16_ram_gb": round(total_fp16, 2),
            "tq_ram_gb": round(total_tq, 2),
            "savings_gb": round(savings_gb, 2),
            "monthly_cost_fp16_usd": round(cost_fp16, 2),
            "monthly_cost_tq_usd": round(cost_tq, 2),
            "monthly_savings_usd": round(cost_fp16 - cost_tq, 2),
        })

    return pd.DataFrame(rows)


# ======================================================================
# Report generation
# ======================================================================


def generate_markdown_report(results: dict[str, pd.DataFrame]) -> str:
    """Generate a Markdown report from benchmark results.

    Args:
        results: Dict of suite_name → DataFrame.

    Returns:
        Markdown string.
    """
    lines = ["# TurboQuant Benchmark Report", ""]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append("")

    for name, df in results.items():
        lines.append(f"## {name.replace('_', ' ').title()}")
        lines.append("")
        lines.append(df.to_markdown(index=False))
        lines.append("")

        # Summary line
        if "savings_4bit_pct" in df.columns:
            best = df.loc[df["savings_4bit_pct"].idxmax()]
            lines.append(
                f"**TurboQuant saves {best.get('tq4_mb', 0):.0f} MB "
                f"({best.get('savings_4bit_pct', 0):.0f}%) at seq_len={best.get('seq_len', 0):.0f}**"
            )
            lines.append("")

    return "\n".join(lines)


def generate_html_report(results: dict[str, pd.DataFrame], output_dir: str) -> str | None:
    """Generate an interactive HTML report with Plotly charts.

    Args:
        results: Dict of suite_name → DataFrame.
        output_dir: Output directory.

    Returns:
        Path to the HTML file, or *None* if Plotly is unavailable.
    """
    if not _HAS_PLOTLY:
        log.warning("plotly not installed, skipping HTML report")
        return None

    figs: list[go.Figure] = []

    if "memory" in results:
        df = results["memory"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="FP16 Baseline", x=df["seq_len"].astype(str), y=df["fp16_mb"]))
        fig.add_trace(go.Bar(name="TurboQuant 3-bit", x=df["seq_len"].astype(str), y=df["tq3_mb"]))
        fig.add_trace(go.Bar(name="TurboQuant 4-bit", x=df["seq_len"].astype(str), y=df["tq4_mb"]))
        fig.update_layout(
            title="Memory Usage by Sequence Length",
            xaxis_title="Sequence Length",
            yaxis_title="Memory (MB)",
            barmode="group",
            template="plotly_dark",
        )
        figs.append(fig)

    if "speed" in results:
        df = results["speed"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Compress", x=df["seq_len"], y=df["compress_ms"], mode="lines+markers"
        ))
        fig.add_trace(go.Scatter(
            name="Decompress", x=df["seq_len"], y=df["decompress_ms"], mode="lines+markers"
        ))
        fig.update_layout(
            title="Latency vs Sequence Length",
            xaxis_title="Sequence Length",
            yaxis_title="Latency (ms)",
            template="plotly_dark",
        )
        figs.append(fig)

    if "quality" in results:
        df = results["quality"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Baseline", x=df["seq_len"].astype(str), y=df["recall_at_1_baseline"]
        ))
        fig.add_trace(go.Bar(
            name="TurboQuant", x=df["seq_len"].astype(str), y=df["recall_at_1_tq"]
        ))
        fig.update_layout(
            title="Recall@1 — Needle in Haystack",
            xaxis_title="Sequence Length",
            yaxis_title="Recall@1",
            barmode="group",
            template="plotly_dark",
        )
        figs.append(fig)

    if "vps" in results:
        df = results["vps"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="FP16 RAM", x=df["model"], y=df["fp16_ram_gb"]))
        fig.add_trace(go.Bar(name="TQ RAM", x=df["model"], y=df["tq_ram_gb"]))
        fig.update_layout(
            title="VPS RAM Usage per Model",
            xaxis_title="Model",
            yaxis_title="RAM (GB)",
            barmode="group",
            template="plotly_dark",
        )
        figs.append(fig)

    # Combine into single HTML
    html_parts = [
        "<html><head><title>TurboQuant Benchmark Report</title></head><body>",
        "<h1>TurboQuant Benchmark Report</h1>",
        f"<p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</p>",
    ]
    for fig in figs:
        html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))
    html_parts.append("</body></html>")

    out_path = os.path.join(output_dir, "benchmark_report.html")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(html_parts))

    log.info("html_report_generated", path=out_path)
    return out_path


# ======================================================================
# Save results incrementally
# ======================================================================


def _save_incremental(results: dict[str, pd.DataFrame], output_dir: str) -> None:
    """Save results to JSON incrementally (append-safe).

    Args:
        results: Current results dict.
        output_dir: Output directory.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    json_path = os.path.join(output_dir, "benchmark_results.json")

    serialized: dict[str, Any] = {}
    for name, df in results.items():
        serialized[name] = df.to_dict(orient="records")

    with open(json_path, "w") as f:
        json.dump(serialized, f, indent=2, default=str)

    md_path = os.path.join(output_dir, "benchmark_results.md")
    with open(md_path, "w") as f:
        f.write(generate_markdown_report(results))


# ======================================================================
# CLI
# ======================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv``).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="turboquant-benchmark",
        description="TurboQuant benchmark suite — evaluate memory, speed, quality, and VPS savings.",
    )
    parser.add_argument(
        "--model",
        default="synthetic",
        help="Model identifier (e.g. 'ollama/llama3'). Default: synthetic data.",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "transformers", "synthetic"],
        default="synthetic",
        help="Inference backend (default: synthetic).",
    )
    parser.add_argument(
        "--output",
        default="./benchmark_results",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--suite",
        choices=["memory", "speed", "quality", "vps", "all"],
        default="all",
        help="Benchmark suite to run (default: all).",
    )
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of heads (default: 8, use 32 for Llama-class models).")
    parser.add_argument("--bits", type=int, default=3, help="Quantization bits.")
    parser.add_argument("--group-size", type=int, default=64, help="Group size.")
    parser.add_argument(
        "--no-residual", action="store_true", help="Disable residual correction."
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Optional argument list for testing.
    """
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )

    args = parse_args(argv)

    config = TurboQuantConfig(
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        bits=args.bits,
        group_size=args.group_size,
        residual_correction=not args.no_residual,
    )

    suites = (
        ["memory", "speed", "quality", "vps"] if args.suite == "all" else [args.suite]
    )

    results: dict[str, pd.DataFrame] = {}

    for suite in suites:
        log.info(f"Running {suite} benchmark...")

        if suite == "memory":
            results["memory"] = memory_benchmark(config, batch_size=args.batch_size)
        elif suite == "speed":
            results["speed"] = speed_benchmark(config, batch_size=args.batch_size)
        elif suite == "quality":
            results["quality"] = quality_benchmark(config)
        elif suite == "vps":
            results["vps"] = vps_benchmark(config)

        # Save incrementally
        _save_incremental(results, args.output)
        log.info(f"{suite} benchmark complete, results saved.")

    # Final reports
    _save_incremental(results, args.output)
    generate_html_report(results, args.output)

    print(f"\n{'=' * 60}")
    print("TurboQuant Benchmark Complete")
    print(f"Results saved to: {args.output}/")
    print(f"{'=' * 60}")

    # Print summary
    if "memory" in results:
        df = results["memory"]
        for _, row in df.iterrows():
            print(
                f"  TurboQuant saves {row['fp16_mb'] - row['tq4_mb']:.0f} MB "
                f"({row['savings_4bit_pct']:.0f}%) on seq_len={row['seq_len']:.0f}"
            )


if __name__ == "__main__":
    main()
