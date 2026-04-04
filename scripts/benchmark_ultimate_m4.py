#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import psutil
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for local runners
    raise SystemExit(
        "Missing dependency: psutil. Install with `pip install psutil` or "
        "`pip install -e '.[benchmark]'`."
    ) from exc

try:
    import requests
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for local runners
    raise SystemExit("Missing dependency: requests. Install with `pip install requests`.") from exc

from turboquant.benchmarks.field_report import render_field_markdown, summarize_field_results

OLLAMA_BASE = "http://127.0.0.1:11434"
TQ_PROXY = "http://127.0.0.1:11435"
DEFAULT_MODELS = ["mistral:latest", "mixtral:latest", "llama3.1:latest"]
NEEDLE = "The secret authentication code is SECURILAYER-9X47-ALPHA"
CONTEXT_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def wait_http(url: str, timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=3)
            if resp.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Timeout waiting for {url}")


def ollama_rss_mb() -> float:
    total = 0.0
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmd = " ".join(proc.info.get("cmdline") or []).lower()
            if "ollama" in name or "ollama" in cmd:
                total += proc.memory_info().rss / (1024**2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def pull_model(base_url: str, model: str) -> None:
    payload = {"name": model, "stream": False}
    resp = requests.post(f"{base_url}/api/pull", json=payload, timeout=1800)
    resp.raise_for_status()


def model_context_limit(base_url: str, model: str) -> int:
    try:
        resp = requests.post(
            f"{base_url}/api/show",
            json={"model": model},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        details = data.get("details", {})
        opts = data.get("model_info", {})
        for key in ("num_ctx", "context_length"):
            if key in details:
                return int(details[key])
        for key in (
            "llama.context_length",
            "mistral.context_length",
            "general.context_length",
            "num_ctx",
        ):
            if key in opts:
                return int(opts[key])
    except (requests.RequestException, ValueError, TypeError):
        pass
    return 4096


def generate_once(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
) -> tuple[float, dict[str, Any], float]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }
    t0 = time.perf_counter()
    first_token_ms = math.nan

    if not stream:
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=1200)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        return dt_ms, resp.json(), first_token_ms

    with requests.post(
        f"{base_url}/api/generate",
        json=payload,
        stream=True,
        timeout=1200,
    ) as resp:
        resp.raise_for_status()
        last = {}
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            chunk = json.loads(line)
            if math.isnan(first_token_ms) and chunk.get("response"):
                first_token_ms = (time.perf_counter() - t0) * 1000.0
            last = chunk
            if chunk.get("done"):
                break
        total_ms = (time.perf_counter() - t0) * 1000.0
        return total_ms, last, first_token_ms


def build_context(target_tokens: int) -> str:
    sentence = (
        "Security telemetry stream analysis with contextual routing and token-level "
        "optimization for robust enterprise inference workloads. "
    )
    words_per_sentence = len(sentence.split())
    approx_words = int(target_tokens * 1.35)
    reps = max(1, approx_words // max(1, words_per_sentence))
    return (sentence * reps).strip()


def needle_prompt(target_tokens: int) -> str:
    context = build_context(target_tokens)
    mid = len(context) // 2
    injected = context[:mid] + " " + NEEDLE + " " + context[mid:]
    return f"{injected}\n\nQuestion: what is the secret authentication code?"


def p95(values: list[float]) -> float:
    if not values:
        return math.nan
    arr = sorted(values)
    idx = min(len(arr) - 1, int(round(0.95 * (len(arr) - 1))))
    return arr[idx]


def run_latency_suite(base_url: str, model: str, runs: int) -> dict[str, Any]:
    prompt = build_context(4096)
    latencies = []
    ttfts = []
    tps = []
    details = []
    mem_before = ollama_rss_mb()

    for _ in range(runs):
        total_ms, stream_payload, first_token_ms = generate_once(
            base_url=base_url,
            model=model,
            prompt=prompt,
            max_tokens=128,
            stream=True,
        )
        eval_count = int(stream_payload.get("eval_count") or 0)
        eval_duration_ns = int(stream_payload.get("eval_duration") or 0)
        if eval_count > 0 and eval_duration_ns > 0:
            tokens_per_sec = eval_count / (eval_duration_ns / 1e9)
        else:
            tokens_per_sec = float("nan")
        latencies.append(total_ms)
        ttfts.append(first_token_ms)
        tps.append(tokens_per_sec)
        details.append(
            {
                "latency_ms": total_ms,
                "ttft_ms": first_token_ms,
                "tokens_per_sec": tokens_per_sec,
                "prompt_eval_count": stream_payload.get("prompt_eval_count"),
                "eval_count": stream_payload.get("eval_count"),
            }
        )

    mem_after = ollama_rss_mb()
    ttft_values = [v for v in ttfts if not math.isnan(v)]
    tps_values = [v for v in tps if not math.isnan(v)]

    return {
        "runs": details,
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": p95(latencies),
        "latency_avg_ms": statistics.fmean(latencies),
        "ttft_avg_ms": statistics.fmean(ttft_values) if ttft_values else math.nan,
        "tokens_per_second_avg": statistics.fmean(tps_values) if tps_values else math.nan,
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "memory_delta_mb": mem_after - mem_before,
    }


def run_needle_suite(base_url: str, model: str, model_ctx_limit: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for clen in CONTEXT_LENGTHS:
        if clen > model_ctx_limit:
            out[str(clen)] = {
                "status": "skipped_context_limit",
                "model_ctx_limit": model_ctx_limit,
                "recall_percent": None,
                "latency_ms": None,
                "prompt_eval_count": None,
                "eval_count": None,
            }
            continue

        print(f"[INFO] Needle test {model}: context={clen}")
        prompt = needle_prompt(clen)
        try:
            total_ms, payload, _ = generate_once(
                base_url=base_url,
                model=model,
                prompt=prompt,
                max_tokens=64,
                stream=False,
            )
            text = payload.get("response", "")
            found = NEEDLE in text
            out[str(clen)] = {
                "status": "ok",
                "recall_percent": 100.0 if found else 0.0,
                "latency_ms": total_ms,
                "prompt_eval_count": payload.get("prompt_eval_count"),
                "eval_count": payload.get("eval_count"),
            }
        except requests.RequestException as exc:
            out[str(clen)] = {
                "status": "request_error",
                "error": str(exc),
                "recall_percent": None,
                "latency_ms": None,
                "prompt_eval_count": None,
                "eval_count": None,
            }
    return out


def read_tq_status(proxy_url: str) -> dict[str, Any]:
    try:
        resp = requests.get(f"{proxy_url}/tq/status", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return {}


def _proxy_port(proxy_url: str) -> str:
    parsed = urlparse(proxy_url)
    if parsed.port is not None:
        return str(parsed.port)
    if _is_localhost_url(proxy_url):
        return "11435"
    if parsed.scheme == "https":
        return "443"
    if parsed.scheme == "http":
        return "80"
    return "11435"


def _is_localhost_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def start_tq_proxy(ollama_base: str, proxy_url: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = ollama_base
    env["PROXY_PORT"] = _proxy_port(proxy_url)
    candidates = [
        Path(env.get("VIRTUAL_ENV", "")) / "bin" / "turboquant-proxy",
        Path(".venv/bin/turboquant-proxy"),
        Path("/usr/local/bin/turboquant-proxy"),
    ]
    cmd: list[str] | None = None
    for candidate in candidates:
        if candidate.exists():
            cmd = [str(candidate)]
            break
    if cmd is None:
        # Fallback for environments where script entrypoint is not installed.
        cmd = [env.get("PYTHON", "python3"), "-m", "turboquant.integrations.ollama"]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    wait_http(f"{proxy_url}/health", timeout_s=120)
    return proc


def stop_tq_proxy(proc: subprocess.Popen[str]) -> None:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out-json", default="benchmark_ultimate_m4.json")
    parser.add_argument("--out-readme", default="README_benchmark.md")
    parser.add_argument("--output-dir", default="results/field_local")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--ollama-url", default=OLLAMA_BASE)
    parser.add_argument("--proxy-url", default=TQ_PROXY)
    args = parser.parse_args()

    ollama_base = args.ollama_url.rstrip("/")
    proxy_url = args.proxy_url.rstrip("/")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / Path(args.out_json).name
    out_readme = out_dir / Path(args.out_readme).name

    wait_http(f"{ollama_base}/api/tags")

    results: dict[str, Any] = {"timestamp": now_iso(), "host": "localhost", "models": {}}
    total_mem_saved = 0.0

    for model in args.models:
        if not args.skip_pull:
            print(f"[INFO] Pull/check model: {model}")
            try:
                pull_model(ollama_base, model)
            except requests.RequestException as exc:
                print(f"[WARN] model pull failed for {model}: {exc}")
                results["models"][model] = {"error": f"pull_failed: {exc}"}
                continue

        ctx_limit = model_context_limit(ollama_base, model)
        print(f"[INFO] Model context limit for {model}: {ctx_limit}")

        print(f"[INFO] Baseline suite: {model}")
        baseline_speed = run_latency_suite(ollama_base, model, args.runs)
        baseline_needle = run_needle_suite(ollama_base, model, ctx_limit)

        tq_speed: dict[str, Any] = {}
        tq_needle: dict[str, Any] = {}
        tq_status: dict[str, Any] = {}
        speedup_x: float | None = None
        mem_saved: float | None = None
        kv_ratio: float | None = None
        proxy_error: str | None = None

        if args.no_proxy:
            proxy_error = "proxy_disabled_by_flag"
        else:
            print(f"[INFO] TurboQuant proxy suite: {model}")
            proc: subprocess.Popen[str] | None = None
            try:
                if _is_localhost_url(proxy_url):
                    proc = start_tq_proxy(ollama_base, proxy_url)
                else:
                    wait_http(f"{proxy_url}/health")
                tq_speed = run_latency_suite(proxy_url, model, args.runs)
                tq_needle = run_needle_suite(proxy_url, model, ctx_limit)
                tq_status = read_tq_status(proxy_url)
                speedup_x = baseline_speed["latency_avg_ms"] / max(1e-9, tq_speed["latency_avg_ms"])
                mem_saved = baseline_speed["memory_delta_mb"] - tq_speed["memory_delta_mb"]
                total_mem_saved += mem_saved if mem_saved is not None else 0.0

                mem = tq_status.get("memory", {})
                if "kv_compression_ratio" in mem:
                    kv_ratio = float(mem["kv_compression_ratio"])
                elif (
                    "total_saved_mb" in mem
                    and "total_kv_mb" in mem
                    and float(mem["total_kv_mb"]) > 0
                ):
                    kv_ratio = float(mem["total_kv_mb"]) / max(
                        1e-9, float(mem["total_kv_mb"]) - float(mem["total_saved_mb"])
                    )
            except Exception as exc:  # noqa: BLE001
                proxy_error = str(exc)
            finally:
                if proc is not None:
                    stop_tq_proxy(proc)

        results["models"][model] = {
            "baseline": baseline_speed,
            "turboquant": tq_speed,
            "model_context_limit": ctx_limit,
            "needle_baseline": baseline_needle,
            "needle_turboquant": tq_needle,
            "tq_status": tq_status,
            "speedup_x": speedup_x,
            "memory_saved_mb": mem_saved,
            "kv_compression_ratio": kv_ratio,
            "proxy_error": proxy_error,
        }

    summary = summarize_field_results(results)
    results["summary"] = summary
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out_readme.write_text(render_field_markdown(results, summary), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nSaved: {out_json.resolve()}")
    print(f"Report: {out_readme.resolve()}")
    print(f"TurboQuant saves {total_mem_saved / 1024.0:.3f} GB on M4 Air")


if __name__ == "__main__":
    main()
