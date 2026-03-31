#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ProbeResult:
    total_requests: int
    success: int
    failed: int
    error_rate_percent: float
    rps: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    latency_ms_mean: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    ttft_ms_p99: float
    ttft_ms_mean: float


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((q / 100.0) * (len(s) - 1)))))
    return s[idx]


async def one_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> tuple[bool, float, float]:
    # Uses OpenAI-compatible completions endpoint.
    # Many serving stacks expose this path (vLLM, TGI wrappers, custom proxies).
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def _call() -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                if resp.status >= 400:
                    return False, body
                return True, body
        except urllib.error.HTTPError as exc:
            return False, str(exc)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    try:
        ok, body = await asyncio.to_thread(_call)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not ok:
            return False, elapsed_ms, 0.0
        data = json.loads(body)
        ttft_ms = elapsed_ms
        _ = data.get("choices", [])
        return True, elapsed_ms, ttft_ms
    except Exception:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return False, elapsed_ms, 0.0


async def run_probe(args: argparse.Namespace) -> ProbeResult:
    latencies: list[float] = []
    ttfts: list[float] = []
    success = 0
    failed = 0
    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()

    async def worker() -> None:
        nonlocal success, failed
        async with semaphore:
            ok, latency_ms, ttft_ms = await one_request(
                base_url=args.base_url,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            latencies.append(latency_ms)
            if ok:
                success += 1
                ttfts.append(ttft_ms)
            else:
                failed += 1

    tasks: list[asyncio.Task[None]] = []
    if args.duration_sec > 0:
        end_time = time.perf_counter() + args.duration_sec
        while time.perf_counter() < end_time:
            tasks.append(asyncio.create_task(worker()))
            await asyncio.sleep(0)
        await asyncio.gather(*tasks)
    else:
        for _ in range(args.requests):
            tasks.append(asyncio.create_task(worker()))
        await asyncio.gather(*tasks)

    elapsed = max(1e-9, time.perf_counter() - start)
    total = success + failed
    return ProbeResult(
        total_requests=total,
        success=success,
        failed=failed,
        error_rate_percent=(failed / total * 100.0) if total else 100.0,
        rps=total / elapsed,
        latency_ms_p50=percentile(latencies, 50),
        latency_ms_p95=percentile(latencies, 95),
        latency_ms_p99=percentile(latencies, 99),
        latency_ms_mean=(statistics.fmean(latencies) if latencies else 0.0),
        ttft_ms_p50=percentile(ttfts, 50),
        ttft_ms_p95=percentile(ttfts, 95),
        ttft_ms_p99=percentile(ttfts, 99),
        ttft_ms_mean=(statistics.fmean(ttfts) if ttfts else 0.0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-compatible load probe")
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="model id exposed by serving stack")
    parser.add_argument("--prompt", default="Summarize TurboQuant in one paragraph.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--duration-sec", type=int, default=0, help="if >0, run time-based soak")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", required=True, help="output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_probe(args))
    output = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "prompt_chars": len(args.prompt),
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "duration_sec": args.duration_sec,
            "timeout_sec": args.timeout,
        },
        "metrics": asdict(result),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
