# Copyright (c) 2026 Denis Remizov. Licensed under BUSL-1.1.
# See LICENSE file for details.


"""Ollama proxy with TurboQuant-MoE support."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from contextlib import suppress

import aiohttp
import aiohttp.web
import psutil
import structlog

from turboquant.core.turboquant_moe import TurboQuantMoE, TurboQuantMoEConfig

LOGGER = structlog.get_logger(__name__)


class OllamaMemoryMonitor:
    """Tracks process and optional GPU memory usage for savings reporting."""

    def __init__(self) -> None:
        self.current_mb = 0.0
        self.peak_mb = 0.0
        self.tq_savings_mb = 0.0
        self.savings_percent = 0.0
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _loop(self) -> None:
        while self._running:
            self._sample()
            await asyncio.sleep(5)

    def _sample(self) -> None:
        rss_mb = 0.0
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                name = (proc.info.get("name") or "").lower()
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                if "ollama" in name or "ollama" in cmdline:
                    rss_mb += proc.memory_info().rss / (1024**2)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        self.current_mb = rss_mb
        self.peak_mb = max(self.peak_mb, rss_mb)

    def report(self) -> dict[str, float]:
        return {
            "current_mb": self.current_mb,
            "peak_mb": self.peak_mb,
            "tq_savings_mb": self.tq_savings_mb,
            "savings_percent": self.savings_percent,
        }


class OllamaTurboQuantProxy:
    """Reverse proxy for Ollama endpoints enriched with TurboQuant endpoints."""

    def __init__(
        self,
        ollama_host: str,
        proxy_port: int,
        tq_moe_config: TurboQuantMoEConfig,
    ) -> None:
        self.ollama_host = ollama_host.rstrip("/")
        self.proxy_port = proxy_port
        self.tq_moe_config = tq_moe_config
        self.tq_manager = TurboQuantMoE(tq_moe_config)
        self.monitor = OllamaMemoryMonitor()

        self._app: aiohttp.web.Application | None = None
        self._runner: aiohttp.web.AppRunner | None = None
        self._site: aiohttp.web.TCPSite | None = None
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self._app = aiohttp.web.Application()
        self._app.router.add_get("/tq/status", self._tq_status)
        self._app.router.add_get("/tq/metrics", self._tq_metrics)
        self._app.router.add_post("/tq/config", self._tq_config)
        self._app.router.add_get("/tq/experts", self._tq_experts)
        self._app.router.add_post("/tq/warmup", self._tq_warmup)
        self._app.router.add_get("/health", self._health)

        for path in ["/api/generate", "/api/chat", "/api/pull", "/api/show", "/api/ps"]:
            self._app.router.add_route("*", path, self._proxy)

        self._runner = aiohttp.web.AppRunner(self._app)
        await self._runner.setup()
        self._site = aiohttp.web.TCPSite(self._runner, host="0.0.0.0", port=self.proxy_port)
        await self._site.start()

        self._session = aiohttp.ClientSession()
        await self.monitor.start()
        LOGGER.info("proxy_started", port=self.proxy_port, host=self.ollama_host)

    async def stop(self) -> None:
        await self.monitor.stop()
        if self._session is not None:
            await self._session.close()
        if self._site is not None:
            await self._site.stop()
        if self._runner is not None:
            await self._runner.cleanup()

    async def _proxy(self, request: aiohttp.web.Request) -> aiohttp.web.StreamResponse:
        assert self._session is not None
        url = f"{self.ollama_host}{request.path_qs}"
        headers = dict(request.headers)
        headers.pop("Host", None)
        body = await request.read()

        delay = 0.25
        for attempt in range(5):
            try:
                async with self._session.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    data=body,
                ) as upstream:
                    response = aiohttp.web.StreamResponse(
                        status=upstream.status,
                        headers=upstream.headers,
                    )
                    await response.prepare(request)
                    async for chunk in upstream.content.iter_chunked(8192):
                        await response.write(chunk)
                    await response.write_eof()
                    return response
            except aiohttp.ClientError:
                if attempt == 4:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 4.0)
        raise RuntimeError("unreachable")

    async def _tq_status(self, _request: aiohttp.web.Request) -> aiohttp.web.Response:
        rep = self.tq_manager.memory_report()
        payload = {
            "config": {
                "model_type": self.tq_moe_config.model_type,
                "bits": self.tq_moe_config.kv_config.bits,
                "gpu_cache_experts": self.tq_moe_config.expert_config.gpu_cache_size,
            },
            "memory": rep.__dict__,
            "monitor": self.monitor.report(),
        }
        return aiohttp.web.json_response(payload)

    async def _tq_metrics(self, _request: aiohttp.web.Request) -> aiohttp.web.Response:
        rep = self.tq_manager.memory_report()
        text = "\n".join(
            [
                f"turboquant_total_saved_mb {rep.total_saved_mb}",
                f"turboquant_expert_hit_rate {rep.expert_hit_rate}",
                f"turboquant_prefetch_accuracy {rep.prefetch_accuracy}",
                "",
            ]
        )
        return aiohttp.web.Response(text=text, content_type="text/plain")

    async def _tq_config(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        payload = await request.json()
        if "enable_expert_prediction" in payload:
            self.tq_moe_config.enable_expert_prediction = bool(payload["enable_expert_prediction"])
        return aiohttp.web.json_response({"ok": True, "updated": payload})

    async def _tq_experts(self, _request: aiohttp.web.Request) -> aiohttp.web.Response:
        stats = self.tq_manager.expert_cache.stats()
        return aiohttp.web.json_response(stats.__dict__)

    async def _tq_warmup(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        payload = await request.json()
        history = payload.get("routing_history", [])
        self.tq_manager.expert_cache.warmup(history)
        return aiohttp.web.json_response({"ok": True})

    async def _health(self, _request: aiohttp.web.Request) -> aiohttp.web.Response:
        return aiohttp.web.json_response({"status": "ok", "ts": time.time()})


def load_proxy_from_env() -> OllamaTurboQuantProxy:
    """Build proxy instance from environment variables."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    port = int(os.getenv("PROXY_PORT", "11435"))
    bits = int(os.getenv("TQ_BITS", "3"))
    gpu_cache = int(os.getenv("TQ_GPU_CACHE_EXPERTS", "4"))

    cfg = TurboQuantMoEConfig.from_pretrained_config(
        type(
            "Cfg", (), {"hidden_size": 4096, "num_attention_heads": 32, "model_type": "mixtral"}
        )(),
        bits=bits,
        gpu_cache_size=gpu_cache,
    )
    return OllamaTurboQuantProxy(host, port, cfg)


async def run_proxy_from_env() -> None:
    """Entrypoint to run proxy with SIGTERM/SIGINT graceful shutdown."""
    proxy = load_proxy_from_env()
    await proxy.start()

    stop_event = asyncio.Event()

    def _stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(RuntimeError):
            loop.add_signal_handler(sig, _stop)

    await stop_event.wait()
    await proxy.stop()


def main() -> None:
    asyncio.run(run_proxy_from_env())


if __name__ == "__main__":
    main()
