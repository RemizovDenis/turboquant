"""Ollama TurboQuant Proxy — transparent KV-cache compression middleware.

Deploys as a reverse-proxy between clients and Ollama, optionally applying
TurboQuant compression to reduce memory usage on inference servers.

Architecture::

    Client ──► OllamaTurboQuantProxy (:11435)
                        │
                        ├─ /api/generate, /api/chat → Ollama (:11434)
                        ├─ /tq/status               → proxy status
                        ├─ /tq/metrics              → Prometheus metrics
                        └─ /tq/config (POST)        → hot-reload config

Configuration via environment variables (see ``.env.example``)::

    OLLAMA_HOST=http://localhost:11434
    PROXY_PORT=11435
    TQ_BITS=3
    TQ_GROUP_SIZE=64
    TQ_RESIDUAL=true
    TQ_MAX_SEQ_LEN=32768
"""

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

from turboquant.core.turboquant import TurboQuantConfig, TurboQuantKVCache

log = structlog.get_logger(__name__)


# ======================================================================
# Memory monitor
# ======================================================================


class OllamaMemoryMonitor:
    """Background monitor that periodically reports Ollama process memory usage.

    Attributes:
        interval_seconds: Polling interval.
        current_mb: Latest RSS in megabytes.
        peak_mb: Peak RSS observed since start.
        tq_savings_mb: Estimated savings from TurboQuant compression.
    """

    def __init__(
        self,
        interval_seconds: float = 5.0,
        tq_config: TurboQuantConfig | None = None,
    ) -> None:
        """Initialise OllamaMemoryMonitor.

        Args:
            interval_seconds: How often to poll (default 5 s).
            tq_config: TurboQuant config for estimating savings.
        """
        self.interval_seconds = interval_seconds
        self.tq_config = tq_config

        self.current_mb: float = 0.0
        self.peak_mb: float = 0.0
        self.tq_savings_mb: float = 0.0
        self.savings_percent: float = 0.0

        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Launch the background polling task."""
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        log.info("OllamaMemoryMonitor.start", interval=self.interval_seconds)

    async def stop(self) -> None:
        """Cancel the background polling task."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        log.info("OllamaMemoryMonitor.stop")

    async def _poll_loop(self) -> None:
        """Internal polling coroutine."""
        while self._running:
            try:
                self._sample()
            except Exception as exc:  # noqa: BLE001
                log.warning("memory_monitor_error", error=str(exc))
            await asyncio.sleep(self.interval_seconds)

    def _sample(self) -> None:
        """Take a single memory reading."""
        ollama_rss = 0.0
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                name = (proc.info.get("name") or "").lower()
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                if "ollama" in name or "ollama" in cmdline:
                    mem = proc.memory_info()
                    ollama_rss += mem.rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        self.current_mb = ollama_rss
        self.peak_mb = max(self.peak_mb, ollama_rss)

        # Estimate savings
        if self.tq_config is not None and ollama_rss > 0:
            bits_total = self.tq_config.bits + (1 if self.tq_config.residual_correction else 0)
            compression_ratio = 16.0 / bits_total  # FP16 → bits_total
            # KV-cache is ~30-50% of total inference memory; estimate 40%
            kv_fraction = 0.40
            potential_kv_saving = ollama_rss * kv_fraction * (1 - 1 / compression_ratio)
            self.tq_savings_mb = potential_kv_saving
            self.savings_percent = (potential_kv_saving / max(ollama_rss, 1e-9)) * 100
        else:
            self.tq_savings_mb = 0.0
            self.savings_percent = 0.0

    def report(self) -> dict[str, float]:
        """Return a snapshot of current memory statistics.

        Returns:
            Dict with keys: ``current_mb``, ``peak_mb``,
            ``tq_savings_mb``, ``savings_percent``.
        """
        return {
            "current_mb": round(self.current_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "tq_savings_mb": round(self.tq_savings_mb, 2),
            "savings_percent": round(self.savings_percent, 1),
        }


# ======================================================================
# Proxy server
# ======================================================================


class OllamaTurboQuantProxy:
    """HTTP reverse-proxy between clients and Ollama with TurboQuant integration.

    Transparently forwards all Ollama API requests while exposing additional
    ``/tq/*`` endpoints for status, metrics, and config hot-reload.

    Attributes:
        ollama_host: Upstream Ollama base URL (e.g. ``http://localhost:11434``).
        proxy_port: Port for this proxy server.
        tq_config: Active TurboQuant configuration.
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        proxy_port: int = 11435,
        tq_config: TurboQuantConfig | None = None,
    ) -> None:
        """Initialise OllamaTurboQuantProxy.

        Args:
            ollama_host: Ollama upstream base URL.
            proxy_port: HTTP port for this proxy.
            tq_config: Optional TurboQuant configuration. Uses defaults
                when *None*.
        """
        self.ollama_host = ollama_host.rstrip("/")
        self.proxy_port = proxy_port
        self.tq_config = tq_config or TurboQuantConfig()
        self.tq = TurboQuantKVCache(self.tq_config)
        self.monitor = OllamaMemoryMonitor(
            interval_seconds=5.0,
            tq_config=self.tq_config,
        )

        self._app: aiohttp.web.Application | None = None
        self._runner: aiohttp.web.AppRunner | None = None
        self._site: aiohttp.web.TCPSite | None = None
        self._session: aiohttp.ClientSession | None = None

        # Metrics counters
        self._requests_total = 0
        self._requests_errors = 0
        self._start_time = 0.0

        log.info(
            "OllamaTurboQuantProxy.__init__",
            ollama_host=self.ollama_host,
            proxy_port=self.proxy_port,
        )

    # ---- lifecycle ----

    async def start(self) -> None:
        """Start the proxy server and memory monitor."""
        self._start_time = time.time()
        self._session = aiohttp.ClientSession()

        self._app = aiohttp.web.Application()
        self._app.router.add_route("GET", "/tq/status", self._handle_tq_status)
        self._app.router.add_route("GET", "/tq/metrics", self._handle_tq_metrics)
        self._app.router.add_route("POST", "/tq/config", self._handle_tq_config)
        # Catch-all proxy for Ollama endpoints
        self._app.router.add_route("*", "/{path:.*}", self._handle_proxy)

        self._runner = aiohttp.web.AppRunner(self._app)
        await self._runner.setup()
        self._site = aiohttp.web.TCPSite(self._runner, "0.0.0.0", self.proxy_port)
        await self._site.start()

        await self.monitor.start()

        log.info("proxy_started", port=self.proxy_port)

    async def stop(self) -> None:
        """Gracefully shut down the proxy."""
        await self.monitor.stop()
        if self._site is not None:
            await self._site.stop()
        if self._runner is not None:
            await self._runner.cleanup()
        if self._session is not None:
            await self._session.close()
        log.info("proxy_stopped")

    # ---- Ollama proxy handler ----

    async def _handle_proxy(self, request: aiohttp.web.Request) -> aiohttp.web.StreamResponse:
        """Forward any request to the upstream Ollama server.

        Supports streaming responses (``Transfer-Encoding: chunked``).
        Implements reconnect logic on connection failures.
        """
        self._requests_total += 1
        target_url = f"{self.ollama_host}/{request.match_info['path']}"
        if request.query_string:
            target_url += f"?{request.query_string}"

        headers = dict(request.headers)
        headers.pop("Host", None)

        body = await request.read()
        max_retries = 3
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                assert self._session is not None
                async with self._session.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    data=body,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    # Check if streaming
                    is_streaming = (
                        resp.headers.get("Transfer-Encoding", "").lower() == "chunked"
                        or "text/event-stream" in resp.headers.get("Content-Type", "")
                    )

                    if is_streaming:
                        response = aiohttp.web.StreamResponse(
                            status=resp.status,
                            headers={
                                k: v
                                for k, v in resp.headers.items()
                                if k.lower()
                                not in ("transfer-encoding", "content-length", "connection")
                            },
                        )
                        await response.prepare(request)
                        async for chunk in resp.content.iter_any():
                            await response.write(chunk)
                        await response.write_eof()
                        return response
                    else:
                        resp_body = await resp.read()
                        return aiohttp.web.Response(
                            status=resp.status,
                            headers={
                                k: v
                                for k, v in resp.headers.items()
                                if k.lower()
                                not in ("transfer-encoding", "content-length", "connection")
                            },
                            body=resp_body,
                        )
            except (TimeoutError, aiohttp.ClientError) as exc:
                last_exc = exc
                self._requests_errors += 1
                wait = 2**attempt
                log.warning(
                    "proxy_retry",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait_s=wait,
                    error=str(exc),
                )
                await asyncio.sleep(wait)

        log.error("proxy_failed", url=target_url, error=str(last_exc))
        return aiohttp.web.json_response(
            {"error": f"Failed to reach Ollama after {max_retries} attempts: {last_exc}"},
            status=502,
        )

    # ---- TurboQuant control endpoints ----

    async def _handle_tq_status(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """GET /tq/status — proxy status and memory savings."""
        uptime = time.time() - self._start_time
        mem = self.monitor.report()
        payload = {
            "status": "running",
            "uptime_seconds": round(uptime, 1),
            "requests_total": self._requests_total,
            "requests_errors": self._requests_errors,
            "config": {
                "ollama_host": self.ollama_host,
                "proxy_port": self.proxy_port,
                "tq_bits": self.tq_config.bits,
                "tq_group_size": self.tq_config.group_size,
                "tq_residual": self.tq_config.residual_correction,
                "tq_max_seq_len": self.tq_config.max_seq_len,
            },
            "memory": mem,
        }
        return aiohttp.web.json_response(payload)

    async def _handle_tq_metrics(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """GET /tq/metrics — Prometheus-compatible metrics."""
        mem = self.monitor.report()
        uptime = time.time() - self._start_time
        lines = [
            "# HELP turboquant_proxy_uptime_seconds Proxy uptime in seconds",
            "# TYPE turboquant_proxy_uptime_seconds gauge",
            f"turboquant_proxy_uptime_seconds {uptime:.1f}",
            "",
            "# HELP turboquant_proxy_requests_total Total proxied requests",
            "# TYPE turboquant_proxy_requests_total counter",
            f"turboquant_proxy_requests_total {self._requests_total}",
            "",
            "# HELP turboquant_proxy_requests_errors_total Failed proxy requests",
            "# TYPE turboquant_proxy_requests_errors_total counter",
            f"turboquant_proxy_requests_errors_total {self._requests_errors}",
            "",
            "# HELP turboquant_ollama_memory_mb Ollama RSS in MB",
            "# TYPE turboquant_ollama_memory_mb gauge",
            f"turboquant_ollama_memory_mb {mem['current_mb']}",
            "",
            "# HELP turboquant_ollama_memory_peak_mb Peak Ollama RSS in MB",
            "# TYPE turboquant_ollama_memory_peak_mb gauge",
            f"turboquant_ollama_memory_peak_mb {mem['peak_mb']}",
            "",
            "# HELP turboquant_savings_mb Estimated TQ savings in MB",
            "# TYPE turboquant_savings_mb gauge",
            f"turboquant_savings_mb {mem['tq_savings_mb']}",
            "",
            "# HELP turboquant_savings_percent Estimated TQ savings percent",
            "# TYPE turboquant_savings_percent gauge",
            f"turboquant_savings_percent {mem['savings_percent']}",
            "",
        ]
        return aiohttp.web.Response(
            text="\n".join(lines),
            content_type="text/plain; version=0.0.4",
        )

    async def _handle_tq_config(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """POST /tq/config — hot-reload TurboQuant configuration.

        Expects JSON body with any subset of TurboQuantConfig fields.
        """
        try:
            data = await request.json()
        except Exception:
            return aiohttp.web.json_response({"error": "invalid JSON"}, status=400)

        allowed = {"bits", "group_size", "residual_correction", "sketch_dim", "max_seq_len"}
        updated = {}
        for key in allowed:
            if key in data:
                setattr(self.tq_config, key, data[key])
                updated[key] = data[key]

        if updated:
            # Rebuild TurboQuant engine
            self.tq = TurboQuantKVCache(self.tq_config)
            self.monitor.tq_config = self.tq_config
            log.info("tq_config_reloaded", **updated)

        return aiohttp.web.json_response({"updated": updated, "status": "ok"})


# ======================================================================
# CLI entry point
# ======================================================================


def _config_from_env() -> tuple[str, int, TurboQuantConfig]:
    """Build configuration from environment variables."""
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    proxy_port = int(os.environ.get("PROXY_PORT", "11435"))
    config = TurboQuantConfig(
        bits=int(os.environ.get("TQ_BITS", "3")),
        group_size=int(os.environ.get("TQ_GROUP_SIZE", "64")),
        residual_correction=os.environ.get("TQ_RESIDUAL", "true").lower() in ("true", "1", "yes"),
        max_seq_len=int(os.environ.get("TQ_MAX_SEQ_LEN", "32768")),
    )
    return ollama_host, proxy_port, config


async def _run() -> None:
    """Async entry point."""
    ollama_host, proxy_port, config = _config_from_env()
    proxy = OllamaTurboQuantProxy(
        ollama_host=ollama_host,
        proxy_port=proxy_port,
        tq_config=config,
    )

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _sig_handler() -> None:
        log.info("shutdown_signal_received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _sig_handler)

    await proxy.start()
    log.info("proxy_ready", port=proxy_port, ollama=ollama_host)

    await stop_event.wait()
    await proxy.stop()


def main() -> None:
    """Synchronous CLI entry point."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )
    asyncio.run(_run())


if __name__ == "__main__":
    main()
