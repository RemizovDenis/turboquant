"""Core TurboQuant KV-cache manager."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import torch

from turboquant.core.polar_quant import PolarQuantConfig, PolarQuantizer
from turboquant.core.qjl import QJLConfig, QJLResidualCorrector

log = structlog.get_logger(__name__)


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuantKVCache."""

    head_dim: int = 128
    num_heads: int = 32
    bits: int = 3
    group_size: int = 64
    residual_correction: bool = True
    sketch_dim: int | None = None
    use_triton_kernels: bool = True
    use_hadamard: bool = True
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    seed: int = 42
    max_seq_len: int = 131072
    cpu_offload_threshold_mb: float = 1024.0

    @classmethod
    def from_model_config(cls, model_config: Any) -> TurboQuantConfig:
        """Build config from HuggingFace model config-like object."""
        head_dim = None
        num_heads = None

        if hasattr(model_config, "hidden_size") and hasattr(model_config, "num_attention_heads"):
            head_dim = int(model_config.hidden_size // model_config.num_attention_heads)
            num_heads = int(model_config.num_attention_heads)

        if hasattr(model_config, "head_dim"):
            head_dim = int(model_config.head_dim)
        if hasattr(model_config, "num_key_value_heads"):
            num_heads = int(model_config.num_key_value_heads)

        if head_dim is None or num_heads is None:
            raise ValueError("Could not infer head_dim/num_heads from model config")

        return cls(head_dim=head_dim, num_heads=num_heads)


@dataclass
class CacheEntry:
    """Compressed KV cache entry."""

    compressed_keys: tuple[torch.Tensor, torch.Tensor]
    compressed_values: tuple[torch.Tensor, torch.Tensor]
    residual_keys: torch.Tensor | None = None
    residual_values: torch.Tensor | None = None
    residual_scales_k: torch.Tensor | None = None
    residual_scales_v: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    access_count: int = 0

    def to(self, device: str | torch.device) -> CacheEntry:
        self.compressed_keys = (
            self.compressed_keys[0].to(device),
            self.compressed_keys[1].to(device),
        )
        self.compressed_values = (
            self.compressed_values[0].to(device),
            self.compressed_values[1].to(device),
        )
        if self.residual_keys is not None:
            self.residual_keys = self.residual_keys.to(device)
        if self.residual_values is not None:
            self.residual_values = self.residual_values.to(device)
        if self.residual_scales_k is not None:
            self.residual_scales_k = self.residual_scales_k.to(device)
        if self.residual_scales_v is not None:
            self.residual_scales_v = self.residual_scales_v.to(device)
        return self


class TurboQuantKVCache:
    """Unified KV-cache compressor with optional QJL residual correction."""

    def __init__(self, config: TurboQuantConfig) -> None:
        self.config = config
        self._lock = threading.RLock()

        if "cuda" in config.device and not torch.cuda.is_available():
            log.warning("cuda requested but unavailable, using cpu")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        qcfg = PolarQuantConfig(
            head_dim=config.head_dim,
            bits=config.bits,
            group_size=config.group_size,
            seed=config.seed,
            use_hadamard=config.use_hadamard,
        )
        self.quantizer = PolarQuantizer(qcfg).to(self.device)

        self.corrector: QJLResidualCorrector | None = None
        if config.residual_correction:
            self.corrector = QJLResidualCorrector(
                QJLConfig(
                    head_dim=config.head_dim,
                    sketch_dim=config.sketch_dim,
                    seed=config.seed + 1,
                )
            ).to(self.device)

        log.info(
            "TurboQuantKVCache.init",
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            residual=config.residual_correction,
            device=str(self.device),
        )

    def __enter__(self) -> TurboQuantKVCache:
        return self

    def __exit__(self, *_: Any) -> None:
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.empty_cache()

    def compress(self, keys: torch.Tensor, values: torch.Tensor) -> CacheEntry:
        """Compress key/value tensors into a cache entry."""
        with self._lock:
            try:
                entry = self._compress_impl(keys.to(self.device), values.to(self.device))
            except torch.cuda.OutOfMemoryError:
                log.warning("OOM during compress, retrying on CPU")
                self.device = torch.device("cpu")
                self.quantizer = self.quantizer.to(self.device)
                if self.corrector is not None:
                    self.corrector = self.corrector.to(self.device)
                entry = self._compress_impl(keys.cpu(), values.cpu())

            mem = self.memory_usage(entry)
            log.info("compress", compression_ratio=mem["compression_ratio"], savings_percent=mem["actual_savings_percent"])
            if mem["total_mb"] > self.config.cpu_offload_threshold_mb:
                entry = self._cpu_offload(entry)
            return entry

    def _compress_impl(self, keys: torch.Tensor, values: torch.Tensor) -> CacheEntry:
        k_packed, k_scales = self.quantizer(keys)
        v_packed, v_scales = self.quantizer(values)

        rk = rv = None
        rsk = rsv = None
        if self.corrector is not None:
            k_hat = self.quantizer.dequantize(k_packed, k_scales, tuple(keys.shape))
            v_hat = self.quantizer.dequantize(v_packed, v_scales, tuple(values.shape))

            r_bits_k, rsk = self.corrector.encode_with_scale((keys - k_hat).to(torch.float16))
            r_bits_v, rsv = self.corrector.encode_with_scale((values - v_hat).to(torch.float16))
            rk, rv = r_bits_k, r_bits_v

        meta = {
            "original_shape": list(keys.shape),
            "original_dtype": str(keys.dtype),
            "seq_len": int(keys.shape[2]) if keys.dim() >= 3 else int(keys.shape[0]),
            "device": str(keys.device),
        }

        return CacheEntry(
            compressed_keys=(k_packed, k_scales),
            compressed_values=(v_packed, v_scales),
            residual_keys=rk,
            residual_values=rv,
            residual_scales_k=rsk,
            residual_scales_v=rsv,
            metadata=meta,
        )

    def decompress(self, entry: CacheEntry) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress cache entry back to key/value tensors."""
        with self._lock:
            entry.access_count += 1
            shape = tuple(int(v) for v in entry.metadata["original_shape"])
            k = self.quantizer.dequantize(entry.compressed_keys[0], entry.compressed_keys[1], shape)
            v = self.quantizer.dequantize(entry.compressed_values[0], entry.compressed_values[1], shape)

            if self.corrector is not None and entry.residual_keys is not None and entry.residual_values is not None:
                k_corr = self.corrector.decode(
                    entry.residual_keys,
                    shape,
                    scale=float(entry.residual_scales_k.mean().item()) if entry.residual_scales_k is not None else 1.0,
                )
                v_corr = self.corrector.decode(
                    entry.residual_values,
                    shape,
                    scale=float(entry.residual_scales_v.mean().item()) if entry.residual_scales_v is not None else 1.0,
                )
                k = (k.float() + k_corr.float()).to(torch.float16)
                v = (v.float() + v_corr.float()).to(torch.float16)
            return k, v

    def update(self, entry: CacheEntry, new_keys: torch.Tensor, new_values: torch.Tensor) -> CacheEntry:
        """Append new tokens (sliding-window constrained by max_seq_len)."""
        with self._lock:
            keys, values = self.decompress(entry)
            keys = torch.cat([keys, new_keys.to(keys.device)], dim=2)
            values = torch.cat([values, new_values.to(values.device)], dim=2)

            if keys.shape[2] > self.config.max_seq_len:
                keys = keys[:, :, -self.config.max_seq_len :, :]
                values = values[:, :, -self.config.max_seq_len :, :]

            return self.compress(keys, values)

    def memory_usage(self, entry: CacheEntry) -> dict[str, float]:
        """Return detailed memory accounting."""

        def nbytes(t: torch.Tensor | None) -> int:
            return 0 if t is None else int(t.nelement() * t.element_size())

        kv_compressed_bytes = (
            nbytes(entry.compressed_keys[0])
            + nbytes(entry.compressed_keys[1])
            + nbytes(entry.compressed_values[0])
            + nbytes(entry.compressed_values[1])
        )
        residual_bytes = nbytes(entry.residual_keys) + nbytes(entry.residual_values)
        scales_bytes = nbytes(entry.residual_scales_k) + nbytes(entry.residual_scales_v)
        total_bytes = kv_compressed_bytes + residual_bytes + scales_bytes

        shape = entry.metadata.get("original_shape", [1, 1, 1, 1])
        fp16_baseline_bytes = int(np.prod(shape)) * 2 * 2
        ratio = total_bytes / max(fp16_baseline_bytes, 1)

        return {
            "kv_compressed_bytes": float(kv_compressed_bytes),
            "residual_bytes": float(residual_bytes),
            "scales_bytes": float(scales_bytes),
            "total_bytes": float(total_bytes),
            "total_mb": total_bytes / (1024**2),
            "fp16_baseline_mb": fp16_baseline_bytes / (1024**2),
            "compression_ratio": ratio,
            "actual_savings_percent": (1.0 - ratio) * 100.0,
            # Legacy aliases for older tests/callers.
            "bytes": float(total_bytes),
            "mb": total_bytes / (1024**2),
            "ratio": ratio,
        }

    def benchmark(
        self,
        seq_lengths: list[int] | None = None,
        batch_size: int = 1,
        num_heads: int | None = None,
    ) -> pd.DataFrame:
        """Benchmark compression/decompression latency and memory."""
        if seq_lengths is None:
            seq_lengths = [1024, 4096, 16384, 65536, 131072]
        heads = num_heads or self.config.num_heads

        rows: list[dict[str, float]] = []
        for sl in seq_lengths:
            xk = torch.randn(batch_size, heads, sl, self.config.head_dim, dtype=torch.float16, device=self.device)
            xv = torch.randn_like(xk)
            t0 = time.perf_counter()
            entry = self.compress(xk, xv)
            compress_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            dk, dv = self.decompress(entry)
            decompress_ms = (time.perf_counter() - t0) * 1000.0

            mse = float(((xk.float() - dk.float()) ** 2).mean().item() + ((xv.float() - dv.float()) ** 2).mean().item()) / 2.0
            mem = self.memory_usage(entry)

            rows.append(
                {
                    "seq_len": float(sl),
                    "compress_ms": compress_ms,
                    "decompress_ms": decompress_ms,
                    "memory_mb": mem["total_mb"],
                    "ratio": mem["compression_ratio"],
                    "mse": mse,
                }
            )
        return pd.DataFrame(rows)

    def export_onnx(self, path: str, example_seq_len: int = 2048) -> None:
        """Export small ONNX wrappers for compress/decompress paths."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        class CompressModule(torch.nn.Module):
            def __init__(self, cache: TurboQuantKVCache) -> None:
                super().__init__()
                self.cache = cache

            def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                k_p, k_s = self.cache.quantizer(k)
                v_p, v_s = self.cache.quantizer(v)
                return k_p, k_s, v_p, v_s

        class DecompressModule(torch.nn.Module):
            def __init__(self, cache: TurboQuantKVCache, shape: tuple[int, int, int, int]) -> None:
                super().__init__()
                self.cache = cache
                self.shape = shape

            def forward(self, k_p: torch.Tensor, k_s: torch.Tensor, v_p: torch.Tensor, v_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                dk = self.cache.quantizer.dequantize(k_p, k_s, self.shape)
                dv = self.cache.quantizer.dequantize(v_p, v_s, self.shape)
                return dk, dv

        shape = (1, self.config.num_heads, example_seq_len, self.config.head_dim)
        k = torch.randn(shape, dtype=torch.float16, device=self.device)
        v = torch.randn(shape, dtype=torch.float16, device=self.device)

        comp = CompressModule(self).to(self.device)
        torch.onnx.export(comp, (k, v), str(p.with_name(p.stem + "_compress.onnx")), opset_version=17)

        k_p, k_s = self.quantizer(k)
        v_p, v_s = self.quantizer(v)
        decomp = DecompressModule(self, shape).to(self.device)
        torch.onnx.export(decomp, (k_p, k_s, v_p, v_s), str(p.with_name(p.stem + "_decompress.onnx")), opset_version=17)

    def _cpu_offload(self, entry: CacheEntry) -> CacheEntry:
        """Move all cache tensors to CPU."""
        log.info("cpu_offload_triggered")
        return entry.to("cpu")
