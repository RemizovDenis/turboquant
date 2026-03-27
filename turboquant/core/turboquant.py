"""TurboQuantKVCache — unified KV-cache compression interface.

Combines PolarQuantizer (3-bit) and QJLResidualCorrector (1-bit) into a
single production-ready module for compressing LLM KV-caches.

Total cost: 3 bits (polar) + 1 bit (QJL) = 4 bits per element
→ **4× memory reduction** versus FP16.

Reference: TurboQuant (arXiv 2504.19874).

Typical usage::

    config = TurboQuantConfig(head_dim=128, num_heads=32)
    with TurboQuantKVCache(config) as tq:
        entry = tq.compress(keys, values)
        keys_hat, values_hat = tq.decompress(entry)
        print(tq.memory_usage(entry))
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn

from turboquant.core.polar_quant import PolarQuantizer
from turboquant.core.qjl import QJLResidualCorrector

log = structlog.get_logger(__name__)


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuantKVCache.

    Attributes:
        head_dim: Dimension of each attention head.
        num_heads: Number of attention heads.
        bits: Quantization bit-width for PolarQuantizer (default 3).
        group_size: Elements per quantization group (default 64).
        residual_correction: Enable QJL residual correction (default True).
        sketch_dim: JL projection dimension. ``None`` → ``head_dim // 4``.
        device: Target device string.
        dtype: Target tensor dtype.
        seed: Random seed for rotation / projection matrices.
        max_seq_len: Maximum sequence length budget.
    """

    head_dim: int = 128
    num_heads: int = 32
    bits: int = 3
    group_size: int = 64
    residual_correction: bool = True
    sketch_dim: int | None = None
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    seed: int = 42
    max_seq_len: int = 131072


@dataclass
class CacheEntry:
    """Container for a single compressed KV-cache snapshot.

    Attributes:
        compressed_keys: ``(quantized_int8, scales_float32)`` pair.
        compressed_values: ``(quantized_int8, scales_float32)`` pair.
        residual_keys: Packed 1-bit residual for keys (or *None*).
        residual_values: Packed 1-bit residual for values (or *None*).
        metadata: Auxiliary information (shapes, dtypes, seq_len, …).
    """

    compressed_keys: tuple[torch.Tensor, torch.Tensor]
    compressed_values: tuple[torch.Tensor, torch.Tensor]
    residual_keys: torch.Tensor | None = None
    residual_values: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: str | torch.device) -> CacheEntry:
        """Move all tensors to *device* and return self."""
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
        return self


# ======================================================================
# Main class
# ======================================================================


class TurboQuantKVCache(nn.Module):
    """Unified KV-cache compressor combining polar quantization and QJL residual correction.

    Thread-safe: the ``update`` method acquires an internal lock.
    Context-manager aware: ``__exit__`` releases GPU tensors.

    Attributes:
        config: Active ``TurboQuantConfig``.
        polar: ``PolarQuantizer`` instance.
        qjl: ``QJLResidualCorrector`` instance (or *None* when disabled).
    """

    def __init__(self, config: TurboQuantConfig) -> None:
        """Initialise TurboQuantKVCache.

        Args:
            config: Complete configuration dataclass.
        """
        super().__init__()
        self.config = config
        self._lock = threading.Lock()

        # Resolve device: fall back to CPU if CUDA is requested but unavailable
        effective_device = config.device
        if "cuda" in effective_device and not torch.cuda.is_available():
            log.info("CUDA not available, falling back to CPU")
            effective_device = "cpu"
        self._device = effective_device

        # Core quantizer
        self.polar = PolarQuantizer(
            head_dim=config.head_dim,
            bits=config.bits,
            group_size=config.group_size,
            seed=config.seed,
        ).to(effective_device)

        # Residual corrector
        self.qjl: QJLResidualCorrector | None = None
        if config.residual_correction:
            self.qjl = QJLResidualCorrector(
                head_dim=config.head_dim,
                sketch_dim=config.sketch_dim,
                seed=config.seed + 1,
            ).to(effective_device)

        log.info(
            "TurboQuantKVCache.__init__",
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            bits=config.bits,
            residual=config.residual_correction,
            device=effective_device,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> TurboQuantKVCache:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release GPU buffers."""
        self._clear_gpu()

    def _clear_gpu(self) -> None:
        """Delete GPU tensors and empty the CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.debug("_clear_gpu")

    # ------------------------------------------------------------------
    # Compress / Decompress
    # ------------------------------------------------------------------

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> CacheEntry:
        """Compress a KV-cache pair.

        Args:
            keys: Key tensor  ``[batch, num_heads, seq_len, head_dim]`` in FP16/BF16.
            values: Value tensor of the same shape.

        Returns:
            ``CacheEntry`` with compressed data and metadata.

        Raises:
            RuntimeError: On GPU OOM — automatically retries on CPU.
        """
        try:
            return self._compress_impl(keys, values)
        except torch.cuda.OutOfMemoryError:
            log.warning("OOM during compress — offloading to CPU")
            return self._compress_impl(keys.cpu(), values.cpu())

    def _compress_impl(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> CacheEntry:
        """Internal compression implementation."""
        device = keys.device
        original_dtype = keys.dtype

        # Polar quantize keys
        qk, sk = self.polar(keys)
        qv, sv = self.polar(values)

        rk: torch.Tensor | None = None
        rv: torch.Tensor | None = None

        if self.qjl is not None:
            # Compute residual from polar dequant
            keys_hat = self.polar.dequantize(qk, sk)
            values_hat = self.polar.dequantize(qv, sv)
            res_k = (keys.float() - keys_hat.float()).to(keys.dtype)
            res_v = (values.float() - values_hat.float()).to(values.dtype)
            rk = self.qjl.encode(res_k)
            rv = self.qjl.encode(res_v)

        meta: dict[str, Any] = {
            "original_shape": list(keys.shape),
            "original_dtype": str(original_dtype),
            "seq_len": keys.shape[2] if keys.dim() >= 3 else keys.shape[0],
            "device": str(device),
            "compressed_at": time.time(),
        }

        log.debug("compress", shape=list(keys.shape), device=str(device))

        return CacheEntry(
            compressed_keys=(qk, sk),
            compressed_values=(qv, sv),
            residual_keys=rk,
            residual_values=rv,
            metadata=meta,
        )

    def decompress(
        self,
        entry: CacheEntry,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress a ``CacheEntry`` back to FP16 key/value tensors.

        Args:
            entry: Previously compressed cache entry.

        Returns:
            Tuple ``(keys, values)`` each as FP16 tensors with the original shape.
        """
        qk, sk = entry.compressed_keys
        qv, sv = entry.compressed_values

        keys_hat = self.polar.dequantize(qk, sk)
        values_hat = self.polar.dequantize(qv, sv)

        if self.qjl is not None and entry.residual_keys is not None:
            original_shape = tuple(entry.metadata.get("original_shape", keys_hat.shape))
            res_k = self.qjl.decode(entry.residual_keys, original_shape)
            res_v = self.qjl.decode(entry.residual_values, original_shape)  # type: ignore[arg-type]
            keys_hat = (keys_hat.float() + res_k.float()).to(torch.float16)
            values_hat = (values_hat.float() + res_v.float()).to(torch.float16)

        log.debug("decompress", shape=list(keys_hat.shape))
        return keys_hat, values_hat

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def update(
        self,
        entry: CacheEntry,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> CacheEntry:
        """Append new key/value tokens to an existing compressed cache.

        This is thread-safe: an internal lock serialises concurrent updates.

        Args:
            entry: Existing ``CacheEntry``.
            new_keys: New key tokens ``[batch, num_heads, new_seq, head_dim]``.
            new_values: Corresponding value tokens.

        Returns:
            Updated ``CacheEntry`` with the appended data.
        """
        with self._lock:
            return self._update_impl(entry, new_keys, new_values)

    def _update_impl(
        self,
        entry: CacheEntry,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> CacheEntry:
        """Non-locked update implementation.

        Strategy: compress the new slice independently and concatenate the
        compressed representations along the sequence dimension.
        """
        # Compress the incremental slice
        new_entry = self._compress_impl(new_keys, new_values)

        # Concatenate quantized tensors along seq dimension (dim=2)
        seq_dim = 2

        merged_qk = torch.cat(
            [entry.compressed_keys[0], new_entry.compressed_keys[0]], dim=seq_dim
        )
        merged_sk = torch.cat(
            [entry.compressed_keys[1], new_entry.compressed_keys[1]], dim=seq_dim
        )
        merged_qv = torch.cat(
            [entry.compressed_values[0], new_entry.compressed_values[0]], dim=seq_dim
        )
        merged_sv = torch.cat(
            [entry.compressed_values[1], new_entry.compressed_values[1]], dim=seq_dim
        )

        merged_rk: torch.Tensor | None = None
        merged_rv: torch.Tensor | None = None
        if (
            entry.residual_keys is not None
            and new_entry.residual_keys is not None
            and entry.residual_values is not None
            and new_entry.residual_values is not None
        ):
            merged_rk = torch.cat([entry.residual_keys, new_entry.residual_keys], dim=seq_dim)
            merged_rv = torch.cat([entry.residual_values, new_entry.residual_values], dim=seq_dim)

        # Update metadata
        old_shape = entry.metadata["original_shape"]
        new_seq = old_shape[2] + new_keys.shape[2]
        merged_shape = list(old_shape)
        merged_shape[2] = new_seq

        meta = dict(entry.metadata)
        meta["original_shape"] = merged_shape
        meta["seq_len"] = new_seq
        meta["updated_at"] = time.time()

        log.debug("update", old_seq=old_shape[2], new_seq=new_seq)

        return CacheEntry(
            compressed_keys=(merged_qk, merged_sk),
            compressed_values=(merged_qv, merged_sv),
            residual_keys=merged_rk,
            residual_values=merged_rv,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Memory reporting
    # ------------------------------------------------------------------

    def memory_usage(self, entry: CacheEntry) -> dict[str, float]:
        """Report memory consumption of a ``CacheEntry``.

        Args:
            entry: Compressed cache entry.

        Returns:
            Dictionary with keys ``bytes``, ``mb``, ``fp16_bytes``,
            ``fp16_mb``, ``ratio``, ``savings_percent``.
        """
        def _tensor_bytes(t: torch.Tensor | None) -> int:
            return t.nelement() * t.element_size() if t is not None else 0

        compressed_bytes = (
            _tensor_bytes(entry.compressed_keys[0])
            + _tensor_bytes(entry.compressed_keys[1])
            + _tensor_bytes(entry.compressed_values[0])
            + _tensor_bytes(entry.compressed_values[1])
            + _tensor_bytes(entry.residual_keys)
            + _tensor_bytes(entry.residual_values)
        )

        # FP16 baseline
        shape = entry.metadata.get("original_shape", [1, 1, 1, 1])
        fp16_bytes = int(np.prod(shape)) * 2 * 2  # keys + values, 2 bytes each

        ratio = compressed_bytes / max(fp16_bytes, 1)

        return {
            "bytes": float(compressed_bytes),
            "mb": compressed_bytes / (1024 * 1024),
            "fp16_bytes": float(fp16_bytes),
            "fp16_mb": fp16_bytes / (1024 * 1024),
            "ratio": ratio,
            "savings_percent": (1.0 - ratio) * 100.0,
        }

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def benchmark(
        self,
        seq_lengths: list[int] | None = None,
        batch_size: int = 1,
        warmup: int = 3,
        iterations: int = 10,
    ) -> pd.DataFrame:
        """Run an automated compression/decompression benchmark.

        Args:
            seq_lengths: List of sequence lengths to test.
                Defaults to ``[1024, 4096, 16384, 65536]``.
            batch_size: Batch size for synthetic data.
            warmup: Number of warm-up iterations (not timed).
            iterations: Number of timed iterations for averaging.

        Returns:
            ``pd.DataFrame`` with columns: ``seq_len``, ``compress_ms``,
            ``decompress_ms``, ``memory_mb``, ``ratio``, ``savings_pct``.
        """
        if seq_lengths is None:
            seq_lengths = [1024, 4096, 16384, 65536]

        device = self._device
        rows: list[dict[str, Any]] = []

        for seq_len in seq_lengths:
            # Cap at max_seq_len
            sl = min(seq_len, self.config.max_seq_len)
            shape = (batch_size, self.config.num_heads, sl, self.config.head_dim)

            try:
                k = torch.randn(shape, dtype=self.config.dtype, device=device)
                v = torch.randn(shape, dtype=self.config.dtype, device=device)
            except torch.cuda.OutOfMemoryError:
                log.warning("benchmark: OOM at seq_len=%d, skipping", sl)
                continue

            # Warmup
            for _ in range(warmup):
                e = self.compress(k, v)
                _ = self.decompress(e)

            # Timed compress
            if "cuda" in device:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iterations):
                e = self.compress(k, v)
            if "cuda" in device:
                torch.cuda.synchronize()
            compress_ms = (time.perf_counter() - t0) / iterations * 1000

            # Timed decompress
            if "cuda" in device:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iterations):
                _ = self.decompress(e)
            if "cuda" in device:
                torch.cuda.synchronize()
            decompress_ms = (time.perf_counter() - t0) / iterations * 1000

            mem = self.memory_usage(e)

            rows.append({
                "seq_len": sl,
                "compress_ms": round(compress_ms, 2),
                "decompress_ms": round(decompress_ms, 2),
                "memory_mb": round(mem["mb"], 2),
                "fp16_mb": round(mem["fp16_mb"], 2),
                "ratio": round(mem["ratio"], 4),
                "savings_pct": round(mem["savings_percent"], 1),
            })

            log.info(
                "benchmark",
                seq_len=sl,
                compress_ms=round(compress_ms, 2),
                decompress_ms=round(decompress_ms, 2),
                savings_pct=round(mem["savings_percent"], 1),
            )

            # Free
            del k, v, e

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def export_onnx(self, path: str) -> None:
        """Export the polar quantizer to ONNX format.

        Args:
            path: Destination ``.onnx`` file path.

        Note:
            Exports only the polar quantizer forward pass. QJL residual
            correction is not included because ONNX does not support
            bit-packing natively.
        """
        from pathlib import Path as _Path

        _Path(path).parent.mkdir(parents=True, exist_ok=True)

        dummy = torch.randn(
            1, self.config.num_heads, 16, self.config.head_dim,
            dtype=torch.float32,
            device="cpu",
        )
        polar_cpu = self.polar.cpu()

        torch.onnx.export(
            polar_cpu,
            (dummy,),
            path,
            input_names=["kv_cache"],
            output_names=["quantized", "scales"],
            dynamic_axes={
                "kv_cache": {0: "batch", 2: "seq_len"},
                "quantized": {0: "batch", 2: "seq_len"},
                "scales": {0: "batch", 2: "seq_len"},
            },
            opset_version=17,
        )
        # Move back to original device
        self.polar.to(self._device)
        log.info("export_onnx", path=path)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        bits_total = self.config.bits + (1 if self.config.residual_correction else 0)
        return (
            f"head_dim={self.config.head_dim}, num_heads={self.config.num_heads}, "
            f"bits_total={bits_total}, device={self._device}"
        )
