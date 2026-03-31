"""Core TurboQuant KV-cache manager v0.3.0.

Unified KV-cache compressor with true packed 3-bit storage,
Hadamard fast path, and async CUDA processing.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import structlog
import torch
import torch.nn.functional as F  # noqa: N812

from turboquant.core.adaptive_bitwidth import (
    AdaptiveBitwidthQuantizer,
    AdaptiveCompressedCache,
)
from turboquant.core.polar_quant import PolarQuantizer
from turboquant.core.qjl import QJLResidualCorrector

log = structlog.get_logger(__name__)

__all__ = [
    "TurboQuantConfig",
    "CacheEntry",
    "TurboQuantKVCache",
    "AdaptiveCompressedCache",
]

# ======================================================================
# Configuration
# ======================================================================


@dataclass
class TurboQuantConfig:
    """Enhanced configuration for TurboQuantKVCache v0.3.0."""

    head_dim: int = 128
    num_heads: int = 32
    bits: int = 3
    group_size: int = 64
    residual_correction: bool = True
    sketch_dim: int | None = None
    use_triton_kernels: bool = True
    use_hadamard: bool = True
    norm_preserving_qjl: bool = True
    benchmark_on_init: bool = False
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    seed: int = 42
    max_seq_len: int = 131072
    enable_semantic_eviction: bool = False
    enable_cross_layer_sharing: bool = False
    enable_adaptive_bitwidth: bool = False
    semantic_eviction_config: Any | None = None
    cross_layer_config: Any | None = None
    adaptive_bitwidth_config: Any | None = None


@dataclass
class CacheEntry:
    """Compressed KV cache entry with true packed storage."""

    compressed_keys: tuple[torch.Tensor, torch.Tensor]  # (packed, scales)
    compressed_values: tuple[torch.Tensor, torch.Tensor]  # (packed, scales)
    residual_keys: torch.Tensor | None = None
    residual_values: torch.Tensor | None = None
    residual_norms_k: torch.Tensor | None = None
    residual_norms_v: torch.Tensor | None = None
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
        if self.residual_norms_k is not None:
            self.residual_norms_k = self.residual_norms_k.to(device)
        if self.residual_norms_v is not None:
            self.residual_norms_v = self.residual_norms_v.to(device)
        return self


# ======================================================================
# Main Cache Engine
# ======================================================================


class TurboQuantKVCache:
    """Master class for TurboQuant KV cache compression and management."""

    def __init__(self, config: TurboQuantConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._latest_report: dict[str, float] = {}

        # Determine device
        if "cuda" in config.device and not torch.cuda.is_available():
            log.warning("cuda requested but unavailable, using cpu")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        # Core components
        self.polar = PolarQuantizer(
            head_dim=config.head_dim,
            bits=config.bits,
            group_size=config.group_size,
            seed=config.seed,
            use_hadamard=config.use_hadamard,
        ).to(self.device)

        self.qjl = None
        if config.residual_correction:
            self.qjl = QJLResidualCorrector(
                head_dim=config.head_dim, sketch_dim=config.sketch_dim, seed=config.seed + 1
            ).to(self.device)

        self.adaptive = None
        if config.enable_adaptive_bitwidth and config.adaptive_bitwidth_config is not None:
            self.adaptive = AdaptiveBitwidthQuantizer(config.adaptive_bitwidth_config)

        log.info("TurboQuantKVCache.init", config=config)

        # Optional benchmark on init
        if self.config.benchmark_on_init and "cuda" in str(self.device):
            from turboquant.kernels.triton_quant import benchmark_triton_kernels

            bench = benchmark_triton_kernels(head_dim=self.config.head_dim)
            log.info("kernel_benchmark", results=bench)

    def __enter__(self) -> TurboQuantKVCache:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""

    @property
    def quantizer(self) -> PolarQuantizer:
        """Legacy compatibility alias for polar quantizer."""
        return self.polar

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        token_ids: torch.Tensor | None = None,
        attention_entropy: torch.Tensor | None = None,
        layer_id: int | None = None,
    ) -> CacheEntry | AdaptiveCompressedCache:
        """Compress keys/values with true packed 3-bit, optionally adaptive."""
        if self.config.enable_adaptive_bitwidth and self.adaptive is not None:
            return self.adaptive.compress(keys, values, token_ids, attention_entropy)

        with self._lock:
            # layer_id is currently unused in standard mode but accepted for MoE synergy
            _ = layer_id
            k, v = keys.to(self.device), values.to(self.device)
            bs, heads, seq, dim = k.shape

            # 1. 3-bit polar quantization
            k_packed, k_scales = self.polar.forward(k)
            v_packed, v_scales = self.polar.forward(v)

            # 2. Residual correction
            rk, rv = None, None
            rnk, rnv = None, None
            if self.qjl is not None:
                k_hat = self.polar.dequantize(k_packed, k_scales)
                v_hat = self.polar.dequantize(v_packed, v_scales)

                rk, rnk = self.qjl.encode((k - k_hat).to(torch.float16))
                rv, rnv = self.qjl.encode((v - v_hat).to(torch.float16))

            entry = CacheEntry(
                compressed_keys=(k_packed, k_scales),
                compressed_values=(v_packed, v_scales),
                residual_keys=rk,
                residual_values=rv,
                residual_norms_k=rnk,
                residual_norms_v=rnv,
                metadata={
                    "original_shape": list(keys.shape),
                    "original_dtype": str(keys.dtype),
                    "seq_len": seq,
                    "device": str(self.device),
                },
            )
            self._latest_report = self.memory_usage(entry)
            return entry

    def update(
        self,
        entry: CacheEntry | AdaptiveCompressedCache,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> CacheEntry | AdaptiveCompressedCache:
        """Update existing CacheEntry by appending new keys/values."""
        with self._lock:
            if isinstance(entry, AdaptiveCompressedCache):
                # For adaptive, we must re-evaluate assignments for the whole sequence
                # to maintain the global bit budget.
                k_old, v_old = self.decompress(entry)
                k_new = torch.cat([k_old, new_keys.to(self.device)], dim=2)
                v_new = torch.cat([v_old, new_values.to(self.device)], dim=2)
                return self.compress(k_new, v_new)

            # For standard PolarQuant, we can compress the NEW part and concatenate
            # assuming the previous sequence length was a multiple of group_size
            # to avoid grain misalignment.
            seq_old = entry.metadata.get("seq_len", 0)
            if seq_old % self.config.group_size == 0:
                new_entry = self.compress(new_keys, new_values)
                if isinstance(new_entry, AdaptiveCompressedCache):
                    # Should not happen given the check above, but for type safety:
                    return new_entry

                # Merge packed tensors
                entry.compressed_keys = (
                    torch.cat([entry.compressed_keys[0], new_entry.compressed_keys[0]], dim=-1),
                    torch.cat([entry.compressed_keys[1], new_entry.compressed_keys[1]], dim=-1),
                )
                entry.compressed_values = (
                    torch.cat([entry.compressed_values[0], new_entry.compressed_values[0]], dim=-1),
                    torch.cat([entry.compressed_values[1], new_entry.compressed_values[1]], dim=-1),
                )
                r_keys, nr_keys = entry.residual_keys, new_entry.residual_keys
                r_vals, nr_vals = entry.residual_values, new_entry.residual_values
                r_n_k, nr_n_k = entry.residual_norms_k, new_entry.residual_norms_k
                r_n_v, nr_n_v = entry.residual_norms_v, new_entry.residual_norms_v

                if (
                    r_keys is not None
                    and nr_keys is not None
                    and r_vals is not None
                    and nr_vals is not None
                    and r_n_k is not None
                    and nr_n_k is not None
                    and r_n_v is not None
                    and nr_n_v is not None
                ):
                    entry.residual_keys = torch.cat([r_keys, nr_keys], dim=2)
                    entry.residual_values = torch.cat([r_vals, nr_vals], dim=2)
                    entry.residual_norms_k = torch.cat([r_n_k, nr_n_k], dim=2)
                    entry.residual_norms_v = torch.cat([r_n_v, nr_n_v], dim=2)

                entry.metadata["seq_len"] = seq_old + new_keys.shape[2]
                return entry

            # Fallback for misaligned sequence lengths
            k_old, v_old = self.decompress(entry)
            k_new = torch.cat([k_old, new_keys.to(self.device)], dim=2)
            v_new = torch.cat([v_old, new_values.to(self.device)], dim=2)
            return self.compress(k_new, v_new)

    def compress_async(
        self, keys: torch.Tensor, values: torch.Tensor, stream: torch.cuda.Stream | None = None
    ) -> CacheEntry | AdaptiveCompressedCache:
        """Non-blocking compression using CUDA stream."""
        if torch.cuda.is_available() and stream is None:
            stream = torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]

        with torch.cuda.stream(stream) if stream else contextlib.nullcontext():
            return self.compress(keys, values)

    def decompress(
        self, entry: CacheEntry | AdaptiveCompressedCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress back to float16."""
        from turboquant.core.adaptive_bitwidth import AdaptiveCompressedCache

        if isinstance(entry, AdaptiveCompressedCache) and self.adaptive is not None:
            return self.adaptive.decompress(entry, entry.original_shape)

        with self._lock:
            # Type narrowing for the rest of the method
            if isinstance(entry, AdaptiveCompressedCache):
                # Should have been handled above if self.adaptive is not None
                # If we are here, something is wrong, but we return to avoid crash
                return self.decompress(entry) if self.adaptive else (torch.empty(0), torch.empty(0))

            entry.access_count += 1

            # 1. Base 3-bit polar dequant
            k = self.polar.dequantize(entry.compressed_keys[0], entry.compressed_keys[1])
            v = self.polar.dequantize(entry.compressed_values[0], entry.compressed_values[1])

            # 2. Residual correction
            if (
                self.qjl is not None
                and entry.residual_keys is not None
                and entry.residual_values is not None
            ):
                shape = k.shape
                k_err = self.qjl.decode(
                    entry.residual_keys, entry.residual_norms_k, original_shape=shape
                )
                v_err = self.qjl.decode(
                    entry.residual_values, entry.residual_norms_v, original_shape=shape
                )

                k = (k.float() + k_err.float()).to(torch.float16)
                v = (v.float() + v_err.float()).to(torch.float16)

            return k, v

    def memory_usage(self, entry: CacheEntry | AdaptiveCompressedCache) -> dict[str, float]:
        """Memory report considering true packed storage."""
        from turboquant.core.adaptive_bitwidth import AdaptiveCompressedCache

        if isinstance(entry, AdaptiveCompressedCache):
            # For adaptive, we use the pre-calculated assignment metrics
            # bytes = (tokens * heads * dim * bits) / 8
            num_elements = (
                entry.assignment.token_indices.numel()
                * self.config.num_heads
                * self.config.head_dim
            )
            # Multiply by 2 because entry contains both Key and Value components
            total_bytes = float(entry.assignment.avg_bits * num_elements * 2 / 8)
            return {
                "total_bytes": total_bytes,
                "avg_bits": entry.assignment.avg_bits,
                "compression_ratio": entry.assignment.estimated_compression_ratio,
            }

        def nbytes(t: torch.Tensor | None) -> int:
            return t.nbytes if t is not None else 0

        packed_bytes = nbytes(entry.compressed_keys[0]) + nbytes(entry.compressed_values[0])
        scales_bytes = nbytes(entry.compressed_keys[1]) + nbytes(entry.compressed_values[1])
        residual_bytes = nbytes(entry.residual_keys) + nbytes(entry.residual_values)
        residual_norms_bytes = nbytes(entry.residual_norms_k) + nbytes(entry.residual_norms_v)

        total = packed_bytes + scales_bytes + residual_bytes + residual_norms_bytes

        # FP16 baseline
        shape = entry.metadata["original_shape"]
        fp16_baseline = math.prod(shape) * 2 * 2  # K + V, 2 bytes/element

        ratio = total / max(fp16_baseline, 1)

        return {
            "total_mb": total / (1024**2),
            "total_bytes": total,
            "bytes": total,  # Legacy support
            "ratio": ratio,
            "savings_percent": (1 - ratio) * 100,
            "packed_mb": packed_bytes / (1024**2),
            "scales_mb": scales_bytes / (1024**2),
            "residual_mb": residual_bytes / (1024**2),
        }

    def latest_memory_usage(self) -> dict[str, float]:
        """Return memory metrics for the most recently registered cache entry."""
        return self._latest_report or {"total_mb": 0.0, "status": 0.0}

    def quality_metrics(
        self, original_keys: torch.Tensor, original_values: torch.Tensor
    ) -> dict[str, float]:
        """Measure reconstruction quality."""
        entry = self.compress(original_keys, original_values)
        k_hat, v_hat = self.decompress(entry)

        k_orig = original_keys.float()
        v_orig = original_values.float()
        k_hat_f = k_hat.float()
        v_hat_f = v_hat.float()

        k_mse = float(((k_orig - k_hat_f) ** 2).mean())
        v_mse = float(((v_orig - v_hat_f) ** 2).mean())

        # Flatten for cosine similarity
        k_cos = float(
            F.cosine_similarity(
                k_orig.reshape(-1, self.config.head_dim), k_hat_f.reshape(-1, self.config.head_dim)
            ).mean()
        )

        v_cos = float(
            F.cosine_similarity(
                v_orig.reshape(-1, self.config.head_dim), v_hat_f.reshape(-1, self.config.head_dim)
            ).mean()
        )

        return {
            "keys_mse": k_mse,
            "values_mse": v_mse,
            "keys_cosine_sim": k_cos,
            "values_cosine_sim": v_cos,
            "compression_ratio_actual": float(1.0 / self.memory_usage(entry)["ratio"]),
        }
