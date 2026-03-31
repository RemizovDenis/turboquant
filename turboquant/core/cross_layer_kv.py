"""Cross-layer KV sharing with delta compression."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import structlog
import torch
import torch.nn.functional as functional

from turboquant.core.polar_quant import PolarQuantConfig, PolarQuantizer
from turboquant.core.turboquant import CacheEntry, TurboQuantKVCache

LOGGER = structlog.get_logger(__name__)


@dataclass
class CrossLayerConfig:
    """Configuration for cross-layer KV sharing."""

    num_layers: int
    anchor_stride: int = 4
    delta_bits: int = 2
    delta_group_size: int = 32
    similarity_threshold: float = 0.80
    adaptive_anchors: bool = True
    anchor_update_freq: int = 100
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


@dataclass
class LayerSimilarityStats:
    """Similarity and compression diagnostics for a layer pair."""

    layer_pair: tuple[int, int]
    cosine_similarity: float
    delta_norm_ratio: float
    is_anchor: bool
    compression_gain: float


@dataclass
class CrossLayerCacheEntry:
    """Cross-layer compressed cache record."""

    anchor_layer_id: int
    layer_id: int
    is_anchor: bool
    anchor_entry: CacheEntry | None
    delta_packed: torch.Tensor | None
    delta_scales: torch.Tensor | None
    delta_norm_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)


class CrossLayerKVCache:
    """Store full KV only for anchors and deltas for non-anchor layers."""

    def __init__(self, config: CrossLayerConfig, base_quantizer: TurboQuantKVCache) -> None:
        """Initialize cross-layer cache wrapper.

        Args:
            config: Cross-layer sharing configuration.
            base_quantizer: Base TurboQuant KV cache used for anchors.
        """
        self.config = config
        self.base_quantizer = base_quantizer
        self.anchor_layers = sorted(
            {i for i in range(0, config.num_layers, config.anchor_stride)} | {0}
        )
        self.similarity_matrix = torch.zeros(
            (config.num_layers, config.num_layers),
            dtype=torch.float32,
        )
        self.anchor_assignments = self._build_initial_assignments()

        delta_cfg = PolarQuantConfig(
            head_dim=base_quantizer.config.head_dim,
            bits=config.delta_bits,
            group_size=config.delta_group_size,
            seed=base_quantizer.config.seed + 17,
            use_hadamard=base_quantizer.config.use_hadamard,
        )
        self.delta_quantizer = PolarQuantizer(delta_cfg).to(base_quantizer.device)
        self.entries: dict[int, CrossLayerCacheEntry] = {}
        self.anchor_entries: dict[int, CacheEntry] = {}
        self._lock = threading.RLock()
        self._step_count = 0
        self._logger = LOGGER.bind(component="CrossLayerKVCache")

    def _build_initial_assignments(self) -> dict[int, int]:
        assignments: dict[int, int] = {}
        for layer in range(self.config.num_layers):
            anchor = max([a for a in self.anchor_layers if a <= layer], default=0)
            assignments[layer] = anchor
        return assignments

    def compress(
        self,
        layer_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        token_ids: torch.Tensor | None = None,
        attention_entropy: torch.Tensor | None = None,
    ) -> CrossLayerCacheEntry:
        """Compress KV for a layer as anchor or delta.

        Args:
            layer_id: Layer index.
            keys: Key tensor `[batch, heads, seq, dim]`.
            values: Value tensor `[batch, heads, seq, dim]`.
            token_ids: Optional token IDs for adaptive classification.
            attention_entropy: Optional attention entropy for adaptive bitwidth.

        Returns:
            :class:`CrossLayerCacheEntry` for the layer.
        """
        with self._lock:
            anchor_layer = self.anchor_assignments.get(layer_id, 0)
            original_shape = list(keys.shape)

            if layer_id == anchor_layer or anchor_layer not in self.anchor_entries:
                anchor_entry = self.base_quantizer.compress(
                    keys, values, token_ids=token_ids, attention_entropy=attention_entropy
                )
                metadata: dict[str, Any] = {
                    "original_shape": original_shape,
                    "fallback_anchor": False,
                }
                out = CrossLayerCacheEntry(
                    anchor_layer_id=layer_id,
                    layer_id=layer_id,
                    is_anchor=True,
                    anchor_entry=anchor_entry,
                    delta_packed=None,
                    delta_scales=None,
                    delta_norm_ratio=0.0,
                    metadata=metadata,
                )
                self.entries[layer_id] = out
                self.anchor_entries[layer_id] = anchor_entry
                return out

            base_anchor = self.anchor_entries[anchor_layer]
            anchor_k, anchor_v = self.base_quantizer.decompress(base_anchor)
            stats = self.measure_similarity(anchor_layer, layer_id, anchor_k, keys)
            if stats.cosine_similarity < float(self.config.similarity_threshold):
                self._logger.warning(
                    "cross_layer_similarity_low_fallback_anchor",
                    layer_id=layer_id,
                    anchor_layer=anchor_layer,
                    similarity=stats.cosine_similarity,
                    threshold=self.config.similarity_threshold,
                )
                anchor_entry = self.base_quantizer.compress(keys, values)
                out = CrossLayerCacheEntry(
                    anchor_layer_id=layer_id,
                    layer_id=layer_id,
                    is_anchor=True,
                    anchor_entry=anchor_entry,
                    delta_packed=None,
                    delta_scales=None,
                    delta_norm_ratio=0.0,
                    metadata={"original_shape": original_shape, "fallback_anchor": True},
                )
                self.entries[layer_id] = out
                self.anchor_entries[layer_id] = anchor_entry
                self.anchor_assignments[layer_id] = layer_id
                return out

            delta_k = (keys - anchor_k).to(dtype=torch.float16)
            delta_v = (values - anchor_v).to(dtype=torch.float16)
            packed_k, scales_k = self.delta_quantizer(delta_k)
            packed_v, scales_v = self.delta_quantizer(delta_v)
            delta_ratio = float(
                delta_k.float().norm().item() / max(1e-8, keys.float().norm().item())
            )
            entry = CrossLayerCacheEntry(
                anchor_layer_id=anchor_layer,
                layer_id=layer_id,
                is_anchor=False,
                anchor_entry=None,
                delta_packed=torch.stack([packed_k, packed_v], dim=0),
                delta_scales=torch.stack([scales_k, scales_v], dim=0),
                delta_norm_ratio=delta_ratio,
                metadata={"original_shape": original_shape},
            )
            self.entries[layer_id] = entry
            self._step_count += 1

            if self._step_count % 50 == 0:
                ratio = self.memory_report().get("total_compression_ratio", 0.0)
                self._logger.info(
                    "cross_layer_compression",
                    layer_id=layer_id,
                    anchor_layer=anchor_layer,
                    delta_norm_ratio=delta_ratio,
                    total_compression_ratio=ratio,
                )
            if (
                self.config.adaptive_anchors
                and self._step_count % max(1, int(self.config.anchor_update_freq)) == 0
            ):
                self.adapt_anchors()
            return entry

    def decompress(self, entry: CrossLayerCacheEntry) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress cross-layer cache entry."""
        with self._lock:
            if entry.is_anchor:
                if entry.anchor_entry is None:
                    raise ValueError("anchor entry is missing for anchor layer")
                return self.base_quantizer.decompress(entry.anchor_entry)

            if entry.delta_packed is None or entry.delta_scales is None:
                raise ValueError("delta tensors are missing for non-anchor entry")

            anchor_entry = self.anchor_entries.get(entry.anchor_layer_id)
            if anchor_entry is None:
                raise KeyError(f"Missing anchor entry for layer {entry.anchor_layer_id}")

            anchor_k, anchor_v = self.base_quantizer.decompress(anchor_entry)
            delta_k = self.delta_quantizer.dequantize(entry.delta_packed[0], entry.delta_scales[0])
            delta_v = self.delta_quantizer.dequantize(entry.delta_packed[1], entry.delta_scales[1])
            return (
                (anchor_k.float() + delta_k.float()).to(torch.float16),
                (anchor_v.float() + delta_v.float()).to(torch.float16),
            )

    def measure_similarity(
        self,
        layer_a: int,
        layer_b: int,
        keys_a: torch.Tensor,
        keys_b: torch.Tensor,
    ) -> LayerSimilarityStats:
        """Measure cosine similarity and delta norm ratio between layers."""
        flat_a = keys_a.float().reshape(-1, keys_a.shape[-1])
        flat_b = keys_b.float().reshape(-1, keys_b.shape[-1])
        n = min(flat_a.shape[0], flat_b.shape[0])
        if n == 0:
            cos = 0.0
            ratio = 1.0
        else:
            cos = float(functional.cosine_similarity(flat_a[:n], flat_b[:n], dim=-1).mean().item())
            ratio = float(
                (flat_b[:n] - flat_a[:n]).norm().item() / max(1e-8, flat_b[:n].norm().item())
            )

        gain = float(1.0 / max(1e-8, ratio))
        self.similarity_matrix[layer_a, layer_b] = cos
        self.similarity_matrix[layer_b, layer_a] = cos
        is_anchor = self.anchor_assignments.get(layer_b, 0) == layer_b
        return LayerSimilarityStats(
            layer_pair=(layer_a, layer_b),
            cosine_similarity=cos,
            delta_norm_ratio=ratio,
            is_anchor=is_anchor,
            compression_gain=gain,
        )

    def adapt_anchors(self) -> dict[int, int]:
        """Reassign anchors via greedy clustering on similarity matrix."""
        with self._lock:
            remaining = set(range(self.config.num_layers))
            assignments: dict[int, int] = {}
            threshold = float(self.config.similarity_threshold)

            while remaining:
                anchor = min(remaining)
                group = [
                    layer
                    for layer in sorted(remaining)
                    if self.similarity_matrix[anchor, layer] >= threshold
                ]
                if anchor not in group:
                    group.insert(0, anchor)
                for layer in group:
                    assignments[layer] = anchor
                    remaining.discard(layer)

            for layer in range(self.config.num_layers):
                assignments.setdefault(
                    layer, max([a for a in assignments.values() if a <= layer], default=0)
                )

            self.anchor_assignments = assignments
            return dict(assignments)

    def memory_report(self) -> dict[str, float]:
        """Return memory usage report with compression ratio."""
        with self._lock:
            total_mb = 0.0
            baseline_mb = 0.0
            report: dict[str, float] = {}

            for layer_id, entry in self.entries.items():
                shape = entry.metadata.get("original_shape", [1, 1, 1, 1])
                layer_baseline = float(torch.tensor(shape).prod().item() * 4 / (1024**2))
                baseline_mb += layer_baseline

                if entry.is_anchor and entry.anchor_entry is not None:
                    mem = self.base_quantizer.memory_usage(entry.anchor_entry)
                    layer_bytes = float(mem["total_bytes"])
                    report[f"layer_{layer_id}_anchor_mb"] = layer_bytes / (1024**2)
                else:
                    layer_bytes = 0.0
                    if entry.delta_packed is not None:
                        layer_bytes += float(
                            entry.delta_packed.numel() * entry.delta_packed.element_size()
                        )
                    if entry.delta_scales is not None:
                        layer_bytes += float(
                            entry.delta_scales.numel() * entry.delta_scales.element_size()
                        )
                    report[f"layer_{layer_id}_delta_mb"] = layer_bytes / (1024**2)

                layer_mb = layer_bytes / (1024**2)
                report[f"layer_{layer_id}_total_mb"] = layer_mb
                report[f"layer_{layer_id}_vs_baseline_mb"] = layer_baseline
                total_mb += layer_mb

            report["total_mb"] = total_mb
            report["baseline_mb"] = baseline_mb
            report["total_compression_ratio"] = baseline_mb / max(1e-8, total_mb)
            return report

    def warmup(self, sample_keys: list[torch.Tensor], sample_values: list[torch.Tensor]) -> None:
        """Warm similarity matrix and anchor assignments from sample batches.

        Args:
            sample_keys: List of sample key tensors for each layer.
            sample_values: List of sample value tensors for each layer.
        """
        if len(sample_keys) < 2:
            return
        with self._lock:
            max_layer = min(self.config.num_layers, len(sample_keys))
            # Similarity is primarily driven by keys in v0.3.0 as it determines
            # the attention structure, but we process both for completeness.
            for layer in range(1, max_layer):
                prev = sample_keys[layer - 1]
                cur = sample_keys[layer]
                # measure_similarity currently uses keys only, which is standard for CL-KV
                self.measure_similarity(layer - 1, layer, prev, cur)

            if self.config.adaptive_anchors:
                self.adapt_anchors()
            self._logger.info("cross_layer_warmup_complete", layers=max_layer)
