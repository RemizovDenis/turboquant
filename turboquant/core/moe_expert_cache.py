"""Dynamic expert cache for MoE inference.

This module keeps active experts on GPU and stores inactive experts on CPU,
optionally compressed, to reduce VRAM pressure for large MoE models.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import structlog
import torch

from turboquant.core.polar_quant import PolarQuantizer

try:
    from safetensors.torch import load_file, save_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

LOGGER = structlog.get_logger(__name__)
MB = 1024.0 * 1024.0


@dataclass
class ExpertCacheConfig:
    """Configuration for dynamic MoE expert cache."""

    num_experts: int
    top_k_experts: int
    num_layers: int
    gpu_cache_size: int
    compress_cpu_experts: bool = True
    prefetch_depth: int = 3
    prefetch_threshold: float = 0.5
    eviction_policy: str = "arc"
    pin_memory: bool = True
    transfer_streams: int = 2
    device: str = "cuda"


@dataclass
class ExpertEntry:
    """Single expert record in cache."""

    expert_id: int
    layer_id: int
    weights: dict[str, torch.Tensor]
    is_on_gpu: bool
    last_access: float
    access_count: int
    compressed: bool
    gpu_memory_mb: float
    cpu_memory_mb: float


@dataclass
class ExpertCacheStats:
    """Mutable counters and derived metrics for cache health."""

    total_requests: int = 0
    gpu_hits: int = 0
    cpu_hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    avg_load_time_ms: float = 0.0
    avg_prefetch_accuracy: float = 0.0
    gpu_memory_used_mb: float = 0.0
    cpu_memory_used_mb: float = 0.0
    gpu_memory_saved_mb: float = 0.0
    transfers_completed: int = 0
    transfers_pending: int = 0


class ARCCache:
    """Adaptive Replacement Cache with O(1) average operations.

    Tracks recency and frequency via four lists:
    `T1` recent, `T2` frequent, `B1` recent ghost, `B2` frequent ghost.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("ARC capacity must be > 0")
        self.capacity = capacity
        self.target_t1 = 0
        self.t1: OrderedDict[tuple[int, int], None] = OrderedDict()
        self.t2: OrderedDict[tuple[int, int], None] = OrderedDict()
        self.b1: OrderedDict[tuple[int, int], None] = OrderedDict()
        self.b2: OrderedDict[tuple[int, int], None] = OrderedDict()

    def get(self, key: tuple[int, int]) -> bool:
        """Return True if key is present in ARC active lists."""
        if key in self.t1:
            self.t1.pop(key)
            self.t2[key] = None
            return True
        if key in self.t2:
            self.t2.move_to_end(key)
            return True
        return False

    def adapt(self, key: tuple[int, int], ghost_hit: Literal["b1", "b2"]) -> None:
        """Adapt recency/frequency target after ghost-list hit."""
        if ghost_hit == "b1":
            delta = (
                1 if len(self.b1) >= len(self.b2) else max(1, len(self.b2) // max(1, len(self.b1)))
            )
            self.target_t1 = min(self.capacity, self.target_t1 + delta)
        else:
            delta = (
                1 if len(self.b2) >= len(self.b1) else max(1, len(self.b1) // max(1, len(self.b2)))
            )
            self.target_t1 = max(0, self.target_t1 - delta)

    def _replace(self, key: tuple[int, int]) -> tuple[int, int] | None:
        if self.t1 and (
            len(self.t1) > self.target_t1 or (key in self.b2 and len(self.t1) == self.target_t1)
        ):
            old, _ = self.t1.popitem(last=False)
            self.b1[old] = None
            return old
        if self.t2:
            old, _ = self.t2.popitem(last=False)
            self.b2[old] = None
            return old
        return None

    def put(self, key: tuple[int, int]) -> tuple[int, int] | None:
        """Insert key and return evicted active key if any."""
        if self.get(key):
            return None

        if key in self.b1:
            self.adapt(key, "b1")
            evicted_key = self._replace(key)
            self.b1.pop(key, None)
            self.t2[key] = None
            self._trim_ghosts()
            return evicted_key

        if key in self.b2:
            self.adapt(key, "b2")
            evicted_key = self._replace(key)
            self.b2.pop(key, None)
            self.t2[key] = None
            self._trim_ghosts()
            return evicted_key

        evicted: tuple[int, int] | None = None
        if len(self.t1) + len(self.b1) == self.capacity:
            if len(self.t1) < self.capacity:
                self.b1.popitem(last=False)
                evicted = self._replace(key)
            else:
                evicted, _ = self.t1.popitem(last=False)
        elif len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= self.capacity:
            if len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= 2 * self.capacity:
                self.b2.popitem(last=False)
            evicted = self._replace(key)

        self.t1[key] = None
        self._trim_ghosts()
        return evicted

    def _trim_ghosts(self) -> None:
        while len(self.b1) > self.capacity:
            self.b1.popitem(last=False)
        while len(self.b2) > self.capacity:
            self.b2.popitem(last=False)

    def state_dict(self) -> dict[str, object]:
        """Serialize ARC state."""
        return {
            "capacity": self.capacity,
            "target_t1": self.target_t1,
            "t1": [list(x) for x in self.t1],
            "t2": [list(x) for x in self.t2],
            "b1": [list(x) for x in self.b1],
            "b2": [list(x) for x in self.b2],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore ARC state."""
        capacity_obj = state.get("capacity")
        target_obj = state.get("target_t1")
        t1_obj = state.get("t1", [])
        t2_obj = state.get("t2", [])
        b1_obj = state.get("b1", [])
        b2_obj = state.get("b2", [])

        if capacity_obj is None or target_obj is None:
            raise ValueError("Invalid ARC state: missing capacity/target_t1")

        self.capacity = int(cast(int | float | str, capacity_obj))
        self.target_t1 = int(cast(int | float | str, target_obj))

        t1_list = cast(list[list[int]], t1_obj)
        t2_list = cast(list[list[int]], t2_obj)
        b1_list = cast(list[list[int]], b1_obj)
        b2_list = cast(list[list[int]], b2_obj)

        self.t1 = OrderedDict(((int(x[0]), int(x[1])), None) for x in t1_list)
        self.t2 = OrderedDict(((int(x[0]), int(x[1])), None) for x in t2_list)
        self.b1 = OrderedDict(((int(x[0]), int(x[1])), None) for x in b1_list)
        self.b2 = OrderedDict(((int(x[0]), int(x[1])), None) for x in b2_list)


class DynamicExpertCache:
    """Thread-safe dynamic cache for MoE experts across CPU/GPU tiers."""

    def __init__(self, config: ExpertCacheConfig, quantizer: PolarQuantizer | None = None) -> None:
        self.config = config
        self.quantizer = quantizer
        self.device = torch.device(config.device)
        self._lock = threading.RLock()
        self._stats_lock = threading.Lock()
        self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, config.transfer_streams), thread_name_prefix="expert-prefetch"
        )

        self._experts: dict[tuple[int, int], ExpertEntry] = {}
        self._gpu_experts: set[tuple[int, int]] = set()
        self._pending_prefetch: set[tuple[int, int]] = set()
        self._prefetch_queue: deque[tuple[int, int]] = deque(
            maxlen=max(1, config.prefetch_depth * config.num_layers)
        )

        self._policy = config.eviction_policy.lower()
        self._arc = ARCCache(config.gpu_cache_size)
        self._lru: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._lfu: dict[tuple[int, int], int] = {}

        self._streams: list[torch.cuda.Stream] = []
        if torch.cuda.is_available() and self.device.type == "cuda":
            cuda_mod: Any = torch.cuda
            self._streams = [
                cuda_mod.Stream(device=self.device) for _ in range(max(1, config.transfer_streams))
            ]

        self._stats = ExpertCacheStats()
        self._stream_index = 0
        self._request_counter = 0
        self._prefetch_total = 0
        self._prefetch_hits = 0
        self._logger = LOGGER.bind(component="DynamicExpertCache")

    def register_expert(
        self, expert_id: int, layer_id: int, weights: dict[str, torch.Tensor]
    ) -> None:
        """Register a new expert; stored initially on CPU tier."""
        key = (layer_id, expert_id)
        with self._lock:
            if key in self._experts:
                raise ValueError(f"Expert already registered: layer={layer_id}, expert={expert_id}")

            cpu_weights = {name: self._prepare_cpu_tensor(t) for name, t in weights.items()}
            compressed = bool(self.config.compress_cpu_experts)
            stored = self._compress_weights(cpu_weights) if compressed else cpu_weights

            gpu_mem_mb = self._weights_size_mb(
                {k: v.to(dtype=torch.float16) for k, v in cpu_weights.items()}
            )
            cpu_mem_mb = self._weights_size_mb(stored)
            entry = ExpertEntry(
                expert_id=expert_id,
                layer_id=layer_id,
                weights=stored,
                is_on_gpu=False,
                last_access=time.monotonic(),
                access_count=0,
                compressed=compressed,
                gpu_memory_mb=gpu_mem_mb,
                cpu_memory_mb=cpu_mem_mb,
            )
            self._experts[key] = entry
            self._refresh_memory_stats_locked()

    def get_expert(self, expert_id: int, layer_id: int) -> dict[str, torch.Tensor]:
        """Return expert weights on GPU in float16 format."""
        start = time.perf_counter()
        key = (layer_id, expert_id)

        with self._lock:
            self._request_counter += 1
            entry = self._experts.get(key)
            if entry is None:
                self._update_stats(load_ms=0.0, gpu_hit=False, cpu_hit=False, miss=True)
                raise KeyError(f"Unknown expert: layer={layer_id}, expert={expert_id}")

            if entry.is_on_gpu and key in self._gpu_experts:
                self._touch_policy(key)
                entry.last_access = time.monotonic()
                entry.access_count += 1
                self._update_stats(load_ms=0.0, gpu_hit=True, cpu_hit=False, miss=False)
                if key in self._pending_prefetch:
                    self._prefetch_hits += 1
                    self._pending_prefetch.discard(key)
                self._maybe_log_info_locked()
                self._logger.debug("expert_get", key=key, source="gpu")
                return {n: t for n, t in entry.weights.items()}

            self._ensure_capacity_locked()
            gpu_weights = self._move_entry_to_gpu_locked(key)
            load_ms = (time.perf_counter() - start) * 1000.0
            self._update_stats(load_ms=load_ms, gpu_hit=False, cpu_hit=True, miss=False)
            self._maybe_log_info_locked()
            self._logger.debug("expert_get", key=key, source="cpu", load_ms=load_ms)
            return gpu_weights

    def prefetch_experts(
        self,
        expert_ids: list[int],
        layer_id: int,
        priority: float = 1.0,
    ) -> concurrent.futures.Future[dict[tuple[int, int], bool]]:
        """Asynchronously prefetch experts for future MoE layers."""

        keys = [(layer_id, expert_id) for expert_id in expert_ids]
        self._prefetch_total += len(keys)

        def _task() -> dict[tuple[int, int], bool]:
            outcome: dict[tuple[int, int], bool] = {}
            for key in keys:
                with self._lock:
                    if key not in self._experts:
                        outcome[key] = False
                        continue
                    if key in self._gpu_experts:
                        outcome[key] = True
                        continue
                    if (
                        len(self._gpu_experts) >= self.config.gpu_cache_size
                        and priority < self.config.prefetch_threshold
                    ):
                        outcome[key] = False
                        continue
                    self._pending_prefetch.add(key)
                    self._prefetch_queue.append(key)
                    self._ensure_capacity_locked()
                    self._move_entry_to_gpu_locked(key, async_transfer=True)
                    outcome[key] = True
            return outcome

        with self._stats_lock:
            self._stats.transfers_pending += len(keys)
        return self._prefetch_executor.submit(_task)

    def evict(self, expert_id: int, layer_id: int) -> None:
        """Force-evict one expert from GPU to CPU tier."""
        key = (layer_id, expert_id)
        with self._lock:
            if key not in self._experts:
                raise KeyError(f"Unknown expert: layer={layer_id}, expert={expert_id}")
            if key not in self._gpu_experts:
                return
            self._evict_locked(key)
            self._logger.debug("expert_evict", key=key)

    def warmup(self, routing_history: list[list[list[int]]]) -> None:
        """Warm cache with most frequently used experts from routing history."""
        counts: dict[tuple[int, int], int] = {}
        for layer_id, layer_steps in enumerate(routing_history):
            for step in layer_steps:
                for expert_id in step:
                    key = (layer_id, expert_id)
                    counts[key] = counts.get(key, 0) + 1

        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        budget = min(self.config.gpu_cache_size, self.config.num_experts)
        for key, _ in ranked[:budget]:
            layer_id, expert_id = key
            self.get_expert(expert_id=expert_id, layer_id=layer_id)

    def stats(self) -> ExpertCacheStats:
        """Return a snapshot of current cache statistics."""
        with self._stats_lock:
            return ExpertCacheStats(**asdict(self._stats))

    def reset_stats(self) -> None:
        """Reset counters without touching cache contents."""
        with self._stats_lock:
            self._stats = ExpertCacheStats(
                gpu_memory_used_mb=self._stats.gpu_memory_used_mb,
                cpu_memory_used_mb=self._stats.cpu_memory_used_mb,
                gpu_memory_saved_mb=self._stats.gpu_memory_saved_mb,
            )
        self._prefetch_total = 0
        self._prefetch_hits = 0

    def save_state(self, path: str) -> None:
        """Persist CPU expert storage via safetensors + metadata JSON."""
        if not HAS_SAFETENSORS:
            raise RuntimeError("safetensors is required for save_state")

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            tensors: dict[str, torch.Tensor] = {}
            entries_meta: dict[str, object] = {}

            for key, entry in self._experts.items():
                layer_id, expert_id = key
                if entry.is_on_gpu:
                    self._evict_locked(key)
                prefix = f"l{layer_id}_e{expert_id}"
                for name, tensor in entry.weights.items():
                    tensors[f"{prefix}::{name}"] = tensor.detach().cpu().contiguous()
                entries_meta[prefix] = {
                    "expert_id": expert_id,
                    "layer_id": layer_id,
                    "is_on_gpu": False,
                    "last_access": entry.last_access,
                    "access_count": entry.access_count,
                    "compressed": entry.compressed,
                    "gpu_memory_mb": entry.gpu_memory_mb,
                    "cpu_memory_mb": entry.cpu_memory_mb,
                    "weight_names": sorted(entry.weights.keys()),
                }

            save_file(tensors, str(out_dir / "experts.safetensors"))
            metadata = {
                "config": asdict(self.config),
                "arc": self._arc.state_dict(),
                "stats": asdict(self._stats),
                "entries": entries_meta,
            }
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load_state(self, path: str) -> None:
        """Load cache state and validate configuration compatibility."""
        if not HAS_SAFETENSORS:
            raise RuntimeError("safetensors is required for load_state")

        base = Path(path)
        metadata = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
        state_cfg = metadata["config"]
        if state_cfg["num_experts"] != self.config.num_experts:
            raise ValueError("State num_experts mismatch")
        if state_cfg["num_layers"] != self.config.num_layers:
            raise ValueError("State num_layers mismatch")

        tensors = load_file(str(base / "experts.safetensors"))
        with self._lock:
            self._experts.clear()
            self._gpu_experts.clear()
            self._lru.clear()
            self._lfu.clear()

            entries = metadata["entries"]
            for prefix, entry_meta in entries.items():
                layer_id = int(entry_meta["layer_id"])
                expert_id = int(entry_meta["expert_id"])
                weight_names = entry_meta["weight_names"]
                weights = {name: tensors[f"{prefix}::{name}"].cpu() for name in weight_names}
                key = (layer_id, expert_id)
                self._experts[key] = ExpertEntry(
                    expert_id=expert_id,
                    layer_id=layer_id,
                    weights=weights,
                    is_on_gpu=False,
                    last_access=float(entry_meta["last_access"]),
                    access_count=int(entry_meta["access_count"]),
                    compressed=bool(entry_meta["compressed"]),
                    gpu_memory_mb=float(entry_meta["gpu_memory_mb"]),
                    cpu_memory_mb=float(entry_meta["cpu_memory_mb"]),
                )

            self._arc.load_state_dict(metadata["arc"])
            self._stats = ExpertCacheStats(**metadata["stats"])
            self._refresh_memory_stats_locked()

    def __repr__(self) -> str:
        st = self.stats()
        gpu_count = len(self._gpu_experts)
        cpu_count = len(self._experts) - gpu_count
        return (
            "DynamicExpertCache("  # pragma: no cover - repr format only
            f"gpu={gpu_count}, cpu={cpu_count}, "
            f"hit_rate={st.hit_rate:.3f}, "
            f"gpu_mem_mb={st.gpu_memory_used_mb:.1f}, "
            f"cpu_mem_mb={st.cpu_memory_used_mb:.1f})"
        )

    def _prepare_cpu_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach().to(device="cpu", dtype=torch.float16, copy=True).contiguous()
        if self.config.pin_memory and torch.cuda.is_available():
            try:
                return out.pin_memory()
            except RuntimeError:
                return out
        return out

    def _compress_weights(self, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        compressed: dict[str, torch.Tensor] = {}
        for name, tensor in weights.items():
            scale = tensor.abs().amax().clamp(min=1e-8) / 127.0
            q = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
            compressed[name] = q
            compressed[f"{name}__scale"] = scale.to(torch.float32)
        return compressed

    def _decompress_weights(self, entry: ExpertEntry) -> dict[str, torch.Tensor]:
        if not entry.compressed:
            return {n: t for n, t in entry.weights.items()}

        out: dict[str, torch.Tensor] = {}
        for name, tensor in entry.weights.items():
            if name.endswith("__scale"):
                continue
            scale = entry.weights[f"{name}__scale"]
            out[name] = tensor.float() * scale.float()
        return out

    def _weights_size_mb(self, weights: dict[str, torch.Tensor]) -> float:
        bytes_total = sum(t.numel() * t.element_size() for t in weights.values())
        return float(bytes_total) / MB

    def _touch_policy(self, key: tuple[int, int]) -> None:
        if self._policy == "arc":
            self._arc.get(key)
            return
        if self._policy == "lru":
            if key in self._lru:
                self._lru.move_to_end(key)
            return
        self._lfu[key] = self._lfu.get(key, 0) + 1

    def _choose_victim_locked(self) -> tuple[int, int] | None:
        if not self._gpu_experts:
            return None
        if self._policy == "arc":
            active = list(self._arc.t1.keys()) + list(self._arc.t2.keys())
            for key in active:
                if key in self._gpu_experts:
                    return key
            return next(iter(self._gpu_experts))
        if self._policy == "lru":
            for key in self._lru:
                if key in self._gpu_experts:
                    return key
            return next(iter(self._gpu_experts))

        min_key = min(self._gpu_experts, key=lambda key: self._lfu.get(key, 0))
        return min_key

    def _ensure_capacity_locked(self) -> None:
        while len(self._gpu_experts) >= self.config.gpu_cache_size:
            victim = self._choose_victim_locked()
            if victim is None:
                break
            self._evict_locked(victim)

    def _next_stream(self) -> torch.cuda.Stream | None:
        if not self._streams:
            return None
        stream = self._streams[self._stream_index % len(self._streams)]
        self._stream_index += 1
        return stream

    def _move_entry_to_gpu_locked(
        self, key: tuple[int, int], async_transfer: bool = False
    ) -> dict[str, torch.Tensor]:
        entry = self._experts[key]
        weights_cpu = self._decompress_weights(entry)
        stream = self._next_stream()
        non_blocking = self.config.pin_memory

        if self.device.type == "cuda" and torch.cuda.is_available():
            if stream is not None:
                with torch.cuda.stream(stream):
                    weights_gpu = {
                        n: t.to(self.device, dtype=torch.float16, non_blocking=non_blocking)
                        for n, t in weights_cpu.items()
                    }
                if not async_transfer:
                    stream.synchronize()
            else:
                weights_gpu = {
                    n: t.to(self.device, dtype=torch.float16) for n, t in weights_cpu.items()
                }
        else:
            weights_gpu = {n: t.to(dtype=torch.float16) for n, t in weights_cpu.items()}

        entry.weights = weights_gpu
        entry.is_on_gpu = True
        entry.last_access = time.monotonic()
        entry.access_count += 1
        self._gpu_experts.add(key)

        if self._policy == "arc":
            victim = self._arc.put(key)
            if victim is not None and victim in self._gpu_experts and victim != key:
                self._evict_locked(victim)
        elif self._policy == "lru":
            self._lru[key] = None
            self._lru.move_to_end(key)
        else:
            self._lfu[key] = self._lfu.get(key, 0) + 1

        with self._stats_lock:
            self._stats.transfers_completed += 1
            self._stats.transfers_pending = max(0, self._stats.transfers_pending - 1)
        self._refresh_memory_stats_locked()
        return weights_gpu

    def _evict_locked(self, key: tuple[int, int]) -> None:
        entry = self._experts[key]
        if not entry.is_on_gpu:
            return

        cpu_weights = {
            name: self._prepare_cpu_tensor(tensor) for name, tensor in entry.weights.items()
        }
        stored = (
            self._compress_weights(cpu_weights) if self.config.compress_cpu_experts else cpu_weights
        )
        entry.weights = stored
        entry.is_on_gpu = False
        entry.compressed = self.config.compress_cpu_experts
        entry.cpu_memory_mb = self._weights_size_mb(stored)

        self._gpu_experts.discard(key)
        self._lru.pop(key, None)
        self._lfu.pop(key, None)
        self._refresh_memory_stats_locked()

    def _update_stats(self, load_ms: float, gpu_hit: bool, cpu_hit: bool, miss: bool) -> None:
        with self._stats_lock:
            self._stats.total_requests += 1
            self._stats.gpu_hits += int(gpu_hit)
            self._stats.cpu_hits += int(cpu_hit)
            self._stats.misses += int(miss)
            self._stats.hit_rate = self._stats.gpu_hits / max(1, self._stats.total_requests)

            n = self._stats.total_requests
            self._stats.avg_load_time_ms = ((self._stats.avg_load_time_ms * (n - 1)) + load_ms) / n
            if self._prefetch_total > 0:
                self._stats.avg_prefetch_accuracy = self._prefetch_hits / self._prefetch_total

    def _refresh_memory_stats_locked(self) -> None:
        gpu_used = 0.0
        cpu_used = 0.0
        all_gpu = 0.0
        for key, entry in self._experts.items():
            all_gpu += entry.gpu_memory_mb
            if key in self._gpu_experts:
                gpu_used += entry.gpu_memory_mb
            else:
                cpu_used += entry.cpu_memory_mb

        with self._stats_lock:
            self._stats.gpu_memory_used_mb = gpu_used
            self._stats.cpu_memory_used_mb = cpu_used
            self._stats.gpu_memory_saved_mb = max(0.0, all_gpu - gpu_used)

    def _maybe_log_info_locked(self) -> None:
        if self._request_counter % 100 != 0:
            return
        st = self.stats()
        self._logger.info(
            "cache_stats",
            total_requests=st.total_requests,
            hit_rate=st.hit_rate,
            gpu_hits=st.gpu_hits,
            cpu_hits=st.cpu_hits,
            misses=st.misses,
            gpu_memory_mb=st.gpu_memory_used_mb,
            cpu_memory_mb=st.cpu_memory_used_mb,
        )
