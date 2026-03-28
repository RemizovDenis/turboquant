"""Markov trajectory predictor for speculative expert prefetching."""

from __future__ import annotations

import concurrent.futures
import json
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn

from turboquant.core.moe_expert_cache import DynamicExpertCache

try:
    from safetensors.torch import load_file, save_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

LOGGER = structlog.get_logger(__name__)


@dataclass
class MarkovPrefetchConfig:
    """Configuration for :class:`MarkovTrajectoryPredictor`."""

    num_layers: int
    num_experts: int
    top_k_experts: int
    lookahead_steps: int = 3
    ema_alpha: float = 0.05
    prefetch_threshold: float = 0.25
    min_prefetch_prob: float = 0.1
    prefetch_priority_decay: float = 0.7
    uncertainty_topk_boost: int = 2
    per_source_topk: int = 1
    max_prefetch_per_layer: int = 0
    wait_timeout_ms: float = 0.25
    async_transfer_streams: int = 2
    max_pending_prefetches: int = 16
    device: str = "cuda"


@dataclass
class PrefetchPrediction:
    """Prefetch recommendation for a future layer."""

    layer_id: int
    expert_ids: list[int]
    probabilities: list[float]
    horizon: int
    confidence: float
    prefetch_started: bool
    estimated_load_ms: float


@dataclass
class MarkovStats:
    """Runtime statistics for Markov prefetcher."""

    total_predictions: int
    correct_predictions: int
    accuracy_at_1: float
    accuracy_at_k: float
    avg_lookahead_accuracy: dict[int, float]
    io_latency_hidden_ms: float
    transition_matrix_entropy: float


class MarkovTrajectoryPredictor(nn.Module):
    """Predict future MoE expert usage via layer-to-layer Markov transitions."""

    def __init__(
        self,
        config: MarkovPrefetchConfig,
        expert_cache: DynamicExpertCache,
    ) -> None:
        super().__init__()
        if config.num_layers <= 0 or config.num_experts <= 0:
            raise ValueError("num_layers and num_experts must be positive")
        self.config = config
        self.expert_cache = expert_cache
        self.device_obj = torch.device(config.device)

        init_prob = 1.0 / float(config.num_experts)
        transition = torch.full(
            (config.num_layers, config.num_experts, config.num_experts),
            fill_value=init_prob,
            dtype=torch.float32,
        )
        counts = torch.zeros(
            (config.num_layers, config.num_experts, config.num_experts),
            dtype=torch.int64,
        )
        self.register_buffer("transition_matrix", transition)
        self.register_buffer("observation_counts", counts)

        self.recent_trajectory: deque[tuple[int, list[int]]] = deque(maxlen=config.num_layers)
        self.pending_prefetches: dict[
            tuple[int, int],
            concurrent.futures.Future[dict[tuple[int, int], bool]],
        ] = {}
        self._prediction_cache: dict[int, list[tuple[int, PrefetchPrediction]]] = defaultdict(list)

        self._streams: list[torch.cuda.Stream] = []
        if torch.cuda.is_available() and self.device_obj.type == "cuda":
            cuda_mod: Any = torch.cuda
            self._streams = [
                cuda_mod.Stream(device=self.device_obj)
                for _ in range(max(1, config.async_transfer_streams))
            ]
        self._stream_idx = 0
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, config.async_transfer_streams),
            thread_name_prefix="markov-prefetch",
        )
        self._lock = threading.RLock()

        self._total_predictions = 0
        self._correct_predictions = 0
        self._correct_at_1 = 0
        self._lookahead_hits: dict[int, int] = defaultdict(int)
        self._lookahead_total: dict[int, int] = defaultdict(int)
        self._io_latency_hidden_ms = 0.0
        self._completed_trajectories = 0
        self._logger = LOGGER.bind(component="MarkovTrajectoryPredictor")

    def _transition_matrix(self) -> torch.Tensor:
        return self.get_buffer("transition_matrix")

    def _observation_counts(self) -> torch.Tensor:
        return self.get_buffer("observation_counts")

    def predict(
        self,
        current_layer: int,
        active_experts: list[int],
        lookahead_steps: int | None = None,
    ) -> dict[int, PrefetchPrediction]:
        """Predict future experts for subsequent layers.

        Args:
            current_layer: Current layer id.
            active_experts: Active experts on current layer.
            lookahead_steps: Optional custom lookahead horizon.

        Returns:
            Mapping `future_layer_id -> PrefetchPrediction`.
        """
        if not active_experts:
            return {}
        if current_layer < 0 or current_layer >= self.config.num_layers:
            return {}

        steps = (
            self.config.lookahead_steps if lookahead_steps is None else max(1, int(lookahead_steps))
        )
        probs = torch.zeros(self.config.num_experts, dtype=torch.float32)
        unique_active = sorted(set(int(x) for x in active_experts))
        probs[unique_active] = 1.0 / float(len(unique_active))

        output: dict[int, PrefetchPrediction] = {}
        with self._lock:
            matrix = self._transition_matrix().detach().cpu()
            for step in range(1, steps + 1):
                transition_layer = current_layer + step - 1
                future_layer = current_layer + step
                if (
                    transition_layer >= self.config.num_layers
                    or future_layer >= self.config.num_layers
                ):
                    break

                probs = probs @ matrix[transition_layer]
                probs = probs / probs.sum().clamp_min(1e-8)

                entropy = float(
                    -(probs * torch.log(probs.clamp_min(1e-8))).sum().item()
                    / math.log(self.config.num_experts)
                )
                confidence = 1.0 - max(0.0, min(1.0, entropy))
                dyn_topk = min(
                    self.config.num_experts,
                    max(
                        self.config.top_k_experts,
                        self.config.top_k_experts
                        + int(round(entropy * max(0, self.config.uncertainty_topk_boost))),
                    ),
                )

                candidate = torch.nonzero(
                    probs >= float(self.config.min_prefetch_prob), as_tuple=False
                ).flatten()
                if candidate.numel() == 0:
                    top_vals, top_idx = torch.topk(
                        probs,
                        k=dyn_topk,
                    )
                    chosen_ids = [int(x) for x in top_idx.tolist()]
                else:
                    cand_probs = probs[candidate]
                    order = torch.argsort(cand_probs, descending=True)
                    top_n = min(dyn_topk, int(order.numel()))
                    chosen_ids = [int(x) for x in candidate[order[:top_n]].tolist()]

                if self.config.per_source_topk > 0:
                    extra_ids: set[int] = set()
                    src_topk = min(self.config.num_experts, self.config.per_source_topk)
                    for src in unique_active:
                        row = matrix[transition_layer, src]
                        src_top = torch.topk(row, k=src_topk).indices.tolist()
                        extra_ids.update(int(x) for x in src_top)
                    merged = sorted(
                        set(chosen_ids) | extra_ids,
                        key=lambda expert_id: float(probs[expert_id].item()),
                        reverse=True,
                    )
                    max_ids = min(self.config.num_experts, dyn_topk + self.config.per_source_topk)
                    chosen_ids = merged[:max_ids]

                chosen_probs = [float(probs[idx].item()) for idx in chosen_ids]
                prediction = PrefetchPrediction(
                    layer_id=future_layer,
                    expert_ids=chosen_ids,
                    probabilities=chosen_probs,
                    horizon=step,
                    confidence=confidence,
                    prefetch_started=False,
                    estimated_load_ms=float(self.expert_cache.stats().avg_load_time_ms),
                )
                output[future_layer] = prediction
                self._prediction_cache[future_layer].append((step, prediction))
                self._total_predictions += len(prediction.expert_ids)
                self._lookahead_total[step] += 1

        self._logger.debug(
            "markov_predict",
            current_layer=current_layer,
            active_experts=unique_active,
            predictions={k: v.expert_ids for k, v in output.items()},
        )
        return output

    def start_prefetch(self, predictions: dict[int, PrefetchPrediction]) -> None:
        """Start asynchronous prefetch for predicted experts.

        Args:
            predictions: Predictions produced by :meth:`predict`.
        """
        if not predictions:
            return

        layer_scores: dict[int, dict[int, float]] = defaultdict(dict)
        layer_preds: dict[int, list[PrefetchPrediction]] = defaultdict(list)
        layer_horizons: dict[int, dict[int, int]] = defaultdict(dict)
        for layer_id, pred in predictions.items():
            if not pred.expert_ids:
                continue
            max_prob = max(pred.probabilities, default=0.0)
            confidence_ok = pred.confidence >= self.config.prefetch_threshold
            probability_ok = max_prob >= max(
                self.config.min_prefetch_prob,
                self.config.prefetch_threshold,
            )
            if not confidence_ok and not probability_ok:
                continue
            layer_preds[layer_id].append(pred)
            for expert_id, prob in zip(pred.expert_ids, pred.probabilities, strict=True):
                effective_confidence = max(pred.confidence, self.config.prefetch_threshold, 0.05)
                score = max(
                    0.0,
                    float(prob)
                    * effective_confidence
                    * (self.config.prefetch_priority_decay ** max(0, pred.horizon - 1)),
                )
                prev = layer_scores[layer_id].get(int(expert_id), 0.0)
                if score >= prev:
                    layer_scores[layer_id][int(expert_id)] = score
                horizon_prev = layer_horizons[layer_id].get(int(expert_id), pred.horizon)
                layer_horizons[layer_id][int(expert_id)] = min(horizon_prev, pred.horizon)

        if not layer_scores:
            return

        per_layer_limit = (
            self.config.max_prefetch_per_layer
            if self.config.max_prefetch_per_layer > 0
            else max(self.config.top_k_experts * 2, self.config.top_k_experts)
        )

        with self._lock:
            budget = self.config.max_pending_prefetches - len(self.pending_prefetches)
            if budget <= 0:
                return

            layer_ready: dict[int, set[int]] = defaultdict(set)
            layer_selected: dict[int, list[int]] = defaultdict(list)
            ranked_global: list[tuple[float, int, int, int]] = []

            for layer_id, scores in layer_scores.items():
                for expert_id, score in scores.items():
                    key = (layer_id, expert_id)
                    if key in self.expert_cache._gpu_experts:
                        layer_ready[layer_id].add(expert_id)
                        continue
                    if key in self.pending_prefetches:
                        continue
                    horizon = layer_horizons[layer_id].get(expert_id, self.config.lookahead_steps)
                    ranked_global.append((float(score), -int(horizon), layer_id, expert_id))

            ranked_global.sort(key=lambda item: (item[0], item[1]), reverse=True)

            for score, _, layer_id, expert_id in ranked_global:
                if budget <= 0:
                    break
                if len(layer_selected[layer_id]) >= per_layer_limit:
                    continue
                key = (layer_id, expert_id)
                if key in self.pending_prefetches or key in self.expert_cache._gpu_experts:
                    continue
                if score <= 0.0:
                    continue
                layer_selected[layer_id].append(expert_id)
                budget -= 1

            for layer_id in sorted(layer_scores):
                selected = layer_selected.get(layer_id, [])
                ready_set = layer_ready.get(layer_id, set())
                if not selected and not ready_set:
                    continue

                if selected:
                    top_score = max(
                        layer_scores[layer_id].get(expert_id, 0.0) for expert_id in selected
                    )
                    fut = self.expert_cache.prefetch_experts(
                        expert_ids=selected,
                        layer_id=layer_id,
                        priority=max(0.1, top_score),
                    )
                    for expert_id in selected:
                        self.pending_prefetches[(layer_id, expert_id)] = fut

                selected_set = set(selected)
                for pred in layer_preds[layer_id]:
                    if set(pred.expert_ids) & (selected_set | ready_set):
                        pred.prefetch_started = True

    def on_layer_complete(self, layer_id: int, actual_experts: list[int]) -> None:
        """Update transitions and prediction accuracy when a layer completes.

        Args:
            layer_id: Layer that has just produced final routing decision.
            actual_experts: Experts actually activated at this layer.
        """
        t0 = time.perf_counter()
        if not actual_experts:
            return
        actual = sorted(set(int(x) for x in actual_experts))

        with self._lock:
            if self.recent_trajectory and self.recent_trajectory[-1][0] == layer_id - 1:
                prev_layer, prev_experts = self.recent_trajectory[-1]
                if 0 <= prev_layer < self.config.num_layers:
                    self._update_transition(prev_layer, prev_experts, actual)

            self.recent_trajectory.append((layer_id, actual))

            preds = self._prediction_cache.pop(layer_id, [])
            if preds:
                actual_set = set(actual)
                cache_stats = self.expert_cache.stats()
                avg_load = float(cache_stats.avg_load_time_ms)
                cpu_load = (
                    float(cache_stats.avg_cpu_load_time_ms)
                    if cache_stats.avg_cpu_load_time_ms > 0.0
                    else avg_load
                )
                for step, pred in preds:
                    predicted_set = set(pred.expert_ids)
                    overlap_count = len(predicted_set & actual_set)
                    if overlap_count > 0:
                        self._correct_predictions += 1
                        self._lookahead_hits[step] += 1
                        if pred.prefetch_started:
                            self._io_latency_hidden_ms += cpu_load * float(overlap_count)
                    if pred.expert_ids and pred.expert_ids[0] in actual_set:
                        self._correct_at_1 += 1

            self._cleanup_pending(layer_id)
            self._completed_trajectories += 1
            if self._completed_trajectories % 50 == 0:
                st = self.stats()
                self._logger.info(
                    "markov_stats",
                    accuracy_at_k=st.accuracy_at_k,
                    accuracy_at_1=st.accuracy_at_1,
                    entropy=st.transition_matrix_entropy,
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._logger.debug("markov_on_layer_complete", layer_id=layer_id, elapsed_ms=elapsed_ms)

    def online_update(self, completed_trajectory: list[list[int]]) -> None:
        """Asynchronously update transition matrix from full trajectory.

        Args:
            completed_trajectory: List of per-layer active experts.
        """

        def _task() -> None:
            with self._lock:
                for layer in range(len(completed_trajectory) - 1):
                    if layer >= self.config.num_layers:
                        break
                    src = completed_trajectory[layer]
                    dst = completed_trajectory[layer + 1]
                    if not src or not dst:
                        continue
                    self._update_transition(layer, src, dst)

        self._thread_pool.submit(_task)

    def adaptive_alpha(self, layer_id: int, expert_id: int) -> float:
        """Compute adaptive EMA alpha for a row update.

        Args:
            layer_id: Transition layer id.
            expert_id: Source expert id.

        Returns:
            Adaptive smoothing coefficient.
        """
        counts = self._observation_counts()[layer_id, expert_id].sum().item()
        return float(self.config.ema_alpha / math.sqrt(1.0 + float(counts)))

    def matrix_entropy(self) -> float:
        """Return mean row entropy of transition matrix."""
        with self._lock:
            matrix = self._transition_matrix().detach()
            p = matrix.clamp_min(1e-8)
            ent = -(p * torch.log(p)).sum(dim=-1) / math.log(self.config.num_experts)
            return float(ent.mean().item())

    def wait_for_layer(self, layer_id: int, timeout_ms: float = 5.0) -> list[int]:
        """Wait for pending prefetches for a specific layer.

        Args:
            layer_id: Layer to wait for.
            timeout_ms: Per-future timeout in milliseconds.

        Returns:
            List of expert ids confirmed as prefetched.
        """
        timeout_s = max(0.0, float(timeout_ms) / 1000.0)
        loaded: list[int] = []
        with self._lock:
            items = [(k, f) for k, f in self.pending_prefetches.items() if k[0] == layer_id]

        for key, fut in items:
            try:
                result = fut.result(timeout=timeout_s)
            except Exception:  # noqa: BLE001
                continue
            ok = bool(result.get(key, False))
            if ok:
                loaded.append(key[1])
            with self._lock:
                self.pending_prefetches.pop(key, None)
        return sorted(set(loaded))

    def stats(self) -> MarkovStats:
        """Compute current Markov predictor statistics."""
        with self._lock:
            total_pred_batches = max(1, sum(self._lookahead_total.values()))
            total_predictions = self._total_predictions
            correct = self._correct_predictions
            acc_k = correct / total_pred_batches
            acc_1 = self._correct_at_1 / total_pred_batches
            lookahead = {
                step: (self._lookahead_hits[step] / max(1, self._lookahead_total[step]))
                for step in sorted(self._lookahead_total)
            }
            entropy = self.matrix_entropy()
            return MarkovStats(
                total_predictions=int(total_predictions),
                correct_predictions=int(correct),
                accuracy_at_1=float(acc_1),
                accuracy_at_k=float(acc_k),
                avg_lookahead_accuracy={int(k): float(v) for k, v in lookahead.items()},
                io_latency_hidden_ms=float(self._io_latency_hidden_ms),
                transition_matrix_entropy=float(entropy),
            )

    def save(self, path: str) -> None:
        """Persist transition matrix and stats to disk.

        Args:
            path: Target directory path.
        """
        dst = Path(path)
        dst.mkdir(parents=True, exist_ok=True)
        tensors = {
            "transition_matrix": self._transition_matrix().cpu(),
            "observation_counts": self._observation_counts().cpu().to(torch.int64),
        }
        if HAS_SAFETENSORS:
            save_file(tensors, str(dst / "markov.safetensors"))
        else:
            torch.save(tensors, dst / "markov.pt")

        with self._lock:
            state = {
                "config": asdict(self.config),
                "stats": asdict(self.stats()),
                "completed_trajectories": self._completed_trajectories,
            }
        (dst / "markov_stats.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        """Load transition matrix and stats from disk.

        Args:
            path: Source directory path.
        """
        src = Path(path)
        if HAS_SAFETENSORS and (src / "markov.safetensors").exists():
            tensors = load_file(str(src / "markov.safetensors"))
        else:
            tensors = torch.load(src / "markov.pt", map_location="cpu")

        with self._lock:
            self._transition_matrix().copy_(tensors["transition_matrix"].to(torch.float32))
            self._observation_counts().copy_(tensors["observation_counts"].to(torch.int64))
            meta = json.loads((src / "markov_stats.json").read_text(encoding="utf-8"))
            stats_obj = meta.get("stats", {})
            self._total_predictions = int(stats_obj.get("total_predictions", 0))
            self._correct_predictions = int(stats_obj.get("correct_predictions", 0))
            self._io_latency_hidden_ms = float(stats_obj.get("io_latency_hidden_ms", 0.0))
            self._completed_trajectories = int(meta.get("completed_trajectories", 0))

    def __repr__(self) -> str:
        st = self.stats()
        return (
            "MarkovTrajectoryPredictor("
            f"layers={self.config.num_layers}, "
            f"experts={self.config.num_experts}, "
            f"accuracy={st.accuracy_at_k * 100:.1f}%, "
            f"entropy={st.transition_matrix_entropy:.2f})"
        )

    def _update_transition(
        self,
        layer_id: int,
        src_experts: list[int],
        dst_experts: list[int],
    ) -> None:
        src = sorted(set(int(x) for x in src_experts if 0 <= int(x) < self.config.num_experts))
        dst = sorted(set(int(x) for x in dst_experts if 0 <= int(x) < self.config.num_experts))
        if not src or not dst or layer_id < 0 or layer_id >= self.config.num_layers:
            return

        matrix = self._transition_matrix()
        counts = self._observation_counts()
        observed = torch.zeros(self.config.num_experts, dtype=torch.float32)
        observed[dst] = 1.0 / float(len(dst))

        for i in src:
            alpha = self.adaptive_alpha(layer_id, i)
            row = matrix[layer_id, i]
            row.mul_(1.0 - alpha).add_(observed * alpha)
            row.div_(row.sum().clamp_min(1e-8))
            counts[layer_id, i, dst] += 1

    def _cleanup_pending(self, current_layer: int) -> None:
        stale: list[tuple[int, int]] = []
        for key, fut in self.pending_prefetches.items():
            if key[0] <= current_layer or fut.done():
                stale.append(key)
        for key in stale:
            self.pending_prefetches.pop(key, None)
