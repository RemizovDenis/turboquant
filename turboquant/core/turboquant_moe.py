# Copyright (c) 2026 Denis Remizov. Licensed under BUSL-1.1.
# See LICENSE file for details.


"""Unified TurboQuant manager for MoE inference optimization."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import structlog
import torch
import torch.nn as nn

from turboquant.core.adaptive_bitwidth import AdaptiveBitwithConfig, AdaptiveCompressedCache
from turboquant.core.cross_layer_kv import CrossLayerConfig
from turboquant.core.expert_predictor import ExpertPredictor, ExpertPredictorConfig
from turboquant.core.markov_prefetch import MarkovPrefetchConfig, MarkovTrajectoryPredictor
from turboquant.core.moe_expert_cache import DynamicExpertCache, ExpertCacheConfig
from turboquant.core.moe_router import MoERouterOptimizer, RouterOptimizerConfig, RouterOutput
from turboquant.core.nash_router import GameTheoreticRouter, NashRouterConfig
from turboquant.core.pid_vram import PIDConfig, VRAM_PID_Controller
from turboquant.core.polar_quant import PolarQuantizer
from turboquant.core.semantic_eviction import SemanticEvictionConfig
from turboquant.core.turboquant import CacheEntry, TurboQuantConfig, TurboQuantKVCache

LOGGER = structlog.get_logger(__name__)


@dataclass
class TurboQuantMoEConfig:
    """Composite config for full TurboQuant MoE pipeline."""

    kv_config: TurboQuantConfig
    expert_config: ExpertCacheConfig
    router_config: RouterOptimizerConfig
    nash_router_config: NashRouterConfig | None = None
    predictor_config: ExpertPredictorConfig | None = None
    markov_prefetch_config: MarkovPrefetchConfig | None = None
    pid_config: PIDConfig | None = None
    semantic_eviction_config: SemanticEvictionConfig | None = None
    cross_layer_config: CrossLayerConfig | None = None
    adaptive_bitwidth_config: AdaptiveBitwithConfig | None = None
    model_type: str = "auto"
    enable_kv_quant: bool = True
    enable_expert_cache: bool = True
    enable_router_opt: bool = True
    enable_expert_prediction: bool = True
    enable_nash_routing: bool = True
    enable_markov_prefetch: bool = True
    enable_pid_vram: bool = True
    enable_semantic_kv_eviction: bool = True
    enable_cross_layer_kv: bool = True
    enable_adaptive_bitwidth: bool = True
    profile_mode: bool = False

    @classmethod
    def from_pretrained_config(
        cls,
        model_config: Any,
        gpu_cache_size: int = 4,
        bits: int = 3,
    ) -> TurboQuantMoEConfig:
        """Infer sensible defaults from HuggingFace-like model config."""
        model_type = str(getattr(model_config, "model_type", "auto")).lower()
        hidden_size = int(getattr(model_config, "hidden_size", 4096))
        num_heads = int(
            getattr(
                model_config,
                "num_attention_heads",
                getattr(model_config, "num_key_value_heads", 32),
            )
        )
        head_dim = int(getattr(model_config, "head_dim", max(1, hidden_size // max(1, num_heads))))

        num_experts = int(
            getattr(
                model_config,
                "num_local_experts",
                getattr(model_config, "num_experts", 8),
            )
        )
        top_k = int(
            getattr(
                model_config,
                "num_experts_per_tok",
                getattr(model_config, "top_k", 2),
            )
        )
        num_layers = int(
            getattr(
                model_config,
                "num_hidden_layers",
                getattr(model_config, "n_layer", 32),
            )
        )

        kv_cfg = TurboQuantConfig(
            head_dim=head_dim,
            num_heads=num_heads,
            bits=bits,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        expert_cfg = ExpertCacheConfig(
            num_experts=num_experts,
            top_k_experts=top_k,
            num_layers=num_layers,
            gpu_cache_size=gpu_cache_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        router_cfg = RouterOptimizerConfig(num_experts=num_experts, top_k=top_k)
        nash_cfg = NashRouterConfig(num_experts=num_experts, top_k=top_k)
        predictor_cfg = ExpertPredictorConfig(
            hidden_dim=hidden_size,
            num_experts=num_experts,
            num_layers=num_layers,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        markov_cfg = MarkovPrefetchConfig(
            num_layers=num_layers,
            num_experts=num_experts,
            top_k_experts=top_k,
            prefetch_threshold=0.25,
            min_prefetch_prob=0.1,
            max_pending_prefetches=max(32, num_layers * top_k),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        pid_cfg = PIDConfig(
            min_cache_size=1,
            max_cache_size=max(4, gpu_cache_size * 2),
        )
        semantic_cfg = SemanticEvictionConfig(
            max_seq_len=kv_cfg.max_seq_len,
            eviction_target_len=max(1, kv_cfg.max_seq_len // 2),
            device=kv_cfg.device,
            dtype=kv_cfg.dtype,
        )
        cross_cfg = CrossLayerConfig(
            num_layers=num_layers,
            device=kv_cfg.device,
            dtype=kv_cfg.dtype,
        )
        adaptive_cfg = AdaptiveBitwithConfig(
            head_dim=head_dim,
            num_heads=num_heads,
            vocab_size=128_000,
            device=kv_cfg.device,
            dtype=kv_cfg.dtype,
        )

        return cls(
            kv_config=kv_cfg,
            expert_config=expert_cfg,
            router_config=router_cfg,
            nash_router_config=nash_cfg,
            predictor_config=predictor_cfg,
            markov_prefetch_config=markov_cfg,
            pid_config=pid_cfg,
            semantic_eviction_config=semantic_cfg,
            cross_layer_config=cross_cfg,
            adaptive_bitwidth_config=adaptive_cfg,
            model_type=model_type,
        )


@dataclass
class MoEStepOutput:
    """Output of a single unified MoE step."""

    cache_entry: CacheEntry | AdaptiveCompressedCache
    router_output: RouterOutput
    active_experts: list[int]
    predicted_experts: list[int]
    prediction_was_correct: bool
    memory_delta_mb: float
    step_latency_ms: float


@dataclass
class MemoryReport:
    """Memory and hit-rate summary across KV and expert components."""

    kv_cache_mb: float
    kv_cache_fp16_baseline_mb: float
    kv_compression_ratio: float
    expert_gpu_mb: float
    expert_cpu_mb: float
    expert_total_baseline_mb: float
    expert_compression_ratio: float
    total_saved_mb: float
    total_saved_percent: float
    expert_hit_rate: float
    prefetch_accuracy: float


class TurboQuantMoE:
    """Unified manager integrating KV quantization and MoE expert caching."""

    def __init__(self, config: TurboQuantMoEConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._logger = LOGGER.bind(component="TurboQuantMoE")

        self.quantizer = PolarQuantizer(
            head_dim=config.kv_config.head_dim,
            bits=config.kv_config.bits,
            group_size=config.kv_config.group_size,
            seed=config.kv_config.seed,
            use_hadamard=config.kv_config.use_hadamard,
        )

        config.kv_config.enable_semantic_eviction = config.enable_semantic_kv_eviction
        config.kv_config.enable_cross_layer_sharing = config.enable_cross_layer_kv
        config.kv_config.enable_adaptive_bitwidth = config.enable_adaptive_bitwidth
        config.kv_config.semantic_eviction_config = config.semantic_eviction_config
        config.kv_config.cross_layer_config = config.cross_layer_config
        config.kv_config.adaptive_bitwidth_config = config.adaptive_bitwidth_config
        self.kv_cache = TurboQuantKVCache(config.kv_config)
        self.expert_cache = DynamicExpertCache(config.expert_config, quantizer=self.quantizer)
        if config.enable_nash_routing:
            nash_cfg = config.nash_router_config
            if nash_cfg is None:
                nash_cfg = NashRouterConfig(
                    num_experts=config.router_config.num_experts,
                    top_k=config.router_config.top_k,
                    pruning_threshold=config.router_config.pruning_threshold,
                    load_balance_alpha=config.router_config.load_balance_alpha,
                    expert_dropout_rate=config.router_config.expert_dropout_rate,
                    capacity_factor=config.router_config.capacity_factor,
                    use_aux_loss=config.router_config.use_aux_loss,
                    use_z_loss=config.router_config.use_z_loss,
                    z_loss_coeff=config.router_config.z_loss_coeff,
                    normalize_expert_weights=config.router_config.normalize_expert_weights,
                    use_noise_during_training=config.router_config.use_noise_during_training,
                    noise_std=config.router_config.noise_std,
                    ema_decay=config.router_config.ema_decay,
                )
            self.router: MoERouterOptimizer = GameTheoreticRouter(nash_cfg)
        else:
            self.router = MoERouterOptimizer(config.router_config)

        self.predictor: ExpertPredictor | None = None
        if config.enable_expert_prediction and config.predictor_config is not None:
            self.predictor = ExpertPredictor(config.predictor_config)
        self.markov_predictor: MarkovTrajectoryPredictor | None = None
        if config.enable_markov_prefetch:
            markov_cfg = config.markov_prefetch_config
            if markov_cfg is None:
                markov_cfg = MarkovPrefetchConfig(
                    num_layers=config.expert_config.num_layers,
                    num_experts=config.expert_config.num_experts,
                    top_k_experts=config.expert_config.top_k_experts,
                    device=config.expert_config.device,
                )
            self.markov_predictor = MarkovTrajectoryPredictor(markov_cfg, self.expert_cache)

        self.pid_controller: VRAM_PID_Controller | None = None
        if config.enable_pid_vram:
            pid_cfg = config.pid_config
            if pid_cfg is None:
                pid_cfg = PIDConfig(
                    min_cache_size=1,
                    max_cache_size=max(config.expert_config.gpu_cache_size * 2, 4),
                )
            self.pid_controller = VRAM_PID_Controller(
                config=pid_cfg,
                expert_cache=self.expert_cache,
                initial_cache_size=config.expert_config.gpu_cache_size,
            )

        self._patched_forwards: dict[int, Any] = {}
        self._step_counter = 0
        self._profile_rows: list[dict[str, float]] = []

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Patch detected MoE layers in-place to route through TurboQuant manager."""
        with self._lock:
            model_type = self._detect_model_type(model)
            patched = 0

            for layer_id, module in enumerate(model.modules()):
                if not self._is_moe_module(module):
                    continue
                mid = id(module)
                if mid in self._patched_forwards:
                    continue

                self._register_module_experts(layer_id, module)
                original_forward = module.forward
                self._patched_forwards[mid] = original_forward

                def _patched_forward(
                    hidden_states: torch.Tensor,
                    *args: Any,
                    __layer_id: int = layer_id,
                    __module: nn.Module = module,
                    __orig: Any = original_forward,
                    **kwargs: Any,
                ) -> Any:
                    router_logits = self._extract_router_logits(__module, hidden_states)
                    if router_logits is None:
                        return __orig(hidden_states, *args, **kwargs)

                    batch = hidden_states.shape[0]
                    seq = hidden_states.shape[1] if hidden_states.ndim == 3 else 1
                    head_dim = self.config.kv_config.head_dim
                    heads = self.config.kv_config.num_heads
                    device = hidden_states.device
                    keys = torch.zeros(
                        (batch, heads, seq, head_dim), dtype=torch.float16, device=device
                    )
                    values = torch.zeros_like(keys)
                    self.step(__layer_id, hidden_states, router_logits, keys, values)
                    return __orig(hidden_states, *args, **kwargs)

                module.forward = _patched_forward
                patched += 1

            if patched == 0:
                self._logger.warning(
                    "wrap_model_no_moe_layers",
                    model_type=model_type,
                    action="kv_quant_only",
                )
            return model

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Restore original forward methods for previously wrapped modules."""
        with self._lock:
            for module in model.modules():
                mid = id(module)
                if mid in self._patched_forwards:
                    module.forward = self._patched_forwards[mid]
            self._patched_forwards.clear()
            return model

    def step(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> MoEStepOutput:
        """Execute one MoE step: prediction, routing, expert load, KV compression."""
        with self._lock:
            t_start = time.perf_counter()
            before = self._total_saved_mb_fast()

            predicted_experts: list[int] = []
            predicted_prefetch_future = None
            if self.predictor is not None and self.config.enable_expert_prediction:
                predicted_experts = self.predictor.predict_experts(hidden_states, layer_id)
                if predicted_experts:
                    predicted_prefetch_future = self.expert_cache.prefetch_experts(
                        predicted_experts,
                        layer_id,
                        priority=1.0,
                    )

            if isinstance(self.router, GameTheoreticRouter):
                locations = torch.zeros(
                    self.config.router_config.num_experts,
                    dtype=torch.bool,
                    device=router_logits.device,
                )
                for expert_id in range(self.config.router_config.num_experts):
                    locations[expert_id] = (layer_id, expert_id) in self.expert_cache._gpu_experts
                router_out = self.router(
                    router_logits=router_logits,
                    expert_locations_mask=locations,
                    expert_current_load=None,
                    training=False,
                )
            else:
                router_out = self.router(router_logits=router_logits, training=False)
            active_experts = sorted({int(x) for x in router_out.expert_indices.flatten().tolist()})

            markov_ready: list[int] = []
            if self.markov_predictor is not None:
                markov_ready = self.markov_predictor.wait_for_layer(
                    layer_id=layer_id,
                    timeout_ms=self.markov_predictor.config.wait_timeout_ms,
                )

            if predicted_prefetch_future is not None:
                with suppress(Exception):
                    predicted_prefetch_future.result(timeout=0.001)

            for expert_id in active_experts:
                try:
                    self.expert_cache.get_expert(expert_id=expert_id, layer_id=layer_id)
                except torch.cuda.OutOfMemoryError:
                    self.expert_cache.evict(expert_id=expert_id, layer_id=layer_id)

            if self.pid_controller is not None and self._step_counter % 4 == 0:
                _, pid_state = self.pid_controller.step()
                if pid_state.current_utilization > self.pid_controller.config.emergency_threshold:
                    self.pid_controller.emergency_evict()

            markov_prefetch: list[int] = markov_ready
            if self.markov_predictor is not None and active_experts:
                future = self.markov_predictor.predict(layer_id, active_experts)
                self.markov_predictor.start_prefetch(future)
                self.markov_predictor.on_layer_complete(layer_id, active_experts)
                markov_prefetch = sorted(
                    set(markov_ready)
                    | {expert_id for pred in future.values() for expert_id in pred.expert_ids}
                )

            entropy: torch.Tensor | None = None
            if self.config.enable_adaptive_bitwidth:
                probs = torch.softmax(router_logits.float(), dim=-1)
                token_entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
                seq_len = int(keys.shape[2])
                if token_entropy.numel() >= seq_len:
                    entropy = token_entropy[:seq_len]

            try:
                cache_entry = self.kv_cache.compress(
                    keys,
                    values,
                    layer_id=layer_id,
                    token_ids=None,
                    attention_entropy=entropy,
                )
            except torch.cuda.OutOfMemoryError:
                keys_cpu = keys.detach().cpu()
                values_cpu = values.detach().cpu()
                cache_entry = self.kv_cache.compress(
                    keys_cpu,
                    values_cpu,
                    layer_id=layer_id,
                    token_ids=None,
                    attention_entropy=entropy,
                )

            if self.predictor is not None and self.config.enable_expert_prediction:
                self.predictor.update_history(layer_id, active_experts)
                if self._step_counter % 10 == 0:
                    self.predictor.online_update(hidden_states, layer_id, active_experts)

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            after = self._total_saved_mb_fast()
            prediction_union = set(predicted_experts) | set(markov_prefetch)
            prediction_ok = set(active_experts).issubset(prediction_union)

            self._step_counter += 1
            if self.config.profile_mode:
                self._profile_rows.append(
                    {
                        "layer_id": float(layer_id),
                        "latency_ms": latency_ms,
                        "active_experts": float(len(active_experts)),
                    }
                )

            if self._step_counter % 100 == 0:
                rep = self.memory_report()
                self._logger.info(
                    "memory_report",
                    total_saved_mb=rep.total_saved_mb,
                    total_saved_percent=rep.total_saved_percent,
                    expert_hit_rate=rep.expert_hit_rate,
                )
                if self.markov_predictor is not None:
                    ms = self.markov_predictor.stats()
                    self._logger.info(
                        "markov_report",
                        accuracy_at_k=ms.accuracy_at_k,
                        entropy=ms.transition_matrix_entropy,
                    )

            return MoEStepOutput(
                cache_entry=cache_entry,
                router_output=router_out,
                active_experts=active_experts,
                predicted_experts=sorted(prediction_union),
                prediction_was_correct=prediction_ok,
                memory_delta_mb=after - before,
                step_latency_ms=latency_ms,
            )

    def memory_report(self) -> MemoryReport:
        """Collect current memory report across all managed components."""
        kv_mem = self.kv_cache.latest_memory_usage()

        ex_stats = self.expert_cache.stats()
        expert_baseline = ex_stats.gpu_memory_mb + ex_stats.gpu_memory_saved_mb
        expert_used = ex_stats.gpu_memory_mb + ex_stats.cpu_memory_mb
        expert_ratio = expert_used / max(1e-8, expert_baseline)

        total_saved = (
            kv_mem["fp16_baseline_mb"] - kv_mem["total_mb"]
        ) + ex_stats.gpu_memory_saved_mb
        total_baseline = kv_mem["fp16_baseline_mb"] + expert_baseline

        return MemoryReport(
            kv_cache_mb=kv_mem["total_mb"],
            kv_cache_fp16_baseline_mb=kv_mem["fp16_baseline_mb"],
            kv_compression_ratio=kv_mem["compression_ratio"],
            expert_gpu_mb=ex_stats.gpu_memory_mb,
            expert_cpu_mb=ex_stats.cpu_memory_mb,
            expert_total_baseline_mb=expert_baseline,
            expert_compression_ratio=expert_ratio,
            total_saved_mb=total_saved,
            total_saved_percent=100.0 * total_saved / max(1e-8, total_baseline),
            expert_hit_rate=ex_stats.hit_rate,
            prefetch_accuracy=ex_stats.avg_prefetch_accuracy,
        )

    def _total_saved_mb_fast(self) -> float:
        kv_mem = self.kv_cache.latest_memory_usage()
        ex_stats = self.expert_cache.stats()
        return (kv_mem["fp16_baseline_mb"] - kv_mem["total_mb"]) + ex_stats.gpu_memory_saved_mb

    def benchmark(
        self,
        seq_lengths: list[int] | None = None,
        batch_size: int = 1,
    ) -> pd.DataFrame:
        """Run synthetic benchmark without loading a full model."""
        if seq_lengths is None:
            seq_lengths = [256, 1024, 4096]

        rows: list[dict[str, float]] = []
        hidden_dim = (
            self.config.predictor_config.hidden_dim
            if self.config.predictor_config is not None
            else self.config.kv_config.head_dim * self.config.kv_config.num_heads
        )

        for sl in seq_lengths:
            hidden = torch.randn(batch_size, sl, hidden_dim, device=self.kv_cache.device)
            router_logits = torch.randn(
                batch_size * sl,
                self.config.router_config.num_experts,
                device=self.kv_cache.device,
            )
            keys = torch.randn(
                batch_size,
                self.config.kv_config.num_heads,
                sl,
                self.config.kv_config.head_dim,
                dtype=torch.float16,
                device=self.kv_cache.device,
            )
            values = torch.randn_like(keys)

            t0 = time.perf_counter()
            out = self.step(
                layer_id=0,
                hidden_states=hidden,
                router_logits=router_logits,
                keys=keys,
                values=values,
            )
            step_ms = (time.perf_counter() - t0) * 1000.0
            rep = self.memory_report()
            rows.append(
                {
                    "seq_len": float(sl),
                    "step_ms": step_ms,
                    "router_dropped": float(out.router_output.dropped_tokens),
                    "active_experts": float(len(out.active_experts)),
                    "kv_ratio": rep.kv_compression_ratio,
                    "total_saved_mb": rep.total_saved_mb,
                }
            )

        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        """Save unified TurboQuantMoE state to directory."""
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)

        cfg = {
            "kv_config": _to_json_dict(self.config.kv_config),
            "expert_config": _to_json_dict(self.config.expert_config),
            "router_config": _to_json_dict(self.config.router_config),
            "nash_router_config": _to_json_dict(self.config.nash_router_config)
            if self.config.nash_router_config
            else None,
            "predictor_config": _to_json_dict(self.config.predictor_config)
            if self.config.predictor_config
            else None,
            "markov_prefetch_config": _to_json_dict(self.config.markov_prefetch_config)
            if self.config.markov_prefetch_config
            else None,
            "pid_config": _to_json_dict(self.config.pid_config) if self.config.pid_config else None,
            "semantic_eviction_config": _to_json_dict(self.config.semantic_eviction_config)
            if self.config.semantic_eviction_config
            else None,
            "cross_layer_config": _to_json_dict(self.config.cross_layer_config)
            if self.config.cross_layer_config
            else None,
            "adaptive_bitwidth_config": _to_json_dict(self.config.adaptive_bitwidth_config)
            if self.config.adaptive_bitwidth_config
            else None,
            "model_type": self.config.model_type,
            "enable_kv_quant": self.config.enable_kv_quant,
            "enable_expert_cache": self.config.enable_expert_cache,
            "enable_router_opt": self.config.enable_router_opt,
            "enable_expert_prediction": self.config.enable_expert_prediction,
            "enable_nash_routing": self.config.enable_nash_routing,
            "enable_markov_prefetch": self.config.enable_markov_prefetch,
            "enable_pid_vram": self.config.enable_pid_vram,
            "enable_semantic_kv_eviction": self.config.enable_semantic_kv_eviction,
            "enable_cross_layer_kv": self.config.enable_cross_layer_kv,
            "enable_adaptive_bitwidth": self.config.enable_adaptive_bitwidth,
            "profile_mode": self.config.profile_mode,
        }
        (base / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        self.expert_cache.save_state(str(base / "expert_cache"))
        torch.save(self.router.state_dict(), base / "router.pt")
        if self.predictor is not None:
            self.predictor.save(str(base / "predictor"))
        if self.markov_predictor is not None:
            self.markov_predictor.save(str(base / "markov"))

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> TurboQuantMoE:
        """Load TurboQuantMoE state from directory path."""
        base = Path(path)
        cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))

        kv_payload = dict(cfg["kv_config"])
        if "dtype" in kv_payload:
            kv_payload["dtype"] = _parse_dtype(kv_payload["dtype"])
        kv_cfg = TurboQuantConfig(**kv_payload)
        kv_cfg.device = device
        exp_cfg = ExpertCacheConfig(**cfg["expert_config"])
        exp_cfg.device = device
        router_cfg = RouterOptimizerConfig(**cfg["router_config"])
        nash_cfg = None
        if cfg.get("nash_router_config") is not None:
            nash_cfg = NashRouterConfig(**cfg["nash_router_config"])

        pred_cfg = None
        if cfg["predictor_config"] is not None:
            pred_cfg = ExpertPredictorConfig(**cfg["predictor_config"])
            pred_cfg.device = device
        markov_cfg = None
        if cfg.get("markov_prefetch_config") is not None:
            markov_cfg = MarkovPrefetchConfig(**cfg["markov_prefetch_config"])
            markov_cfg.device = device
        pid_cfg = None
        if cfg.get("pid_config") is not None:
            pid_cfg = PIDConfig(**cfg["pid_config"])

        semantic_cfg = None
        if cfg.get("semantic_eviction_config") is not None:
            sem_payload = dict(cfg["semantic_eviction_config"])
            if "dtype" in sem_payload:
                sem_payload["dtype"] = _parse_dtype(sem_payload["dtype"])
            semantic_cfg = SemanticEvictionConfig(**sem_payload)
            semantic_cfg.device = device

        cross_cfg = None
        if cfg.get("cross_layer_config") is not None:
            cross_payload = dict(cfg["cross_layer_config"])
            if "dtype" in cross_payload:
                cross_payload["dtype"] = _parse_dtype(cross_payload["dtype"])
            cross_cfg = CrossLayerConfig(**cross_payload)
            cross_cfg.device = device

        adaptive_cfg = None
        if cfg.get("adaptive_bitwidth_config") is not None:
            adaptive_payload = dict(cfg["adaptive_bitwidth_config"])
            if "dtype" in adaptive_payload:
                adaptive_payload["dtype"] = _parse_dtype(adaptive_payload["dtype"])
            adaptive_cfg = AdaptiveBitwithConfig(**adaptive_payload)
            adaptive_cfg.device = device

        tcfg = TurboQuantMoEConfig(
            kv_config=kv_cfg,
            expert_config=exp_cfg,
            router_config=router_cfg,
            nash_router_config=nash_cfg,
            predictor_config=pred_cfg,
            markov_prefetch_config=markov_cfg,
            pid_config=pid_cfg,
            semantic_eviction_config=semantic_cfg,
            cross_layer_config=cross_cfg,
            adaptive_bitwidth_config=adaptive_cfg,
            model_type=cfg["model_type"],
            enable_kv_quant=bool(cfg["enable_kv_quant"]),
            enable_expert_cache=bool(cfg["enable_expert_cache"]),
            enable_router_opt=bool(cfg["enable_router_opt"]),
            enable_expert_prediction=bool(cfg["enable_expert_prediction"]),
            enable_nash_routing=bool(cfg.get("enable_nash_routing", False)),
            enable_markov_prefetch=bool(cfg.get("enable_markov_prefetch", False)),
            enable_pid_vram=bool(cfg.get("enable_pid_vram", False)),
            enable_semantic_kv_eviction=bool(cfg.get("enable_semantic_kv_eviction", False)),
            enable_cross_layer_kv=bool(cfg.get("enable_cross_layer_kv", False)),
            enable_adaptive_bitwidth=bool(cfg.get("enable_adaptive_bitwidth", False)),
            profile_mode=bool(cfg["profile_mode"]),
        )

        obj = cls(tcfg)
        obj.expert_cache.load_state(str(base / "expert_cache"))
        obj.router.load_state_dict(torch.load(base / "router.pt", map_location="cpu"))
        if obj.predictor is not None and (base / "predictor").exists():
            obj.predictor.load(str(base / "predictor"))
        if obj.markov_predictor is not None and (base / "markov").exists():
            obj.markov_predictor.load(str(base / "markov"))
        return obj

    def __enter__(self) -> TurboQuantMoE:
        return self

    def __exit__(self, *_: Any) -> None:
        with self._lock:
            if self.pid_controller is not None:
                self.pid_controller.stop_background()
            for key in list(self.expert_cache._gpu_experts):
                layer_id, expert_id = key
                self.expert_cache.evict(expert_id=expert_id, layer_id=layer_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __repr__(self) -> str:
        rep = self.memory_report()
        return (
            "TurboQuantMoE("
            f"kv_ratio={rep.kv_compression_ratio:.3f}, "
            f"expert_hit_rate={rep.expert_hit_rate:.3f}, "
            f"saved_mb={rep.total_saved_mb:.1f}, "
            f"saved_pct={rep.total_saved_percent:.1f})"
        )

    def _detect_model_type(self, model: nn.Module) -> str:
        configured = self.config.model_type.lower()
        if configured != "auto":
            return configured
        model_type = getattr(getattr(model, "config", None), "model_type", "auto")
        return str(model_type).lower()

    def _is_moe_module(self, module: nn.Module) -> bool:
        attrs = set(dir(module))
        has_experts = "experts" in attrs
        has_router = "gate" in attrs or "router" in attrs
        return has_experts and has_router

    def _register_module_experts(self, layer_id: int, module: nn.Module) -> None:
        if not hasattr(module, "experts"):
            return
        experts = cast(Iterable[Any], module.experts)
        for expert_id, expert in enumerate(experts):
            if not isinstance(expert, nn.Module):
                continue
            weights = {
                name: p.detach() for name, p in expert.state_dict().items() if p.is_floating_point()
            }
            if not weights:
                continue
            try:
                self.expert_cache.register_expert(
                    expert_id=expert_id, layer_id=layer_id, weights=weights
                )
            except ValueError:
                continue

    def _extract_router_logits(
        self, module: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        gate = getattr(module, "gate", None)
        if isinstance(gate, nn.Module):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            return cast(torch.Tensor, gate(flat))
        router = getattr(module, "router", None)
        if isinstance(router, nn.Module):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            return cast(torch.Tensor, router(flat))
        return None


def _to_json_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    data = asdict(obj)
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.dtype):
            out[key] = str(value).replace("torch.", "")
        else:
            out[key] = value
    return out


def _parse_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        candidate = value.replace("torch.", "")
        if hasattr(torch, candidate):
            attr = getattr(torch, candidate)
            if isinstance(attr, torch.dtype):
                return attr
    return torch.float16
