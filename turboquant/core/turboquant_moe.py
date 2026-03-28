"""Unified TurboQuant manager for MoE inference optimization."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import structlog
import torch
import torch.nn as nn

from turboquant.core.expert_predictor import ExpertPredictor, ExpertPredictorConfig
from turboquant.core.moe_expert_cache import DynamicExpertCache, ExpertCacheConfig
from turboquant.core.moe_router import MoERouterOptimizer, RouterOptimizerConfig, RouterOutput
from turboquant.core.polar_quant import PolarQuantConfig, PolarQuantizer
from turboquant.core.turboquant import CacheEntry, TurboQuantConfig, TurboQuantKVCache

LOGGER = structlog.get_logger(__name__)


@dataclass
class TurboQuantMoEConfig:
    """Composite config for full TurboQuant MoE pipeline."""

    kv_config: TurboQuantConfig
    expert_config: ExpertCacheConfig
    router_config: RouterOptimizerConfig
    predictor_config: ExpertPredictorConfig | None = None
    model_type: str = "auto"
    enable_kv_quant: bool = True
    enable_expert_cache: bool = True
    enable_router_opt: bool = True
    enable_expert_prediction: bool = True
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
            getattr(model_config, "num_attention_heads", getattr(model_config, "num_key_value_heads", 32))
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
        predictor_cfg = ExpertPredictorConfig(
            hidden_dim=hidden_size,
            num_experts=num_experts,
            num_layers=num_layers,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        return cls(
            kv_config=kv_cfg,
            expert_config=expert_cfg,
            router_config=router_cfg,
            predictor_config=predictor_cfg,
            model_type=model_type,
        )


@dataclass
class MoEStepOutput:
    """Output of a single unified MoE step."""

    cache_entry: CacheEntry
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

        pq_cfg = PolarQuantConfig(
            head_dim=config.kv_config.head_dim,
            bits=config.kv_config.bits,
            group_size=config.kv_config.group_size,
            seed=config.kv_config.seed,
            use_hadamard=config.kv_config.use_hadamard,
        )
        self.quantizer = PolarQuantizer(pq_cfg)

        self.kv_cache = TurboQuantKVCache(config.kv_config)
        self.expert_cache = DynamicExpertCache(config.expert_config, quantizer=self.quantizer)
        self.router = MoERouterOptimizer(config.router_config)

        self.predictor: ExpertPredictor | None = None
        if (
            config.enable_expert_prediction
            and config.predictor_config is not None
        ):
            self.predictor = ExpertPredictor(config.predictor_config)

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
                    keys = torch.zeros((batch, heads, seq, head_dim), dtype=torch.float16, device=device)
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
            before = self.memory_report().total_saved_mb

            predicted_experts: list[int] = []
            if self.predictor is not None and self.config.enable_expert_prediction:
                predicted_experts = self.predictor.predict_experts(hidden_states, layer_id)
                if predicted_experts:
                    self.expert_cache.prefetch_experts(predicted_experts, layer_id, priority=1.0)

            router_out = self.router(router_logits=router_logits, training=False)
            active_experts = sorted({int(x) for x in router_out.expert_indices.flatten().tolist()})

            for expert_id in active_experts:
                try:
                    self.expert_cache.get_expert(expert_id=expert_id, layer_id=layer_id)
                except torch.cuda.OutOfMemoryError:
                    self.expert_cache.evict(expert_id=expert_id, layer_id=layer_id)

            try:
                cache_entry = self.kv_cache.compress(keys, values)
            except torch.cuda.OutOfMemoryError:
                keys_cpu = keys.detach().cpu()
                values_cpu = values.detach().cpu()
                cache_entry = self.kv_cache.compress(keys_cpu, values_cpu)

            if self.predictor is not None and self.config.enable_expert_prediction:
                self.predictor.update_history(layer_id, active_experts)
                if self._step_counter % 10 == 0:
                    self.predictor.online_update(hidden_states, layer_id, active_experts)

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            after = self.memory_report().total_saved_mb
            prediction_ok = set(predicted_experts) >= set(active_experts) if predicted_experts else False

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

            return MoEStepOutput(
                cache_entry=cache_entry,
                router_output=router_out,
                active_experts=active_experts,
                predicted_experts=predicted_experts,
                prediction_was_correct=prediction_ok,
                memory_delta_mb=after - before,
                step_latency_ms=latency_ms,
            )

    def memory_report(self) -> MemoryReport:
        """Collect current memory report across all managed components."""
        dummy = torch.zeros(
            (1, self.config.kv_config.num_heads, 1, self.config.kv_config.head_dim),
            dtype=torch.float16,
            device=self.kv_cache.device,
        )
        entry = self.kv_cache.compress(dummy, dummy)
        kv_mem = self.kv_cache.memory_usage(entry)

        ex_stats = self.expert_cache.stats()
        expert_baseline = ex_stats.gpu_memory_used_mb + ex_stats.gpu_memory_saved_mb
        expert_used = ex_stats.gpu_memory_used_mb + ex_stats.cpu_memory_used_mb
        expert_ratio = expert_used / max(1e-8, expert_baseline)

        total_saved = (kv_mem["fp16_baseline_mb"] - kv_mem["total_mb"]) + ex_stats.gpu_memory_saved_mb
        total_baseline = kv_mem["fp16_baseline_mb"] + expert_baseline

        return MemoryReport(
            kv_cache_mb=kv_mem["total_mb"],
            kv_cache_fp16_baseline_mb=kv_mem["fp16_baseline_mb"],
            kv_compression_ratio=kv_mem["compression_ratio"],
            expert_gpu_mb=ex_stats.gpu_memory_used_mb,
            expert_cpu_mb=ex_stats.cpu_memory_used_mb,
            expert_total_baseline_mb=expert_baseline,
            expert_compression_ratio=expert_ratio,
            total_saved_mb=total_saved,
            total_saved_percent=100.0 * total_saved / max(1e-8, total_baseline),
            expert_hit_rate=ex_stats.hit_rate,
            prefetch_accuracy=ex_stats.avg_prefetch_accuracy,
        )

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
            "predictor_config": _to_json_dict(self.config.predictor_config) if self.config.predictor_config else None,
            "model_type": self.config.model_type,
            "enable_kv_quant": self.config.enable_kv_quant,
            "enable_expert_cache": self.config.enable_expert_cache,
            "enable_router_opt": self.config.enable_router_opt,
            "enable_expert_prediction": self.config.enable_expert_prediction,
            "profile_mode": self.config.profile_mode,
        }
        (base / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        self.expert_cache.save_state(str(base / "expert_cache"))
        torch.save(self.router.state_dict(), base / "router.pt")
        if self.predictor is not None:
            self.predictor.save(str(base / "predictor"))

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> TurboQuantMoE:
        """Load TurboQuantMoE state from directory path."""
        base = Path(path)
        cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))

        kv_cfg = TurboQuantConfig(**cfg["kv_config"])
        kv_cfg.device = device
        exp_cfg = ExpertCacheConfig(**cfg["expert_config"])
        exp_cfg.device = device
        router_cfg = RouterOptimizerConfig(**cfg["router_config"])

        pred_cfg = None
        if cfg["predictor_config"] is not None:
            pred_cfg = ExpertPredictorConfig(**cfg["predictor_config"])
            pred_cfg.device = device

        tcfg = TurboQuantMoEConfig(
            kv_config=kv_cfg,
            expert_config=exp_cfg,
            router_config=router_cfg,
            predictor_config=pred_cfg,
            model_type=cfg["model_type"],
            enable_kv_quant=bool(cfg["enable_kv_quant"]),
            enable_expert_cache=bool(cfg["enable_expert_cache"]),
            enable_router_opt=bool(cfg["enable_router_opt"]),
            enable_expert_prediction=bool(cfg["enable_expert_prediction"]),
            profile_mode=bool(cfg["profile_mode"]),
        )

        obj = cls(tcfg)
        obj.expert_cache.load_state(str(base / "expert_cache"))
        obj.router.load_state_dict(torch.load(base / "router.pt", map_location="cpu"))
        if obj.predictor is not None and (base / "predictor").exists():
            obj.predictor.load(str(base / "predictor"))
        return obj

    def __enter__(self) -> TurboQuantMoE:
        return self

    def __exit__(self, *_: Any) -> None:
        with self._lock:
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
            weights = {name: p.detach() for name, p in expert.state_dict().items() if p.is_floating_point()}
            if not weights:
                continue
            try:
                self.expert_cache.register_expert(expert_id=expert_id, layer_id=layer_id, weights=weights)
            except ValueError:
                continue

    def _extract_router_logits(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor | None:
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
