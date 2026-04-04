# Copyright (c) 2026 Denis Remizov. Licensed under BUSL-1.1.
# See LICENSE file for details.


"""Transformers integration for TurboQuant and TurboQuant-MoE.

Example:
    >>> from transformers import AutoModelForCausalLM
    >>> from turboquant.integrations.transformers import auto_config, patch_moe_model
    >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    >>> config = auto_config(model, bits=3, gpu_cache_experts=4)
    >>> model = patch_moe_model(model, config)
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.util import find_spec
from typing import Any, cast

import structlog
import torch

from turboquant.core.moe_router import RouterOutput
from turboquant.core.turboquant import (
    AdaptiveCompressedCache,
    CacheEntry,
    TurboQuantConfig,
    TurboQuantKVCache,
)
from turboquant.core.turboquant_moe import TurboQuantMoE, TurboQuantMoEConfig

LOGGER = structlog.get_logger(__name__)

HAS_TRANSFORMERS = find_spec("transformers") is not None


def _require_transformers() -> None:
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers>=4.40.0 is required. Install with: pip install turboquant[transformers]"
        )


class TurboQuantCache:
    """Cache implementation compatible with HuggingFace DynamicCache API."""

    def __init__(self, tq_cache: TurboQuantKVCache) -> None:
        super().__init__()
        self.tq_cache = tq_cache
        self.entries: dict[int, CacheEntry | AdaptiveCompressedCache] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if layer_idx in self.entries:
            self.entries[layer_idx] = self.tq_cache.update(
                self.entries[layer_idx], key_states, value_states
            )
        else:
            self.entries[layer_idx] = self.tq_cache.compress(key_states, value_states)
        return self.tq_cache.decompress(self.entries[layer_idx])

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self.entries:
            return 0
        return int(self.entries[layer_idx].metadata.get("seq_len", 0))

    def get_max_length(self) -> int | None:
        return self.tq_cache.config.max_seq_len

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for layer_idx, entry in list(self.entries.items()):
            keys, values = self.tq_cache.decompress(entry)
            keys = keys.index_select(0, beam_idx.to(keys.device))
            values = values.index_select(0, beam_idx.to(values.device))
            self.entries[layer_idx] = self.tq_cache.compress(keys, values)


class TurboQuantMoECache(TurboQuantCache):
    """TurboQuant cache variant that also stores router outputs per layer."""

    def __init__(self, tq_cache: TurboQuantKVCache) -> None:
        super().__init__(tq_cache)
        self.router_outputs: dict[int, RouterOutput] = {}

    def update_router(self, layer_idx: int, router_output: RouterOutput) -> None:
        self.router_outputs[layer_idx] = router_output


def patch_model(model: Any, config: TurboQuantConfig) -> Any:
    """Patch non-MoE model with TurboQuant KV cache only."""
    _require_transformers()
    patched_model = copy.copy(model)
    patched_any = cast(Any, patched_model)

    tq_cache = TurboQuantKVCache(config)
    patched_any._turboquant_cache = TurboQuantCache(tq_cache)

    original_prepare = getattr(model, "prepare_inputs_for_generation", None)
    patched_any._turboquant_original_prepare = original_prepare

    if callable(original_prepare):

        def _prepare_inputs_for_generation(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("past_key_values", patched_any._turboquant_cache)
            return original_prepare(*args, **kwargs)

        patched_any.prepare_inputs_for_generation = _prepare_inputs_for_generation

    model_type = str(getattr(getattr(model, "config", None), "model_type", "unknown"))
    attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
    LOGGER.info("patch_model", model_type=model_type, attn_implementation=attn_impl)
    return patched_model


def patch_moe_model(model: Any, moe_config: TurboQuantMoEConfig) -> Any:
    """Patch MoE model with KV quantization + expert cache + router optimization."""
    _require_transformers()
    patched_model = copy.copy(model)
    patched_any = cast(Any, patched_model)

    manager = TurboQuantMoE(moe_config)
    model_type = str(getattr(getattr(model, "config", None), "model_type", "unknown")).lower()
    supported = {"mixtral", "deepseek_v2", "qwen2_moe", "olmoe", "arctic"}

    if any(name in model_type for name in supported):
        manager.wrap_model(patched_model)
    else:
        warnings.warn(
            f"Unknown MoE architecture '{model_type}', applying KV patch only.",
            stacklevel=2,
        )

    patched_any._turboquant_moe = manager
    patched_any._turboquant_cache = TurboQuantMoECache(manager.kv_cache)

    original_prepare = getattr(patched_model, "prepare_inputs_for_generation", None)
    patched_any._turboquant_original_prepare = original_prepare
    if callable(original_prepare):

        def _prepare_inputs_for_generation(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("past_key_values", patched_any._turboquant_cache)
            return original_prepare(*args, **kwargs)

        patched_any.prepare_inputs_for_generation = _prepare_inputs_for_generation

    return patched_model


def unpatch_model(model: Any) -> Any:
    """Restore original non-MoE model behavior."""
    original_prepare = getattr(model, "_turboquant_original_prepare", None)
    if callable(original_prepare):
        cast(Any, model).prepare_inputs_for_generation = original_prepare
    for attr in ["_turboquant_cache", "_turboquant_original_prepare"]:
        if hasattr(model, attr):
            delattr(model, attr)
    return model


def unpatch_moe_model(model: Any) -> Any:
    """Restore original MoE model behavior."""
    manager: TurboQuantMoE | None = getattr(model, "_turboquant_moe", None)
    if manager is not None:
        manager.unwrap_model(model)
    return unpatch_model(model)


@contextmanager
def turboquant_inference(
    model: Any,
    config: TurboQuantConfig,
) -> Iterator[Any]:
    """Context manager for temporary KV patching."""
    patched = patch_model(model, config)
    try:
        yield patched
    finally:
        unpatch_model(patched)


@contextmanager
def turboquant_moe_inference(
    model: Any,
    moe_config: TurboQuantMoEConfig,
) -> Iterator[Any]:
    """Context manager for temporary MoE patching."""
    patched = patch_moe_model(model, moe_config)
    try:
        yield patched
    finally:
        unpatch_moe_model(patched)


def auto_config(
    model: Any,
    bits: int = 3,
    gpu_cache_experts: int = 4,
) -> TurboQuantMoEConfig:
    """Create automatic TurboQuantMoEConfig from a pretrained model."""
    _require_transformers()

    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model has no config attribute")

    auto = TurboQuantMoEConfig.from_pretrained_config(
        cfg,
        gpu_cache_size=gpu_cache_experts,
        bits=bits,
    )

    if torch.cuda.is_available():
        free_bytes, _total = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
        if free_gb < 16:
            auto.expert_config.gpu_cache_size = min(auto.expert_config.gpu_cache_size, 2)
        elif free_gb < 24:
            auto.expert_config.gpu_cache_size = min(auto.expert_config.gpu_cache_size, 4)

    return auto
