"""HuggingFace Transformers integration for TurboQuant KV-cache compression.

Provides a drop-in replacement for the HuggingFace ``DynamicCache`` that
transparently compresses KV-cache using TurboQuant.

Quick start::

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquant.integrations.transformers import patch_model
    from turboquant.core.turboquant import TurboQuantConfig

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3-8B", torch_dtype=torch.float16
    )
    config = TurboQuantConfig(head_dim=128, num_heads=32)
    model = patch_model(model, config)
    # Done. The model now uses TurboQuant KV-cache.
"""

from __future__ import annotations

import copy
import warnings
from contextlib import contextmanager
from typing import Any

import structlog
import torch

from turboquant.core.turboquant import CacheEntry, TurboQuantConfig, TurboQuantKVCache

log = structlog.get_logger(__name__)

try:
    from transformers import PreTrainedModel  # type: ignore[import-untyped]

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    PreTrainedModel = object  # type: ignore[assignment,misc]


def _check_transformers() -> None:
    """Raise if transformers is not installed."""
    if not _HAS_TRANSFORMERS:
        raise ImportError(
            "transformers>=4.40.0 is required for this integration. "
            "Install with: pip install turboquant[transformers]"
        )


# ======================================================================
# TurboQuantCache — drop-in replacement for DynamicCache
# ======================================================================


class TurboQuantCache:
    """Drop-in replacement for ``transformers.DynamicCache`` that transparently
    compresses KV-cache via TurboQuant.

    Each layer stores a ``CacheEntry`` instead of raw FP16 key/value tensors.

    Attributes:
        tq: Underlying ``TurboQuantKVCache`` instance.
        _entries: Per-layer list of ``CacheEntry`` objects.
        _seen_tokens: Running count of processed tokens (per ``DynamicCache`` API).
    """

    def __init__(self, tq: TurboQuantKVCache, num_layers: int | None = None) -> None:
        """Initialise TurboQuantCache.

        Args:
            tq: Configured ``TurboQuantKVCache``.
            num_layers: Number of transformer layers. When *None*, entries
                are created lazily on first ``update`` call.
        """
        self.tq = tq
        self._entries: list[CacheEntry | None] = (
            [None] * num_layers if num_layers else []
        )
        self._seen_tokens: int = 0

    # ---- DynamicCache interface ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the cache for *layer_idx* with new key/value states.

        Compresses the incoming KV slice, stores it, and returns the full
        decompressed cache for attention computation.

        Args:
            key_states: ``[batch, num_heads, new_seq, head_dim]``.
            value_states: Same shape.
            layer_idx: Transformer layer index.
            cache_kwargs: Unused (kept for API compatibility).

        Returns:
            ``(full_keys, full_values)`` decompressed as FP16.
        """
        # Expand entries list if needed
        while len(self._entries) <= layer_idx:
            self._entries.append(None)

        existing = self._entries[layer_idx]
        if existing is None:
            entry = self.tq.compress(key_states, value_states)
            self._entries[layer_idx] = entry
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[2]
        else:
            entry = self.tq.update(existing, key_states, value_states)
            self._entries[layer_idx] = entry
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[2]

        # Return full decompressed cache for attention
        return self.tq.decompress(entry)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the current sequence length stored."""
        if layer_idx < len(self._entries) and self._entries[layer_idx] is not None:
            return int(self._entries[layer_idx].metadata.get("seq_len", 0))  # type: ignore[union-attr]
        return 0

    def get_max_length(self) -> int | None:
        """Return the maximum sequence length (budget from config)."""
        return self.tq.config.max_seq_len

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder cache entries for beam search.

        Re-decompresses, reorders along batch dim, re-compresses.

        Args:
            beam_idx: Index tensor for beam reordering.
        """
        for i, entry in enumerate(self._entries):
            if entry is not None:
                keys, values = self.tq.decompress(entry)
                keys = keys.index_select(0, beam_idx.to(keys.device))
                values = values.index_select(0, beam_idx.to(values.device))
                self._entries[i] = self.tq.compress(keys, values)

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """Return how much of the cache can be used (for ``DynamicCache`` compat)."""
        return self.get_seq_length(layer_idx)

    @property
    def seen_tokens(self) -> int:
        """Total tokens processed."""
        return self._seen_tokens

    def __len__(self) -> int:
        """Number of layers with cached data."""
        return sum(1 for e in self._entries if e is not None)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return decompressed (key, value) for a layer."""
        entry = self._entries[layer_idx]
        if entry is None:
            raise KeyError(f"No cache entry for layer {layer_idx}")
        return self.tq.decompress(entry)

    def __iter__(self):
        """Iterate over (key, value) tuples for each cached layer."""
        for entry in self._entries:
            if entry is not None:
                yield self.tq.decompress(entry)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert to legacy HuggingFace cache format."""
        result = []
        for entry in self._entries:
            if entry is not None:
                result.append(self.tq.decompress(entry))
            else:
                result.append(
                    (torch.empty(0), torch.empty(0))
                )
        return tuple(result)


# ======================================================================
# Model patching
# ======================================================================

# Attention class names supported for monkey-patching
_SUPPORTED_ATTENTION_CLASSES = {
    "llama": "LlamaAttention",
    "mistral": "MistralAttention",
    "qwen2": "Qwen2Attention",
    "gemma": "GemmaAttention",
    "gemma2": "Gemma2Attention",
    "phi3": "Phi3Attention",
}


def _detect_model_type(model: PreTrainedModel) -> str | None:
    """Auto-detect the HuggingFace model type from ``model.config``."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return model_type


def patch_model(
    model: PreTrainedModel,
    config: TurboQuantConfig,
    inplace: bool = False,
) -> PreTrainedModel:
    """Patch a HuggingFace model to use TurboQuant KV-cache.

    Creates a ``TurboQuantCache`` and wraps each attention layer's forward
    so that ``past_key_value`` uses the compressed cache.

    Args:
        model: A ``PreTrainedModel`` (e.g. from ``AutoModelForCausalLM``).
        config: TurboQuant configuration.
        inplace: If *False* (default), patches a deep copy of the model.

    Returns:
        The patched model. The original is not modified unless *inplace=True*.

    Note:
        If the model architecture is not in the supported list, a warning is
        emitted and the model is returned unmodified (graceful degradation).
    """
    _check_transformers()

    model_type = _detect_model_type(model)
    if model_type is None or model_type not in _SUPPORTED_ATTENTION_CLASSES:
        supported = ", ".join(sorted(_SUPPORTED_ATTENTION_CLASSES.keys()))
        warnings.warn(
            f"Model type '{model_type}' is not explicitly supported. "
            f"Supported: {supported}. Returning model unmodified.",
            stacklevel=2,
        )
        return model

    if not inplace:
        model = copy.deepcopy(model)

    tq = TurboQuantKVCache(config)

    # Detect number of layers
    num_layers = getattr(model.config, "num_hidden_layers", None)
    cache = TurboQuantCache(tq, num_layers=num_layers)

    # Store references on the model for later retrieval / unpatching
    model._turboquant_cache = cache  # type: ignore[attr-defined]
    model._turboquant_config = config  # type: ignore[attr-defined]
    model._turboquant_original_forward = {}  # type: ignore[attr-defined]

    # Monkey-patch attention layers
    attn_class_name = _SUPPORTED_ATTENTION_CLASSES[model_type]
    patched = 0
    for name, module in model.named_modules():
        if type(module).__name__ == attn_class_name:
            _patch_attention_module(model, name, module, cache)
            patched += 1

    log.info(
        "patch_model",
        model_type=model_type,
        attn_class=attn_class_name,
        layers_patched=patched,
    )

    return model


def _patch_attention_module(
    model: PreTrainedModel,
    name: str,
    module: torch.nn.Module,
    cache: TurboQuantCache,
) -> None:
    """Patch a single attention module to use ``TurboQuantCache``.

    We override the module's ``forward`` to inject ``past_key_value=cache``.
    """
    original_forward = module.forward

    def patched_forward(*args: Any, **kwargs: Any) -> Any:
        # Inject our cache if the caller didn't supply one
        if "past_key_value" not in kwargs or kwargs["past_key_value"] is None or kwargs.get("use_cache", True):
            kwargs["past_key_value"] = cache
        return original_forward(*args, **kwargs)

    model._turboquant_original_forward[name] = original_forward  # type: ignore[attr-defined]
    module.forward = patched_forward  # type: ignore[assignment]


def unpatch_model(model: PreTrainedModel) -> PreTrainedModel:
    """Restore a patched model to its original behaviour.

    Args:
        model: Model previously patched with ``patch_model(…, inplace=True)``.

    Returns:
        The model with original forwards restored.
    """
    _check_transformers()

    originals: dict[str, Any] = getattr(model, "_turboquant_original_forward", {})
    if not originals:
        warnings.warn("Model does not appear to be patched by TurboQuant.", stacklevel=2)
        return model

    for name, original_forward in originals.items():
        parts = name.split(".")
        mod = model
        for p in parts:
            mod = getattr(mod, p)
        mod.forward = original_forward  # type: ignore[assignment]

    # Clean up references
    for attr in ("_turboquant_cache", "_turboquant_config", "_turboquant_original_forward"):
        if hasattr(model, attr):
            delattr(model, attr)

    log.info("unpatch_model", layers_restored=len(originals))
    return model


# ======================================================================
# Context manager
# ======================================================================


@contextmanager
def turboquant_inference(
    model: PreTrainedModel,
    config: TurboQuantConfig,
):
    """Context manager that temporarily patches a model with TurboQuant.

    Usage::

        with turboquant_inference(model, config) as tq_model:
            outputs = tq_model.generate(input_ids, max_new_tokens=100)

    Args:
        model: HuggingFace ``PreTrainedModel``.
        config: TurboQuant configuration.

    Yields:
        The patched model. The model is unpatched on exit.
    """
    _check_transformers()
    patched = patch_model(model, config, inplace=True)
    try:
        yield patched
    finally:
        unpatch_model(patched)
