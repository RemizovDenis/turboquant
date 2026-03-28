"""vLLM integration for TurboQuant-MoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import structlog
import torch
import torch.nn.functional as functional

from turboquant.core.turboquant import TurboQuantConfig, TurboQuantKVCache
from turboquant.core.turboquant_moe import TurboQuantMoE, TurboQuantMoEConfig

LOGGER = structlog.get_logger(__name__)


@dataclass
class TurboQuantKVCacheConfig:
    """Block-level config for vLLM KV cache."""

    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    tq_config: TurboQuantConfig
    block_size: int = 16


class TurboQuantKVCacheOp:
    """Block operations over compressed KV blocks."""

    def __init__(self, config: TurboQuantKVCacheConfig) -> None:
        self.config = config
        self.kv = TurboQuantKVCache(config.tq_config)

    def swap_blocks(
        self,
        src: dict[int, Any],
        dst: dict[int, Any],
        block_mapping: dict[int, int],
    ) -> None:
        for src_idx, dst_idx in block_mapping.items():
            dst[dst_idx] = src[src_idx]
            del src[src_idx]

    def copy_blocks(
        self,
        kv_caches: dict[int, Any],
        block_mapping: dict[int, int],
    ) -> None:
        for src_idx, dst_idx in block_mapping.items():
            kv_caches[dst_idx] = kv_caches[src_idx]

    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: dict[int, Any],
        slot_mapping: dict[int, int],
    ) -> None:
        for token_idx, block_idx in slot_mapping.items():
            key_token = key[:, :, token_idx : token_idx + 1, :]
            val_token = value[:, :, token_idx : token_idx + 1, :]
            kv_cache[block_idx] = self.kv.compress(key_token, val_token)


class TurboQuantPagedAttention:
    """Paged-attention wrapper with on-demand block decompression."""

    def __init__(self, kv_op: TurboQuantKVCacheOp) -> None:
        self.kv_op = kv_op

    def forward(
        self,
        query: torch.Tensor,
        key_cache: dict[int, Any],
        value_cache: dict[int, Any],
        block_indices: list[int],
        causal: bool = True,
    ) -> torch.Tensor:
        keys = []
        values = []
        for idx in block_indices:
            key_entry = key_cache[idx]
            val_entry = value_cache[idx]
            k, _ = self.kv_op.kv.decompress(key_entry)
            v, _ = self.kv_op.kv.decompress(val_entry)
            keys.append(k)
            values.append(v)

        k_all = torch.cat(keys, dim=2)
        v_all = torch.cat(values, dim=2)
        out = functional.scaled_dot_product_attention(query, k_all, v_all, is_causal=causal)
        return out.to(torch.float16)


def create_turboquant_llm(
    model: str,
    tq_moe_config: TurboQuantMoEConfig,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 32768,
    tensor_parallel_size: int = 1,
    **vllm_kwargs: Any,
) -> Any:
    """Create vLLM LLM with TurboQuant-MoE hooks while preserving LLM API."""
    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError(
            "vllm>=0.4.0 is required. Install with: pip install turboquant[vllm]"
        ) from exc

    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        **vllm_kwargs,
    )
    llm_any = cast(Any, llm)
    llm_any._turboquant_moe = TurboQuantMoE(tq_moe_config)
    llm_any._turboquant_metrics = {
        "turboquant_block_compression_ratio": 0.0,
        "turboquant_expert_hit_rate": 0.0,
        "turboquant_prefetch_accuracy": 0.0,
    }
    LOGGER.info("create_turboquant_llm", model=model, tp=tensor_parallel_size)
    return llm


class TurboQuantWorkerMixin:
    """Mixin adding expert prefetch hooks to vLLM worker lifecycle."""

    def init_model(self, *args: Any, **kwargs: Any) -> Any:
        parent = super()
        init_fn = getattr(parent, "init_model", None)
        result = init_fn(*args, **kwargs) if callable(init_fn) else None
        manager: TurboQuantMoE | None = getattr(self, "_turboquant_moe", None)
        if manager is not None and hasattr(self, "model"):
            manager.wrap_model(self.model)
        return result

    def execute_model(self, *args: Any, **kwargs: Any) -> Any:
        manager: TurboQuantMoE | None = getattr(self, "_turboquant_moe", None)
        if manager is not None and manager.predictor is not None:
            manager.expert_cache.prefetch_experts([0], layer_id=0, priority=1.0)
        parent = super()
        exec_fn = getattr(parent, "execute_model", None)
        result = exec_fn(*args, **kwargs) if callable(exec_fn) else None
        if manager is not None and manager.predictor is not None:
            manager.predictor.update_history(layer_id=0, activated_experts=[0])
        return result
