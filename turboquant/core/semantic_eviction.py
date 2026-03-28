"""Semantic KV eviction based on token importance estimation."""

from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import structlog
import torch
import torch.nn as nn

LOGGER = structlog.get_logger(__name__)


def _maybe_compile(fn: Any) -> Any:
    """Compile callable with `torch.compile` when available."""
    if not torch.cuda.is_available():
        return fn
    if hasattr(torch, "compile"):
        try:
            return torch.compile(fn, dynamic=True, backend="eager")
        except Exception:  # pragma: no cover - backend dependent
            return fn
    return fn


@dataclass
class SemanticEvictionConfig:
    """Configuration for semantic KV eviction."""

    max_seq_len: int = 131072
    eviction_target_len: int = 65536
    scorer_hidden_dim: int = 64
    scorer_num_layers: int = 2
    importance_threshold: float = 0.1
    sink_token_count: int = 4
    recent_token_count: int = 256
    eviction_chunk_size: int = 1024
    online_update_freq: int = 32
    online_lr: float = 1e-4
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    use_attention_history: bool = True
    history_window: int = 8


@dataclass
class EvictionResult:
    """Output of semantic token selection."""

    kept_indices: torch.Tensor
    evicted_indices: torch.Tensor
    importance_scores: torch.Tensor
    sink_tokens_kept: int
    recent_tokens_kept: int
    semantic_tokens_kept: int
    eviction_ratio: float


class ImportanceScorer(nn.Module):
    """Lightweight model that predicts token importance in `[0, 1]`."""

    def __init__(self, head_dim: int, config: SemanticEvictionConfig) -> None:
        """Initialize scorer.

        Args:
            head_dim: Attention head dimension.
            config: Semantic eviction configuration.
        """
        super().__init__()
        self.config = config
        self.head_dim = head_dim

        self.attention_stats_encoder = nn.Linear(config.history_window, 32)
        self.kv_encoder = nn.Linear(head_dim * 2, 32)
        self.combiner = nn.Sequential(
            nn.Linear(64, config.scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.scorer_hidden_dim, config.scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.scorer_hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _positional_encoding(positions: torch.Tensor, dim: int = 32) -> torch.Tensor:
        """Build sinusoidal position encoding."""
        freqs = torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32)
        freqs = torch.exp(-math.log(10000.0) * (freqs / max(1, dim)))
        phase = positions.float().unsqueeze(1) * freqs.unsqueeze(0)
        out = torch.zeros((positions.shape[0], dim), device=positions.device, dtype=torch.float32)
        out[:, 0::2] = torch.sin(phase)
        out[:, 1::2] = torch.cos(phase)
        return out

    @_maybe_compile
    def forward(
        self,
        kv_repr: torch.Tensor,
        attention_history: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token importance.

        Args:
            kv_repr: Concatenated K/V representation `[seq_len, 2*head_dim]`.
            attention_history: Optional attention history `[seq_len, history_window]`.
            positions: Token positions `[seq_len]`.

        Returns:
            Importance scores `[seq_len]` in range `[0, 1]`.
        """
        kv_feat = self.kv_encoder(kv_repr.float())
        if attention_history is None:
            attn_feat = self._positional_encoding(positions, dim=32)
        else:
            attn_feat = self.attention_stats_encoder(attention_history.float())
        fused = torch.cat([kv_feat, attn_feat], dim=-1)
        logits = self.combiner(fused).squeeze(-1)
        return torch.sigmoid(logits).float()


class SemanticKVEviction:
    """Semantic token eviction engine for long-context KV cache."""

    def __init__(
        self,
        config: SemanticEvictionConfig,
        head_dim: int,
        num_heads: int,
    ) -> None:
        """Initialize eviction pipeline.

        Args:
            config: Semantic eviction settings.
            head_dim: Head dimension for KV tensors.
            num_heads: Number of attention heads.
        """
        self.config = config
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.device = torch.device(config.device)

        self.scorer = ImportanceScorer(head_dim=head_dim, config=config).to(self.device)
        self.online_optimizer = torch.optim.AdamW(self.scorer.parameters(), lr=config.online_lr)

        self.attention_history: dict[int, deque[torch.Tensor]] = defaultdict(
            lambda: deque(maxlen=config.history_window)
        )
        self.scorer_loss_history: deque[float] = deque(maxlen=100)
        self._lock = threading.RLock()
        self._logger = LOGGER.bind(component="SemanticKVEviction")
        self.eviction_count = 0
        self.tokens_saved = 0
        self._kept_score_sum = 0.0
        self._evicted_score_sum = 0.0
        self._kept_score_count = 0
        self._evicted_score_count = 0
        self._online_steps = 0

    def _layer_history_tensor(
        self, layer_id: int, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        with self._lock:
            raw = list(self.attention_history.get(layer_id, []))
        if not raw:
            return torch.zeros(
                (seq_len, self.config.history_window),
                device=device,
                dtype=torch.float32,
            )

        cols: list[torch.Tensor] = []
        for vec in raw[-self.config.history_window :]:
            if vec.numel() >= seq_len:
                col = vec[-seq_len:]
            else:
                pad = torch.zeros((seq_len - vec.numel(),), dtype=vec.dtype, device=vec.device)
                col = torch.cat([pad, vec], dim=0)
            cols.append(col.to(device=device, dtype=torch.float32))

        while len(cols) < self.config.history_window:
            cols.insert(0, torch.zeros((seq_len,), dtype=torch.float32, device=device))

        return torch.stack(cols, dim=-1)

    def compute_importance(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Compute token importance scores for current KV block."""
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys/values must have shape [batch, heads, seq_len, head_dim]")

        keys = keys.to(self.device)
        values = values.to(self.device)
        mean_k = keys.float().mean(dim=(0, 1))
        mean_v = values.float().mean(dim=(0, 1))
        kv_repr = torch.cat([mean_k, mean_v], dim=-1).to(dtype=torch.float16)

        seq_len = kv_repr.shape[0]
        positions = torch.arange(seq_len, device=self.device, dtype=torch.int64)
        history = None
        if self.config.use_attention_history:
            history = self._layer_history_tensor(layer_id, seq_len, self.device)
        scores = cast(torch.Tensor, self.scorer(kv_repr, history, positions))
        return scores.to(torch.float32)

    def select_tokens_to_keep(
        self,
        importance_scores: torch.Tensor,
        current_seq_len: int,
    ) -> EvictionResult:
        """Select token indices to keep according to semantic importance."""
        if current_seq_len <= 0:
            empty = torch.empty((0,), dtype=torch.long, device=importance_scores.device)
            return EvictionResult(
                kept_indices=empty,
                evicted_indices=empty,
                importance_scores=importance_scores,
                sink_tokens_kept=0,
                recent_tokens_kept=0,
                semantic_tokens_kept=0,
                eviction_ratio=0.0,
            )

        seq_len = int(current_seq_len)
        sink_n = min(int(self.config.sink_token_count), seq_len)
        recent_n = min(int(self.config.recent_token_count), max(0, seq_len - sink_n))

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=importance_scores.device)
        if sink_n > 0:
            keep_mask[:sink_n] = True
        if recent_n > 0:
            keep_mask[-recent_n:] = True

        budget = max(0, int(self.config.eviction_target_len) - sink_n - recent_n)
        mid_start = sink_n
        mid_end = max(mid_start, seq_len - recent_n)
        if budget > 0 and mid_end > mid_start:
            mid_idx = torch.arange(mid_start, mid_end, device=importance_scores.device)
            mid_scores = importance_scores[mid_idx]
            top_n = min(budget, int(mid_idx.numel()))
            if top_n > 0:
                _, rel = torch.topk(mid_scores, k=top_n, dim=0)
                chosen = mid_idx[rel]
                threshold = float(self.config.importance_threshold)
                if threshold > 0.0:
                    chosen = chosen[importance_scores[chosen] >= threshold]
                keep_mask[chosen] = True

        kept_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
        evicted_indices = torch.nonzero(~keep_mask, as_tuple=False).flatten()
        semantic_kept = int(max(0, kept_indices.numel() - sink_n - recent_n))
        ratio = float(evicted_indices.numel()) / max(1, seq_len)

        return EvictionResult(
            kept_indices=kept_indices,
            evicted_indices=evicted_indices,
            importance_scores=importance_scores,
            sink_tokens_kept=sink_n,
            recent_tokens_kept=recent_n,
            semantic_tokens_kept=semantic_kept,
            eviction_ratio=ratio,
        )

    def evict(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, EvictionResult]:
        """Evict low-importance tokens when sequence is over target length."""
        seq_len = int(keys.shape[2]) if keys.ndim == 4 else int(keys.shape[0])
        scores = self.compute_importance(keys, values, layer_id)
        result = self.select_tokens_to_keep(scores, seq_len)
        if seq_len <= int(self.config.eviction_target_len):
            return (keys, values, result)

        kept_keys = keys[:, :, result.kept_indices, :]
        kept_values = values[:, :, result.kept_indices, :]

        kept_scores = scores[result.kept_indices]
        evicted_scores = scores[result.evicted_indices]
        self.eviction_count += 1
        self.tokens_saved += int(result.evicted_indices.numel())
        self._kept_score_sum += float(kept_scores.sum().item())
        self._evicted_score_sum += float(evicted_scores.sum().item())
        self._kept_score_count += int(kept_scores.numel())
        self._evicted_score_count += int(evicted_scores.numel())

        if self.eviction_count % 10 == 0:
            self._logger.info(
                "semantic_eviction",
                eviction_ratio=result.eviction_ratio,
                tokens_saved=self.tokens_saved,
                seq_len=seq_len,
                kept_tokens=int(result.kept_indices.numel()),
            )
        return (kept_keys, kept_values, result)

    def update_attention_history(
        self,
        layer_id: int,
        attention_weights: torch.Tensor,
    ) -> None:
        """Update rolling attention-received history for a layer."""
        if attention_weights.ndim != 4:
            raise ValueError("attention_weights must have shape [batch, heads, seq_len, seq_len]")

        received = attention_weights.float().sum(dim=2).mean(dim=(0, 1))
        with self._lock:
            self.attention_history[layer_id].append(received.detach().cpu())

    def online_update(
        self,
        kept_keys: torch.Tensor,
        kept_values: torch.Tensor,
        future_attention: torch.Tensor,
        layer_id: int,
    ) -> float:
        """Train scorer online on future attention signal.

        Returns:
            Scalar loss value for current update step.
        """
        self._online_steps += 1
        if self._online_steps % int(self.config.online_update_freq) != 0:
            return 0.0

        scores = self.compute_importance(kept_keys, kept_values, layer_id).clamp(1e-6, 1 - 1e-6)
        logits = torch.log(scores / (1.0 - scores))

        if future_attention.ndim == 4:
            target_vec = future_attention.float().sum(dim=2).mean(dim=(0, 1))
        elif future_attention.ndim == 1:
            target_vec = future_attention.float()
        else:
            raise ValueError("future_attention must be [batch, heads, seq, seq] or [seq]")

        if target_vec.numel() >= logits.numel():
            target_vec = target_vec[-logits.numel() :]
        else:
            pad = torch.zeros((logits.numel() - target_vec.numel(),), dtype=torch.float32)
            target_vec = torch.cat([pad, target_vec], dim=0)
        target = (target_vec > target_vec.mean()).to(dtype=torch.float32, device=logits.device)

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, target)
        self.online_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.online_optimizer.step()

        value = float(loss.item())
        self.scorer_loss_history.append(value)
        return value

    def stats(self) -> dict[str, float | int | list[float]]:
        """Return eviction and scorer training metrics."""
        avg_ratio = self.tokens_saved / max(
            1, self.eviction_count * self.config.eviction_target_len
        )
        kept_avg = self._kept_score_sum / max(1, self._kept_score_count)
        evicted_avg = self._evicted_score_sum / max(1, self._evicted_score_count)
        return {
            "total_evictions": self.eviction_count,
            "tokens_saved": self.tokens_saved,
            "avg_eviction_ratio": float(avg_ratio),
            "avg_importance_score_kept": float(kept_avg),
            "avg_importance_score_evicted": float(evicted_avg),
            "scorer_loss_history": list(self.scorer_loss_history),
        }

    def save_scorer(self, path: str) -> None:
        """Save scorer weights and attention history."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            history = {
                layer: [t.tolist() for t in tensors]
                for layer, tensors in self.attention_history.items()
            }
        payload = {
            "state_dict": self.scorer.state_dict(),
            "history": history,
            "loss_history": list(self.scorer_loss_history),
        }
        torch.save(payload, out)

    def load_scorer(self, path: str) -> None:
        """Load scorer checkpoint and continue online updates."""
        payload = torch.load(Path(path), map_location=self.device)
        state_dict = payload["state_dict"]
        self.scorer.load_state_dict(state_dict)
        with self._lock:
            self.attention_history.clear()
            raw_history = payload.get("history", {})
            for key, values in raw_history.items():
                layer = int(key)
                bucket: deque[torch.Tensor] = deque(maxlen=self.config.history_window)
                for vec in values[-self.config.history_window :]:
                    bucket.append(torch.tensor(vec, dtype=torch.float32))
                self.attention_history[layer] = bucket
        self.scorer_loss_history.clear()
        for value in payload.get("loss_history", [])[-100:]:
            self.scorer_loss_history.append(float(value))
