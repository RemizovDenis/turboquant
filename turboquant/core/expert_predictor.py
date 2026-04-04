# Copyright (c) 2026 Denis Remizov. Licensed under BUSL-1.1.
# See LICENSE file for details.


"""Expert activation predictor for proactive MoE prefetching."""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as functional

LOGGER = structlog.get_logger(__name__)


@dataclass
class ExpertPredictorConfig:
    """Configuration for :class:`ExpertPredictor`."""

    hidden_dim: int
    num_experts: int
    num_layers: int
    history_len: int = 8
    predictor_hidden: int = 64
    prediction_threshold: float = 0.6
    online_learning_rate: float = 0.01
    use_layer_embeddings: bool = True
    device: str = "cuda"


class ExpertPredictor(nn.Module):
    """Lightweight predictor of future expert activations for MoE routing."""

    def __init__(self, config: ExpertPredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.device_obj = torch.device(config.device)

        self.layer_embed = (
            nn.Embedding(config.num_layers, 16) if config.use_layer_embeddings else None
        )
        self.routing_history_encoder = nn.Linear(config.num_experts * config.history_len, 64)
        self.hidden_state_encoder = nn.Linear(config.hidden_dim, 64)

        combiner_in = 64 + 64 + (16 if config.use_layer_embeddings else 0)
        self.combiner = nn.Linear(combiner_in, config.predictor_hidden)
        self.output = nn.Linear(config.predictor_hidden, config.num_experts)
        self.act = nn.ReLU()

        self._init_weights()

        self.routing_history: dict[int, deque[list[int]]] = {
            layer_id: deque(maxlen=config.history_len) for layer_id in range(config.num_layers)
        }
        self._state_lock = threading.RLock()

        self.online_optimizer = torch.optim.SGD(self.parameters(), lr=config.online_learning_rate)

        self.prediction_accuracy: deque[float] = deque(maxlen=1000)
        self.precision_history: deque[float] = deque(maxlen=1000)
        self.recall_history: deque[float] = deque(maxlen=1000)
        self._last_prediction: dict[int, set[int]] = {}
        self._step_counter = 0
        self._logger = LOGGER.bind(component="ExpertPredictor")

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, hidden_states: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Predict activation probability for each expert.

        Args:
            hidden_states: `[batch, seq_len, hidden_dim]` or `[batch, hidden_dim]`.
            layer_id: MoE layer index.

        Returns:
            Probabilities tensor `[num_experts]`.
        """
        if hidden_states.ndim == 3:
            pooled = hidden_states.mean(dim=1)
        elif hidden_states.ndim == 2:
            pooled = hidden_states
        else:
            raise ValueError("hidden_states must be 2D or 3D")

        if pooled.shape[-1] != self.config.hidden_dim:
            raise ValueError("hidden dimension mismatch")

        hidden_feat = self.hidden_state_encoder(pooled.float()).mean(dim=0)
        hist_feat = self.routing_history_encoder(self._encode_history(layer_id).float())

        features = [hidden_feat, hist_feat]
        if self.layer_embed is not None:
            layer_idx = int(layer_id)
            if layer_idx < 0 or layer_idx >= self.config.num_layers:
                raise ValueError("layer_id out of range")
            features.append(self.layer_embed.weight[layer_idx])

        combined = torch.cat(features, dim=-1)
        logits = self.output.forward(self.act.forward(self.combiner.forward(combined)))
        probs = torch.sigmoid(logits)
        return probs

    def predict_experts(
        self,
        hidden_states: torch.Tensor,
        layer_id: int,
        threshold: float | None = None,
    ) -> list[int]:
        """Predict expert ids expected to be activated by router."""
        thr = self.config.prediction_threshold if threshold is None else threshold
        with torch.inference_mode():
            probs = self.forward(hidden_states, layer_id)
            chosen = cast(list[int], torch.nonzero(probs > thr, as_tuple=False).flatten().tolist())
            if not chosen:
                min_k = min(2, self.config.num_experts)
                topk = torch.topk(probs, k=min_k)
                chosen = cast(list[int], topk.indices.tolist())

        with self._state_lock:
            self._last_prediction[layer_id] = set(chosen)
        return chosen

    def update_history(self, layer_id: int, activated_experts: list[int]) -> None:
        """Append routing decision and update rolling accuracy metrics."""
        actual_set = set(activated_experts)
        with self._state_lock:
            self.routing_history[layer_id].append(sorted(actual_set))
            pred_set = self._last_prediction.get(layer_id, set())
            if pred_set:
                inter = len(pred_set & actual_set)
                precision = inter / max(1, len(pred_set))
                recall = inter / max(1, len(actual_set))
                exact = 1.0 if pred_set == actual_set else 0.0
                self.prediction_accuracy.append(exact)
                self.precision_history.append(precision)
                self.recall_history.append(recall)

            self._step_counter += 1
            if self._step_counter % 100 == 0:
                acc = self.get_accuracy()
                self._logger.info(
                    "predictor_accuracy",
                    rolling_accuracy=acc["rolling_accuracy"],
                    precision_at_k=acc["precision_at_k"],
                    recall_at_k=acc["recall_at_k"],
                )

    def online_update(
        self,
        hidden_states: torch.Tensor,
        layer_id: int,
        actual_experts: list[int],
    ) -> float:
        """Run one lightweight online SGD step against actual routed experts."""
        target = torch.zeros(
            self.config.num_experts,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        if actual_experts:
            idx = torch.tensor(actual_experts, dtype=torch.long, device=hidden_states.device)
            target.scatter_(0, idx, 1.0)

        logits = self._forward_logits(hidden_states, layer_id)
        probs = torch.sigmoid(logits)
        predicted = set(
            torch.nonzero(probs > self.config.prediction_threshold, as_tuple=False)
            .flatten()
            .tolist()
        )
        actual_set = set(actual_experts)
        reward = 1.0 if predicted == actual_set else 0.0

        loss = functional.binary_cross_entropy_with_logits(logits, target)
        scaled_loss = loss * (2.0 - reward)
        self.online_optimizer.zero_grad(set_to_none=True)
        params = [p for p in self.parameters() if p.requires_grad]
        grads = torch.autograd.grad(scaled_loss, params, allow_unused=True)
        for param, grad in zip(params, grads, strict=False):
            param.grad = grad
        self.online_optimizer.step()
        return float(scaled_loss.detach().item())

    def get_accuracy(self) -> dict[str, float]:
        """Return rolling metrics over recent predictor decisions."""
        with self._state_lock:
            rolling = (
                float(sum(self.prediction_accuracy) / len(self.prediction_accuracy))
                if self.prediction_accuracy
                else 0.0
            )
            precision = (
                float(sum(self.precision_history) / len(self.precision_history))
                if self.precision_history
                else 0.0
            )
            recall = (
                float(sum(self.recall_history) / len(self.recall_history))
                if self.recall_history
                else 0.0
            )
        return {
            "rolling_accuracy": rolling,
            "precision_at_k": precision,
            "recall_at_k": recall,
        }

    def save(self, path: str) -> None:
        """Save model weights, routing history and predictor metrics."""
        dst = Path(path)
        dst.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), dst / "predictor.pt")

        with self._state_lock:
            history_obj = {layer_id: list(hist) for layer_id, hist in self.routing_history.items()}
            stats = {
                "prediction_accuracy": list(self.prediction_accuracy),
                "precision_history": list(self.precision_history),
                "recall_history": list(self.recall_history),
                "step_counter": self._step_counter,
                "history": history_obj,
            }
        (dst / "predictor_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        """Load model state and continue from accumulated history/statistics."""
        src = Path(path)
        state = torch.load(src / "predictor.pt", map_location="cpu")
        self.load_state_dict(state)

        stats = json.loads((src / "predictor_stats.json").read_text(encoding="utf-8"))
        with self._state_lock:
            self.prediction_accuracy = deque(stats.get("prediction_accuracy", []), maxlen=1000)
            self.precision_history = deque(stats.get("precision_history", []), maxlen=1000)
            self.recall_history = deque(stats.get("recall_history", []), maxlen=1000)
            self._step_counter = int(stats.get("step_counter", 0))

            history = stats.get("history", {})
            self.routing_history = {
                layer_id: deque(history.get(str(layer_id), []), maxlen=self.config.history_len)
                for layer_id in range(self.config.num_layers)
            }

    def _forward_logits(self, hidden_states: torch.Tensor, layer_id: int) -> torch.Tensor:
        if hidden_states.ndim == 3:
            pooled = hidden_states.mean(dim=1)
        elif hidden_states.ndim == 2:
            pooled = hidden_states
        else:
            raise ValueError("hidden_states must be 2D or 3D")

        hidden_feat = self.hidden_state_encoder(pooled.float()).mean(dim=0)
        hist_feat = self.routing_history_encoder(self._encode_history(layer_id).float())

        features = [hidden_feat, hist_feat]
        if self.layer_embed is not None:
            layer_idx = int(layer_id)
            if layer_idx < 0 or layer_idx >= self.config.num_layers:
                raise ValueError("layer_id out of range")
            features.append(self.layer_embed.weight[layer_idx])

        combined = torch.cat(features, dim=-1)
        return self.output.forward(self.act.forward(self.combiner.forward(combined)))

    def _encode_history(self, layer_id: int) -> torch.Tensor:
        with self._state_lock:
            history = list(self.routing_history[layer_id])

        vec = torch.zeros(
            (self.config.history_len, self.config.num_experts),
            dtype=torch.float32,
            device=self.device_obj,
        )
        if history:
            recent = history[-self.config.history_len :]
            offset = self.config.history_len - len(recent)
            for idx, experts in enumerate(recent):
                if not experts:
                    continue
                row = offset + idx
                vec[row, experts] = 1.0
        return vec.reshape(-1)
