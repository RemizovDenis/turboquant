# Copyright (c) 2026 Denis Remizov. Licensed under BUSL-1.1.
# See LICENSE file for details.


"""MoE router optimizer with pruning, capacity control, and balancing losses."""

from __future__ import annotations

import math
from dataclasses import dataclass

import structlog
import torch
import torch.nn as nn

LOGGER = structlog.get_logger(__name__)


@dataclass
class RouterOptimizerConfig:
    """Configuration for :class:`MoERouterOptimizer`."""

    num_experts: int
    top_k: int
    pruning_threshold: float = 0.01
    load_balance_alpha: float = 0.01
    expert_dropout_rate: float = 0.0
    capacity_factor: float = 1.25
    use_aux_loss: bool = True
    use_z_loss: bool = True
    z_loss_coeff: float = 1e-3
    normalize_expert_weights: bool = True
    use_noise_during_training: bool = True
    noise_std: float = 1.0
    ema_decay: float = 0.99


@dataclass
class RouterOutput:
    """Router outputs used for dispatch and diagnostics."""

    dispatch_mask: torch.Tensor
    combine_weights: torch.Tensor
    expert_indices: torch.Tensor
    expert_load: torch.Tensor
    aux_loss: torch.Tensor | None
    z_loss: torch.Tensor | None
    dropped_tokens: int


class MoERouterOptimizer(nn.Module):
    """Top-k MoE router with differentiable STE path and balancing losses."""

    def __init__(self, config: RouterOptimizerConfig) -> None:
        super().__init__()
        if config.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if config.top_k <= 0 or config.top_k > config.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")

        self.config = config
        uniform = torch.full((config.num_experts,), 1.0 / config.num_experts, dtype=torch.float32)
        self.register_buffer("expert_ema_load", uniform)
        self._logger = LOGGER.bind(component="MoERouterOptimizer")

    def _ema_load(self) -> torch.Tensor:
        return self.get_buffer("expert_ema_load")

    def forward(self, router_logits: torch.Tensor, training: bool = False) -> RouterOutput:
        """Route tokens to top-k experts and compute balancing diagnostics."""
        if router_logits.ndim != 2:
            raise ValueError("router_logits must have shape [num_tokens, num_experts]")
        num_tokens, num_experts = router_logits.shape
        if num_experts != self.config.num_experts:
            raise ValueError("router_logits last dim mismatch with config.num_experts")

        logits = router_logits.float()
        if training and self.config.use_noise_during_training:
            logits = logits + self._gumbel_noise_like(logits) * self.config.noise_std

        probs = torch.softmax(logits, dim=-1)
        probs = self.prune_router_weights(probs)

        topk_weights, topk_indices = torch.topk(probs, k=self.config.top_k, dim=-1)
        if self.config.normalize_expert_weights:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        ste_weights = (
            topk_weights + (topk_weights.detach() * 0 + topk_weights) - topk_weights.detach()
        )
        dispatch_mask, ste_weights, dropped_tokens = self._apply_capacity(
            expert_indices=topk_indices,
            expert_weights=ste_weights,
            num_tokens=num_tokens,
        )

        expert_load = dispatch_mask.float().sum(dim=0)

        aux_loss = None
        if training and self.config.use_aux_loss:
            aux_loss = self.compute_aux_loss(router_probs=probs, dispatch_mask=dispatch_mask)

        z_loss = None
        if training and self.config.use_z_loss:
            z_loss = self.compute_z_loss(router_logits=router_logits)

        ema_load = self._ema_load()
        current_load = (expert_load / max(1, num_tokens)).to(ema_load.dtype)
        with torch.no_grad():
            ema_load.mul_(self.config.ema_decay).add_(current_load * (1.0 - self.config.ema_decay))

        imbalance_ratio = float(
            expert_load.max().item() / expert_load.mean().clamp_min(1e-8).item()
        )
        self._logger.debug(
            "router_forward",
            dropped_tokens=dropped_tokens,
            imbalance_ratio=imbalance_ratio,
            num_tokens=num_tokens,
        )

        return RouterOutput(
            dispatch_mask=dispatch_mask,
            combine_weights=ste_weights,
            expert_indices=topk_indices,
            expert_load=expert_load,
            aux_loss=aux_loss,
            z_loss=z_loss,
            dropped_tokens=dropped_tokens,
        )

    def compute_aux_loss(
        self, router_probs: torch.Tensor, dispatch_mask: torch.Tensor
    ) -> torch.Tensor:
        """Switch Transformer auxiliary load balancing loss."""
        tokens = max(1, dispatch_mask.shape[0])
        f_i = dispatch_mask.float().sum(dim=0) / tokens
        p_i = router_probs.mean(dim=0)
        return self.config.load_balance_alpha * self.config.num_experts * torch.sum(f_i * p_i)

    def compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """ST-MoE z-loss regularizer for router logit stability."""
        z = torch.logsumexp(router_logits.float(), dim=-1)
        return self.config.z_loss_coeff * torch.mean(z.square())

    def prune_router_weights(
        self, probs: torch.Tensor, threshold: float | None = None
    ) -> torch.Tensor:
        """Zero out small routing probabilities and renormalize rows."""
        thr = self.config.pruning_threshold if threshold is None else threshold
        if thr <= 0:
            return probs

        mask = probs >= thr
        pruned = probs * mask
        row_sum = pruned.sum(dim=-1, keepdim=True)
        needs_fix = row_sum.squeeze(-1) <= 0

        if needs_fix.any():
            topk_vals, topk_idx = torch.topk(probs[needs_fix], k=self.config.top_k, dim=-1)
            repaired = torch.zeros_like(probs[needs_fix])
            repaired.scatter_(1, topk_idx, topk_vals)
            pruned = pruned.clone()
            pruned[needs_fix] = repaired
            row_sum = pruned.sum(dim=-1, keepdim=True)

        return pruned / row_sum.clamp_min(1e-8)

    def get_expert_utilization(self) -> dict[int, float]:
        """Return EMA utilization per expert plus imbalance ratio key -1."""
        ema = self._ema_load().detach().cpu()
        mean_load = float(ema.mean().item())
        utilization = {idx: float(val.item()) for idx, val in enumerate(ema)}
        utilization[-1] = float(ema.max().item() / max(1e-8, mean_load))
        return utilization

    def reset_stats(self) -> None:
        """Reset EMA utilization to a uniform prior."""
        with torch.no_grad():
            self._ema_load().fill_(1.0 / self.config.num_experts)

    def extra_repr(self) -> str:
        """Readable module representation for print(model)."""
        return (
            f"num_experts={self.config.num_experts}, top_k={self.config.top_k}, "
            f"threshold={self.config.pruning_threshold}, capacity_factor={self.config.capacity_factor}"
        )

    @staticmethod
    def _gumbel_noise_like(tensor: torch.Tensor) -> torch.Tensor:
        eps = 1e-9
        u = torch.rand_like(tensor).clamp_(min=eps, max=1.0 - eps)
        return -torch.log(-torch.log(u))

    def _apply_capacity(
        self,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Apply per-expert capacity and drop low-score overflow assignments."""
        capacity = int(
            math.ceil(
                self.config.top_k
                * num_tokens
                / self.config.num_experts
                * self.config.capacity_factor
            )
        )
        capacity = max(1, capacity)

        mask = torch.ones_like(expert_weights, dtype=torch.bool)
        kept_per_expert = torch.zeros(
            (self.config.num_experts,), dtype=torch.int32, device=expert_indices.device
        )

        flat_scores = expert_weights.reshape(-1)
        flat_experts = expert_indices.reshape(-1)
        flat_tokens = (
            torch.arange(num_tokens, device=expert_indices.device)
            .unsqueeze(1)
            .expand(num_tokens, self.config.top_k)
            .reshape(-1)
        )
        order = torch.argsort(flat_scores, descending=True)

        keep_flat = torch.zeros_like(flat_scores, dtype=torch.bool)
        for idx in order.tolist():
            expert = int(flat_experts[idx].item())
            if kept_per_expert[expert] < capacity:
                keep_flat[idx] = True
                kept_per_expert[expert] += 1

        mask = keep_flat.reshape_as(expert_weights)
        expert_weights = expert_weights * mask.float()
        row_sum = expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights / row_sum.clamp_min(1e-8)

        dispatch = torch.zeros(
            (num_tokens, self.config.num_experts),
            dtype=torch.bool,
            device=expert_indices.device,
        )

        keep_tokens = flat_tokens[keep_flat]
        keep_experts = flat_experts[keep_flat]
        dispatch[keep_tokens, keep_experts] = True

        dropped_tokens = int((~dispatch.any(dim=-1)).sum().item())
        return dispatch, expert_weights, dropped_tokens
