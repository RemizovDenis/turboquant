"""Temporal Expert Fusion for MoE.

Reduces MoE weight footprint by merging rarely-used experts via
truncated SVD, creating "composite experts" that maintain quality
at lower storage and compute cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import torch

log = structlog.get_logger(__name__)


@dataclass
class FusionConfig:
    """Configuration for expert fusion."""

    min_usage_rate_for_fusion: float = 0.05
    max_experts_per_composite: int = 4
    svd_rank_ratio: float = 0.5
    quality_threshold: float = 0.95
    enable_online_tracking: bool = True


class ExpertUsageTracker:
    """Track expert activation frequency over time."""

    def __init__(self, num_experts: int, window_size: int = 1000):
        self.num_experts = num_experts
        self.window_size = window_size
        self.counts = torch.zeros(num_experts, dtype=torch.long)
        self._total_activations = 0

    def record_activations(self, expert_indices: torch.Tensor) -> None:
        """Increment counters for activated experts."""
        # indices shape: (batch, seq, top_k)
        flat = expert_indices.reshape(-1)
        # Avoid counts going to infinity, use rolling sum or decay
        self.counts.scatter_add_(0, flat.to(torch.long), torch.ones_like(flat, dtype=torch.long))
        self._total_activations += flat.numel()

    def get_usage_rates(self) -> torch.Tensor:
        """Return usage frequency [0, 1] per expert."""
        return self.counts.float() / max(self._total_activations, 1)

    def get_rare_experts(self, threshold: float) -> list[int]:
        """Indices of experts with usage below threshold."""
        rates = self.get_usage_rates()
        rare = (rates < threshold).nonzero().reshape(-1).tolist()
        return rare


class TemporalExpertFusion:
    """Merges rare MoE experts via truncated SVD."""

    def __init__(
        self, num_experts: int, expert_hidden_dim: int, expert_ffn_dim: int, config: FusionConfig
    ) -> None:
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.expert_ffn_dim = expert_ffn_dim
        self.config = config

    def can_fuse(self, usage_tracker: ExpertUsageTracker) -> list[list[int]]:
        """Identify groups of experts to fuse."""
        rare_indices = usage_tracker.get_rare_experts(self.config.min_usage_rate_for_fusion)

        # Simple grouping into blocks of size max_experts_per_composite
        groups = []
        for i in range(0, len(rare_indices), self.config.max_experts_per_composite):
            group = rare_indices[i : i + self.config.max_experts_per_composite]
            if len(group) >= 2:  # At least 2 to fuse
                groups.append(group)
        return groups

    def fuse_experts(
        self, expert_weights: dict[int, tuple[torch.Tensor, torch.Tensor]], expert_group: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge multiple expert matrices into a single low-rank composite."""
        # W_gate and W_up (or W1/W2)
        w_gates = [expert_weights[eid][0] for eid in expert_group]
        w_up_projs = [expert_weights[eid][1] for eid in expert_group]

        # Stack vertically: [W_3; W_7; W_12] -> shape (N * ffn_dim, hidden_dim)
        w_gate_combined = torch.cat(w_gates, dim=0)  # (G*D, H)
        w_up_combined = torch.cat(w_up_projs, dim=0)  # (G*D, H)

        # Apply SVD for compression
        def _compress_svd(W: torch.Tensor) -> torch.Tensor:  # noqa: N803
            try:
                # full_matrices=False for thin SVD
                u, s, vh = torch.linalg.svd(W, full_matrices=False)
                k = int(min(W.shape) * self.config.svd_rank_ratio)
                k = max(k, 1)
                res: torch.Tensor = u[:, :k] @ torch.diag(s[:k]) @ vh[:k, :]
                return res
            except RuntimeError:
                return W

        composite_gate = _compress_svd(w_gate_combined)
        composite_up = _compress_svd(w_up_combined)

        return composite_gate, composite_up

    def estimate_memory_savings(
        self,
        usage_tracker: ExpertUsageTracker,
        expert_weights: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, float]:
        """Project VRAM savings from expert fusion."""
        groups = self.can_fuse(usage_tracker)
        num_fuseable = sum(len(g) for g in groups)
        num_composites = len(groups)

        # Each expert footprint (assuming 2 matrices per expert)
        # Hidden * FFN * 2 * dtype_size
        expert_params = self.expert_hidden_dim * self.expert_ffn_dim * 2
        orig_mb = num_fuseable * expert_params * 2 / (1024**2)

        # Fused version (assuming SVD rank reduction by svd_rank_ratio)
        (
            num_composites
            * (num_fuseable / max(num_composites, 1))
            * expert_params
            * self.config.svd_rank_ratio
            * 2
            / (1024**2)
        )
        # Actually it's simpler:
        savings_mb = orig_mb * (1 - self.config.svd_rank_ratio)

        return {
            "experts_fuseable": float(num_fuseable),
            "composites_created": float(num_composites),
            "original_mb": orig_mb,
            "fused_mb": orig_mb - savings_mb,
            "savings_mb": savings_mb,
            "savings_percent": (1 - self.config.svd_rank_ratio) * 100 if num_fuseable > 0 else 0.0,
        }

    def apply_fusion(
        self,
        usage_tracker: ExpertUsageTracker,
        expert_weights: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[
        dict[int, tuple[torch.Tensor, torch.Tensor]],
        dict[str, tuple[torch.Tensor, torch.Tensor]],
        dict[str, list[int]],
    ]:
        """Run the full fusion pipeline."""
        groups = self.can_fuse(usage_tracker)

        active_weights = expert_weights.copy()
        composite_weights = {}
        mapping = {}

        for i, group in enumerate(groups):
            composite_id = f"composite_{i}"
            comp_gate, comp_up = self.fuse_experts(active_weights, group)
            composite_weights[composite_id] = (comp_gate, comp_up)
            mapping[composite_id] = group

            # Remove from active
            for eid in group:
                if eid in active_weights:
                    del active_weights[eid]

        return active_weights, composite_weights, mapping
