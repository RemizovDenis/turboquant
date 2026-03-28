"""Game-theoretic Nash router for MoE expert selection."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import structlog
import torch

from turboquant.core.moe_router import MoERouterOptimizer, RouterOptimizerConfig, RouterOutput

LOGGER = structlog.get_logger(__name__)


@dataclass
class NashRouterConfig(RouterOptimizerConfig):
    """Configuration for :class:`GameTheoreticRouter`.

    Extends :class:`RouterOptimizerConfig` with utility exponents and Nash loop
    controls.
    """

    token_power: float = 1.0
    expert_power: float = 0.5
    locality_power: float = 1.2
    dynamic_temperature: bool = True
    base_temperature: float = 1.0
    nash_iterations: int = 3
    convergence_threshold: float = 1e-4
    load_smoothing: float = 0.1


@dataclass
class NashEquilibriumState:
    """Per-forward Nash optimization diagnostics."""

    iteration: int
    nash_scores: torch.Tensor
    token_utilities: torch.Tensor
    expert_utilities: torch.Tensor
    system_utilities: torch.Tensor
    converged: bool
    convergence_delta: float
    temperature_used: float


class GameTheoreticRouter(MoERouterOptimizer):
    """MoE router with Nash-style utility balancing.

    The router combines three utilities:
    1. token affinity utility from router logits,
    2. expert load utility (discourages overloaded experts),
    3. system locality utility (prefers GPU-resident experts).
    """

    def __init__(self, config: NashRouterConfig) -> None:
        super().__init__(config)
        self.nash_config = config
        self.register_buffer("load_ema", torch.zeros(config.num_experts, dtype=torch.float32))
        self.register_buffer("imbalance_ema", torch.tensor(0.0, dtype=torch.float32))
        self._state_lock = threading.RLock()
        self._nash_iterations_total = 0
        self._nash_batches_total = 0
        self._nash_converged_total = 0
        self._gpu_selected_total = 0
        self._cpu_selected_total = 0
        self._last_state: NashEquilibriumState | None = None
        self._logger = LOGGER.bind(component="GameTheoreticRouter")

    def _load_ema(self) -> torch.Tensor:
        return self.get_buffer("load_ema")

    def _imbalance_ema(self) -> torch.Tensor:
        return self.get_buffer("imbalance_ema")

    def forward(
        self,
        router_logits: torch.Tensor,
        training: bool = False,
        expert_locations_mask: torch.Tensor | None = None,
        expert_current_load: torch.Tensor | None = None,
    ) -> RouterOutput:
        """Route tokens with Nash equilibrium iterations.

        Args:
            router_logits: Logits tensor `[num_tokens, num_experts]`.
            expert_locations_mask: Boolean tensor `[num_experts]`, `True` for GPU.
            expert_current_load: Optional current expert load `[num_experts]`.
            training: If `True`, compute auxiliary losses.

        Returns:
            Standard :class:`RouterOutput` compatible with existing pipeline.
        """
        if router_logits.ndim != 2:
            raise ValueError("router_logits must have shape [num_tokens, num_experts]")
        num_tokens, num_experts = router_logits.shape
        if expert_locations_mask is None:
            expert_locations_mask = torch.ones(
                num_experts,
                dtype=torch.bool,
                device=router_logits.device,
            )
        if expert_locations_mask.ndim != 1:
            raise ValueError("expert_locations_mask must have shape [num_experts]")
        if num_experts != self.nash_config.num_experts:
            raise ValueError("router_logits last dim mismatch with config.num_experts")
        if expert_locations_mask.shape[0] != num_experts:
            raise ValueError("expert_locations_mask size mismatch with num_experts")

        logits = router_logits.float()
        if training and self.nash_config.use_noise_during_training:
            logits = logits + self._gumbel_noise_like(logits) * self.nash_config.noise_std

        if expert_current_load is None:
            current_load = self._load_ema().detach() * float(max(num_tokens, 1))
        else:
            current_load = expert_current_load.to(device=logits.device, dtype=torch.float32)

        temperature = (
            self.compute_dynamic_temperature()
            if self.nash_config.dynamic_temperature
            else float(self.nash_config.base_temperature)
        )

        prev_indices: torch.Tensor | None = None
        converged = False
        convergence_delta = float("inf")
        topk_indices = torch.zeros(
            (num_tokens, self.nash_config.top_k),
            dtype=torch.long,
            device=logits.device,
        )
        nash_scores = torch.zeros_like(logits)
        u_token = torch.zeros_like(logits)
        u_expert = torch.zeros_like(logits)
        u_system = torch.zeros_like(logits)
        iteration_count = 0

        for itr in range(self.nash_config.nash_iterations):
            iteration_count = itr + 1
            u_token, u_expert, u_system = self.compute_nash_utilities(
                router_logits=logits,
                expert_locations_mask=expert_locations_mask,
                current_assignments=current_load,
                temperature=temperature,
            )
            log_scores = (
                self.nash_config.token_power * torch.log(u_token.clamp_min(1e-8))
                + self.nash_config.expert_power * torch.log(u_expert.clamp_min(1e-8))
                + self.nash_config.locality_power * torch.log(u_system.clamp_min(1e-8))
            )
            # Keep values in numerically safe range before exp.
            log_scores = log_scores - log_scores.max(dim=-1, keepdim=True).values
            nash_scores = torch.exp(log_scores)

            topk_indices, current_load = self._capacity_aware_topk(nash_scores)

            if prev_indices is not None:
                convergence_delta = float((topk_indices != prev_indices).float().mean().item())
                if convergence_delta <= max(self.nash_config.convergence_threshold, 1e-3):
                    converged = True
                    break
            prev_indices = topk_indices.detach().clone()

        topk_logits = torch.gather(logits, 1, topk_indices)
        topk_weights = torch.softmax(topk_logits, dim=-1)
        if self.nash_config.normalize_expert_weights:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        dispatch_mask, combine_weights, dropped_tokens = self._apply_capacity(
            expert_indices=topk_indices,
            expert_weights=topk_weights,
            num_tokens=num_tokens,
        )
        expert_load = dispatch_mask.float().sum(dim=0)

        aux_loss = None
        if training and self.nash_config.use_aux_loss:
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            aux_loss = self.compute_aux_loss(router_probs=probs, dispatch_mask=dispatch_mask)

        z_loss = None
        if training and self.nash_config.use_z_loss:
            z_loss = self.compute_z_loss(router_logits=logits)

        with self._state_lock:
            ema_load = self._ema_load()
            current_load_norm = (expert_load / max(1, num_tokens)).to(ema_load.dtype)
            ema_load.mul_(self.nash_config.ema_decay).add_(
                current_load_norm * (1.0 - self.nash_config.ema_decay)
            )

            load_ema = self._load_ema()
            load_ema.mul_(1.0 - self.nash_config.load_smoothing).add_(
                expert_load.to(load_ema.dtype) * self.nash_config.load_smoothing
            )

            imbalance = (expert_load.max() - expert_load.mean()) / expert_load.mean().clamp_min(
                1e-8
            )
            imbalance_ema = self._imbalance_ema()
            imbalance_ema.mul_(1.0 - self.nash_config.load_smoothing).add_(
                imbalance.to(imbalance_ema.dtype) * self.nash_config.load_smoothing
            )

            self._nash_iterations_total += iteration_count
            self._nash_batches_total += 1
            if converged:
                self._nash_converged_total += 1

            gpu_selected = int(dispatch_mask[:, expert_locations_mask].sum().item())
            cpu_selected = int(dispatch_mask[:, ~expert_locations_mask].sum().item())
            self._gpu_selected_total += gpu_selected
            self._cpu_selected_total += cpu_selected

            self._last_state = NashEquilibriumState(
                iteration=iteration_count,
                nash_scores=nash_scores.detach(),
                token_utilities=u_token.detach(),
                expert_utilities=u_expert.detach(),
                system_utilities=u_system.detach(),
                converged=converged,
                convergence_delta=convergence_delta,
                temperature_used=temperature,
            )

        self._logger.debug(
            "nash_forward",
            iterations=iteration_count,
            converged=converged,
            convergence_delta=convergence_delta,
            dropped_tokens=dropped_tokens,
            temperature=temperature,
        )

        return RouterOutput(
            dispatch_mask=dispatch_mask,
            combine_weights=combine_weights,
            expert_indices=topk_indices,
            expert_load=expert_load,
            aux_loss=aux_loss,
            z_loss=z_loss,
            dropped_tokens=dropped_tokens,
        )

    def _capacity_aware_topk(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k experts while discouraging overload per expert.

        Args:
            scores: Nash scores `[num_tokens, num_experts]`.

        Returns:
            Tuple `(expert_indices, expert_load)`.
        """
        num_tokens, num_experts = scores.shape
        k = int(self.nash_config.top_k)
        capacity = int(
            math.ceil(k * num_tokens / max(1, num_experts) * self.nash_config.capacity_factor)
        )
        capacity = max(1, capacity)

        sorted_experts = torch.argsort(scores, dim=-1, descending=True)
        confidence = scores.max(dim=-1).values
        token_order = torch.argsort(confidence, descending=True)
        loads = torch.zeros(num_experts, dtype=torch.int64, device=scores.device)
        out = torch.empty((num_tokens, k), dtype=torch.long, device=scores.device)

        for token_idx in token_order.tolist():
            chosen: list[int] = []
            for expert_idx in sorted_experts[token_idx].tolist():
                if int(loads[expert_idx].item()) < capacity:
                    chosen.append(expert_idx)
                    loads[expert_idx] += 1
                if len(chosen) == k:
                    break
            if len(chosen) < k:
                for expert_idx in sorted_experts[token_idx].tolist():
                    if expert_idx in chosen:
                        continue
                    chosen.append(expert_idx)
                    loads[expert_idx] += 1
                    if len(chosen) == k:
                        break
            out[token_idx] = torch.tensor(chosen, dtype=torch.long, device=scores.device)

        return out, loads.to(dtype=torch.float32)

    def compute_nash_utilities(
        self,
        router_logits: torch.Tensor,
        expert_locations_mask: torch.Tensor,
        current_assignments: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute token/expert/system utility matrices.

        Args:
            router_logits: Router logits `[num_tokens, num_experts]`.
            expert_locations_mask: Expert location flags `[num_experts]`.
            current_assignments: Current load estimate `[num_experts]` or dispatch matrix.
            temperature: Softmax temperature.

        Returns:
            Tuple `(U_token, U_expert, U_system)` each with shape
            `[num_tokens, num_experts]`.
        """
        logits = router_logits.float()
        temp = max(float(temperature), 1e-6)
        u_token = torch.softmax(logits / temp, dim=-1)

        if current_assignments.ndim == 2:
            load = current_assignments.float().sum(dim=0)
        else:
            load = current_assignments.float()

        num_tokens, num_experts = logits.shape
        capacity = int(
            math.ceil(
                self.nash_config.top_k
                * num_tokens
                / self.nash_config.num_experts
                * self.nash_config.capacity_factor
            )
        )
        capacity = max(1, capacity)
        load_ratio = (load / float(capacity)).clamp(0.0, 1.0)
        per_expert_utility = 1.0 - load_ratio
        u_expert = per_expert_utility.unsqueeze(0).expand(num_tokens, num_experts)

        gpu_util = torch.ones_like(expert_locations_mask, dtype=torch.float32, device=logits.device)
        cpu_penalty = max(0.0, min(0.95, self.nash_config.locality_power - 1.0))
        cpu_util_val = 1.0 - cpu_penalty
        cpu_util = torch.full_like(gpu_util, fill_value=cpu_util_val)
        per_system = torch.where(expert_locations_mask.to(logits.device), gpu_util, cpu_util)
        u_system = per_system.unsqueeze(0).expand(num_tokens, num_experts)

        return u_token, u_expert, u_system

    def compute_dynamic_temperature(self) -> float:
        """Compute dynamic softmax temperature from EMA imbalance.

        Returns:
            Temperature value clipped to `[0.5, 2.0]`.
        """
        imbalance = float(self._imbalance_ema().item())
        raw = self.nash_config.base_temperature / (1.0 + max(0.0, imbalance))
        temperature = float(min(2.0, max(0.5, raw)))
        self._logger.debug("nash_temperature", imbalance_ema=imbalance, temperature=temperature)
        return temperature

    def get_nash_stats(self) -> dict[str, float]:
        """Return aggregate Nash routing statistics."""
        with self._state_lock:
            batches = max(1, self._nash_batches_total)
            convergence_rate = self._nash_converged_total / batches
            avg_iterations = self._nash_iterations_total / batches
            gpu_pref = self._gpu_selected_total / max(
                1, self._gpu_selected_total + self._cpu_selected_total
            )
            load_imb = float(self._imbalance_ema().item())
        return {
            "nash_convergence_rate": float(convergence_rate),
            "avg_iterations": float(avg_iterations),
            "gpu_expert_preference": float(gpu_pref),
            "load_imbalance_ratio": float(load_imb),
        }

    def overhead_ms(
        self,
        num_tokens: int = 512,
        num_experts: int = 8,
        n_warmup: int = 10,
        n_iters: int = 100,
    ) -> float:
        """Measure Nash routing overhead over standard top-k.

        Args:
            num_tokens: Number of tokens in synthetic benchmark.
            num_experts: Number of experts.
            n_warmup: Warmup iterations.
            n_iters: Measured iterations.

        Returns:
            Overhead in milliseconds: `nash_ms - baseline_ms`.
        """
        device = self._ema_load().device
        logits = torch.randn(num_tokens, num_experts, device=device)
        mask = torch.zeros(num_experts, dtype=torch.bool, device=device)
        mask[: max(1, num_experts // 2)] = True

        for _ in range(n_warmup):
            probs = torch.softmax(logits, dim=-1)
            torch.topk(probs, k=self.nash_config.top_k, dim=-1)
            self.forward(logits, training=False, expert_locations_mask=mask)

        if logits.is_cuda:
            torch.cuda.synchronize(device=device)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            probs = torch.softmax(logits, dim=-1)
            torch.topk(probs, k=self.nash_config.top_k, dim=-1)
        if logits.is_cuda:
            torch.cuda.synchronize(device=device)
        baseline_ms = (time.perf_counter() - t0) * 1000.0 / max(1, n_iters)

        if logits.is_cuda:
            torch.cuda.synchronize(device=device)
        t1 = time.perf_counter()
        for _ in range(n_iters):
            self.forward(logits, training=False, expert_locations_mask=mask)
        if logits.is_cuda:
            torch.cuda.synchronize(device=device)
        nash_ms = (time.perf_counter() - t1) * 1000.0 / max(1, n_iters)

        overhead = max(0.0, nash_ms - baseline_ms)
        if overhead > 0.1 and logits.is_cuda:
            self._logger.warning(
                "nash_overhead_high",
                overhead_ms=overhead,
                advice="consider lowering nash_iterations",
            )
        return overhead
