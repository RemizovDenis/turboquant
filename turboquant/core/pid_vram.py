"""PID-based VRAM controller for dynamic MoE expert cache sizing."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from statistics import mean, pstdev

import structlog
import torch

from turboquant.core.moe_expert_cache import DynamicExpertCache

LOGGER = structlog.get_logger(__name__)
MB = 1024.0 * 1024.0

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


@dataclass
class PIDConfig:
    """Configuration for :class:`VRAM_PID_Controller`."""

    target_vram_utilization: float = 0.88
    kp: float = 0.15
    ki: float = 0.008
    kd: float = 0.04
    min_cache_size: int = 1
    max_cache_size: int = 32
    dead_band: float = 0.02
    integral_window: int = 20
    integral_max: float = 10.0
    control_scale: float = 1.0
    measurement_interval_ms: float = 100.0
    emergency_threshold: float = 0.97
    recovery_threshold: float = 0.80
    hysteresis_steps: int = 3
    smoothing_alpha: float = 0.3


@dataclass
class PIDState:
    """Single PID step state snapshot."""

    current_utilization: float
    target_utilization: float
    error: float
    p_term: float
    i_term: float
    d_term: float
    control_signal: float
    recommended_cache_size: int
    emergency_mode: bool
    steps_since_last_change: int
    timestamp: float


class VRAM_PID_Controller:  # noqa: N801
    """Dynamic controller that resizes GPU expert cache from VRAM feedback.

    The controller applies classical PID terms with anti-windup and hysteresis.
    It can run in foreground (`step`) or continuously in a background thread.
    """

    def __init__(
        self,
        config: PIDConfig,
        expert_cache: DynamicExpertCache,
        initial_cache_size: int = 4,
    ) -> None:
        """Initialize controller state.

        Args:
            config: PID tuning and runtime thresholds.
            expert_cache: Expert cache to resize/evict.
            initial_cache_size: Initial GPU resident expert budget.
        """
        self.config = config
        self.expert_cache = expert_cache

        self.error_history: deque[float] = deque(maxlen=config.integral_window)
        self.prev_error: float = 0.0
        self.prev_utilization: float = 0.0
        self.integral_accumulator: float = 0.0
        self.current_cache_size = int(
            max(config.min_cache_size, min(config.max_cache_size, initial_cache_size))
        )
        self.steps_since_last_change: int = 0
        self.emergency_mode: bool = False
        self.pid_history: deque[PIDState] = deque(maxlen=1000)

        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._total_evictions = 0
        self._total_expansions = 0
        self._emergency_events = 0
        self._logger = LOGGER.bind(component="VRAM_PID_Controller")

        self.expert_cache.config.gpu_cache_size = self.current_cache_size

    def measure_vram(self) -> tuple[float, float, float]:
        """Measure smoothed VRAM utilization.

        Returns:
            Tuple of `(utilization, allocated_mb, reserved_mb)`.
            Returns `(0.0, 0.0, 0.0)` when CUDA is unavailable.
        """
        with self._lock:
            if not torch.cuda.is_available():
                return (0.0, 0.0, 0.0)

            allocated_mb = torch.cuda.memory_allocated() / MB
            reserved_mb = torch.cuda.memory_reserved() / MB
            total_mb = (
                torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / MB
            )
            raw_util = max(allocated_mb, reserved_mb) / max(total_mb, 1e-8)

            alpha = float(self.config.smoothing_alpha)
            smoothed = alpha * raw_util + (1.0 - alpha) * self.prev_utilization
            self.prev_utilization = smoothed
            return (smoothed, allocated_mb, reserved_mb)

    def step(self) -> tuple[int, PIDState]:
        """Run one PID control step and optionally resize cache.

        Returns:
            Tuple `(new_cache_size, state)`.
        """
        with self._lock:
            utilization, _, _ = self.measure_vram()
            error = float(self.config.target_vram_utilization - utilization)

            p_term = float(self.config.kp * error)

            self.error_history.append(error)
            self.integral_accumulator += error
            self.integral_accumulator = max(
                -float(self.config.integral_max),
                min(float(self.config.integral_max), self.integral_accumulator),
            )
            i_term = float(self.config.ki * self.integral_accumulator)

            dt = max(float(self.config.measurement_interval_ms) / 1000.0, 1e-6)
            d_term = float(self.config.kd * (error - self.prev_error) / dt)

            control_signal = p_term + i_term + d_term
            signal_for_resize = control_signal
            if abs(signal_for_resize) < float(self.config.dead_band):
                signal_for_resize = 0.0
            if self.steps_since_last_change < int(self.config.hysteresis_steps):
                signal_for_resize = 0.0

            delta = int(round(signal_for_resize * float(self.config.control_scale)))
            new_size = max(
                int(self.config.min_cache_size),
                min(int(self.config.max_cache_size), self.current_cache_size + delta),
            )

            if new_size != self.current_cache_size:
                prev_size = self.current_cache_size
                if new_size < self.current_cache_size:
                    self._evict_to_size(new_size)
                    self._total_evictions += prev_size - new_size
                else:
                    self._total_expansions += new_size - prev_size

                self.current_cache_size = new_size
                self.expert_cache.config.gpu_cache_size = new_size
                self.steps_since_last_change = 0
                self._logger.info(
                    "pid_cache_resize",
                    previous_size=prev_size,
                    new_size=new_size,
                    utilization=utilization,
                    signal=signal_for_resize,
                )
            else:
                self.steps_since_last_change += 1

            self.prev_error = error
            state = PIDState(
                current_utilization=utilization,
                target_utilization=float(self.config.target_vram_utilization),
                error=error,
                p_term=p_term,
                i_term=i_term,
                d_term=d_term,
                control_signal=control_signal,
                recommended_cache_size=new_size,
                emergency_mode=self.emergency_mode,
                steps_since_last_change=self.steps_since_last_change,
                timestamp=time.time(),
            )
            self.pid_history.append(state)
            self._logger.debug(
                "pid_step",
                utilization=utilization,
                target=state.target_utilization,
                error=error,
                p_term=p_term,
                i_term=i_term,
                d_term=d_term,
                control_signal=control_signal,
                resized_to=new_size,
            )
            return (new_size, state)

    def emergency_evict(self) -> int:
        """Immediately evict 50% of GPU experts when utilization is critical.

        Returns:
            New cache size after emergency action.
        """
        with self._lock:
            utilization, _, _ = self.measure_vram()
            if utilization <= float(self.config.emergency_threshold):
                return self.current_cache_size

            self.emergency_mode = True
            self._emergency_events += 1
            target_size = max(int(self.config.min_cache_size), self.current_cache_size // 2)
            prev_size = self.current_cache_size
            self._evict_to_size(target_size)
            self.current_cache_size = target_size
            self.expert_cache.config.gpu_cache_size = target_size
            self.steps_since_last_change = 0
            self._total_evictions += max(0, prev_size - target_size)
            self._logger.warning(
                "pid_emergency_evict",
                utilization=utilization,
                emergency_threshold=self.config.emergency_threshold,
                previous_size=prev_size,
                new_size=target_size,
            )
            return target_size

    def _evict_to_size(self, target_size: int) -> None:
        """Evict least-recently used GPU experts until target size is met.

        Args:
            target_size: Target number of experts kept on GPU.
        """
        target_size = max(0, int(target_size))
        with self.expert_cache._lock:
            gpu_keys = list(self.expert_cache._gpu_experts)
            if len(gpu_keys) <= target_size:
                return
            ranked = sorted(
                gpu_keys,
                key=lambda key: self.expert_cache._experts[key].last_access,
            )
            to_evict = ranked[: len(gpu_keys) - target_size]

        for layer_id, expert_id in to_evict:
            try:
                self.expert_cache.evict(expert_id=expert_id, layer_id=layer_id)
            except KeyError:
                continue

    def run_background(self) -> None:
        """Start PID controller loop in a background thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()

            def _loop() -> None:
                interval_s = float(self.config.measurement_interval_ms) / 1000.0
                while not self._stop_event.is_set():
                    _, state = self.step()
                    if state.current_utilization > float(self.config.emergency_threshold):
                        self.emergency_evict()
                    elif self.emergency_mode and state.current_utilization < float(
                        self.config.recovery_threshold
                    ):
                        with self._lock:
                            self.emergency_mode = False
                    self._stop_event.wait(interval_s)

            self._thread = threading.Thread(target=_loop, daemon=True, name="vram-pid-controller")
            self._thread.start()

    def stop_background(self) -> None:
        """Stop background PID loop and wait briefly for thread exit."""
        with self._lock:
            self._stop_event.set()
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=1.0)

    def tune_pid(
        self,
        utilization_history: list[float],
        cache_size_history: list[int],
    ) -> PIDConfig:
        """Estimate PID gains with a Ziegler-Nichols style heuristic.

        This method is intended for offline tuning from recorded histories.

        Args:
            utilization_history: Historical measured utilizations.
            cache_size_history: Historical cache sizes.

        Returns:
            New PIDConfig with tuned gains.
        """
        if len(utilization_history) < 8 or len(cache_size_history) < 8:
            return self.config

        target = float(self.config.target_vram_utilization)
        centered = [u - target for u in utilization_history]
        sign_changes = 0
        for idx in range(1, len(centered)):
            if centered[idx - 1] == 0.0:
                continue
            if centered[idx] == 0.0:
                continue
            if (centered[idx - 1] > 0) != (centered[idx] > 0):
                sign_changes += 1

        osc_factor = max(1.0, sign_changes / max(1.0, len(centered) / 6.0))
        ku = max(0.05, min(3.0, pstdev(centered) * 8.0 * osc_factor))
        pu = max(1.0, len(centered) / max(1.0, sign_changes / 2.0))

        kp = 0.6 * ku
        ki = 1.2 * ku / pu
        kd = 0.075 * ku * pu

        return PIDConfig(
            target_vram_utilization=self.config.target_vram_utilization,
            kp=kp,
            ki=ki,
            kd=kd,
            min_cache_size=self.config.min_cache_size,
            max_cache_size=self.config.max_cache_size,
            dead_band=self.config.dead_band,
            integral_window=self.config.integral_window,
            integral_max=self.config.integral_max,
            control_scale=self.config.control_scale,
            measurement_interval_ms=self.config.measurement_interval_ms,
            emergency_threshold=self.config.emergency_threshold,
            recovery_threshold=self.config.recovery_threshold,
            hysteresis_steps=self.config.hysteresis_steps,
            smoothing_alpha=self.config.smoothing_alpha,
        )

    def plot_history(self, last_n: int = 500) -> None:
        """Write PID diagnostics plot to `/tmp/pid_history.html`.

        Args:
            last_n: Number of latest points to render.
        """
        with self._lock:
            if not HAS_PLOTLY:
                self._logger.warning("plotly_not_installed", output="/tmp/pid_history.html")
                return
            history = list(self.pid_history)[-max(1, int(last_n)) :]
            if not history:
                return

        x = [h.timestamp for h in history]
        util = [h.current_utilization for h in history]
        target = [h.target_utilization for h in history]
        cache = [h.recommended_cache_size for h in history]
        p_term = [h.p_term for h in history]
        i_term = [h.i_term for h in history]
        d_term = [h.d_term for h in history]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04)
        fig.add_trace(go.Scatter(x=x, y=util, name="utilization"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=target, name="target"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=cache, name="cache_size"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=p_term, name="P"), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=i_term, name="I"), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=d_term, name="D"), row=3, col=1)
        fig.update_layout(height=900, title="VRAM PID Controller History")
        fig.write_html("/tmp/pid_history.html")

    def stats(self) -> dict[str, float]:
        """Return aggregate controller metrics."""
        with self._lock:
            if not self.pid_history:
                return {
                    "avg_utilization": 0.0,
                    "std_utilization": 0.0,
                    "min_utilization": 0.0,
                    "max_utilization": 0.0,
                    "total_evictions": float(self._total_evictions),
                    "total_expansions": float(self._total_expansions),
                    "emergency_events_count": float(self._emergency_events),
                    "time_above_target_percent": 0.0,
                    "time_below_target_percent": 0.0,
                }

            util = [h.current_utilization for h in self.pid_history]
            target = float(self.config.target_vram_utilization)
            above = sum(1 for x in util if x > target)
            below = sum(1 for x in util if x < target)
            total = max(1, len(util))

            return {
                "avg_utilization": float(mean(util)),
                "std_utilization": float(pstdev(util)) if len(util) > 1 else 0.0,
                "min_utilization": float(min(util)),
                "max_utilization": float(max(util)),
                "total_evictions": float(self._total_evictions),
                "total_expansions": float(self._total_expansions),
                "emergency_events_count": float(self._emergency_events),
                "time_above_target_percent": 100.0 * above / total,
                "time_below_target_percent": 100.0 * below / total,
            }
