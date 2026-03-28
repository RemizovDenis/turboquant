"""Fast Walsh-Hadamard Transform (FWHT) for randomised orthogonal rotation.

Replaces the O(n²) matrix-multiply rotation with O(n log n) butterfly
operations. For ``head_dim=128``: matmul = 16 384 ops, FWHT = 896 ops (18× speedup).

The Randomized Hadamard Transform ``HD`` uses a diagonal sign matrix ``D``
(entries ±1, deterministic for a given seed) followed by the normalised
Walsh-Hadamard transform.

Example::

    from turboquant.kernels.hadamard import randomized_hadamard_transform

    x = torch.randn(2, 32, 128, 128, dtype=torch.float16, device="cuda")
    y = randomized_hadamard_transform(x, seed=42)
    x_hat = randomized_hadamard_transform(y, seed=42, inverse=True)
    assert torch.allclose(x.float(), x_hat.float(), atol=1e-3)
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from typing import Any

import structlog
import torch

log = structlog.get_logger(__name__)


# ======================================================================
# Core FWHT
# ======================================================================


def fwht_iterative(x: torch.Tensor) -> torch.Tensor:
    """In-place iterative Fast Walsh-Hadamard Transform.

    Applies the butterfly algorithm and normalises by ``1/√n``.

    Args:
        x: Tensor with last dimension a power of 2.

    Returns:
        Transformed tensor of the same shape and dtype.

    Raises:
        ValueError: If the last dimension is not a power of 2.
    """
    n = x.shape[-1]
    if n == 0:
        return x
    if n & (n - 1) != 0:
        raise ValueError(f"Last dimension must be a power of 2, got {n}")

    # Work in float32 for numerical stability
    original_dtype = x.dtype
    result = x.float().clone()

    length = 2
    while length <= n:
        half = length // 2
        # Reshape for butterfly blocks on the last dimension.
        shape_prefix = result.shape[:-1]
        blocks = n // length
        reshaped = result.reshape(*shape_prefix, blocks, length)

        left = reshaped.narrow(-1, 0, half).clone()
        right = reshaped.narrow(-1, half, half).clone()
        reshaped.narrow(-1, 0, half).copy_(left + right)
        reshaped.narrow(-1, half, half).copy_(left - right)

        result = reshaped.reshape(*shape_prefix, n)
        length *= 2

    result = result / math.sqrt(n)
    return result.to(original_dtype)


# ======================================================================
# Padding
# ======================================================================


def pad_to_power_of_2(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pad the last dimension to the nearest power of 2 with zeros.

    Args:
        x: Input tensor.

    Returns:
        ``(padded_tensor, original_n)`` for subsequent trimming.
    """
    n = x.shape[-1]
    if n == 0:
        return x, 0
    if n & (n - 1) == 0:
        return x, n

    next_pow2 = 1 << (n - 1).bit_length()
    padding = next_pow2 - n
    padded = torch.nn.functional.pad(x, (0, padding), value=0.0)
    return padded, n


# ======================================================================
# Sign vector cache
# ======================================================================


class HadamardCache:
    """Cache for deterministic sign vectors ``D``.

    Given ``(head_dim, seed)``, produces a vector of ±1 that is fully
    reproducible.

    Args:
        max_cache_size: Maximum number of cached sign vectors.
    """

    def __init__(self, max_cache_size: int = 32) -> None:
        self._max = max_cache_size
        self._cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def get_sign_vector(
        self, head_dim: int, seed: int, device: torch.device
    ) -> torch.Tensor:
        """Return a cached ±1 sign vector for the given parameters.

        Args:
            head_dim: Dimension of the sign vector.
            seed: Random seed.
            device: Target torch device.

        Returns:
            Tensor of shape ``(head_dim,)`` with values in ``{-1, +1}``.
        """
        key = (head_dim, seed, str(device))
        if key in self._cache:
            return self._cache[key]

        gen = torch.Generator(device="cpu").manual_seed(seed)
        signs = torch.randint(0, 2, (head_dim,), generator=gen, dtype=torch.float32)
        signs = signs * 2 - 1  # {0,1} → {-1,+1}
        signs = signs.to(device)

        # Evict oldest if over capacity
        if len(self._cache) >= self._max:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = signs
        return signs


# Module-level singleton
_hadamard_cache = HadamardCache()


@lru_cache(maxsize=64)
def _get_cached_sign_vector(
    head_dim: int, seed: int, device_str: str
) -> torch.Tensor:
    """LRU-cached sign vector retrieval.

    Args:
        head_dim: Vector dimension.
        seed: RNG seed.
        device_str: String representation of device.

    Returns:
        Sign vector on the correct device.
    """
    return _hadamard_cache.get_sign_vector(
        head_dim, seed, torch.device(device_str)
    )


# ======================================================================
# Randomised Hadamard Transform
# ======================================================================


def randomized_hadamard_transform(
    x: torch.Tensor,
    seed: int = 42,
    inverse: bool = False,
) -> torch.Tensor:
    """Randomised Fast Walsh-Hadamard Transform.

    Equivalent to multiplying by a random orthogonal matrix but runs in
    O(n log n) instead of O(n²).

    * **Forward**: ``FWHT(x * D)``
    * **Inverse**: ``FWHT(x) * D``  (since H is self-inverse when normalised)

    If *head_dim* is not a power of 2 the tensor is zero-padded, transformed,
    and trimmed back.

    Args:
        x: Input tensor with last dim = head_dim.
        seed: Seed for deterministic sign vector.
        inverse: If ``True``, apply the inverse transform.

    Returns:
        Transformed tensor of the same shape as *x*, same dtype.
    """
    original_dtype = x.dtype
    head_dim = x.shape[-1]

    if head_dim == 0:
        return x

    t0 = time.perf_counter_ns()

    # Pad if needed
    padded, original_n = pad_to_power_of_2(x)
    padded_dim = padded.shape[-1]

    # Get sign vector (padded length)
    device_str = str(x.device)
    d = _get_cached_sign_vector(padded_dim, seed, device_str).to(x.device)

    if inverse:
        # Inverse: FWHT first, then multiply by D
        result = fwht_iterative(padded)
        result = result * d
    else:
        # Forward: multiply by D, then FWHT
        result = padded * d
        result = fwht_iterative(result)

    # Trim if we padded
    if padded_dim != original_n:
        result = result.narrow(-1, 0, original_n)

    elapsed_ns = time.perf_counter_ns() - t0
    log.debug(
        "randomized_hadamard_transform",
        head_dim=head_dim,
        inverse=inverse,
        elapsed_ns=elapsed_ns,
    )

    return result.to(original_dtype)


# ======================================================================
# Benchmark
# ======================================================================


def benchmark_hadamard_vs_matmul(
    head_dims: list[int] | None = None,
    n_warmup: int = 10,
    n_iters: int = 100,
) -> dict[str, Any]:
    """Compare FWHT vs random matmul performance.

    Args:
        head_dims: List of dimensions to benchmark.
        n_warmup: Warm-up iterations.
        n_iters: Timed iterations.

    Returns:
        Dict mapping head_dim → speedup and throughput metrics.
    """
    if head_dims is None:
        head_dims = [64, 128, 256, 512]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda_events = device == "cuda"
    results: dict[str, Any] = {}

    for dim in head_dims:
        batch_shape = (4, 32, 128)
        x = torch.randn(*batch_shape, dim, dtype=torch.float16, device=device)

        # --- FWHT ---
        for _ in range(n_warmup):
            _ = randomized_hadamard_transform(x, seed=42)

        if use_cuda_events:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(n_iters):
            _ = randomized_hadamard_transform(x, seed=42)

        if use_cuda_events:
            torch.cuda.synchronize()
        fwht_ms = (time.perf_counter() - t0) / n_iters * 1000

        # --- Matmul ---
        gen = torch.Generator().manual_seed(42)
        mat = torch.randn(dim, dim, generator=gen, dtype=torch.float16, device=device)

        for _ in range(n_warmup):
            _ = x @ mat.T

        if use_cuda_events:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(n_iters):
            _ = x @ mat.T

        if use_cuda_events:
            torch.cuda.synchronize()
        matmul_ms = (time.perf_counter() - t0) / n_iters * 1000

        total_bytes = x.nelement() * x.element_size()
        fwht_gbs = total_bytes / (fwht_ms * 1e6) if fwht_ms > 0 else 0
        matmul_gbs = total_bytes / (matmul_ms * 1e6) if matmul_ms > 0 else 0
        speedup = matmul_ms / fwht_ms if fwht_ms > 0 else float("inf")

        results[str(dim)] = {
            "fwht_ms": round(fwht_ms, 4),
            "matmul_ms": round(matmul_ms, 4),
            "speedup": round(speedup, 2),
            "fwht_gb_s": round(fwht_gbs, 2),
            "matmul_gb_s": round(matmul_gbs, 2),
        }

    # Print table
    print(f"\n{'dim':>6} | {'FWHT (ms)':>10} | {'Matmul (ms)':>12} | {'Speedup':>8}")
    print("-" * 46)
    for dim_str, d in results.items():
        print(
            f"{dim_str:>6} | {d['fwht_ms']:>10.4f} | {d['matmul_ms']:>12.4f} | {d['speedup']:>7.2f}x"
        )
    print()

    return results
