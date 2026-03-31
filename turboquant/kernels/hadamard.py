"""Fast Walsh-Hadamard Transform for TurboQuant.

Provides efficient O(n log n) orthonormal transforms to replace O(n^2) 
random orthogonal matrices.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Any

def is_power_of_two(n: int) -> bool:
    """Check if n is a positive power of 2."""
    return n > 0 and (n & (n - 1)) == 0

def next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n."""
    if n <= 0:
        return 1
    return 2**math.ceil(math.log2(n))

def hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Apply Fast Walsh-Hadamard Transform along the last dimension.
    
    Algorithm: Vectorized butterfly pattern, O(n log n).
    
    Args:
        x: Input tensor, last dim must be a power of 2.
        normalize: If True, divide by sqrt(n) for an orthonormal transform.
        
    Returns:
        Transformed tensor of the same shape and dtype as input.
        
    Raises:
        ValueError: If last dim is not a power of 2.
    """
    n = x.shape[-1]
    if not is_power_of_two(n):
        raise ValueError(f"Last dimension must be a power of 2, got {n}")

    # Use float32 for internal computation to maintain precision
    orig_dtype = x.dtype
    x = x.float()
    
    # Vectorized butterfly implementation
    # Reshape and operate on blocks to avoid Python loops over n
    # The number of stages is log2(n)
    batch_shape = x.shape[:-1]
    n = x.shape[-1]
    # ...
    stages = int(math.log2(n))
    for stage in range(stages):
        step = 2**stage
        x = x.view(*batch_shape, -1, 2, step)
        a, b = x.unbind(dim=-2)
        x = torch.stack([a + b, a - b], dim=-2)
    
    x = x.view(*batch_shape, n)
    
    if normalize:
        x = x / math.sqrt(n)
        
    return x.to(orig_dtype)

def hadamard_transform_padded(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """FWHT with automatic zero-padding to the next power of 2.
    
    Args:
        x: Input tensor of any shape.
        normalize: Whether to normalize the transform.
        
    Returns:
        Transformed tensor, truncated back to the original head_dim.
    """
    d = x.shape[-1]
    if is_power_of_two(d):
        return hadamard_transform(x, normalize=normalize)
    
    n = next_power_of_two(d)
    pad = n - d
    x_padded = torch.nn.functional.pad(x, (0, pad))
    transformed = hadamard_transform(x_padded, normalize=normalize)
    return transformed[..., :d]

def randomized_hadamard_transform(x: torch.Tensor, seed: int = 42, inverse: bool = False) -> torch.Tensor:
    """Apply a randomized Hadamard transform (random signs + FWHT).
    
    Acts as a structured replacement for a random orthogonal matrix.
    
    Args:
        x: Input tensor.
        seed: Seed for random signs.
        inverse: If True, applies inverse (which is the same for Hadamard).
        
    Returns:
        Transformed tensor.
    """
    d = x.shape[-1]
    n = next_power_of_two(d)
    
    # Generate random signs ±1
    gen = torch.Generator(device=x.device).manual_seed(seed)
    # We need n signs even if d is not n
    signs = torch.randint(0, 2, (d,), generator=gen, device=x.device, dtype=x.dtype) * 2 - 1
    
    if not inverse:
        # Pre-multiply by signs
        x = x * signs
        return hadamard_transform_padded(x, normalize=True)
    else:
        # FWHT is self-inverse (with normalization)
        x = hadamard_transform_padded(x, normalize=True)
        # Post-multiply by signs (signs are self-inverse)
        return x * signs

def benchmark_hadamard_vs_matmul(head_dim: int, n_iters: int = 100) -> dict[str, float]:
    """Benchmark FWHT vs standard matmul for a given dimension."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 32, 1024, head_dim, device=device, dtype=torch.float16)
    
    # Pre-generate rotation matrix for matmul
    if is_power_of_two(head_dim):
        n = head_dim
    else:
        n = next_power_of_two(head_dim)
    
    # Warmup
    for _ in range(10):
        hadamard_transform_padded(x)
        
    # Matmul benchmark
    matrix = torch.randn(head_dim, head_dim, device=device, dtype=torch.float16)
    if device == "cuda":
        torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = x @ matrix
    if device == "cuda":
        torch.cuda.synchronize()
    matmul_ms = (time.perf_counter() - t0) * 1000 / n_iters
    
    # Hadamard benchmark
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(n_iters):
        _ = hadamard_transform_padded(x)
    if device == "cuda":
        torch.cuda.synchronize()
    hadamard_ms = (time.perf_counter() - t1) * 1000 / n_iters
    
    return {
        "hadamard_ms": hadamard_ms,
        "matmul_ms": matmul_ms,
        "speedup": matmul_ms / max(hadamard_ms, 1e-9)
    }

import time # For benchmark
