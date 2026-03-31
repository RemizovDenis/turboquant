"""Kernel helpers for TurboQuant v0.3.0."""

from turboquant.kernels.hadamard import (
    hadamard_transform,
    hadamard_transform_padded,
    randomized_hadamard_transform,
    benchmark_hadamard_vs_matmul,
)
from turboquant.kernels.triton_quant import (
    HAS_TRITON,
    benchmark_triton_kernels,
    dequant_3bit,
)

__all__ = [
    "HAS_TRITON",
    "hadamard_transform",
    "hadamard_transform_padded",
    "randomized_hadamard_transform",
    "benchmark_hadamard_vs_matmul",
    "benchmark_triton_kernels",
    "dequant_3bit",
]
