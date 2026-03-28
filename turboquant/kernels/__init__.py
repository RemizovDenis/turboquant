"""Kernel helpers for TurboQuant."""

from turboquant.kernels.hadamard import (
    benchmark_hadamard_vs_matmul,
    fwht_iterative,
    randomized_hadamard_transform,
)
from turboquant.kernels.triton_quant import (
    HAS_TRITON,
    benchmark_triton_kernels,
    dequantize_3bit_triton,
    quantize_3bit_triton,
)

__all__ = [
    "HAS_TRITON",
    "benchmark_hadamard_vs_matmul",
    "benchmark_triton_kernels",
    "dequantize_3bit_triton",
    "fwht_iterative",
    "quantize_3bit_triton",
    "randomized_hadamard_transform",
]
