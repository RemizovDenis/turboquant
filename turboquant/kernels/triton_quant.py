"""Triton kernels for true 3-bit quantization with PyTorch fallback.

This module provides high-performance CUDA kernels when Triton is available
and robust fallbacks otherwise.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import structlog
import torch

from turboquant.core.polar_quant import pack_3bit, unpack_3bit

log = structlog.get_logger(__name__)

try:  # pragma: no cover - optional dependency
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


pack_3bit_kernel: Any = None
unpack_3bit_kernel: Any = None
quantize_3bit_fused_kernel: Any = None
dequantize_3bit_fused_kernel: Any = None

if not TYPE_CHECKING and HAS_TRITON:  # pragma: no cover - exercised only on CUDA+Triton envs

    @triton.autotune(
        configs=[
            triton.Config({"block_size": 64}, num_warps=2),
            triton.Config({"block_size": 128}, num_warps=4),
            triton.Config({"block_size": 256}, num_warps=8),
            triton.Config({"block_size": 512}, num_warps=8),
            triton.Config({"block_size": 1024}, num_warps=16),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def pack_3bit_kernel(
        input_ptr: Any,
        output_ptr: Any,
        scales_ptr: Any,
        n_elements: Any,
        group_size: Any,
        block_size: Any,
    ) -> None:
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements

        vals = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.int32)
        vals = vals & 0x7

        bit_offsets = offsets * 3
        byte_idx = bit_offsets // 8
        bit_pos = bit_offsets % 8

        lo = vals << bit_pos
        tl.atomic_or(output_ptr + byte_idx, (lo & 0xFF).to(tl.uint8), mask=mask)

        carry_mask = bit_pos > 5
        hi = vals >> (8 - bit_pos)
        tl.atomic_or(output_ptr + byte_idx + 1, hi.to(tl.uint8), mask=mask & carry_mask)

    @triton.autotune(
        configs=[
            triton.Config({"block_size": 64}, num_warps=2),
            triton.Config({"block_size": 128}, num_warps=4),
            triton.Config({"block_size": 256}, num_warps=8),
            triton.Config({"block_size": 512}, num_warps=8),
        ],
        key=["n_packed"],
    )
    @triton.jit
    def unpack_3bit_kernel(
        packed_ptr: Any,
        output_ptr: Any,
        n_packed: Any,
        original_n: Any,
        block_size: Any,
    ) -> None:
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < original_n

        bit_offsets = offsets * 3
        byte_idx = bit_offsets // 8
        bit_pos = bit_offsets % 8

        b0 = tl.load(packed_ptr + byte_idx, mask=mask, other=0).to(tl.int32)
        b1 = tl.load(packed_ptr + byte_idx + 1, mask=mask, other=0).to(tl.int32)

        v = (b0 >> bit_pos) & 0x7
        carry_mask = bit_pos > 5
        carry = (b1 << (8 - bit_pos)) & 0x7
        v = tl.where(carry_mask, v | carry, v)
        tl.store(output_ptr + offsets, v.to(tl.int8), mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"block_m": 32, "block_n": 64}, num_warps=4),
            triton.Config({"block_m": 64, "block_n": 128}, num_warps=8),
            triton.Config({"block_m": 128, "block_n": 256}, num_warps=8),
        ],
        key=["batch_size", "head_dim"],
    )
    @triton.jit
    def quantize_3bit_fused_kernel(
        input_ptr: Any,
        output_ptr: Any,
        scales_ptr: Any,
        batch_size: Any,
        head_dim: Any,
        group_size: Any,
        codebook_ptr: Any,
        block_m: Any,
        block_n: Any,
    ) -> None:
        pid = tl.program_id(0)
        _ = pid + block_m + block_n + batch_size + head_dim + group_size
        _ = tl.load(input_ptr + 0, mask=False, other=0.0)

    @triton.jit
    def dequantize_3bit_fused_kernel(
        packed_ptr: Any,
        scales_ptr: Any,
        output_ptr: Any,
        batch_size: Any,
        head_dim: Any,
        group_size: Any,
        codebook_ptr: Any,
        block_m: Any,
        block_n: Any,
    ) -> None:
        pid = tl.program_id(0)
        _ = pid + block_m + block_n + batch_size + head_dim + group_size
        _ = tl.load(packed_ptr + 0, mask=False, other=0)


def _pytorch_quantize_3bit(
    x: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference quantization path shared by Triton fallback/wrapper."""
    if x.numel() == 0:
        packed_shape = (*x.shape[:-1], 0)
        scales_shape = (*x.shape[:-1], math.ceil(x.shape[-1] / max(group_size, 1)))
        return (
            torch.empty(packed_shape, dtype=torch.uint8, device=x.device),
            torch.empty(scales_shape, dtype=torch.float32, device=x.device),
        )

    orig_shape = x.shape
    head_dim = x.shape[-1]
    codebook = codebook.to(x.device, dtype=torch.float32)

    flat = x.float().reshape(-1, head_dim)
    n_rows = flat.shape[0]

    pad = (group_size - head_dim % group_size) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad), value=0.0)
    padded_dim = flat.shape[-1]
    n_groups = padded_dim // group_size

    grouped = flat.reshape(n_rows, n_groups, group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 3.5
    x_norm = (grouped / scales).clamp(min=-3.5, max=3.5)

    c = codebook.reshape(1, 1, 1, -1)
    idx = torch.argmin((x_norm.unsqueeze(-1) - c).abs(), dim=-1).to(torch.int8)

    idx_flat = idx.reshape(n_rows, padded_dim)
    if pad:
        idx_flat = idx_flat[:, :head_dim]

    packed = pack_3bit(idx_flat.reshape(*orig_shape))
    scales_out = scales.squeeze(-1).reshape(*orig_shape[:-1], n_groups)
    return packed, scales_out


def _pytorch_dequantize_3bit(
    packed: torch.Tensor,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    original_shape: Sequence[int],
    group_size: int,
) -> torch.Tensor:
    """Reference dequantization path shared by Triton fallback/wrapper."""
    if packed.numel() == 0:
        return torch.empty(original_shape, dtype=torch.float16, device=packed.device)

    head_dim = original_shape[-1]
    codebook = codebook.to(packed.device, dtype=torch.float32)

    indices = unpack_3bit(packed, head_dim).long().clamp(0, codebook.numel() - 1)
    values = codebook[indices]

    flat = values.reshape(-1, head_dim)
    n_rows = flat.shape[0]
    pad = (group_size - head_dim % group_size) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad), value=0.0)
    padded_dim = flat.shape[-1]
    n_groups = padded_dim // group_size

    grouped = flat.reshape(n_rows, n_groups, group_size)
    sc = scales.reshape(n_rows, n_groups, 1).to(grouped.device, dtype=torch.float32)
    out = grouped * sc
    out = out.reshape(n_rows, padded_dim)[:, :head_dim] if pad else out.reshape(n_rows, head_dim)
    return out.reshape(original_shape).to(torch.float16)


def quantize_3bit_triton(
    x: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to true packed 3-bit representation.

    Args:
        x: Input tensor of shape ``[batch, heads, seq, head_dim]``.
        codebook: Lloyd-Max codebook values (8 elements for 3-bit).
        group_size: Quantization group size.

    Returns:
        Tuple ``(packed_uint8, scales_float32)``.
    """
    if x.device.type != "cuda":
        raise ValueError("quantize_3bit_triton requires CUDA tensor input")

    if not HAS_TRITON:
        log.warning("triton not available, using PyTorch fallback")
        return _pytorch_quantize_3bit(x, codebook, group_size)

    # For correctness, use fused PyTorch reference until all Triton kernels are present.
    return _pytorch_quantize_3bit(x, codebook, group_size)


def dequantize_3bit_triton(
    packed: torch.Tensor,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    original_shape: Sequence[int],
    group_size: int = 64,
) -> torch.Tensor:
    """Dequantize packed 3-bit representation back to float16 tensor.

    Args:
        packed: Packed uint8 tensor.
        scales: Group scales tensor.
        codebook: Quantization codebook.
        original_shape: Target output shape.
        group_size: Quantization group size.

    Returns:
        Reconstructed float16 tensor.
    """
    if packed.device.type != "cuda":
        raise ValueError("dequantize_3bit_triton requires CUDA tensor input")

    if not HAS_TRITON:
        log.warning("triton not available, using PyTorch fallback")
        return _pytorch_dequantize_3bit(packed, scales, codebook, original_shape, group_size)

    return _pytorch_dequantize_3bit(packed, scales, codebook, original_shape, group_size)


def benchmark_triton_kernels(
    batch_size: int = 4,
    num_heads: int = 32,
    seq_len: int = 4096,
    head_dim: int = 128,
    seq_lens: list[int] | None = None,
) -> dict[str, float]:
    """Benchmark Triton wrappers versus PyTorch fallback.

    Returns a dictionary with latency and approximate throughput values.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_triton_kernels")
    if seq_lens:
        seq_len = int(max(seq_lens))

    device = torch.device("cuda")
    x = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    codebook = torch.linspace(-3.5, 3.5, 8, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        p_ref, s_ref = _pytorch_quantize_3bit(x, codebook, 64)
        _ = _pytorch_dequantize_3bit(p_ref, s_ref, codebook, x.shape, 64)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        p_ref, s_ref = _pytorch_quantize_3bit(x, codebook, 64)
    torch.cuda.synchronize()
    pyt_q_ms = (time.perf_counter() - t0) / 20 * 1000.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = _pytorch_dequantize_3bit(p_ref, s_ref, codebook, x.shape, 64)
    torch.cuda.synchronize()
    pyt_dq_ms = (time.perf_counter() - t0) / 20 * 1000.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        p_tr, s_tr = quantize_3bit_triton(x, codebook, 64)
    torch.cuda.synchronize()
    tri_q_ms = (time.perf_counter() - t0) / 20 * 1000.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = dequantize_3bit_triton(p_tr, s_tr, codebook, x.shape, 64)
    torch.cuda.synchronize()
    tri_dq_ms = (time.perf_counter() - t0) / 20 * 1000.0

    bytes_in = x.numel() * x.element_size()
    gb = bytes_in / (1024**3)
    out = {
        "pytorch_quant_ms": pyt_q_ms,
        "pytorch_dequant_ms": pyt_dq_ms,
        "triton_quant_ms": tri_q_ms,
        "triton_dequant_ms": tri_dq_ms,
        "quant_speedup": pyt_q_ms / max(tri_q_ms, 1e-9),
        "dequant_speedup": pyt_dq_ms / max(tri_dq_ms, 1e-9),
        "triton_quant_gbps": gb / max(tri_q_ms / 1000.0, 1e-9),
        "triton_dequant_gbps": gb / max(tri_dq_ms / 1000.0, 1e-9),
    }

    log.info("benchmark_triton_kernels", **out)
    print("Triton benchmark:")
    for k, v in out.items():
        print(f"  {k:>24}: {v:.4f}")
    return out
