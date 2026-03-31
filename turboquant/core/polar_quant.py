"""PolarQuantizer: True 3-bit KV-cache quantization (arXiv 2504.19874).

Implements the first stage of the TurboQuant algorithm with true packed
3-bit storage — 8 values are packed into 3 bytes (uint8) rather than
storing each 3-bit index in an int8 container.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import structlog
import torch
import torch.nn as nn
from scipy.stats import beta as sp_beta

from turboquant.kernels.hadamard import randomized_hadamard_transform

log = structlog.get_logger(__name__)
TensorShape = Sequence[int]


def _pack_1bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 8 1-bit values into 1 byte."""
    n = indices.shape[-1]
    if n % 8 != 0:
        pad = 8 - (n % 8)
        indices = torch.nn.functional.pad(indices, (0, pad), value=0)
    v = indices.to(torch.uint8).view(*indices.shape[:-1], -1, 8)
    packed = (
        (v[..., 0] << 0)
        | (v[..., 1] << 1)
        | (v[..., 2] << 2)
        | (v[..., 3] << 3)
        | (v[..., 4] << 4)
        | (v[..., 5] << 5)
        | (v[..., 6] << 6)
        | (v[..., 7] << 7)
    )
    return packed


def _unpack_1bit(packed: torch.Tensor, original_dim: int) -> torch.Tensor:
    """Unpack 1-bit values from uint8."""
    v = packed.unsqueeze(-1)
    unpacked = torch.stack([(v >> i) & 1 for i in range(8)], dim=-1)
    return unpacked.view(*packed.shape[:-1], -1).narrow(-1, 0, original_dim).to(torch.int8)


def _pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4 2-bit values into 1 byte."""
    n = indices.shape[-1]
    if n % 4 != 0:
        pad = 4 - (n % 4)
        indices = torch.nn.functional.pad(indices, (0, pad), value=0)
    v = indices.to(torch.uint8).view(*indices.shape[:-1], -1, 4)
    packed = (
        (v[..., 0] & 0x03)
        | ((v[..., 1] & 0x03) << 2)
        | ((v[..., 2] & 0x03) << 4)
        | ((v[..., 3] & 0x03) << 6)
    )
    return packed


def _unpack_2bit(packed: torch.Tensor, original_dim: int) -> torch.Tensor:
    """Unpack 2-bit values from uint8."""
    v = packed.unsqueeze(-1)
    unpacked = torch.stack([(v >> (2 * i)) & 0x03 for i in range(4)], dim=-1)
    return unpacked.view(*packed.shape[:-1], -1).narrow(-1, 0, original_dim).to(torch.int8)


def _pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """8 3-bit values packed into 3 bytes (uint8[3])."""
    n = indices.shape[-1]
    if n % 8 != 0:
        pad = 8 - (n % 8)
        indices = torch.nn.functional.pad(indices, (0, pad), value=0)
    indices = indices.to(torch.uint8)
    v = indices.view(*indices.shape[:-1], -1, 8)
    byte0 = (v[..., 0] & 0x07) | ((v[..., 1] & 0x07) << 3) | ((v[..., 2] & 0x03) << 6)
    byte1 = (
        ((v[..., 2] & 0x04) >> 2)
        | ((v[..., 3] & 0x07) << 1)
        | ((v[..., 4] & 0x07) << 4)
        | ((v[..., 5] & 0x01) << 7)
    )
    byte2 = ((v[..., 5] & 0x06) >> 1) | ((v[..., 6] & 0x07) << 2) | ((v[..., 7] & 0x07) << 5)
    packed = torch.stack([byte0, byte1, byte2], dim=-1)
    return packed.view(*indices.shape[:-2], -1)


def _unpack_3bit(packed: torch.Tensor, original_dim: int) -> torch.Tensor:
    """Unpack 3-bit values from dense uint8."""
    n_groups = packed.shape[-1] // 3
    p = packed.view(*packed.shape[:-1], n_groups, 3)
    byte0, byte1, byte2 = p[..., 0], p[..., 1], p[..., 2]
    v0 = byte0 & 0x07
    v1 = (byte0 >> 3) & 0x07
    v2 = (byte0 >> 6) | ((byte1 & 0x01) << 2)
    v3 = (byte1 >> 1) & 0x07
    v4 = (byte1 >> 4) & 0x07
    v5 = (byte1 >> 7) | ((byte2 & 0x03) << 1)
    v6 = (byte2 >> 2) & 0x07
    v7 = (byte2 >> 5) & 0x07
    unpacked = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1)
    return unpacked.view(*packed.shape[:-1], -1).narrow(-1, 0, original_dim).to(torch.int8)


@dataclass
class PolarQuantConfig:
    """Configuration for PolarQuantizer."""

    head_dim: int
    bits: int = 3
    group_size: int = 64
    seed: int = 42
    use_hadamard: bool = True


class PolarQuantizer(nn.Module):
    """Principal ML version of PolarQuantizer with true 3-bit packing."""

    Pi: torch.Tensor
    levels: torch.Tensor
    boundaries: torch.Tensor
    cal_mean: torch.Tensor
    cal_m2: torch.Tensor
    cal_count: torch.Tensor
    calibrated: torch.Tensor

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 3,
        group_size: int = 64,
        seed: int = 42,
        use_hadamard: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(head_dim, PolarQuantConfig):
            config = head_dim
            head_dim = config.head_dim
            bits = config.bits
            group_size = config.group_size
            seed = config.seed
            use_hadamard = config.use_hadamard

        if head_dim <= 0 or bits <= 0 or group_size <= 0:
            raise ValueError("Dimensions and bits must be positive")

        self.head_dim = head_dim
        self.bits = bits
        self.num_levels = 1 << bits
        self.group_size = group_size
        self.seed = seed
        self.use_hadamard = use_hadamard

        # Rotation Π
        rotation = self._generate_rotation(head_dim, seed)
        self.register_buffer("Pi", rotation)

        # Buffers
        self.register_buffer("levels", torch.zeros(self.num_levels))
        self.register_buffer("boundaries", torch.zeros(self.num_levels - 1))
        self.register_buffer("cal_mean", torch.tensor(0.0))
        self.register_buffer("cal_m2", torch.tensor(0.0))
        self.register_buffer("cal_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("calibrated", torch.tensor(0, dtype=torch.long))

        # Init codebook
        self._lloyd_max_from_beta_fast(0.5, 0.5)

    @property
    def rotation(self) -> torch.Tensor:
        """Legacy compatibility alias for Pi."""
        return self.Pi

    def _generate_rotation(self, dim: int, seed: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(dim, dim, generator=gen)
        q, r = torch.linalg.qr(z)
        d = r.diagonal().sign()
        d[d == 0] = 1.0
        return cast(torch.Tensor, q * d.unsqueeze(0))

    def _apply_rotation(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if self.use_hadamard and (self.head_dim & (self.head_dim - 1) == 0):
            return randomized_hadamard_transform(x, seed=self.seed, inverse=inverse)
        return x @ (self.Pi if inverse else self.Pi.T)

    def _lloyd_max_from_beta_fast(self, a: float, b: float) -> None:
        n_bins = 10000
        grid = np.linspace(1e-6, 1.0 - 1e-6, n_bins)
        pdf = sp_beta.pdf(grid, a, b)
        cdf = sp_beta.cdf(grid, a, b)
        num_levels = self.num_levels
        levels = np.linspace(0, 1, num_levels + 2)[1:-1]

        for _ in range(100):
            bnd = (levels[:-1] + levels[1:]) / 2.0
            new_levels = []
            for i in range(num_levels):
                lo = bnd[i - 1] if i > 0 else 0.0
                hi = bnd[i] if i < num_levels - 1 else 1.0
                mask = (grid >= lo) & (grid <= hi)
                if not mask.any():
                    new_levels.append((lo + hi) / 2.0)
                    continue
                denom = (
                    cdf[min(n_bins - 1, np.searchsorted(grid, hi))]
                    - cdf[min(n_bins - 1, np.searchsorted(grid, lo))]
                )
                if denom < 1e-9:
                    new_levels.append((lo + hi) / 2.0)
                else:
                    try:
                        from scipy.integrate import trapezoid
                    except ImportError:
                        from scipy.integrate import trapz as trapezoid
                    num = trapezoid(grid[mask] * pdf[mask], grid[mask])
                    new_levels.append(num / (denom + 1e-12))
            levels = np.array(new_levels)

        self.levels.copy_(torch.from_numpy(levels * 7.0 - 3.5).float())
        self.boundaries.copy_(torch.from_numpy(bnd * 7.0 - 3.5).float())

    def calibrate_batch(self, tensor: torch.Tensor) -> None:
        x_rot = self._apply_rotation(tensor.float(), inverse=False)
        flat = x_rot.abs().reshape(-1)
        val_max = flat.max().item()
        if val_max > 0:
            flat = flat / val_max
        for v in flat:
            self.cal_count += 1
            delta = v - self.cal_mean
            self.cal_mean += delta / self.cal_count
            delta2 = v - self.cal_mean
            self.cal_m2 += delta * delta2

    def calibrate_finalize(self) -> None:
        if self.cal_count.item() < 2:
            return
        m, v_val = self.cal_mean.item(), (self.cal_m2 / (self.cal_count - 1)).item()
        v_val = max(float(v_val), 1e-9)
        common = (m * (1.0 - m) / v_val) - 1.0
        common = max(common, 1e-6)
        alpha, beta_val = m * common, (1.0 - m) * common
        self._lloyd_max_from_beta_fast(float(alpha), float(beta_val))
        self.calibrated.fill_(1)

    def calibrate(self, data_list: list[torch.Tensor]) -> None:
        if not data_list:
            raise ValueError("data_list cannot be empty")
        for t in data_list:
            self.calibrate_batch(t)
        self.calibrate_finalize()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.numel() == 0:
            # Handle empty tensors
            return torch.zeros(0, dtype=torch.uint8, device=x.device), torch.zeros(
                0, dtype=torch.float32, device=x.device
            )

        if x.ndim == 2:
            x = x.unsqueeze(1).unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(f"Expected 2D or 4D tensor, got {x.ndim}D")

        bs, heads, seq, dim = x.shape
        x_rot = self._apply_rotation(x.float(), inverse=False)
        flat = x_rot.reshape(-1, dim)
        pad_dim = (self.group_size - dim % self.group_size) % self.group_size
        if pad_dim > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_dim))
        num_groups = flat.shape[-1] // self.group_size
        grouped = flat.view(-1, num_groups, self.group_size)
        scales = grouped.abs().amax(dim=-1).clamp(min=1e-8) / 3.5
        normalised = (grouped / scales.unsqueeze(-1)).clamp(-3.5, 3.5)
        indices = torch.searchsorted(self.boundaries, normalised.reshape(-1)).clamp(
            0, self.num_levels - 1
        )
        indices = indices.view(normalised.shape).to(torch.int8)
        indices_flat = indices.view(-1, num_groups * self.group_size)[..., :dim]

        if self.bits == 1:
            packed = _pack_1bit(indices_flat)
        elif self.bits == 2:
            packed = _pack_2bit(indices_flat)
        elif self.bits == 3:
            packed = _pack_3bit(indices_flat)
        else:
            # Fallback to int8
            packed = indices_flat.to(torch.uint8)

        return packed.view(bs, heads, seq, -1), scales.view(bs, heads, seq, num_groups).to(
            torch.float16
        )

    def dequantize(self, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        if packed.numel() == 0:
            return torch.zeros(0, dtype=torch.float16, device=packed.device)

        # Unpack
        if self.bits == 1:
            indices = _unpack_1bit(packed, self.head_dim)
        elif self.bits == 2:
            indices = _unpack_2bit(packed, self.head_dim)
        elif self.bits == 3:
            indices = _unpack_3bit(packed, self.head_dim)
        else:
            indices = packed.to(torch.int8)

        vals = self.levels[indices.reshape(-1).long()].view(*indices.shape)

        # Rescale
        if scales.ndim == 4:
            bs, heads, seq, num_groups = scales.shape
        else:
            # Fallback for 2D scales or other shapes
            num_groups = scales.numel() // (indices.numel() // self.head_dim)
            bs, heads, seq = 1, 1, indices.numel() // self.head_dim

        dim = self.head_dim
        flat = vals.view(-1, dim)
        pad_dim = num_groups * self.group_size - dim
        if pad_dim > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_dim))
        grouped = flat.view(-1, num_groups, self.group_size)
        rescaled = grouped * scales.reshape(-1, num_groups).unsqueeze(-1)
        rescaled = rescaled.view(-1, num_groups * self.group_size)[..., :dim]

        reconstructed = self._apply_rotation(rescaled.view(bs, heads, seq, dim), inverse=True)
        return reconstructed.to(torch.float16)

    def memory_footprint_bytes(self, seq_len: int, batch: int, heads: int) -> int:
        packed_bytes = math.ceil(self.head_dim * self.bits / 8)
        num_groups = math.ceil(self.head_dim / self.group_size)
        scale_bytes = num_groups * 2  # float16
        return (packed_bytes + scale_bytes) * seq_len * batch * heads

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, bits={self.bits}, use_hadamard={self.use_hadamard}"
