"""PolarQuantizer: True 3-bit KV-cache quantization (arXiv 2504.19874).

Implements the first stage of the TurboQuant algorithm with true packed
3-bit storage — 8 values are packed into 3 bytes (uint8) rather than
storing each 3-bit index in an int8 container.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn as nn
from scipy.stats import beta as sp_beta

from turboquant.kernels.hadamard import randomized_hadamard_transform

log = structlog.get_logger(__name__)
TensorShape = Sequence[int]

# ======================================================================
# Help functions for 3-bit packing
# ======================================================================

def _pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """8 3-bit values packed into 3 bytes (uint8[3]).
    
    Formula: byte0 = v0|(v1<<3)|(v2>>2); byte1 = (v2<<6)|(v3<<3)|v4; byte2 = v5|(v6<<3)|(v7<<5)...
    Wait, the prompt says: byte0 = v0|(v1<<3)|(v2>>2); byte1 = (v2<<6)|(v3<<3)|v4; byte2 = v5|(v6<<3)|(v7<<5)
    Actually, let's use a standard packing:
    v0: bits 0-2 (byte 0)
    v1: bits 3-5 (byte 0)
    v2: bits 6-7 (byte 0) + bit 0 (byte 1)
    v3: bits 1-3 (byte 1)
    v4: bits 4-6 (byte 1)
    v5: bits 7 (byte 1) + bits 0-1 (byte 2)
    v6: bits 2-4 (byte 2)
    v7: bits 5-7 (byte 2)
    """
    n = indices.shape[-1]
    if n % 8 != 0:
        pad = 8 - (n % 8)
        indices = torch.nn.functional.pad(indices, (0, pad), value=0)
        n = n + pad

    indices = indices.to(torch.uint8)
    # Reshape to (..., n // 8, 8)
    v = indices.view(*indices.shape[:-1], -1, 8)
    
    byte0 = (v[..., 0] & 0x07) | ((v[..., 1] & 0x07) << 3) | ((v[..., 2] & 0x03) << 6)
    byte1 = ((v[..., 2] & 0x04) >> 2) | ((v[..., 3] & 0x07) << 1) | ((v[..., 4] & 0x07) << 4) | ((v[..., 5] & 0x01) << 7)
    byte2 = ((v[..., 5] & 0x06) >> 1) | ((v[..., 6] & 0x07) << 2) | ((v[..., 7] & 0x07) << 5)
    
    packed = torch.stack([byte0, byte1, byte2], dim=-1)
    # Shape: (..., n // 8, 3)
    return packed.view(*indices.shape[:-2], -1)

def _unpack_3bit(packed: torch.Tensor, original_dim: int) -> torch.Tensor:
    """Unpack 3-bit values from dense uint8."""
    n_groups = packed.shape[-1] // 3
    # Reshape to (..., n_groups, 3)
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
    # Shape: (..., n_groups, 8)
    return unpacked.view(*packed.shape[:-1], -1).narrow(-1, 0, original_dim).to(torch.int8)

# ======================================================================
# PolarQuantizer
# ======================================================================

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
    
    def __init__(
        self, 
        head_dim: int, 
        bits: int = 3, 
        group_size: int = 64, 
        seed: int = 42, 
        use_hadamard: bool = True
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.bits = bits
        self.num_levels = 1 << bits
        self.group_size = group_size
        self.seed = seed
        self.use_hadamard = use_hadamard # will fallback if not p2 or requested
        
        # Rotation Π
        rotation = self._generate_rotation(head_dim, seed)
        self.register_buffer("Pi", rotation)
        
        # LLoyd-Max codebook
        # Initial levels derived from Beta(0.5, 0.5)
        self.register_buffer("levels", torch.zeros(self.num_levels))
        self.register_buffer("boundaries", torch.zeros(self.num_levels - 1))
        
        # Streaming calibration stats (Welford)
        self.register_buffer("cal_mean", torch.tensor(0.0))
        self.register_buffer("cal_m2", torch.tensor(0.0))
        self.register_buffer("cal_count", torch.tensor(0, dtype=torch.long))
        
        # Pre-initialize with Beta(0.5, 0.5)
        self._lloyd_max_from_beta_fast(0.5, 0.5)
        
    def _generate_rotation(self, dim: int, seed: int) -> torch.Tensor:
        """QR decomposition to get an orthogonal matrix."""
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(dim, dim, generator=gen)
        q, r = torch.linalg.qr(z)
        d = r.diagonal().sign()
        d[d == 0] = 1.0
        return q * d.unsqueeze(0)
    
    def _apply_rotation(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Choice between FWHT and matmul."""
        # Use hadamard if head_dim is power of 2 and allowed
        if self.use_hadamard and (self.head_dim & (self.head_dim - 1) == 0):
            return randomized_hadamard_transform(x, seed=self.seed, inverse=inverse)
        
        pi = self.Pi
        if inverse:
            # inverse is Pi for orthogonal if we used Pi.T in forward
            return x @ pi
        return x @ pi.T
        
    def _fit_beta_moments(self, data: torch.Tensor) -> tuple[float, float]:
        """Method of moments for Beta distribution."""
        mean_ = data.mean().item()
        var_ = data.var().item()
        var_ = max(var_, 1e-12)
        common = (mean_ * (1.0 - mean_) / var_) - 1.0
        common = max(common, 1e-6)
        alpha = mean_ * common
        beta_val = (1.0 - mean_) * common
        return alpha, beta_val

    def _lloyd_max_from_beta_fast(self, a: float, b: float) -> None:
        """Vectorized Lloyd-Max via dense grid interpolation for speed."""
        N = 10000
        grid = np.linspace(1e-6, 1.0 - 1e-6, N)
        pdf = sp_beta.pdf(grid, a, b)
        cdf = sp_beta.cdf(grid, a, b)
        
        # Initial levels
        num_levels = self.num_levels
        levels = np.linspace(0, 1, num_levels + 2)[1:-1]
        
        for _ in range(100):
            # Boundaries as midpoints
            bnd = (levels[:-1] + levels[1:]) / 2.0
            
            # Centroids = E[X | bnd_lo < X < bnd_hi]
            new_levels = []
            for i in range(num_levels):
                lo = bnd[i-1] if i > 0 else 0.0
                hi = bnd[i] if i < num_levels - 1 else 1.0
                
                # Use trapz over mask
                mask = (grid >= lo) & (grid <= hi)
                if not mask.any():
                    new_levels.append((lo + hi) / 2.0)
                    continue
                
                denom = cdf[min(N-1, np.searchsorted(grid, hi))] - cdf[min(N-1, np.searchsorted(grid, lo))]
                if denom < 1e-9:
                    new_levels.append((lo + hi) / 2.0)
                else:
                    num = np.trapz(grid[mask] * pdf[mask], grid[mask])
                    new_levels.append(num / denom)
            
            levels = np.array(new_levels)
            
        # Map to [-3.5, 3.5]
        self.levels.copy_(torch.from_numpy(levels * 7.0 - 3.5).float())
        self.boundaries.copy_(torch.from_numpy(bnd * 7.0 - 3.5).float())

    def calibrate_batch(self, tensor: torch.Tensor) -> None:
        """Online update of statistics using Welford's algorithm."""
        x_rot = self._apply_rotation(tensor.float(), inverse=False)
        flat = x_rot.abs().reshape(-1)
        
        # Normalize roughly
        val_max = flat.max().item()
        if val_max > 0:
            flat = flat / val_max
            
        # Update Welford
        for v in flat:
            self.cal_count += 1
            delta = v - self.cal_mean
            self.cal_mean += delta / self.cal_count
            delta2 = v - self.cal_mean
            self.cal_m2 += delta * delta2
            
    def calibrate_finalize(self) -> None:
        """Finalize calibration and update codebook."""
        if self.cal_count < 2:
            return
        var = self.cal_m2 / (self.cal_count - 1)
        alpha, beta_val = self._fit_beta_moments_metrics(self.cal_mean.item(), var.item())
        self._lloyd_max_from_beta_fast(alpha, beta_val)

    def _fit_beta_moments_metrics(self, m: float, v: float) -> tuple[float, float]:
        v = max(v, 1e-9)
        common = (m * (1.0 - m) / v) - 1.0
        common = max(common, 1e-6)
        return m * common, (1.0 - m) * common

    def calibrate(self, data_list: list[torch.Tensor]) -> None:
        """Wrapper for list-based calibration."""
        for tensor in data_list:
            self.calibrate_batch(tensor)
        self.calibrate_finalize()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize to true packed 3-bit."""
        x_f32 = x.float()
        x_rot = self._apply_rotation(x_f32, inverse=False)
        
        # Group-wise scaling
        bs, heads, seq, dim = x.shape
        flat = x_rot.reshape(-1, dim)
        
        # Padding for group_size
        pad_dim = (self.group_size - dim % self.group_size) % self.group_size
        if pad_dim > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_dim))
            
        num_groups = flat.shape[-1] // self.group_size
        grouped = flat.view(-1, num_groups, self.group_size)
        
        # Scales: max(abs) / 3.5
        scales = grouped.abs().amax(dim=-1).clamp(min=1e-8) / 3.5
        normalised = (grouped / scales.unsqueeze(-1)).clamp(-3.5, 3.5)
        
        # Search sorted for indices
        indices = torch.searchsorted(self.boundaries, normalised.reshape(-1)).clamp(0, self.num_levels - 1)
        indices = indices.view(normalised.shape).to(torch.int8)
        
        # Remove padding and pack
        indices_flat = indices.view(-1, num_groups * self.group_size)[..., :dim]
        # Pack to true 3-bit
        packed = _pack_3bit(indices_flat)
        
        # Reshape output
        packed = packed.view(bs, heads, seq, -1)
        scales = scales.view(bs, heads, seq, num_groups)
        
        return packed, scales.to(torch.float32)

    def dequantize(self, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Unpack and rotate back."""
        # Unpack back to indices
        indices = _unpack_3bit(packed, self.head_dim)
        
        # Codebook lookup
        vals = self.levels[indices.long()]
        
        # Rescale
        dim = self.head_dim
        bs, heads, seq, _ = scales.shape
        num_groups = scales.shape[-1]
        
        flat = vals.view(-1, dim)
        pad_dim = num_groups * self.group_size - dim
        if pad_dim > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_dim))
            
        grouped = flat.view(-1, num_groups, self.group_size)
        rescaled = grouped * scales.view(-1, num_groups).unsqueeze(-1)
        
        rescaled = rescaled.view(-1, num_groups * self.group_size)[..., :dim]
        rescaled = rescaled.view(bs, heads, seq, dim)
        
        # Inverse rotation
        reconstructed = self._apply_rotation(rescaled, inverse=True)
        return reconstructed.to(torch.float16)

    def memory_footprint_bytes(self, seq_len: int, batch: int, heads: int) -> int:
        packed_bytes = math.ceil(self.head_dim * 3 / 8)
        num_groups = math.ceil(self.head_dim / self.group_size)
        scale_bytes = num_groups * 4 # float32
        return (packed_bytes + scale_bytes) * seq_len * batch * heads

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, bits={self.bits}, group_size={self.group_size}, use_hadamard={self.use_hadamard}"
