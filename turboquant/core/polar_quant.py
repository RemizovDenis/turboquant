"""PolarQuantizer: True 3-bit KV-cache quantization (arXiv 2504.19874).

Implements the first stage of the TurboQuant algorithm with true packed
3-bit storage — 8 values are packed into 3 bytes (uint8) rather than
storing each 3-bit index in an int8 container.

Compression: ~81% memory savings vs FP16 (true 3-bit packing).

Example::

    config = PolarQuantConfig(head_dim=128, bits=3, group_size=64)
    quantizer = PolarQuantizer(config)
    x = torch.randn(2, 32, 4096, 128, dtype=torch.float16, device="cuda")
    packed, scales = quantizer(x)
    # packed.nbytes / x.nbytes ≈ 0.19 (true 3-bit storage)
    x_hat = quantizer.dequantize(packed, scales, x.shape)
    mse = ((x - x_hat) ** 2).mean()
    # mse < 0.001 after calibration on real data
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn as nn

log = structlog.get_logger(__name__)
TensorShape = Sequence[int]

# ---------------------------------------------------------------------------
# Optional imports (degrade gracefully)
# ---------------------------------------------------------------------------
try:
    from scipy.stats import beta as _scipy_beta

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from safetensors.torch import load_file, save_file

    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class PolarQuantConfig:
    """Configuration for :class:`PolarQuantizer`.

    Attributes:
        head_dim: Dimension of each attention head.
        bits: Quantization bit-width (default 3 → 8 levels).
        group_size: Number of elements per quantization group.
        seed: RNG seed for reproducible rotation matrix.
        beta_alpha: Initial Beta-distribution α (refined via :meth:`calibrate`).
        beta_beta: Initial Beta-distribution β (refined via :meth:`calibrate`).
        lloyd_max_iters: Iterations for Lloyd-Max codebook optimisation.
        use_hadamard: Use FWHT instead of matmul when *head_dim* is a power of 2.
    """

    head_dim: int
    bits: int = 3
    group_size: int = 64
    seed: int = 42
    beta_alpha: float = 0.5
    beta_beta: float = 0.5
    lloyd_max_iters: int = 100
    use_hadamard: bool = True


# ======================================================================
# Rotation helpers
# ======================================================================


def _random_orthogonal(dim: int, seed: int) -> torch.Tensor:
    """Generate a random orthogonal matrix via QR decomposition.

    Args:
        dim: Matrix dimension.
        seed: Random seed.

    Returns:
        Orthogonal matrix of shape ``(dim, dim)`` in float32.
    """
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(dim, dim, generator=gen)
    q, r = torch.linalg.qr(z)
    # Ensure deterministic sign convention
    d = torch.diag(r)
    ph = d.sign()
    ph[ph == 0] = 1.0
    q = q * ph.unsqueeze(0)
    return torch.as_tensor(q, dtype=torch.float32)


# ======================================================================
# True 3-bit packing
# ======================================================================


def pack_3bit(x: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit values (0-7) into dense uint8 representation.

    Every 8 input values are packed into 3 bytes:

    * Byte 0: ``v0[2:0] | v1[2:0] | v2[1:0]``
    * Byte 1: ``v2[0]   | v3[2:0] | v4[2:0] | v5[0]``
    * Byte 2: ``v5[1:0] | v6[2:0] | v7[2:0]``

    Args:
        x: Int8 tensor with values in ``[0, 7]``.  Last dimension is packed.

    Returns:
        Uint8 tensor with last dimension ``ceil(N*3/8)``.
    """
    if x.numel() == 0:
        out_n = 0
        out_shape = list(x.shape[:-1]) + [out_n]
        return torch.empty(out_shape, dtype=torch.uint8, device=x.device)

    leading = x.shape[:-1]
    n = x.shape[-1]
    flat = x.reshape(-1, n).to(torch.int32)

    # Pad to multiple of 8
    pad_n = (8 - n % 8) % 8
    if pad_n:
        flat = torch.nn.functional.pad(flat, (0, pad_n), value=0)
    padded_n = flat.shape[-1]
    groups = padded_n // 8
    grouped = flat.reshape(-1, groups, 8)  # (B, G, 8)

    g0 = grouped.select(-1, 0)
    g1 = grouped.select(-1, 1)
    g2 = grouped.select(-1, 2)
    g3 = grouped.select(-1, 3)
    g4 = grouped.select(-1, 4)
    g5 = grouped.select(-1, 5)
    g6 = grouped.select(-1, 6)
    g7 = grouped.select(-1, 7)

    b0 = (g0 & 0x7) | ((g1 & 0x7) << 3) | ((g2 & 0x3) << 6)
    b1 = ((g2 >> 2) & 0x1) | ((g3 & 0x7) << 1) | ((g4 & 0x7) << 4) | ((g5 & 0x1) << 7)
    b2 = ((g5 >> 1) & 0x3) | ((g6 & 0x7) << 2) | ((g7 & 0x7) << 5)

    packed = torch.stack([b0, b1, b2], dim=-1)  # (B, G, 3)
    packed = packed.reshape(-1, groups * 3).to(torch.uint8)

    out_bytes = math.ceil(n * 3 / 8)
    packed = packed.narrow(-1, 0, out_bytes)
    return packed.reshape(*leading, out_bytes)


def unpack_3bit(packed: torch.Tensor, original_n: int) -> torch.Tensor:
    """Unpack dense uint8 back to 3-bit int8 values.

    Args:
        packed: Uint8 tensor from :func:`pack_3bit`.
        original_n: Original number of values in the last dimension.

    Returns:
        Int8 tensor with last dimension ``original_n``, values in ``[0, 7]``.
    """
    if packed.numel() == 0:
        out_shape = list(packed.shape[:-1]) + [original_n]
        return torch.empty(out_shape, dtype=torch.int8, device=packed.device)

    leading = packed.shape[:-1]
    flat = packed.reshape(-1, packed.shape[-1]).to(torch.int32)

    # Pad to multiple of 3
    pad_bytes = (3 - flat.shape[-1] % 3) % 3
    if pad_bytes:
        flat = torch.nn.functional.pad(flat, (0, pad_bytes), value=0)
    groups = flat.shape[-1] // 3
    grouped = flat.reshape(-1, groups, 3)  # (B, G, 3)

    b0 = grouped.select(-1, 0)
    b1 = grouped.select(-1, 1)
    b2 = grouped.select(-1, 2)

    v0 = b0 & 0x7
    v1 = (b0 >> 3) & 0x7
    v2 = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
    v3 = (b1 >> 1) & 0x7
    v4 = (b1 >> 4) & 0x7
    v5 = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
    v6 = (b2 >> 2) & 0x7
    v7 = (b2 >> 5) & 0x7

    unpacked = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1)
    unpacked = unpacked.reshape(-1, groups * 8)
    unpacked = unpacked.narrow(-1, 0, original_n)
    return unpacked.reshape(*leading, original_n).to(torch.int8)


# ======================================================================
# PolarQuantizer
# ======================================================================


class PolarQuantizer(nn.Module):
    """3-bit polar quantizer with true packed storage.

    Applies random orthogonal rotation followed by group-wise Lloyd-Max
    quantization with Beta-distribution-optimal codebook.  Values are
    packed into true 3-bit format (8 values → 3 bytes) for maximum
    memory efficiency.

    Args:
        config: :class:`PolarQuantConfig` instance.

    Example::

        config = PolarQuantConfig(head_dim=128)
        q = PolarQuantizer(config)
        x = torch.randn(1, 8, 64, 128, dtype=torch.float16)
        packed, scales = q(x)
        recon = q.dequantize(packed, scales, x.shape)
    """

    def __init__(
        self,
        config: PolarQuantConfig | None = None,
        *,
        head_dim: int | None = None,
        bits: int = 3,
        group_size: int = 64,
        seed: int = 42,
        beta_alpha: float = 0.5,
        beta_beta: float = 0.5,
        lloyd_max_iters: int = 100,
        use_hadamard: bool = True,
    ) -> None:
        super().__init__()
        # Backward-compatible constructor:
        # Legacy constructor support for PolarQuantizer(head_dim=?, bits=?).
        self._legacy_api = config is None
        if config is None:
            if head_dim is None:
                raise ValueError("Either config or head_dim must be provided")
            config = PolarQuantConfig(
                head_dim=head_dim,
                bits=bits,
                group_size=group_size,
                seed=seed,
                beta_alpha=beta_alpha,
                beta_beta=beta_beta,
                lloyd_max_iters=lloyd_max_iters,
                use_hadamard=use_hadamard,
            )

        if config.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {config.head_dim}")
        if config.bits < 1 or config.bits > 8:
            raise ValueError(f"bits must be 1-8, got {config.bits}")
        if config.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {config.group_size}")

        self.config = config
        self.head_dim = config.head_dim
        self.bits = config.bits
        self.num_levels = 1 << config.bits  # 2^bits
        self.group_size = config.group_size
        self.seed = config.seed
        self.use_hadamard = config.use_hadamard and _is_power_of_2(config.head_dim)

        # Rotation matrix Π
        pi = _random_orthogonal(config.head_dim, config.seed)
        self.register_buffer("Pi", pi)

        # Beta distribution parameters
        self.register_buffer("beta_alpha", torch.tensor(config.beta_alpha))
        self.register_buffer("beta_beta", torch.tensor(config.beta_beta))

        # Lloyd-Max codebook
        levels, boundaries = self._init_codebook(config.beta_alpha, config.beta_beta)
        self.register_buffer("levels", levels)
        self.register_buffer("boundaries", boundaries)

        # Calibration flag
        self.register_buffer("calibrated", torch.tensor(0, dtype=torch.int32))

        log.debug(
            "PolarQuantizer.__init__",
            head_dim=self.head_dim,
            bits=self.bits,
            group_size=self.group_size,
            num_levels=self.num_levels,
            use_hadamard=self.use_hadamard,
        )

    def _pi(self) -> torch.Tensor:
        return self.get_buffer("Pi")

    def _levels(self) -> torch.Tensor:
        return self.get_buffer("levels")

    def _boundaries(self) -> torch.Tensor:
        return self.get_buffer("boundaries")

    def _beta_alpha(self) -> torch.Tensor:
        return self.get_buffer("beta_alpha")

    def _beta_beta(self) -> torch.Tensor:
        return self.get_buffer("beta_beta")

    def _calibrated(self) -> torch.Tensor:
        return self.get_buffer("calibrated")

    @property
    def rotation(self) -> torch.Tensor:
        """Backward-compatible alias for the rotation matrix."""
        return self._pi()

    # ------------------------------------------------------------------
    # Lloyd-Max codebook
    # ------------------------------------------------------------------

    def _init_codebook(self, alpha: float, beta_val: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Lloyd-Max codebook for the given Beta distribution.

        Args:
            alpha: Beta distribution α parameter.
            beta_val: Beta distribution β parameter.

        Returns:
            ``(levels, boundaries)`` tensors.
        """
        num_levels = self.num_levels

        # Initial uniform levels in [0, 1] for the absolute values
        levels_np = np.linspace(0, 1, num_levels + 2)[1:-1].astype(np.float64)
        boundaries_np = np.zeros(num_levels - 1, dtype=np.float64)

        for _ in range(self.config.lloyd_max_iters):
            # Boundaries = midpoints between centroids
            for i in range(num_levels - 1):
                boundaries_np[i] = (levels_np[i] + levels_np[i + 1]) / 2.0

            # Centroids = conditional expectation of Beta over each region
            for i in range(num_levels):
                lo = boundaries_np[i - 1] if i > 0 else 0.0
                hi = boundaries_np[i] if i < num_levels - 1 else 1.0
                lo = float(np.clip(lo, 1e-12, 1.0 - 1e-12))
                hi = float(np.clip(hi, lo + 1e-12, 1.0 - 1e-12))
                levels_np[i] = self._beta_conditional_mean(alpha, beta_val, lo, hi)

        # Map levels from [0,1] to [-3.5, 3.5] for the clipped normalised values
        levels_mapped = (levels_np * 7.0 - 3.5).astype(np.float32)
        boundaries_mapped = (boundaries_np * 7.0 - 3.5).astype(np.float32)

        return (
            torch.from_numpy(levels_mapped),
            torch.from_numpy(boundaries_mapped),
        )

    @staticmethod
    def _beta_conditional_mean(alpha: float, beta_val: float, lo: float, hi: float) -> float:
        """E[X | lo < X < hi] for X ~ Beta(alpha, beta_val).

        Args:
            alpha: Beta α.
            beta_val: Beta β.
            lo: Lower bound.
            hi: Upper bound.

        Returns:
            Conditional mean as float.
        """
        if _HAS_SCIPY:
            from scipy.integrate import quad
            from scipy.stats import beta as sp_beta

            dist = sp_beta(alpha, beta_val)
            prob = dist.cdf(hi) - dist.cdf(lo)
            if prob < 1e-15:
                return (lo + hi) / 2.0
            numerator, _ = quad(lambda x: x * dist.pdf(x), lo, hi)
            return float(numerator / prob)
        else:
            # Fallback: simple midpoint
            return (lo + hi) / 2.0

    # ------------------------------------------------------------------
    # Forward (quantize)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize input tensor to packed 3-bit representation.

        Args:
            x: Input tensor ``[batch, heads, seq_len, head_dim]`` float16/bfloat16.

        Returns:
            ``(packed, scales)`` where:

            * ``packed``: uint8 tensor ``[batch, heads, seq_len, ceil(head_dim*3/8)]``
            * ``scales``: float32 tensor ``[batch, heads, seq_len, num_groups]``
        """
        if x.numel() == 0:
            packed_dim = math.ceil(self.head_dim * self.bits / 8)
            num_groups = math.ceil(self.head_dim / self.group_size)
            packed = torch.empty(*x.shape[:-1], packed_dim, dtype=torch.uint8, device=x.device)
            scales = torch.empty(*x.shape[:-1], num_groups, dtype=torch.float32, device=x.device)
            if self._legacy_api:
                quantized = torch.empty(*x.shape, dtype=torch.int8, device=x.device)
                return quantized, scales
            return packed, scales

        original_shape = x.shape
        # Sanitise: NaN/Inf → 0 with warning
        if torch.isnan(x).any() or torch.isinf(x).any():
            log.warning("forward: NaN/Inf detected in input, replacing with 0.0")
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))

        # Convert to float32 for computation
        x_f32 = x.float()

        # 1. Apply rotation
        x_rot = self._rotate(x_f32, inverse=False)

        # 2. Group-wise scale + normalise
        flat = x_rot.reshape(-1, self.head_dim)
        padded_dim = math.ceil(self.head_dim / self.group_size) * self.group_size
        if padded_dim > self.head_dim:
            flat = torch.nn.functional.pad(flat, (0, padded_dim - self.head_dim))
        num_groups = padded_dim // self.group_size
        grouped = flat.reshape(-1, num_groups, self.group_size)  # (N, G, gs)

        scales = grouped.abs().amax(dim=-1).clamp(min=1e-8) / 3.5  # (N, G)
        normalised = (grouped / scales.unsqueeze(-1)).clamp(-3.5, 3.5)

        # 3. Quantize via codebook
        bnd = self._boundaries().to(normalised.device)
        flat_norm = normalised.reshape(-1)
        indices = torch.searchsorted(bnd, flat_norm).clamp(0, self.num_levels - 1)
        indices = indices.reshape(normalised.shape).to(torch.int8)

        # Trim padding
        indices = indices.reshape(-1, padded_dim).narrow(-1, 0, self.head_dim)
        indices = indices.reshape(*original_shape[:-1], self.head_dim)

        # 4. Pack 3-bit
        packed = pack_3bit(indices)

        # Reshape scales
        scales = scales.reshape(*original_shape[:-1], num_groups)

        log.debug(
            "PolarQuantizer.forward",
            input_shape=list(original_shape),
            packed_shape=list(packed.shape),
            compression_bytes=f"{packed.nbytes}/{x.nbytes}",
        )

        if self._legacy_api:
            return indices.to(torch.int8), scales
        return packed, scales

    # ------------------------------------------------------------------
    # Dequantize
    # ------------------------------------------------------------------

    def dequantize(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        original_shape: TensorShape | None = None,
    ) -> torch.Tensor:
        """Dequantize packed 3-bit tensor back to float16.

        Args:
            packed: Uint8 packed tensor from :meth:`forward`.
            scales: Float32 scales from :meth:`forward`.
            original_shape: Original input shape.

        Returns:
            Reconstructed float16 tensor of *original_shape*.
        """
        if packed.numel() == 0:
            if original_shape is None:
                original_shape = tuple(packed.shape)
            return torch.empty(original_shape, dtype=torch.float16, device=packed.device)

        if original_shape is None:
            if packed.dtype == torch.int8 and packed.shape[-1] == self.head_dim:
                original_shape = tuple(packed.shape)
            else:
                raise ValueError("original_shape is required for packed quantized input")

        head_dim = original_shape[-1]
        # 1. Unpack
        if packed.dtype == torch.int8 and packed.shape[-1] == head_dim:
            indices = packed
        else:
            indices = unpack_3bit(packed, head_dim)  # int8, 0-7

        # 2. Codebook lookup
        lvl = self._levels().to(indices.device)
        idx = indices.long().clamp(0, self.num_levels - 1)
        values = lvl[idx]  # float32

        # 3. Rescale
        flat = values.reshape(-1, head_dim)
        padded_dim = math.ceil(head_dim / self.group_size) * self.group_size
        if padded_dim > head_dim:
            flat = torch.nn.functional.pad(flat, (0, padded_dim - head_dim))
        num_groups = padded_dim // self.group_size
        grouped = flat.reshape(-1, num_groups, self.group_size)

        sc = scales.reshape(-1, num_groups).unsqueeze(-1)  # (N, G, 1)
        rescaled = grouped * sc
        rescaled_flat = rescaled.reshape(-1, padded_dim).narrow(-1, 0, head_dim)
        rescaled_out = rescaled_flat.reshape(original_shape)

        # 4. Inverse rotation
        result = self._rotate(rescaled_out, inverse=True)
        return result.to(torch.float16)

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def _rotate(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply rotation (or inverse rotation) to last dimension.

        Args:
            x: Tensor with last dim == head_dim.
            inverse: If True, apply Πᵀ (inverse rotation).

        Returns:
            Rotated tensor of same shape.
        """
        if self.use_hadamard:
            try:
                from turboquant.kernels.hadamard import randomized_hadamard_transform

                return randomized_hadamard_transform(x, seed=self.seed, inverse=inverse)
            except ImportError:
                log.debug("hadamard_kernel_unavailable_fallback")

        pi = self._pi().to(x.device)
        if inverse:
            return x @ pi  # Π^T @ Π = I, so x @ Π = x_rot @ Π
        return x @ pi.T

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, calibration_data: list[torch.Tensor]) -> None:
        """Calibrate codebook from real KV-cache data.

        Fits Beta distribution parameters via MLE (or Method of Moments
        if scipy is unavailable) on the absolute normalised values from
        the calibration data, then recomputes the Lloyd-Max codebook.

        Args:
            calibration_data: List of tensors with last dimension = head_dim.

        Raises:
            ValueError: If *calibration_data* is empty.
        """
        if not calibration_data:
            raise ValueError("calibration_data must be a non-empty list of tensors.")

        all_vals: list[torch.Tensor] = []
        pi = self._pi().to(calibration_data[0].device)
        for tensor in calibration_data:
            t = tensor.float()
            t_rot = t @ pi.T
            all_vals.append(t_rot.abs().reshape(-1))

        merged = torch.cat(all_vals)
        if merged.numel() == 0:
            return

        # Normalise to [0, 1] approximately
        max_val = merged.max().item()
        normalised = (merged / max(max_val, 1e-12)).clamp(1e-6, 1.0 - 1e-6)
        np_vals = normalised.cpu().numpy()

        if _HAS_SCIPY:
            a, b, _, _ = _scipy_beta.fit(np_vals, floc=0, fscale=1)
        else:
            mean_ = float(np_vals.mean())
            var_ = float(np_vals.var())
            var_ = max(var_, 1e-12)
            common = (mean_ * (1.0 - mean_) / var_) - 1.0
            common = max(common, 1e-6)
            a = mean_ * common
            b = (1.0 - mean_) * common

        # Compute MSE before recalibration
        old_mse = self._estimate_mse(merged)

        # Update codebook
        levels, boundaries = self._init_codebook(a, b)
        self._beta_alpha().fill_(a)
        self._beta_beta().fill_(b)
        lvl = self._levels()
        bnd = self._boundaries()
        lvl.copy_(levels.to(lvl.device))
        bnd.copy_(boundaries.to(bnd.device))
        self._calibrated().fill_(1)

        new_mse = self._estimate_mse(merged)

        log.info(
            "calibrate",
            alpha=round(a, 4),
            beta=round(b, 4),
            mse_before=round(old_mse, 6),
            mse_after=round(new_mse, 6),
            improvement=f"{(1 - new_mse / max(old_mse, 1e-15)) * 100:.1f}%",
            n_samples=merged.numel(),
        )

    def _estimate_mse(self, values: torch.Tensor) -> float:
        """Estimate quantization MSE on a flat tensor of absolute values."""
        bnd = self._boundaries().to(values.device)
        lvl = self._levels().to(values.device)
        # Simple uniform quantization approximation
        sample = values[:10000] if values.numel() > 10000 else values
        normalised = (sample / sample.max().clamp(min=1e-12) * 3.5).clamp(-3.5, 3.5)
        indices = torch.searchsorted(bnd, normalised).clamp(0, self.num_levels - 1)
        recon = lvl[indices]
        return float(((normalised - recon) ** 2).mean().item())

    # ------------------------------------------------------------------
    # Compression ratio
    # ------------------------------------------------------------------

    def compression_ratio(self) -> dict[str, float]:
        """Return theoretical and actual compression ratios.

        Returns:
            Dict with keys ``theoretical``, ``actual``, ``overhead_scales_pct``.
        """
        bits_per_val = self.bits
        fp16_bits = 16
        num_groups = math.ceil(self.head_dim / self.group_size)
        scale_bits_per_val = (num_groups * 32) / self.head_dim  # float32 per group

        theoretical = fp16_bits / bits_per_val
        actual = fp16_bits / (bits_per_val + scale_bits_per_val)
        overhead = scale_bits_per_val / (bits_per_val + scale_bits_per_val) * 100

        return {
            "theoretical": round(theoretical, 2),
            "actual": round(actual, 2),
            "overhead_scales_pct": round(overhead, 2),
        }

    # ------------------------------------------------------------------
    # Save / Load calibration
    # ------------------------------------------------------------------

    def save_calibration(self, path: str) -> None:
        """Save calibration data (codebook, rotation, Beta params).

        Args:
            path: File path (safetensors format).

        Raises:
            ImportError: If safetensors is not installed.
        """
        if not _HAS_SAFETENSORS:
            raise ImportError("safetensors is required. Install: pip install safetensors")
        tensors = {
            "levels": self._levels().cpu(),
            "boundaries": self._boundaries().cpu(),
            "beta_alpha": self._beta_alpha().cpu().unsqueeze(0),
            "beta_beta": self._beta_beta().cpu().unsqueeze(0),
            "calibrated": self._calibrated().cpu().unsqueeze(0).float(),
            "Pi": self._pi().cpu(),
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(p))

        # Save metadata alongside
        meta = {
            "head_dim": self.head_dim,
            "bits": self.bits,
            "group_size": self.group_size,
            "seed": self.seed,
            "num_levels": self.num_levels,
        }
        meta_path = p.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2))

        log.info("save_calibration", path=str(p))

    def load_calibration(self, path: str) -> None:
        """Load calibration data and validate compatibility.

        Args:
            path: Path to safetensors file.

        Raises:
            ImportError: If safetensors is not installed.
            FileNotFoundError: If file does not exist.
            ValueError: If head_dim does not match.
        """
        if not _HAS_SAFETENSORS:
            raise ImportError("safetensors is required. Install: pip install safetensors")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        # Validate metadata if present
        meta_path = p.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("head_dim") != self.head_dim:
                raise ValueError(
                    f"head_dim mismatch: file has {meta['head_dim']}, config has {self.head_dim}"
                )

        tensors = load_file(str(p))
        lvl = self._levels()
        bnd = self._boundaries()
        a = self._beta_alpha()
        b = self._beta_beta()
        c = self._calibrated()
        pimat = self._pi()
        lvl.copy_(tensors["levels"].to(lvl.device))
        bnd.copy_(tensors["boundaries"].to(bnd.device))
        a.copy_(tensors["beta_alpha"].squeeze())
        b.copy_(tensors["beta_beta"].squeeze())
        c.fill_(int(tensors["calibrated"].squeeze().item()))
        if "Pi" in tensors:
            pimat.copy_(tensors["Pi"].to(pimat.device))

        log.info("load_calibration", path=str(p))

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Extra repr for ``print(module)``."""
        cal = bool(self._calibrated().item())
        return (
            f"head_dim={self.head_dim}, bits={self.bits}, "
            f"group_size={self.group_size}, seed={self.seed}, "
            f"calibrated={cal}, use_hadamard={self.use_hadamard}"
        )


# ======================================================================
# Utilities
# ======================================================================


def _is_power_of_2(n: int) -> bool:
    """Check if *n* is a positive power of 2."""
    return n > 0 and (n & (n - 1)) == 0
