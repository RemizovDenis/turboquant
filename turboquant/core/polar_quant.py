"""Polar Quantizer — 3-bit KV-cache quantization with random orthogonal rotation.

Implements the first stage of TurboQuant (arXiv 2504.19874):
random rotation → Beta-distribution Lloyd-Max quantization → 3-bit encoding.

Typical usage::

    quantizer = PolarQuantizer(head_dim=128, bits=3, group_size=64, seed=42)
    quantizer.calibrate([sample_tensor_1, sample_tensor_2])
    quantized, scales = quantizer(kv_cache_tensor)
    reconstructed = quantizer.dequantize(quantized, scales)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import structlog
import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file, save_file
except ImportError:  # pragma: no cover
    save_file = None  # type: ignore[assignment]
    load_file = None  # type: ignore[assignment]

log = structlog.get_logger(__name__)


class PolarQuantizer(nn.Module):
    """3-bit polar quantizer with random orthogonal rotation.

    Attributes:
        head_dim: Dimension of each attention head.
        bits: Number of quantization bits (default 3 → 8 levels).
        group_size: Number of elements per quantization group (for per-group scaling).
        seed: Random seed for reproducible rotation matrix generation.

    Buffers (registered, moved with ``.to()`` / ``.cuda()``):
        rotation: Orthogonal rotation matrix ``Π`` of shape ``(head_dim, head_dim)``.
        levels: Quantization levels of shape ``(2**bits,)``.
        boundaries: Decision boundaries of shape ``(2**bits - 1,)``.
        beta_alpha: Beta-distribution ``α`` parameter (scalar).
        beta_beta: Beta-distribution ``β`` parameter (scalar).
        calibrated: Flag ``1`` if calibration has been performed, else ``0``.
    """

    def __init__(
        self,
        head_dim: int,
        bits: int = 3,
        group_size: int = 64,
        seed: int = 42,
    ) -> None:
        """Initialise PolarQuantizer.

        Args:
            head_dim: Dimension of each attention head.
            bits: Quantization bit-width (default 3).
            group_size: Elements per quantization group (default 64).
            seed: Seed for orthogonal rotation matrix (default 42).
        """
        super().__init__()
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if bits < 1 or bits > 8:
            raise ValueError(f"bits must be in [1, 8], got {bits}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")

        self.head_dim = head_dim
        self.bits = bits
        self.group_size = group_size
        self.seed = seed
        self.num_levels = 2**bits

        # --- Orthogonal rotation matrix Π ---
        rotation = self._generate_rotation(head_dim, seed)
        self.register_buffer("rotation", rotation)

        # --- Quantization codebook (initialised as uniform; updated by calibrate) ---
        init_levels = torch.linspace(-1.0, 1.0, self.num_levels, dtype=torch.float32)
        self.register_buffer("levels", init_levels)
        init_boundaries = (init_levels[:-1] + init_levels[1:]) / 2.0
        self.register_buffer("boundaries", init_boundaries)

        # --- Beta distribution parameters ---
        self.register_buffer("beta_alpha", torch.tensor(2.0, dtype=torch.float32))
        self.register_buffer("beta_beta", torch.tensor(2.0, dtype=torch.float32))
        self.register_buffer("calibrated", torch.tensor(0, dtype=torch.int32))

        log.debug(
            "PolarQuantizer.__init__",
            head_dim=head_dim,
            bits=bits,
            group_size=group_size,
            num_levels=self.num_levels,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_rotation(dim: int, seed: int) -> torch.Tensor:
        """Generate a random orthogonal matrix via QR decomposition.

        Args:
            dim: Matrix dimension.
            seed: Random seed for reproducibility.

        Returns:
            Orthogonal matrix of shape ``(dim, dim)`` in float32.
        """
        gen = torch.Generator().manual_seed(seed)
        gaussian = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
        q, r = torch.linalg.qr(gaussian)
        # Ensure unique decomposition by fixing sign of diagonal of R
        diag_sign = torch.sign(torch.diag(r))
        diag_sign[diag_sign == 0] = 1.0
        q = q * diag_sign.unsqueeze(0)
        return cast(torch.Tensor, q)

    # ------------------------------------------------------------------
    # Lloyd-Max codebook via Beta distribution
    # ------------------------------------------------------------------

    def _fit_beta_moments(self, data: torch.Tensor) -> tuple[float, float]:
        """Estimate Beta(α, β) parameters via method of moments.

        The data is first normalised to [0, 1] (min-max) and then the first
        two moments are matched to Beta parameters analytically.

        Args:
            data: 1-D tensor of observed values.

        Returns:
            Tuple ``(alpha, beta)`` of the estimated Beta distribution.
        """
        d = data.float()
        d_min = d.min()
        d_max = d.max()
        if (d_max - d_min).abs() < 1e-12:
            return 2.0, 2.0

        # Normalise to (0, 1)
        d_norm = (d - d_min) / (d_max - d_min)
        d_norm = d_norm.clamp(1e-6, 1.0 - 1e-6)

        mean = d_norm.mean().item()
        var = d_norm.var(unbiased=False).item()
        var = max(var, 1e-12)

        common = (mean * (1.0 - mean) / var) - 1.0
        common = max(common, 1e-6)
        alpha = mean * common
        beta = (1.0 - mean) * common
        alpha = max(alpha, 0.01)
        beta = max(beta, 0.01)
        return alpha, beta

    def _lloyd_max_from_beta(
        self,
        alpha: float,
        beta: float,
        num_levels: int,
        iterations: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Lloyd-Max quantization codebook from a Beta distribution.

        Uses iterative Lloyd-Max algorithm initialised with uniform partitions
        of the Beta CDF.

        Args:
            alpha: Beta distribution α parameter.
            beta: Beta distribution β parameter.
            num_levels: Number of quantization levels.
            iterations: Optimisation iterations.

        Returns:
            Tuple of (levels, boundaries) tensors in float32.
        """
        from scipy.stats import beta as sp_beta  # type: ignore[import-untyped]

        dist = sp_beta(alpha, beta)

        # Initialise centroids at CDF quantiles
        quantiles = np.linspace(0.0, 1.0, num_levels + 2)[1:-1]  # skip 0 and 1
        centroids = np.asarray([float(dist.ppf(q)) for q in quantiles])

        for _ in range(iterations):
            # boundaries = midpoints
            boundaries = (centroids[:-1] + centroids[1:]) / 2.0

            # Recompute centroids as conditional expectations
            full_bounds = np.concatenate(([0.0], boundaries, [1.0]))
            new_centroids = np.empty_like(centroids)
            for i in range(num_levels):
                lo, hi = full_bounds[i], full_bounds[i + 1]
                cdf_lo = dist.cdf(lo)
                cdf_hi = dist.cdf(hi)
                if cdf_hi - cdf_lo < 1e-15:
                    new_centroids[i] = (lo + hi) / 2.0
                else:
                    # E[X | lo < X < hi] via numerical quadrature
                    from scipy.integrate import quad  # type: ignore[import-untyped]

                    num, _ = quad(lambda x: x * dist.pdf(x), lo, hi)
                    den = cdf_hi - cdf_lo
                    new_centroids[i] = num / den
            centroids = new_centroids

        # Map from [0, 1] to [-1, 1]
        lvl = torch.tensor(centroids * 2.0 - 1.0, dtype=torch.float32)
        bnd = torch.tensor(boundaries * 2.0 - 1.0, dtype=torch.float32)
        return lvl, bnd

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, calibration_data: list[torch.Tensor]) -> None:
        """Calibrate quantizer from sample KV-cache tensors.

        Collects rotated element magnitudes, fits a Beta distribution via
        method of moments, and computes the optimal Lloyd-Max codebook.

        Args:
            calibration_data: List of KV-cache tensors, each of shape
                ``[batch, num_heads, seq_len, head_dim]`` or any shape whose
                last dimension equals ``head_dim``.
        """
        if not calibration_data:
            raise ValueError("calibration_data must be a non-empty list of tensors.")

        all_vals: list[torch.Tensor] = []
        rotation = cast(torch.Tensor, self.rotation)
        for tensor in calibration_data:
            t = tensor.float().to(rotation.device)
            t_rot = t @ rotation.T
            all_vals.append(t_rot.reshape(-1))

        merged = torch.cat(all_vals)
        if merged.numel() == 0:
            log.info("calibrate: empty calibration data, keeping defaults")
            return

        alpha, beta_param = self._fit_beta_moments(merged)
        beta_alpha = cast(torch.Tensor, self.beta_alpha)
        beta_beta = cast(torch.Tensor, self.beta_beta)
        levels_buf = cast(torch.Tensor, self.levels)
        boundaries_buf = cast(torch.Tensor, self.boundaries)
        calibrated_buf = cast(torch.Tensor, self.calibrated)
        beta_alpha.fill_(alpha)
        beta_beta.fill_(beta_param)

        levels, boundaries = self._lloyd_max_from_beta(alpha, beta_param, self.num_levels)
        levels_buf.copy_(levels.to(levels_buf.device))
        boundaries_buf.copy_(boundaries.to(boundaries_buf.device))
        calibrated_buf.fill_(1)

        log.info(
            "calibrate",
            alpha=round(alpha, 4),
            beta=round(beta_param, 4),
            num_levels=self.num_levels,
            samples=merged.numel(),
        )

    # ------------------------------------------------------------------
    # Forward (quantize)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a KV-cache tensor to ``self.bits``-bit representation.

        Steps:
            1. Apply random orthogonal rotation.
            2. Compute per-group scale (max absolute value).
            3. Normalise to [-1, 1] within each group.
            4. Map to nearest Lloyd-Max level → integer index stored as int8.

        Args:
            x: Input tensor of shape ``[batch, num_heads, seq_len, head_dim]``
                with dtype float16 or bfloat16.

        Returns:
            Tuple ``(quantized_int8, scales_float32)`` where:
                - ``quantized_int8``: int8 tensor of the same shape as *x*,
                  values in ``[0, 2**bits - 1]``.
                - ``scales_float32``: float32 tensor of shape
                  ``[..., num_groups]`` containing the per-group scale factors.
        """
        if x.numel() == 0:
            return (
                torch.empty_like(x, dtype=torch.int8),
                torch.empty(
                    *x.shape[:-1], 0, dtype=torch.float32, device=x.device
                ),
            )

        original_shape = x.shape
        original_dtype = x.dtype

        # 1. Rotate
        x_float = x.float()
        # rotation is (head_dim, head_dim), x has last dim = head_dim
        rotation = cast(torch.Tensor, self.rotation)
        x_rot = x_float @ rotation.T

        # 2. Reshape for grouping: (..., num_groups, group_size)
        flat = x_rot.reshape(-1, self.head_dim)
        total_elements = flat.shape[0]

        # Pad if head_dim is not divisible by group_size
        remainder = self.head_dim % self.group_size
        if remainder != 0:
            pad_size = self.group_size - remainder
            flat = torch.nn.functional.pad(flat, (0, pad_size), value=0.0)
            padded_dim = flat.shape[-1]
        else:
            pad_size = 0
            padded_dim = self.head_dim

        num_groups = padded_dim // self.group_size
        grouped = flat.reshape(total_elements, num_groups, self.group_size)

        # 3. Per-group scale
        scales = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # (N, G, 1)
        normalized = grouped / scales  # in [-1, 1]

        # 4. Quantize: find nearest level
        # boundaries shape: (num_levels - 1,)
        # Use searchsorted to map each normalized value to a bucket index
        boundaries = cast(torch.Tensor, self.boundaries)
        bnd = boundaries.to(normalized.device)
        flat_norm = normalized.reshape(-1)
        indices = torch.searchsorted(bnd, flat_norm).to(torch.int8)
        quantized_grouped = indices.reshape(grouped.shape)

        # Remove padding from last group
        scales_out = scales.squeeze(-1)  # (N, G)

        if pad_size > 0:
            # Reconstruct original dim via slicing
            quantized_flat = quantized_grouped.reshape(total_elements, padded_dim)
            quantized_flat = quantized_flat[:, :self.head_dim]
        else:
            quantized_flat = quantized_grouped.reshape(total_elements, self.head_dim)

        quantized = quantized_flat.reshape(original_shape)
        scales_out = scales_out.reshape(*original_shape[:-1], num_groups)

        log.debug(
            "forward",
            input_shape=list(original_shape),
            input_dtype=str(original_dtype),
            num_groups=num_groups,
            group_size=self.group_size,
        )

        return quantized, scales_out

    # ------------------------------------------------------------------
    # Dequantize
    # ------------------------------------------------------------------

    def dequantize(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct a float16 tensor from quantized representation.

        Args:
            quantized: int8 tensor of level indices, same shape as original.
            scales: float32 per-group scales, shape ``[..., num_groups]``.

        Returns:
            Reconstructed tensor in float16, same shape as *quantized*.
        """
        if quantized.numel() == 0:
            return torch.empty_like(quantized, dtype=torch.float16)

        original_shape = quantized.shape
        levels = cast(torch.Tensor, self.levels).to(quantized.device)

        # Map indices back to level values
        idx = quantized.long().clamp(0, self.num_levels - 1)
        dequant_values = levels[idx]  # float32, same shape as quantized

        # Reshape for ungrouping
        flat = dequant_values.reshape(-1, self.head_dim)
        total_elements = flat.shape[0]

        remainder = self.head_dim % self.group_size
        if remainder != 0:
            pad_size = self.group_size - remainder
            flat = torch.nn.functional.pad(flat, (0, pad_size), value=0.0)
            padded_dim = flat.shape[-1]
        else:
            pad_size = 0
            padded_dim = self.head_dim

        num_groups = padded_dim // self.group_size
        grouped = flat.reshape(total_elements, num_groups, self.group_size)

        # Rescale
        sc = scales.reshape(total_elements, num_groups, 1).to(grouped.device)
        rescaled = grouped * sc

        if pad_size > 0:
            rescaled_flat = rescaled.reshape(total_elements, padded_dim)
            rescaled_flat = rescaled_flat[:, :self.head_dim]
        else:
            rescaled_flat = rescaled.reshape(total_elements, self.head_dim)

        rescaled_out = rescaled_flat.reshape(original_shape)

        # Inverse rotation: X = X_rot @ Π  (since Π is orthogonal, Π^{-1} = Π^T)
        rotation_inv = cast(torch.Tensor, self.rotation).to(rescaled_out.device)  # Π
        result = rescaled_out @ rotation_inv  # X_rot @ Π = X_rot @ (Π^T)^T  → original space

        return result.to(torch.float16)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_calibration(self, path: str) -> None:
        """Save calibration state (levels, boundaries, Beta params) to a safetensors file.

        Args:
            path: Destination file path (will be created / overwritten).

        Raises:
            RuntimeError: If safetensors is not installed.
        """
        if save_file is None:
            raise RuntimeError(
                "safetensors is required for save_calibration. "
                "Install it with: pip install safetensors"
            )
        tensors: dict[str, torch.Tensor] = {
            "levels": cast(torch.Tensor, self.levels).cpu(),
            "boundaries": cast(torch.Tensor, self.boundaries).cpu(),
            "beta_alpha": cast(torch.Tensor, self.beta_alpha).cpu().unsqueeze(0),
            "beta_beta": cast(torch.Tensor, self.beta_beta).cpu().unsqueeze(0),
            "calibrated": cast(torch.Tensor, self.calibrated).cpu().unsqueeze(0).float(),
            "rotation": cast(torch.Tensor, self.rotation).cpu(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file_fn = cast(Any, save_file)
        save_file_fn(tensors, path)
        log.info("save_calibration", path=path)

    def load_calibration(self, path: str) -> None:
        """Load calibration state from a safetensors file.

        Args:
            path: Source file path.

        Raises:
            FileNotFoundError: If *path* does not exist.
            RuntimeError: If safetensors is not installed.
        """
        if load_file is None:
            raise RuntimeError(
                "safetensors is required for load_calibration. "
                "Install it with: pip install safetensors"
            )
        if not Path(path).exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        load_file_fn = cast(Any, load_file)
        tensors = cast(dict[str, torch.Tensor], load_file_fn(path))
        levels = cast(torch.Tensor, self.levels)
        boundaries = cast(torch.Tensor, self.boundaries)
        beta_alpha = cast(torch.Tensor, self.beta_alpha)
        beta_beta = cast(torch.Tensor, self.beta_beta)
        calibrated = cast(torch.Tensor, self.calibrated)
        rotation = cast(torch.Tensor, self.rotation)

        levels.copy_(tensors["levels"].to(levels.device))
        boundaries.copy_(tensors["boundaries"].to(boundaries.device))
        beta_alpha.copy_(tensors["beta_alpha"].squeeze().to(beta_alpha.device))
        beta_beta.copy_(tensors["beta_beta"].squeeze().to(beta_beta.device))
        calibrated.fill_(int(tensors["calibrated"].squeeze().item()))
        if "rotation" in tensors:
            rotation.copy_(tensors["rotation"].to(rotation.device))
        log.info("load_calibration", path=path)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        calibrated = cast(torch.Tensor, self.calibrated)
        return (
            f"head_dim={self.head_dim}, bits={self.bits}, "
            f"group_size={self.group_size}, seed={self.seed}, "
            f"calibrated={bool(calibrated.item())}"
        )


if __name__ == "__main__":
    # Quick smoke test
    print("=== PolarQuantizer smoke test ===")
    q = PolarQuantizer(head_dim=128, bits=3, group_size=64, seed=42)
    x = torch.randn(2, 4, 32, 128, dtype=torch.float16)
    q.calibrate([x])
    quantized, scales = q(x)
    print(f"Input:     {x.shape} {x.dtype}")
    print(f"Quantized: {quantized.shape} {quantized.dtype}")
    print(f"Scales:    {scales.shape} {scales.dtype}")
    recon = q.dequantize(quantized, scales)
    print(f"Recon:     {recon.shape} {recon.dtype}")
    mse = ((x.float() - recon.float()) ** 2).mean().item()
    print(f"MSE:       {mse:.6f}")
    print("✓ PolarQuantizer OK")
    rotation: torch.Tensor
    levels: torch.Tensor
    boundaries: torch.Tensor
    beta_alpha: torch.Tensor
    beta_beta: torch.Tensor
    calibrated: torch.Tensor
