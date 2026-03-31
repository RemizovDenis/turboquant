"""QJL residual correction using 1-bit random projections.

Optimised for norm-preserving encoding and adaptive sketching in TurboQuant v0.3.0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import structlog
import torch
import torch.nn as nn

log = structlog.get_logger(__name__)
TensorShape = Sequence[int]


@dataclass
class QJLConfig:
    """Configuration for :class:`QJLResidualCorrector`."""

    head_dim: int
    sketch_dim: int | None = None
    seed: int = 42
    sketch_type: str = "rademacher"


class QJLResidualCorrector(nn.Module):
    """1-bit residual corrector using JL random projections (v0.3.0)."""

    def __init__(
        self,
        config: QJLConfig | None = None,
        *,
        head_dim: int | None = None,
        sketch_dim: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if config is None:
            if head_dim is None:
                raise ValueError("Either config or head_dim must be provided")
            config = QJLConfig(head_dim=head_dim, sketch_dim=sketch_dim, seed=seed)

        self.config = config
        self.head_dim = config.head_dim
        self.sketch_dim = config.sketch_dim or max(config.head_dim // 4, 1)

        sketch = self._build_sketch()
        self.register_buffer("S", sketch)
        self.register_buffer(
            "powers", torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
        )

    def _build_sketch(self) -> torch.Tensor:
        """Standard Rademacher sketch."""
        gen = torch.Generator().manual_seed(self.config.seed)
        k = self.sketch_dim
        d = self.head_dim
        s = torch.randint(0, 2, (k, d), generator=gen, dtype=torch.float32)
        s = s * 2.0 - 1.0
        s = s / math.sqrt(k)
        return s

    def compress_ratio(self) -> float:
        """Returns the compression factor for the residual."""
        return (self.head_dim * 16) / self.sketch_dim

    def encode(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (packed_signs, norms_float16)."""
        if residual.numel() == 0:
            out_bytes = math.ceil(self.sketch_dim / 8)
            return (
                torch.empty(
                    *residual.shape[:-1], out_bytes, dtype=torch.uint8, device=residual.device
                ),
                torch.empty(*residual.shape[:-1], dtype=torch.float16, device=residual.device),
            )

        # 1. Compute norms per-vector for norm-preserving reconstruction
        # residual shape: (..., D)
        norms = residual.float().norm(dim=-1).to(torch.float16)

        # 2. Project
        sketch = cast(torch.Tensor, self.S).to(residual.device)
        proj = residual.float() @ sketch.T

        # 3. Packed signs
        # (..., k)
        signs01 = (proj > 0).to(torch.uint8)
        packed = self._pack_bits(signs01)

        return packed, norms

    def decode(
        self,
        packed_bits: torch.Tensor,
        norms: torch.Tensor | None = None,
        original_shape: TensorShape | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Decode with norm-restoration or uniform scale (backward compatibility)."""
        if packed_bits.numel() == 0:
            return torch.empty(
                original_shape or (0,), dtype=torch.float16, device=packed_bits.device
            )

        # 1. Unpack signs
        signs01 = self._unpack_bits(packed_bits, self.sketch_dim)
        signs = signs01 * 2.0 - 1.0  # (..., k)

        # 2. Project back
        sketch = cast(torch.Tensor, self.S).to(packed_bits.device)
        correction = signs @ sketch  # (..., D)

        # 3. Rescale
        if norms is not None:
            # Norm-preserving scaling: Ensure reconstruction norm matches original norm.
            # E[ ||signs @ S||^2 ] = sketch_dim * (1/sqrt(sketch_dim))^2 * ||residual||^2 = ||residual||^2
            # But the signs are {-1, 1}, and S has normalization 1/sqrt(k).
            # So (signs @ S) has expected squared norm = D (roughly).
            # We want current_norm * scale = target_norm
            current_norms = correction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            correction = correction * (norms.float().unsqueeze(-1) / current_norms)
        else:
            # Fallback for old API
            s = scale if scale is not None else 1.0
            correction = correction * (float(s) / math.sqrt(self.sketch_dim))

        if original_shape:
            correction = correction.view(original_shape)

        return correction.to(torch.float16)

    def _pack_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """Pack binary (0, 1) tensor to uint8 bytes."""
        k = bits.shape[-1]
        pad = (8 - k % 8) % 8
        if pad > 0:
            bits = torch.nn.functional.pad(bits, (0, pad), value=0)

        # Reshape and bits to byte
        res = bits.reshape(*bits.shape[:-1], -1, 8)
        packed = (res * self.powers).sum(dim=-1).to(torch.uint8)
        return packed

    def _unpack_bits(self, packed: torch.Tensor, k: int) -> torch.Tensor:
        """Unpack uint8 bytes to binary (0, 1) tensor."""
        unpacked = (packed.unsqueeze(-1) & self.powers).gt(0).to(torch.float32)
        return unpacked.view(*packed.shape[:-1], -1)[..., :k]

    def encode_with_scale(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Wrapper for old API compatibility."""
        return self.encode(residual)


class AdaptiveQJLCorrector(nn.Module):
    """QJL corrector with importance-based sketch_dim."""

    def __init__(self, head_dim: int, sketch_dim_low: int = 16, sketch_dim_high: int = 64) -> None:
        super().__init__()
        self.low = QJLResidualCorrector(head_dim=head_dim, sketch_dim=sketch_dim_low, seed=42)
        self.high = QJLResidualCorrector(head_dim=head_dim, sketch_dim=sketch_dim_high, seed=43)

    def encode_with_importance(
        self, residual: torch.Tensor, importance_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized importance-based routing."""
        high_mask = importance_scores > 0.7
        res_high = residual[high_mask]
        res_low = residual[~high_mask]

        p_h, n_h = self.high.encode(res_high) if res_high.numel() > 0 else (None, None)
        p_l, n_l = self.low.encode(res_low) if res_low.numel() > 0 else (None, None)

        # Return a flat tuple for JIT/TorchScript compatibility instead of a dict
        return high_mask, p_h, n_h, p_l, n_l  # type: ignore

    def decode_with_importance(
        self,
        high_mask: torch.Tensor,
        p_h: torch.Tensor | None,
        n_h: torch.Tensor | None,
        p_l: torch.Tensor | None,
        n_l: torch.Tensor | None,
        original_shape: tuple[int, ...],
    ) -> torch.Tensor:
        out = torch.zeros(original_shape, device=high_mask.device, dtype=torch.float16)
        if p_h is not None and n_h is not None:
            out[high_mask] = self.high.decode(p_h, n_h)
        if p_l is not None and n_l is not None:
            out[~high_mask] = self.low.decode(p_l, n_l)
        return out
