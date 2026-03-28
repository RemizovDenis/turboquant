"""QJL residual correction using 1-bit random projections."""

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
    sketch_type: str = "rademacher"
    sparsity: float = 1.0 / 3.0
    seed: int = 42


class QJLResidualCorrector(nn.Module):
    """1-bit residual corrector using JL random projections."""

    def __init__(
        self,
        config: QJLConfig | None = None,
        *,
        head_dim: int | None = None,
        sketch_dim: int | None = None,
        seed: int = 42,
        sketch_type: str = "rademacher",
        sparsity: float = 1.0 / 3.0,
    ) -> None:
        super().__init__()
        if config is None:
            if head_dim is None:
                raise ValueError("Either config or head_dim must be provided")
            config = QJLConfig(
                head_dim=head_dim,
                sketch_dim=sketch_dim,
                sketch_type=sketch_type,
                sparsity=sparsity,
                seed=seed,
            )

        if config.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if config.sketch_type not in {"rademacher", "gaussian", "sparse"}:
            raise ValueError("sketch_type must be one of: rademacher, gaussian, sparse")

        self.config = config
        self.head_dim = config.head_dim
        self.sketch_dim = config.sketch_dim or max(config.head_dim // 4, 1)

        sketch = self._build_sketch()
        self.register_buffer("S", sketch)

        sparse_mask = (sketch != 0).to(torch.bool)
        self.register_buffer("sparse_mask", sparse_mask)
        self._logged_ratio = False

        log.debug(
            "QJLResidualCorrector.__init__",
            head_dim=self.head_dim,
            sketch_dim=self.sketch_dim,
            sketch_type=self.config.sketch_type,
        )

    def _build_sketch(self) -> torch.Tensor:
        gen = torch.Generator().manual_seed(self.config.seed)
        k = self.sketch_dim
        d = self.head_dim

        if self.config.sketch_type == "rademacher":
            s = torch.randint(0, 2, (k, d), generator=gen, dtype=torch.float32)
            s = s * 2.0 - 1.0
            s = s / math.sqrt(k)
            return s

        if self.config.sketch_type == "gaussian":
            return torch.randn(k, d, generator=gen, dtype=torch.float32) / math.sqrt(k)

        # sparse (Achlioptas-like)
        prob_non_zero = float(max(min(1.0 / max(self.config.sparsity, 1e-6), 1.0), 1e-6))
        mask = torch.rand(k, d, generator=gen) < prob_non_zero
        signs = torch.randint(0, 2, (k, d), generator=gen, dtype=torch.float32)
        signs = signs * 2.0 - 1.0
        scale = math.sqrt(max(self.config.sparsity, 1e-6) / k)
        return signs * mask.to(torch.float32) * scale

    @staticmethod
    def _pack_bits(signs01: torch.Tensor) -> torch.Tensor:
        """Pack {0,1} float tensor into uint8 along last dim."""
        n = signs01.shape[-1]
        pad = (8 - n % 8) % 8
        if pad:
            signs01 = torch.nn.functional.pad(signs01, (0, pad), value=0)
        bits = signs01.reshape(*signs01.shape[:-1], -1, 8).to(torch.uint8)
        powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device=bits.device, dtype=torch.uint8)
        packed = (bits * powers).sum(dim=-1).to(torch.uint8)
        return packed

    @staticmethod
    def _unpack_bits(packed: torch.Tensor, n_bits: int) -> torch.Tensor:
        bytes_ = packed.to(torch.uint8)
        powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device=bytes_.device, dtype=torch.uint8)
        unpacked = ((bytes_.unsqueeze(-1) & powers) > 0).to(torch.float32)
        unpacked = unpacked.reshape(*packed.shape[:-1], -1)
        return unpacked.narrow(-1, 0, n_bits)

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        """Encode residual into packed 1-bit sketch representation."""
        if residual.numel() == 0:
            out_bytes = math.ceil(self.sketch_dim / 8)
            return torch.empty(*residual.shape[:-1], out_bytes, dtype=torch.uint8, device=residual.device)

        if residual.shape[-1] != self.head_dim:
            raise ValueError(f"Expected last dim {self.head_dim}, got {residual.shape[-1]}")

        sketch = cast(torch.Tensor, self.S).to(residual.device)
        proj = residual.float() @ sketch.T
        hard_signs = (proj > 0).to(proj.dtype)
        signs01 = proj + (hard_signs - proj).detach()
        packed = self._pack_bits(signs01)

        if not self._logged_ratio:
            ratio = (packed.numel()) / max(residual.numel() * 2, 1)
            log.debug("qjl_encode_ratio", ratio=ratio)
            self._logged_ratio = True

        return packed

    def decode(
        self,
        packed_bits: torch.Tensor,
        original_shape: TensorShape,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Decode packed signs into approximate correction tensor."""
        if packed_bits.numel() == 0:
            return torch.empty(original_shape, dtype=torch.float16, device=packed_bits.device)

        signs01 = self._unpack_bits(packed_bits, self.sketch_dim)
        signs = signs01 * 2.0 - 1.0

        sketch = cast(torch.Tensor, self.S).to(packed_bits.device)
        correction = signs @ sketch
        correction = correction * (float(scale) / max(self.sketch_dim, 1))
        return correction.reshape(original_shape).to(torch.float16)

    def encode_with_scale(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode residual and return packed bits plus per-vector scale."""
        packed = self.encode(residual)
        scale = residual.float().norm(dim=-1, keepdim=True) / math.sqrt(self.sketch_dim)
        return packed, scale.to(torch.float16)

    def compress_ratio(self) -> float:
        """Return legacy-compatible compression ratio vs FP16 bits."""
        return float((self.head_dim * 16) / max(self.sketch_dim, 1))

    def mutual_information_bound(self, snr: float = 1.0) -> float:
        """Lower bound for I(R; sign(SR)) under simplified SNR assumption."""
        snr = max(float(snr), 1e-9)
        return float(self.sketch_dim / 2.0 * math.log(1.0 + snr))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, sketch_dim={self.sketch_dim}, "
            f"sketch_type={self.config.sketch_type}"
        )
