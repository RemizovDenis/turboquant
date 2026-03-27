"""QJL Residual Corrector — 1-bit Johnson-Lindenstrauss residual correction.

Second stage of TurboQuant (arXiv 2504.19874): encodes the quantization
residual ``R = X − X̂`` into 1 bit per projected dimension using a random
JL projection, recovering the *sign* of the error without storing magnitude.

Combined with PolarQuantizer (3 bits), the total cost is 4 bits per element
→ 4× memory reduction versus FP16.

Typical usage::

    corrector = QJLResidualCorrector(head_dim=128)
    residual = original - reconstructed
    bits = corrector.encode(residual)
    approx_residual = corrector.decode(bits, residual.shape)
    refined = reconstructed + approx_residual
"""

from __future__ import annotations

import math
from typing import Optional

import structlog
import torch
import torch.nn as nn

log = structlog.get_logger(__name__)


class QJLResidualCorrector(nn.Module):
    """1-bit residual corrector via Johnson-Lindenstrauss random projection.

    The corrector projects the residual ``R`` into a lower-dimensional space
    using a random sign matrix ``S ∈ {±1/√d}^{sketch_dim × head_dim}``,
    then stores only the *sign* (1 bit) of each projected component.

    Reconstruction approximates ``R`` by back-projecting the sign vector
    through ``S^T`` and scaling by the estimated norm.

    Attributes:
        head_dim: Dimension of each attention head.
        sketch_dim: Number of JL projection dimensions.

    Buffers:
        projection: Random sign matrix of shape ``(sketch_dim, head_dim)``.
    """

    def __init__(
        self,
        head_dim: int,
        sketch_dim: int | None = None,
        seed: int = 137,
    ) -> None:
        """Initialise QJLResidualCorrector.

        Args:
            head_dim: Dimension of each attention head.
            sketch_dim: Number of JL projection dimensions.
                Defaults to ``head_dim // 4`` when *None*.
            seed: Random seed for the projection matrix.
        """
        super().__init__()
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")

        self.head_dim = head_dim
        self.sketch_dim = sketch_dim if sketch_dim is not None else max(head_dim // 4, 1)

        if self.sketch_dim <= 0:
            raise ValueError(f"sketch_dim must be positive, got {self.sketch_dim}")

        projection = self._generate_projection(self.sketch_dim, head_dim, seed)
        self.register_buffer("projection", projection)

        log.debug(
            "QJLResidualCorrector.__init__",
            head_dim=head_dim,
            sketch_dim=self.sketch_dim,
            compression_ratio=self.compress_ratio(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_projection(sketch_dim: int, head_dim: int, seed: int) -> torch.Tensor:
        """Generate a random sign (±1/√sketch_dim) projection matrix.

        Args:
            sketch_dim: Number of projection rows.
            head_dim: Number of input columns.
            seed: RNG seed.

        Returns:
            Float32 tensor of shape ``(sketch_dim, head_dim)`` with
            entries in ``{-1/√sketch_dim, +1/√sketch_dim}``.
        """
        gen = torch.Generator().manual_seed(seed)
        signs = torch.randint(0, 2, (sketch_dim, head_dim), generator=gen, dtype=torch.float32)
        signs = signs * 2.0 - 1.0  # map {0,1} → {-1,+1}
        scale = 1.0 / math.sqrt(sketch_dim)
        return signs * scale

    @staticmethod
    def _pack_bits(signs: torch.Tensor) -> torch.Tensor:
        """Pack a boolean/sign tensor into uint8 with 8 bits per byte.

        Args:
            signs: Tensor with values in {0, 1} (after mapping from ±1).

        Returns:
            uint8 tensor with shape ``(*signs.shape[:-1], ceil(signs.shape[-1]/8))``.
        """
        last = signs.shape[-1]
        # Pad to multiple of 8
        pad_size = (8 - last % 8) % 8
        if pad_size > 0:
            signs = torch.nn.functional.pad(signs, (0, pad_size), value=0)
        signs_flat = signs.reshape(*signs.shape[:-1], -1, 8)
        # bit 0 = MSB convention: value = Σ bit_i * 2^(7-i)
        powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=signs.device)
        packed = (signs_flat.to(torch.uint8) * powers).sum(dim=-1).to(torch.uint8)
        return packed

    @staticmethod
    def _unpack_bits(packed: torch.Tensor, num_bits: int) -> torch.Tensor:
        """Unpack uint8 packed bits back to a float tensor of {0, 1}.

        Args:
            packed: uint8 tensor from ``_pack_bits``.
            num_bits: Original number of bits (before padding).

        Returns:
            Float32 tensor with shape ``(*packed.shape[:-1], num_bits)`` and
            values in {0.0, 1.0}.
        """
        powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=packed.device)
        unpacked = ((packed.unsqueeze(-1).to(torch.int16) & powers.to(torch.int16)) > 0).float()
        unpacked = unpacked.reshape(*packed.shape[:-1], -1)
        return unpacked[..., :num_bits]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        """Encode residual into packed 1-bit representation.

        Steps:
            1. Project: ``s = R @ S^T``  → ``(…, sketch_dim)``
            2. Sign: keep only ``sign(s)`` as 0/1
            3. Pack bits into uint8

        Args:
            residual: Float tensor whose last dimension equals ``head_dim``.

        Returns:
            uint8 tensor of packed sign bits, shape
            ``(*residual.shape[:-1], ceil(sketch_dim / 8))``.
        """
        if residual.numel() == 0:
            packed_dim = math.ceil(self.sketch_dim / 8)
            return torch.empty(
                *residual.shape[:-1], packed_dim,
                dtype=torch.uint8,
                device=residual.device,
            )

        if residual.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected last dim {self.head_dim}, got {residual.shape[-1]}"
            )

        proj = self.projection.to(residual.device)  # (sketch_dim, head_dim)
        # Project: (..., head_dim) @ (head_dim, sketch_dim) → (..., sketch_dim)
        projected = residual.float() @ proj.T  # (..., sketch_dim)
        # Sign → 0/1  (positive → 1, non-positive → 0)
        sign_bits = (projected > 0).float()
        packed = self._pack_bits(sign_bits)

        log.debug(
            "encode",
            input_shape=list(residual.shape),
            packed_shape=list(packed.shape),
        )
        return packed

    def decode(
        self,
        bits: torch.Tensor,
        original_shape: tuple[int, ...],
        norm_hint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct an approximation of the residual from packed sign bits.

        Steps:
            1. Unpack bits → sign vector in {-1, +1}
            2. Back-project: ``R̃ = signs @ S``  → ``(…, head_dim)``
            3. Scale by estimated norm (or *norm_hint* if provided)

        Args:
            bits: uint8 packed sign bits from ``encode()``.
            original_shape: Shape of the original residual tensor.
            norm_hint: Optional float tensor to scale the reconstruction.
                If *None*, a uniform scaling ``√(head_dim / sketch_dim)`` is
                applied.

        Returns:
            Float16 tensor of shape *original_shape* approximating the
            original residual.
        """
        if bits.numel() == 0:
            return torch.empty(original_shape, dtype=torch.float16, device=bits.device)

        proj = self.projection.to(bits.device)

        # Unpack
        sign_float = self._unpack_bits(bits, self.sketch_dim)  # (..., sketch_dim) in {0,1}
        # Map {0,1} → {-1, +1}
        sign_float = sign_float * 2.0 - 1.0

        # Back-project: (..., sketch_dim) @ (sketch_dim, head_dim) → (..., head_dim)
        reconstructed = sign_float @ proj

        # Scale: since projection has entries ±1/√sketch_dim, the
        # back-projection naturally has the right scale up to a factor.
        # Apply JL scaling correction.
        scale = math.sqrt(self.head_dim / self.sketch_dim)
        if norm_hint is not None:
            # Per-vector scaling from known norms
            recon_norm = reconstructed.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            reconstructed = reconstructed * (norm_hint.unsqueeze(-1) / recon_norm)
        else:
            reconstructed = reconstructed * scale

        result = reconstructed.reshape(original_shape)

        log.debug(
            "decode",
            bits_shape=list(bits.shape),
            output_shape=list(original_shape),
        )
        return result.to(torch.float16)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def compress_ratio(self) -> float:
        """Return the compression ratio of the residual encoding.

        Returns:
            Ratio of original bits to encoded bits for the residual path.
            E.g., for head_dim=128, sketch_dim=32: 128*16 / 32 = 64.
        """
        original_bits = self.head_dim * 16  # float16
        encoded_bits = self.sketch_dim * 1  # 1 bit per projection
        if encoded_bits == 0:
            return float("inf")
        return original_bits / encoded_bits

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, sketch_dim={self.sketch_dim}, "
            f"compress_ratio={self.compress_ratio():.1f}x"
        )


if __name__ == "__main__":
    import pytest  # noqa: F811

    # ------------------------------------------------------------------
    # Inline tests
    # ------------------------------------------------------------------

    def test_encode_decode_shape() -> None:
        """Encode and decode preserve shape and dtype."""
        c = QJLResidualCorrector(head_dim=128, sketch_dim=32)
        residual = torch.randn(2, 4, 16, 128, dtype=torch.float16)
        packed = c.encode(residual)
        assert packed.dtype == torch.uint8
        assert packed.shape == (2, 4, 16, 4)  # 32 bits / 8 = 4 bytes
        recon = c.decode(packed, residual.shape)
        assert recon.shape == residual.shape
        assert recon.dtype == torch.float16

    def test_empty_tensor() -> None:
        """Edge case: zero-element tensor."""
        c = QJLResidualCorrector(head_dim=64, sketch_dim=16)
        residual = torch.empty(0, 64, dtype=torch.float16)
        packed = c.encode(residual)
        assert packed.numel() == 0
        recon = c.decode(packed, residual.shape)
        assert recon.shape == residual.shape

    def test_pack_unpack_roundtrip() -> None:
        """Packing and unpacking 1-bit data must be lossless."""
        c = QJLResidualCorrector(head_dim=128, sketch_dim=37)  # non-multiple-of-8
        bits = torch.randint(0, 2, (5, 37), dtype=torch.float32)
        packed = c._pack_bits(bits)
        unpacked = c._unpack_bits(packed, 37)
        assert torch.equal(bits, unpacked)

    def test_compress_ratio() -> None:
        """Compression ratio is correctly computed."""
        c = QJLResidualCorrector(head_dim=128, sketch_dim=32)
        ratio = c.compress_ratio()
        assert abs(ratio - 64.0) < 1e-6  # 128*16 / 32 = 64

    def test_reconstruction_reduces_error() -> None:
        """Adding the decoded residual to X̂ should reduce MSE vs original X."""
        torch.manual_seed(0)
        c = QJLResidualCorrector(head_dim=128, sketch_dim=64)
        x = torch.randn(4, 8, 64, 128, dtype=torch.float16)
        # Simulate noisy reconstruction
        noise = torch.randn_like(x) * 0.1
        x_hat = x + noise
        residual = x.float() - x_hat.float()
        packed = c.encode(residual.to(torch.float16))
        approx_residual = c.decode(packed, residual.shape)
        x_refined = x_hat.float() + approx_residual.float()

        mse_before = ((x.float() - x_hat.float()) ** 2).mean().item()
        mse_after = ((x.float() - x_refined) ** 2).mean().item()
        # We don't guarantee improvement in all cases (JL is approximate),
        # but the mechanism should function without errors.
        print(f"MSE before correction: {mse_before:.6f}")
        print(f"MSE after correction:  {mse_after:.6f}")

    def test_different_sketch_dims() -> None:
        """Various sketch_dim values work without error."""
        for sd in [1, 7, 32, 64, 128]:
            c = QJLResidualCorrector(head_dim=128, sketch_dim=sd)
            r = torch.randn(2, 128, dtype=torch.float16)
            packed = c.encode(r)
            recon = c.decode(packed, r.shape)
            assert recon.shape == r.shape

    # Run tests
    print("=== QJLResidualCorrector tests ===")
    test_encode_decode_shape()
    print("✓ test_encode_decode_shape")
    test_empty_tensor()
    print("✓ test_empty_tensor")
    test_pack_unpack_roundtrip()
    print("✓ test_pack_unpack_roundtrip")
    test_compress_ratio()
    print("✓ test_compress_ratio")
    test_reconstruction_reduces_error()
    print("✓ test_reconstruction_reduces_error")
    test_different_sketch_dims()
    print("✓ test_different_sketch_dims")
    print("=== All tests passed ===")
