"""Cryptographic Watermarks for KV Cache.

Embeds and detects cryptographic watermarks in compressed KV scales
for attribution, anti-theft, and usage auditing purposes by modifying
the LSBs (least significant bits) of the float16 scales.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import numpy as np  # for unpackbits
import structlog
import torch

from turboquant.core.turboquant import CacheEntry

log = structlog.get_logger(__name__)


@dataclass
class WatermarkConfig:
    """Configuration for KV watermarking."""

    secret_key: str
    bits_per_scale_group: int = 2
    detection_threshold: float = 0.8
    enable_error_correction: bool = True


class KVCacheWatermarker:
    """Principal ML version of KV watermarker with HMAC-based bit generation."""

    def __init__(self, config: WatermarkConfig) -> None:
        self.config = config
        self._hmac_key = config.secret_key.encode()

    def _generate_watermark_bits(self, scales_shape: torch.Size, sequence_id: str) -> torch.Tensor:
        """Generate deterministic watermark bits based on secret key and sequence identity."""
        # Using HMAC-SHA256(key, sequence_id)
        h = hmac.new(self._hmac_key, sequence_id.encode(), hashlib.sha256)
        # Expand bits if necessary for scales_shape numel
        num_bits = scales_shape.numel() * self.config.bits_per_scale_group

        # Simple expansion via repeated hashing (pseudo-random stream)
        bits_raw: list[bytes] = []
        curr_hash: bytes = h.digest()
        while len(bits_raw) * 8 < num_bits:
            bits_raw.append(curr_hash)
            curr_hash = hashlib.sha256(curr_hash).digest()

        # Convert to bit tensor
        all_bytes = b"".join(bits_raw)
        bits_np = np.frombuffer(all_bytes, dtype=np.uint8)
        # Bit decomposition - explicitly use little-endian to match our LSB packing
        bits_tensor = torch.from_numpy(np.unpackbits(bits_np, bitorder="little"))[:num_bits]
        return bits_tensor.view(*scales_shape, self.config.bits_per_scale_group).to(torch.int16)

    def embed(self, entry: CacheEntry, sequence_id: str) -> CacheEntry:
        """Embed watermark into scales by modifying the last bits of float16 values."""
        # Modify scales from keys
        s_k = entry.compressed_keys[1]
        s_v = entry.compressed_values[1]

        # 1. Generate bits for both K and V scales
        bits_k = self._generate_watermark_bits(s_k.shape, sequence_id + "_k").to(s_k.device)
        bits_v = self._generate_watermark_bits(s_v.shape, sequence_id + "_v").to(s_v.device)

        # 2. Embed into float16 LSBs
        def _embed_in_lsb(scales: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
            # View as int16 to manipulate bits directly
            orig_dtype = scales.dtype
            s_bits = scales.float().to(torch.float16).view(torch.int16)

            # Mask out the last bits
            # If 2 bits per scale group: 0xFFFC (all ones except last 2 bits)
            mask = ~((1 << self.config.bits_per_scale_group) - 1)
            s_bits &= mask

            # Combine bits (assuming bits are 0..3)
            # Flatten bits to match scales numel if using multiple bits
            # For simplicity, each scale group (single float16) gets 'bits_per_scale_group' bits
            # But the 'bits' tensor here is (S, G, n_bits). We need to pack them into int16
            packed_bits = torch.zeros_like(s_bits)
            for i in range(self.config.bits_per_scale_group):
                packed_bits |= (bits[..., i] & 1) << i

            s_bits |= packed_bits

            return s_bits.view(torch.float16).to(orig_dtype)

        new_sk = _embed_in_lsb(s_k, bits_k)
        new_sv = _embed_in_lsb(s_v, bits_v)

        # Also handle residual scales if they exist
        new_rk, new_rv = entry.residual_norms_k, entry.residual_norms_v
        if new_rk is not None:
            bits_rk = self._generate_watermark_bits(new_rk.shape, sequence_id + "_rk").to(
                new_rk.device
            )
            new_rk = _embed_in_lsb(new_rk, bits_rk)
        if new_rv is not None:
            bits_rv = self._generate_watermark_bits(new_rv.shape, sequence_id + "_rv").to(
                new_rv.device
            )
            new_rv = _embed_in_lsb(new_rv, bits_rv)

        # Deepcopy entry with new scales
        new_entry = CacheEntry(
            compressed_keys=(entry.compressed_keys[0], new_sk),
            compressed_values=(entry.compressed_values[0], new_sv),
            residual_keys=entry.residual_keys,
            residual_values=entry.residual_values,
            residual_norms_k=new_rk,
            residual_norms_v=new_rv,
            metadata=entry.metadata.copy(),
        )
        return new_entry

    def detect(self, entry: CacheEntry, sequence_id: str) -> dict[str, float | bool | str]:
        """Detect watermark by extracting LSBs and comparing with deterministic expected bits."""
        s_k = entry.compressed_keys[1]

        # 1. Generate expected bits
        expected_bits = self._generate_watermark_bits(s_k.shape, sequence_id + "_k").to(s_k.device)

        # 2. Extract LSBs from current scales
        s_bits = s_k.float().to(torch.float16).view(torch.int16)

        total_bits = s_k.numel() * self.config.bits_per_scale_group

        extracted_bits = torch.zeros_like(expected_bits)
        for i in range(self.config.bits_per_scale_group):
            extracted_bits[..., i] = (s_bits >> i) & 1

        # Count matches
        matches = (extracted_bits == expected_bits).sum().item()
        match_rate = matches / total_bits

        detected = match_rate > self.config.detection_threshold
        # Confidence calculation: match_rate of 0.5 is random. 1.0 is certain.
        # Simple linear: (match_rate - 0.5) * 2.0
        confidence = max(0.0, (match_rate - 0.5) * 2.0)

        log.info("watermark_detection", detected=detected, match_rate=match_rate)

        return {
            "watermark_detected": detected,
            "confidence": confidence,
            "bit_match_rate": match_rate,
            "sequence_id": sequence_id,
        }

    def quality_impact(
        self, original_entry: CacheEntry, watermarked_entry: CacheEntry
    ) -> dict[str, float]:
        """Measure the maximum and relative difference in scales after watermarking."""
        s0 = original_entry.compressed_keys[1].float()
        s1 = watermarked_entry.compressed_keys[1].float()

        max_delta = (s1 - s0).abs().max().item()
        rel_error = ((s1 - s0).abs() / s0.clamp(min=1e-8)).mean().item()

        return {
            "scales_max_delta": max_delta,
            "scales_relative_error": rel_error,
        }
