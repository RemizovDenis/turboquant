"""Adaptive per-token bitwidth quantization for KV cache."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, cast

import structlog
import torch
import torch.nn as nn

from turboquant.core.polar_quant import PolarQuantizer

LOGGER = structlog.get_logger(__name__)


def _maybe_compile(fn: Any) -> Any:
    if not torch.cuda.is_available():
        return fn
    if hasattr(torch, "compile"):
        try:
            return torch.compile(fn, dynamic=True, backend="eager")
        except Exception:  # pragma: no cover - backend dependent
            return fn
    return fn


@dataclass
class AdaptiveBitwithConfig:
    """Configuration for adaptive bitwidth quantization."""

    head_dim: int
    num_heads: int
    vocab_size: int
    min_bits: int = 1
    max_bits: int = 4
    target_avg_bits: float = 2.3
    entropy_low_threshold: float = 0.3
    entropy_high_threshold: float = 0.7
    use_token_classifier: bool = True
    use_attention_entropy: bool = True
    classifier_embedding_dim: int = 16
    calibration_tokens: int = 10000
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


AdaptiveBitwidthConfig = AdaptiveBitwithConfig


@dataclass
class BitwithAssignment:
    """Per-token bit allocation plan."""

    token_indices: torch.Tensor
    bits_per_token: torch.Tensor
    avg_bits: float
    tokens_at_1bit: int
    tokens_at_2bit: int
    tokens_at_3bit: int
    tokens_at_4bit: int
    estimated_compression_ratio: float


@dataclass
class AdaptiveCompressedCache:
    """Compressed KV payload for adaptive per-token bitwidth."""

    components: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]
    original_shape: tuple[int, ...]
    assignment: BitwithAssignment
    metadata: dict[str, Any] = field(default_factory=dict)


class TokenImportanceClassifier(nn.Module):
    """Embedding-lookup token importance estimator."""

    def __init__(self, vocab_size: int, embedding_dim: int = 16) -> None:
        """Initialize classifier.

        Args:
            vocab_size: Token vocabulary size.
            embedding_dim: Unused compatibility argument for API stability.
        """
        super().__init__()
        del embedding_dim
        self.importance_embedding = nn.Embedding(vocab_size, 1)
        with torch.no_grad():
            self.importance_embedding.weight.fill_(3.0)

    @_maybe_compile
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Predict per-token importance in `[1, 4]`."""
        raw = self.importance_embedding(token_ids.long()).squeeze(-1)
        out = cast(torch.Tensor, raw.float().clamp(1.0, 4.0))
        return out

    def calibrate(self, token_ids: torch.Tensor, attention_weights: torch.Tensor) -> None:
        """Calibrate token importances from attention statistics via EMA."""
        if attention_weights.ndim != 3:
            raise ValueError("attention_weights must have shape [heads, seq_len, seq_len]")

        received = attention_weights.float().sum(dim=1).mean(dim=0)
        lo = torch.quantile(received, 0.05)
        hi = torch.quantile(received, 0.95)
        denom = (hi - lo).clamp_min(1e-8)
        norm = ((received - lo) / denom).clamp(0.0, 1.0)
        mapped = 1.0 + norm * 3.0

        token_ids = token_ids.long()
        if token_ids.numel() != mapped.numel():
            n = min(token_ids.numel(), mapped.numel())
            token_ids = token_ids[:n]
            mapped = mapped[:n]

        with torch.no_grad():
            weights = self.importance_embedding.weight.data
            current = weights[token_ids, 0]
            updated = 0.95 * current + 0.05 * mapped.to(current.device)
            weights[token_ids, 0] = updated.to(weights.dtype)


class AdaptiveBitwidthQuantizer:
    """Assign and apply dynamic bitwidths per token."""

    def __init__(self, config: AdaptiveBitwithConfig) -> None:
        """Initialize adaptive quantizer and per-bit quantizer bank."""
        self.config = config
        self.device = torch.device(config.device)
        self.classifier = TokenImportanceClassifier(
            vocab_size=config.vocab_size,
            embedding_dim=config.classifier_embedding_dim,
        ).to(self.device)
        self.quantizers: dict[int, PolarQuantizer] = {
            bits: PolarQuantizer(head_dim=config.head_dim, bits=bits).to(self.device)
            for bits in range(config.min_bits, config.max_bits + 1)
        }
        self._compress_count = 0
        self._lock = threading.RLock()
        self._logger = LOGGER.bind(component="AdaptiveBitwidthQuantizer")

    def _entropy_to_bits(self, attention_entropy: torch.Tensor) -> torch.Tensor:
        low = float(self.config.entropy_low_threshold)
        high = float(self.config.entropy_high_threshold)
        norm = ((attention_entropy.float() - low) / max(1e-8, high - low)).clamp(0.0, 1.0)
        return float(self.config.min_bits) + norm * float(
            self.config.max_bits - self.config.min_bits
        )

    def _apply_budget_control(self, bits: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        avg_bits = float(bits.float().mean().item()) if bits.numel() > 0 else 0.0
        target = float(self.config.target_avg_bits)
        if avg_bits <= target:
            return bits

        adjusted = bits.clone()
        excess = int(round((avg_bits - target) * max(1, bits.numel())))
        order = torch.argsort(importance, descending=False)
        min_bits = int(self.config.min_bits)
        for idx in order.tolist():
            if excess <= 0:
                break
            while adjusted[idx] > min_bits and excess > 0:
                adjusted[idx] -= 1
                excess -= 1
        return adjusted

    def assign_bitwidths(
        self,
        token_ids: torch.Tensor | None,
        attention_entropy: torch.Tensor | None,
        seq_len: int,
    ) -> BitwithAssignment:
        """Assign bitwidth to each token using classifier and/or entropy."""
        seq_len = max(0, int(seq_len))
        indices = torch.arange(seq_len, device=self.device, dtype=torch.long)

        raw_bits: torch.Tensor | None = None
        if self.config.use_token_classifier and token_ids is not None:
            token_ids = token_ids.to(self.device).long()
            if token_ids.numel() < seq_len:
                pad = torch.zeros(
                    (seq_len - token_ids.numel(),), device=self.device, dtype=torch.long
                )
                token_ids = torch.cat([token_ids, pad], dim=0)
            raw_bits = self.classifier(token_ids[:seq_len])

        entropy_bits: torch.Tensor | None = None
        if self.config.use_attention_entropy and attention_entropy is not None:
            entropy = attention_entropy.to(self.device).float()
            if entropy.numel() < seq_len:
                pad = torch.zeros(
                    (seq_len - entropy.numel(),), device=self.device, dtype=torch.float32
                )
                entropy = torch.cat([pad, entropy], dim=0)
            entropy_bits = self._entropy_to_bits(entropy[:seq_len])

        if raw_bits is not None and entropy_bits is not None:
            bits_f = 0.6 * raw_bits + 0.4 * entropy_bits
            importance = bits_f.clone()
        elif raw_bits is not None:
            bits_f = raw_bits
            importance = bits_f.clone()
        elif entropy_bits is not None:
            bits_f = entropy_bits
            importance = bits_f.clone()
        else:
            bits_f = torch.full((seq_len,), 3.0, device=self.device, dtype=torch.float32)
            importance = bits_f.clone()

        bits_i = bits_f.round().clamp(self.config.min_bits, self.config.max_bits).to(torch.int64)
        bits_i = self._apply_budget_control(bits_i, importance)
        bits_u8 = bits_i.to(torch.uint8)

        avg = float(bits_i.float().mean().item()) if seq_len > 0 else 0.0
        c1 = int((bits_i == 1).sum().item())
        c2 = int((bits_i == 2).sum().item())
        c3 = int((bits_i == 3).sum().item())
        c4 = int((bits_i == 4).sum().item())

        return BitwithAssignment(
            token_indices=indices,
            bits_per_token=bits_u8,
            avg_bits=avg,
            tokens_at_1bit=c1,
            tokens_at_2bit=c2,
            tokens_at_3bit=c3,
            tokens_at_4bit=c4,
            estimated_compression_ratio=float(16.0 / max(1e-8, avg)),
        )

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        token_ids: torch.Tensor | None = None,
        attention_entropy: torch.Tensor | None = None,
    ) -> AdaptiveCompressedCache:
        """Compress KV with adaptive per-token bitwidth."""
        keys = keys.to(self.device)
        values = values.to(self.device)
        seq_len = int(keys.shape[2])
        assignment = self.assign_bitwidths(token_ids, attention_entropy, seq_len)
        bits = assignment.bits_per_token.to(dtype=torch.int64, device=self.device)

        components: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        for bitwidth in range(self.config.min_bits, self.config.max_bits + 1):
            mask = bits == bitwidth
            if not bool(mask.any()):
                continue
            sub_keys = keys[:, :, mask, :]
            sub_vals = values[:, :, mask, :]
            packed_k, scales_k = self.quantizers[bitwidth](sub_keys)
            packed_v, scales_v = self.quantizers[bitwidth](sub_vals)
            packed = torch.stack([packed_k, packed_v], dim=0)
            scales = torch.stack([scales_k, scales_v], dim=0)
            components.append((packed, scales, mask.to(torch.uint8), bitwidth))

        self._compress_count += 1
        if self._compress_count % 100 == 0:
            self._logger.info(
                "adaptive_bitwidth_stats",
                avg_bits=assignment.avg_bits,
                bits_distribution={
                    "1": assignment.tokens_at_1bit,
                    "2": assignment.tokens_at_2bit,
                    "3": assignment.tokens_at_3bit,
                    "4": assignment.tokens_at_4bit,
                },
            )

        return AdaptiveCompressedCache(
            components=components,
            original_shape=tuple(int(x) for x in keys.shape),
            assignment=assignment,
            metadata={"seq_len": seq_len},
        )

    def decompress(
        self,
        compressed: AdaptiveCompressedCache,
        original_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress adaptive representation back to KV tensors."""
        batch, heads, seq_len, head_dim = (int(x) for x in original_shape)
        keys = torch.zeros(original_shape, dtype=torch.float16, device=self.device)
        values = torch.zeros_like(keys)

        for packed, scales, mask_u8, bitwidth in compressed.components:
            mask = mask_u8.to(torch.bool)
            token_count = int(mask.sum().item())
            if token_count == 0:
                continue
            sub_k = self.quantizers[bitwidth].dequantize(packed[0], scales[0])
            sub_v = self.quantizers[bitwidth].dequantize(packed[1], scales[1])
            keys[:, :, mask, :] = sub_k
            values[:, :, mask, :] = sub_v

        return (keys, values)

    def calibrate_on_dataset(
        self,
        token_ids_list: list[torch.Tensor],
        attention_weights_list: list[torch.Tensor],
    ) -> None:
        """Calibrate classifier online on a dataset of tokens and attention maps."""
        n = min(len(token_ids_list), len(attention_weights_list))
        if n == 0:
            return

        pre = []
        post = []
        for idx in range(n):
            tokens = token_ids_list[idx].to(self.device)
            attn = attention_weights_list[idx].to(self.device)
            before = self.assign_bitwidths(tokens, None, seq_len=int(tokens.numel())).avg_bits
            self.classifier.calibrate(tokens, attn)
            after = self.assign_bitwidths(tokens, None, seq_len=int(tokens.numel())).avg_bits
            pre.append(before)
            post.append(after)

        self._logger.info(
            "adaptive_calibration",
            avg_bits_before=float(sum(pre) / max(1, len(pre))),
            avg_bits_after=float(sum(post) / max(1, len(post))),
        )

    def actual_avg_bits(self, compressed: AdaptiveCompressedCache) -> float:
        """Compute effective average bits including scale overhead."""
        shape = compressed.original_shape
        batch, heads, seq_len, head_dim = (int(x) for x in shape)
        total_scalars = batch * heads * seq_len * head_dim * 2
        if total_scalars <= 0:
            return 0.0

        total_bits = 0.0
        for packed, scales, _, _ in compressed.components:
            total_bits += float(packed.numel() * packed.element_size() * 8)
            total_bits += float(scales.numel() * scales.element_size() * 8)
        return total_bits / max(1.0, float(total_scalars))

    def memory_savings_vs_uniform(
        self,
        compressed: AdaptiveCompressedCache,
        seq_len: int,
    ) -> dict[str, float]:
        """Compare adaptive storage against uniform 3-bit baseline."""
        del seq_len
        avg_bits = self.actual_avg_bits(compressed)
        additional = (3.0 - avg_bits) / 3.0 * 100.0
        assignment = compressed.assignment
        return {
            "additional_savings_percent": float(additional),
            "avg_bits": float(avg_bits),
            "bits_distribution_1": float(assignment.tokens_at_1bit),
            "bits_distribution_2": float(assignment.tokens_at_2bit),
            "bits_distribution_3": float(assignment.tokens_at_3bit),
            "bits_distribution_4": float(assignment.tokens_at_4bit),
        }
