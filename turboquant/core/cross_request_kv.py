"""Cross-Request KV Cache Sharing for server inference.

Optimizes VRAM usage by sharing compressed KV blocks for common prefixes
(system prompts, repeated context) across multiple concurrent requests.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field

import structlog
import torch

from turboquant.core.turboquant import CacheEntry, TurboQuantConfig, TurboQuantKVCache

log = structlog.get_logger(__name__)


@dataclass
class SharedKVBlock:
    """Immutable compressed KV block shared across requests."""

    entry: CacheEntry
    prefix_hash: str
    token_count: int
    ref_count: int = 1
    created_at: float = field(default_factory=time.monotonic)
    access_count: int = 1
    last_access: float = field(default_factory=time.monotonic)


class CrossRequestKVCache:
    """Copy-on-write KV cache with prefix sharing.

    Thread-safe for concurrent server requests.
    """

    def __init__(
        self,
        tq_config: TurboQuantConfig,
        max_shared_blocks: int = 64,
        min_prefix_len: int = 32,
        eviction_policy: str = "lru",
        ttl_seconds: float = 3600.0,
    ) -> None:
        self.tq = TurboQuantKVCache(tq_config)
        self.max_shared_blocks = max_shared_blocks
        self.min_prefix_len = min_prefix_len
        self.eviction_policy = eviction_policy
        self.ttl_seconds = ttl_seconds

        self._shared_blocks: dict[str, SharedKVBlock] = {}
        self._lock = threading.RLock()

        # Stats
        self._total_requests = 0
        self._total_hits = 0

    def register_prefix(
        self,
        prefix_tokens: list[int],
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> str:
        """Compress and register a prefix KV block. Returns block_id (hash)."""
        if len(prefix_tokens) < self.min_prefix_len:
            # Skip registering short prefixes
            return ""

        block_id = self._compute_hash(prefix_tokens)

        with self._lock:
            self._total_requests += 1
            if block_id in self._shared_blocks:
                block = self._shared_blocks[block_id]
                block.ref_count += 1
                block.access_count += 1
                block.last_access = time.monotonic()
                self._total_hits += 1
                return block_id

            # Check for eviction if full
            if len(self._shared_blocks) >= self.max_shared_blocks:
                self._evict()

            # Compress and store
            entry = self.tq.compress(keys, values)
            self._shared_blocks[block_id] = SharedKVBlock(
                entry=entry, prefix_hash=block_id, token_count=len(prefix_tokens)
            )

            log.info(
                "prefix_registered",
                block_id=block_id,
                tokens=len(prefix_tokens),
                shared_count=len(self._shared_blocks),
            )
            return block_id

    def get_prefix_entry(self, block_id: str) -> CacheEntry | None:
        """Retrieve shared compressed prefix block."""
        with self._lock:
            if block_id in self._shared_blocks:
                block = self._shared_blocks[block_id]
                block.access_count += 1
                block.last_access = time.monotonic()
                return block.entry
            return None

    def release_prefix(self, block_id: str) -> None:
        """Decrement ref_count. Candidates for eviction if ref_count == 0."""
        with self._lock:
            if block_id in self._shared_blocks:
                block = self._shared_blocks[block_id]
                block.ref_count = max(0, block.ref_count - 1)

    def extend_with_private(
        self,
        block_id: str,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> CacheEntry:
        """Create private extension of shared prefix (private part only)."""
        # The shared prefix is NOT copied. New tokens are compressed separately.
        return self.tq.compress(new_keys, new_values)

    def decompress_full(
        self,
        block_id: str,
        private_entry: CacheEntry | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress shared prefix + optional private part into full KV."""
        prefix_entry = self.get_prefix_entry(block_id)
        if prefix_entry is None:
            raise KeyError(f"Shared prefix block {block_id} not found.")

        k_pre, v_pre = self.tq.decompress(prefix_entry)

        if private_entry is not None:
            k_pri, v_pri = self.tq.decompress(private_entry)
            # Concatenate along seq_len (dim=2 for [B, H, S, D])
            k_full = torch.cat([k_pre, k_pri], dim=2)
            v_full = torch.cat([v_pre, v_pri], dim=2)
            return k_full, v_full

        return k_pre, v_pre

    def stats(self) -> dict[str, float | int]:
        """Return sharing statistics and VRAM savings estimate."""
        with self._lock:
            total_saved_bytes = 0
            total_refs = 0
            for block in self._shared_blocks.values():
                # Saving: (ref_count - 1) * footprint
                mem = self.tq.memory_usage(block.entry)
                total_saved_bytes += (block.ref_count - 1) * mem["total_mb"] * (1024**2)
                total_refs += block.ref_count

            hit_rate = self._total_hits / max(self._total_requests, 1)

            return {
                "shared_blocks": len(self._shared_blocks),
                "total_refs": total_refs,
                "hit_rate": hit_rate,
                "vram_saved_mb": total_saved_bytes / (1024**2),
                "avg_prefix_len": sum(b.token_count for b in self._shared_blocks.values())
                / max(len(self._shared_blocks), 1),
            }

    def _evict(self) -> None:
        """Evict least recently used block with ref_count == 0."""
        # For simplicity, LRU of ref_count == 0 candidates.
        # If all are in use, evict LRU of all.
        candidates = [bid for bid, block in self._shared_blocks.items() if block.ref_count == 0]
        if not candidates:
            # TTL check
            now = time.monotonic()
            candidates = [
                bid
                for bid, block in self._shared_blocks.items()
                if now - block.last_access > self.ttl_seconds
            ]
            if not candidates:
                # Force LRU
                candidates = list(self._shared_blocks.keys())

        # LRU find
        victim_id = min(candidates, key=lambda bid: self._shared_blocks[bid].last_access)
        del self._shared_blocks[victim_id]
        log.info("prefix_evicted", block_id=victim_id)

    def _compute_hash(self, tokens: list[int]) -> str:
        """SHA256 of token sequence."""
        # Using string representation for quick hashing
        data = ",".join(map(str, tokens)).encode()
        return hashlib.sha256(data).hexdigest()[:16]


def compute_prefix_hash(tokens: list[int]) -> str:
    """Standalone hash function."""
    data = ",".join(map(str, tokens)).encode()
    return hashlib.sha256(data).hexdigest()[:16]
