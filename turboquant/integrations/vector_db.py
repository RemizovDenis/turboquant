# Copyright (c) 2026 Denis Remizov. Licensed under BUSL-1.1.
# See LICENSE file for details.


"""TurboQuant adapters for vector databases."""

from __future__ import annotations

import abc
import asyncio
import time
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from turboquant.core.turboquant import TurboQuantConfig, TurboQuantKVCache


@dataclass
class CompressedVectors:
    """Compressed embedding block container."""

    packed: np.ndarray[Any, np.dtype[np.uint8]]
    scales: np.ndarray[Any, np.dtype[np.float32]]
    original_shape: tuple[int, int]
    original_dtype: np.dtype[Any]
    metadata: dict[str, Any]


@dataclass
class SearchResult:
    """Single nearest-neighbor search result."""

    id: str
    score: float
    payload: dict[str, Any]


class TurboQuantVectorAdapter(abc.ABC):
    """Abstract base adapter for compressed vector backends."""

    @abc.abstractmethod
    def compress_embeddings(
        self, vectors: np.ndarray[Any, np.dtype[np.float32]]
    ) -> CompressedVectors:
        raise RuntimeError("Abstract method")

    @abc.abstractmethod
    def decompress_embeddings(
        self, compressed: CompressedVectors
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        raise RuntimeError("Abstract method")

    @abc.abstractmethod
    def search(
        self, query: np.ndarray[Any, np.dtype[np.float32]], top_k: int
    ) -> list[SearchResult]:
        raise RuntimeError("Abstract method")

    @abc.abstractmethod
    def search_async(
        self, query: np.ndarray[Any, np.dtype[np.float32]], top_k: int
    ) -> Awaitable[list[SearchResult]]:
        raise RuntimeError("Abstract method")

    @abc.abstractmethod
    def index_size_mb(self) -> float:
        raise RuntimeError("Abstract method")

    @abc.abstractmethod
    def compression_ratio(self) -> float:
        raise RuntimeError("Abstract method")


class InMemoryTurboQuant(TurboQuantVectorAdapter):
    """Numpy-based in-memory adapter for datasets up to ~100k vectors."""

    def __init__(self, tq_config: TurboQuantConfig) -> None:
        self.tq_config = tq_config
        self.cache = TurboQuantKVCache(tq_config)
        self._compressed: CompressedVectors | None = None
        self._ids: list[str] = []
        self._payloads: list[dict[str, Any]] = []
        self._raw_nbytes = 0

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray[Any, np.dtype[np.float32]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if len(ids) != embeddings.shape[0]:
            raise ValueError("ids size mismatch")
        self._ids = ids
        self._payloads = metadatas or [{} for _ in ids]
        self._raw_nbytes = int(embeddings.nbytes)
        self._compressed = self.compress_embeddings(embeddings)

    def compress_embeddings(
        self, vectors: np.ndarray[Any, np.dtype[np.float32]]
    ) -> CompressedVectors:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D")
        n, d = vectors.shape
        self._raw_nbytes = int(vectors.nbytes)
        if self.tq_config.head_dim != d:
            self.tq_config.head_dim = d
            self.cache = TurboQuantKVCache(self.tq_config)

        # Reshape for KV cache compression
        tensor = torch.from_numpy(vectors.astype(np.float32)).unsqueeze(1).unsqueeze(1)
        entry = self.cache.compress(tensor, tensor)
        from turboquant.core.turboquant import CacheEntry

        if not isinstance(entry, CacheEntry):
            raise TypeError(f"Vector DB requires standard CacheEntry, got {type(entry)}")
        packed, scales = entry.compressed_keys
        compressed = CompressedVectors(
            packed=packed.cpu().numpy(),
            scales=scales.cpu().numpy(),
            original_shape=(n, d),
            original_dtype=vectors.dtype,
            metadata={"n": n, "d": d},
        )
        if not self._ids or len(self._ids) != n:
            self._ids = [str(uuid.uuid4()) for _ in range(n)]
            self._payloads = [{} for _ in range(n)]
        self._compressed = compressed
        return compressed

    def decompress_embeddings(
        self, compressed: CompressedVectors
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        n, d = compressed.original_shape
        packed = torch.from_numpy(compressed.packed)
        scales = torch.from_numpy(compressed.scales)

        # Accessing the internal quantizer correctly
        restored = self.cache.polar.dequantize(
            packed.to(self.cache.device), scales.to(self.cache.device)
        )
        # Handle original shape correctly during decompression
        res: np.ndarray[Any, np.dtype[np.float32]] = (
            restored.squeeze(1).squeeze(1).cpu().numpy().astype(np.float32)
        )
        return res

    @staticmethod
    def _safe_unit_normalize(
        matrix: np.ndarray[Any, np.dtype[np.float32]],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Normalize rows while guarding against NaN/Inf and zero-norm vectors."""
        x = np.asarray(matrix, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # Use float64 accumulation to avoid overflow in norm calculation.
        norms = np.linalg.norm(x.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
        norms = np.where(norms > 1e-9, norms, 1.0).astype(np.float32)
        normalized = x / norms
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    def search(
        self, query: np.ndarray[Any, np.dtype[np.float32]], top_k: int
    ) -> list[SearchResult]:
        if self._compressed is None:
            return []
        if top_k <= 0:
            return []

        vectors = self.decompress_embeddings(self._compressed)
        if query.ndim != 1:
            raise ValueError("query must be 1D")
        if vectors.ndim != 2:
            raise ValueError("index vectors must be 2D")
        if query.shape[0] != vectors.shape[1]:
            raise ValueError(
                f"query dim mismatch: got {query.shape[0]}, expected {vectors.shape[1]}"
            )

        q = self._safe_unit_normalize(query.reshape(1, -1))
        v = self._safe_unit_normalize(vectors)
        v = np.clip(v, -1.0, 1.0)
        q = np.clip(q, -1.0, 1.0)
        # Use float64 accumulation in similarity to avoid overflow warnings on extreme values.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims64 = np.matmul(v.astype(np.float64, copy=False), q.T.astype(np.float64, copy=False))
        sims = np.nan_to_num(
            sims64.squeeze(-1).astype(np.float32, copy=False),
            nan=-1.0,
            posinf=1.0,
            neginf=-1.0,
        )

        valid_n = min(len(self._ids), len(self._payloads), int(sims.shape[0]))
        if valid_n == 0:
            return []
        if valid_n < sims.shape[0]:
            sims = sims[:valid_n]

        k = min(int(top_k), int(valid_n))
        top_idx = np.argsort(-sims)[:k]

        out: list[SearchResult] = []
        for i in top_idx.tolist():
            out.append(
                SearchResult(id=self._ids[i], score=float(sims[i]), payload=self._payloads[i])
            )
        return out

    def search_async(
        self, query: np.ndarray[Any, np.dtype[np.float32]], top_k: int
    ) -> Awaitable[list[SearchResult]]:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, lambda: self.search(query, top_k))

    def index_size_mb(self) -> float:
        if self._compressed is None:
            return 0.0
        nbytes = self._compressed.packed.nbytes + self._compressed.scales.nbytes
        return float(nbytes) / float(1024**2)

    def compression_ratio(self) -> float:
        if self._raw_nbytes == 0:
            return 1.0
        return (self.index_size_mb() * 1024 * 1024) / self._raw_nbytes


class ChromaDBTurboQuant(InMemoryTurboQuant):
    """Chroma adapter storing compressed vectors in metadata payload."""

    def __init__(
        self,
        collection_name: str,
        tq_config: TurboQuantConfig,
        client: Any | None = None,
    ) -> None:
        super().__init__(tq_config)
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("chromadb is required for ChromaDBTurboQuant") from exc

        self.client = client or chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray[Any, np.dtype[np.float32]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        super().add(ids, embeddings, metadatas)
        assert self._compressed is not None
        payloads = metadatas or [{} for _ in ids]
        for i, payload in enumerate(payloads):
            payload["tq_packed"] = self._compressed.packed[i].tolist()
            payload["tq_scales"] = self._compressed.scales[i].tolist()
        self.collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=payloads)


class QdrantTurboQuant(InMemoryTurboQuant):
    """Qdrant adapter with batch upsert and compressed payload storage."""

    def __init__(
        self,
        collection_name: str,
        tq_config: TurboQuantConfig,
        client: Any | None = None,
    ) -> None:
        super().__init__(tq_config)
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError("qdrant-client is required for QdrantTurboQuant") from exc

        self.client = client or QdrantClient(":memory:")
        self.collection_name = collection_name

    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray[Any, np.dtype[np.float32]],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        super().add(ids, embeddings, payloads)
        assert self._compressed is not None
        payloads = payloads or [{} for _ in ids]
        points = []
        for i, _id in enumerate(ids):
            payload = dict(payloads[i])
            payload["__tq_compressed"] = {
                "packed": self._compressed.packed[i].tolist(),
                "scales": self._compressed.scales[i].tolist(),
            }
            points.append({"id": _id, "vector": embeddings[i].tolist(), "payload": payload})

        batch_size = 256
        for start in range(0, len(points), batch_size):
            chunk = points[start : start + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=chunk)


def create_adapter(
    backend: str,
    config: TurboQuantConfig,
    **kwargs: Any,
) -> TurboQuantVectorAdapter:
    """Factory for vector DB adapters."""
    name = backend.lower()
    if name == "memory":
        return InMemoryTurboQuant(config)
    if name == "chroma":
        return ChromaDBTurboQuant(kwargs["collection_name"], config, kwargs.get("client"))
    if name == "qdrant":
        return QdrantTurboQuant(kwargs["collection_name"], config, kwargs.get("client"))
    raise ValueError("backend must be one of: chroma, qdrant, memory")


def benchmark_vector_db(
    adapter: TurboQuantVectorAdapter,
    dataset_size: int = 10000,
    dim: int = 1536,
    top_k: int = 10,
) -> dict[str, float]:
    """Benchmark indexing/search speed, memory, recall and compression ratio."""
    rng = np.random.default_rng(42)
    # npt.NDArray can be used as well, but np.ndarray[Any, np.dtype[...]] is more standard for --strict
    vectors: np.ndarray[Any, np.dtype[np.float32]] = rng.standard_normal(
        (dataset_size, dim), dtype=np.float32
    )
    query: np.ndarray[Any, np.dtype[np.float32]] = rng.standard_normal((dim,), dtype=np.float32)

    ids = [str(uuid.uuid4()) for _ in range(dataset_size)]

    t0 = time.perf_counter()
    if isinstance(adapter, (InMemoryTurboQuant, ChromaDBTurboQuant)):
        adapter.add(ids, vectors)
    elif isinstance(adapter, QdrantTurboQuant):
        adapter.upsert(ids, vectors)
    else:
        adapter.compress_embeddings(vectors)
    index_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    results = adapter.search(query, top_k=top_k)
    search_ms = (time.perf_counter() - t1) * 1000.0

    exact_scores = vectors @ query
    exact_idx = np.argsort(-exact_scores)[:10]
    found_ids = {res.id for res in results}
    exact_ids = {ids[i] for i in exact_idx}
    recall_at_10 = len(found_ids & exact_ids) / max(1, len(exact_ids))

    return {
        "index_time_ms": index_ms,
        "search_time_ms": search_ms,
        "memory_mb": adapter.index_size_mb(),
        "recall_at_10": recall_at_10,
        "compression_ratio": adapter.compression_ratio(),
    }
