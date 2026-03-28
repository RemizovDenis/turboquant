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

    packed: np.ndarray
    scales: np.ndarray
    original_shape: tuple[int, int]
    original_dtype: np.dtype
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
    def compress_embeddings(self, vectors: np.ndarray) -> CompressedVectors:
        raise RuntimeError("Abstract method")

    @abc.abstractmethod
    def decompress_embeddings(self, compressed: CompressedVectors) -> np.ndarray:
        raise RuntimeError("Abstract method")

    @abc.abstractmethod
    def search(self, query: np.ndarray, top_k: int) -> list[SearchResult]:
        raise RuntimeError("Abstract method")

    @abc.abstractmethod
    def search_async(self, query: np.ndarray, top_k: int) -> Awaitable[list[SearchResult]]:
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
        embeddings: np.ndarray,
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

    def compress_embeddings(self, vectors: np.ndarray) -> CompressedVectors:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D")
        n, d = vectors.shape
        self._raw_nbytes = int(vectors.nbytes)
        if self.tq_config.head_dim != d:
            self.tq_config.head_dim = d
            self.cache = TurboQuantKVCache(self.tq_config)

        tensor = torch.from_numpy(vectors.astype(np.float32)).unsqueeze(1).unsqueeze(1)
        entry = self.cache.compress(tensor, tensor)
        packed, scales = entry.compressed_keys
        compressed = CompressedVectors(
            packed=packed.cpu().numpy(),
            scales=scales.cpu().numpy(),
            original_shape=(n, d),
            original_dtype=vectors.dtype,
            metadata={"n": n, "d": d},
        )
        # Allow direct usage via compress_embeddings() without an explicit add().
        if not self._ids or len(self._ids) != n:
            self._ids = [str(uuid.uuid4()) for _ in range(n)]
            self._payloads = [{} for _ in range(n)]
        self._compressed = compressed
        return compressed

    def decompress_embeddings(self, compressed: CompressedVectors) -> np.ndarray:
        n, d = compressed.original_shape
        packed = torch.from_numpy(compressed.packed)
        scales = torch.from_numpy(compressed.scales)
        restored = self.cache.quantizer.dequantize(packed, scales, (n, 1, 1, d))
        return restored.squeeze(1).squeeze(1).cpu().numpy().astype(np.float32)

    def search(self, query: np.ndarray, top_k: int) -> list[SearchResult]:
        if self._compressed is None:
            return []
        vectors = self.decompress_embeddings(self._compressed)
        q = query.reshape(1, -1).astype(np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        v = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        sims = (v @ q.T).squeeze(-1)
        top_idx = np.argsort(-sims)[:top_k]

        out: list[SearchResult] = []
        for i in top_idx.tolist():
            out.append(
                SearchResult(id=self._ids[i], score=float(sims[i]), payload=self._payloads[i])
            )
        return out

    def search_async(self, query: np.ndarray, top_k: int) -> Awaitable[list[SearchResult]]:
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
        embeddings: np.ndarray,
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
        embeddings: np.ndarray,
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
    vectors = rng.standard_normal((dataset_size, dim), dtype=np.float32)
    query = rng.standard_normal((dim,), dtype=np.float32)

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
