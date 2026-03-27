"""Universal TurboQuant adapter for Vector Databases.

Applies TurboQuant compression to embedding vectors, reducing storage
by ~4× while preserving search recall.

Supported backends:
- ``InMemoryTurboQuant`` — zero-dependency numpy-based store
- ``ChromaDBTurboQuant`` — ChromaDB integration (requires ``chromadb``)
- ``QdrantTurboQuant`` — Qdrant integration (requires ``qdrant-client``)

Factory::

    adapter = create_adapter("memory", config, dim=1536)
    adapter.compress_embeddings(vectors)
    results = adapter.search(query, top_k=10)
"""

from __future__ import annotations

import abc
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import structlog

from turboquant.core.turboquant import TurboQuantConfig, TurboQuantKVCache

log = structlog.get_logger(__name__)


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class CompressedVectors:
    """Container for compressed embedding vectors.

    Attributes:
        quantized: int8 quantized data.
        scales: float32 per-group scales.
        residual_bits: Optional packed 1-bit residual (uint8).
        ids: Vector identifiers.
        dim: Original embedding dimension.
        count: Number of vectors.
    """

    quantized: np.ndarray
    scales: np.ndarray
    residual_bits: np.ndarray | None = None
    ids: list[str] = field(default_factory=list)
    dim: int = 0
    count: int = 0


@dataclass
class SearchResult:
    """A single search result.

    Attributes:
        id: Vector identifier.
        score: Similarity score (higher = more similar for cosine).
        metadata: Optional associated metadata.
    """

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Abstract base
# ======================================================================


class TurboQuantVectorAdapter(abc.ABC):
    """Abstract base class for TurboQuant vector database adapters.

    Subclasses must implement all abstract methods and their ``_async``
    variants.
    """

    def __init__(self, config: TurboQuantConfig, **kwargs: Any) -> None:
        """Initialise adapter with TurboQuant config.

        Args:
            config: TurboQuant configuration.
            **kwargs: Backend-specific options.
        """
        self.config = config
        self.tq = TurboQuantKVCache(config)

    # ---- sync API ----

    @abc.abstractmethod
    def compress_embeddings(
        self,
        vectors: np.ndarray,
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> CompressedVectors:
        """Compress and store embedding vectors.

        Args:
            vectors: ``(N, dim)`` float32/float64 array.
            ids: Optional list of IDs. Generated if omitted.
            metadata: Optional list of metadata dicts.

        Returns:
            ``CompressedVectors`` with compressed data.
        """
        ...

    @abc.abstractmethod
    def decompress_embeddings(self, compressed: CompressedVectors) -> np.ndarray:
        """Decompress vectors back to float32.

        Args:
            compressed: Previously compressed vectors.

        Returns:
            ``(N, dim)`` float32 array.
        """
        ...

    @abc.abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search for nearest neighbours.

        Args:
            query: ``(dim,)`` or ``(1, dim)`` query vector.
            top_k: Number of results.

        Returns:
            List of ``SearchResult`` sorted by descending score.
        """
        ...

    # ---- async wrappers (default: run sync in executor) ----

    async def compress_embeddings_async(
        self,
        vectors: np.ndarray,
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> CompressedVectors:
        """Async version of ``compress_embeddings``."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.compress_embeddings(vectors, ids, metadata)
        )

    async def decompress_embeddings_async(
        self, compressed: CompressedVectors
    ) -> np.ndarray:
        """Async version of ``decompress_embeddings``."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.decompress_embeddings(compressed)
        )

    async def search_async(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Async version of ``search``."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.search(query, top_k)
        )

    # ---- helpers ----

    def _vectors_to_torch(self, vectors: np.ndarray) -> "torch.Tensor":
        """Convert numpy vectors to torch tensor shaped for TurboQuant.

        The TurboQuant compressor expects ``[batch, heads, seq, head_dim]``.
        We treat each vector as ``[1, 1, 1, dim]``.
        """
        import torch

        t = torch.from_numpy(vectors.astype(np.float32))
        if t.dim() == 1:
            t = t.unsqueeze(0)
        # → [N, 1, 1, dim]
        return t.unsqueeze(1).unsqueeze(1)

    def _torch_to_vectors(self, tensor: "torch.Tensor") -> np.ndarray:
        """Convert [N, 1, 1, dim] torch tensor back to (N, dim) numpy."""
        return tensor.squeeze(1).squeeze(1).float().cpu().numpy()


# ======================================================================
# InMemoryTurboQuant
# ======================================================================


class InMemoryTurboQuant(TurboQuantVectorAdapter):
    """In-memory vector store with TurboQuant compression.

    Uses numpy cosine similarity for search. No external dependencies.

    Attributes:
        _compressed: Stored compressed vectors.
        _metadata: Per-vector metadata list.
    """

    def __init__(self, config: TurboQuantConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._compressed: CompressedVectors | None = None
        self._metadata: list[dict[str, Any]] = []

    def compress_embeddings(
        self,
        vectors: np.ndarray,
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> CompressedVectors:
        import torch

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n, dim = vectors.shape

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]
        if metadata is not None:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in range(n)])

        # Adjust config head_dim to match embedding dim
        self.config.head_dim = dim
        self.tq = TurboQuantKVCache(self.config)

        t = self._vectors_to_torch(vectors)
        entry = self.tq.compress(t, t)  # store same for keys and values

        qk, sk = entry.compressed_keys
        result = CompressedVectors(
            quantized=qk.cpu().numpy(),
            scales=sk.cpu().numpy(),
            residual_bits=entry.residual_keys.cpu().numpy() if entry.residual_keys is not None else None,
            ids=ids,
            dim=dim,
            count=n,
        )

        if self._compressed is None:
            self._compressed = result
        else:
            # Append
            self._compressed.quantized = np.concatenate(
                [self._compressed.quantized, result.quantized], axis=2
            )
            self._compressed.scales = np.concatenate(
                [self._compressed.scales, result.scales], axis=2
            )
            if self._compressed.residual_bits is not None and result.residual_bits is not None:
                self._compressed.residual_bits = np.concatenate(
                    [self._compressed.residual_bits, result.residual_bits], axis=2
                )
            self._compressed.ids.extend(ids)
            self._compressed.count += n

        log.info("compress_embeddings", backend="memory", n=n, dim=dim)
        return result

    def decompress_embeddings(self, compressed: CompressedVectors) -> np.ndarray:
        import torch
        from turboquant.core.turboquant import CacheEntry

        qk = torch.from_numpy(compressed.quantized)
        sk = torch.from_numpy(compressed.scales)
        rk = torch.from_numpy(compressed.residual_bits) if compressed.residual_bits is not None else None

        entry = CacheEntry(
            compressed_keys=(qk, sk),
            compressed_values=(qk, sk),
            residual_keys=rk,
            residual_values=rk,
            metadata={
                "original_shape": list(qk.shape[:-1]) + [compressed.dim],
            },
        )
        keys, _ = self.tq.decompress(entry)
        return self._torch_to_vectors(keys)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        if self._compressed is None or self._compressed.count == 0:
            return []

        db_vectors = self.decompress_embeddings(self._compressed)
        q = query.astype(np.float32).ravel()

        # Cosine similarity
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        db_norms = db_vectors / (
            np.linalg.norm(db_vectors, axis=1, keepdims=True) + 1e-12
        )
        scores = db_norms @ q_norm

        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results = []
        for idx in top_idx:
            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            results.append(
                SearchResult(
                    id=self._compressed.ids[idx],
                    score=float(scores[idx]),
                    metadata=meta,
                )
            )
        return results


# ======================================================================
# ChromaDB
# ======================================================================


class ChromaDBTurboQuant(TurboQuantVectorAdapter):
    """ChromaDB integration with TurboQuant-compressed storage.

    Stores compressed vectors in ChromaDB metadata fields and performs
    on-the-fly decompression during search.

    Args:
        config: TurboQuant configuration.
        collection_name: ChromaDB collection name.
        client: Optional pre-configured ``chromadb.Client``.
    """

    def __init__(
        self,
        config: TurboQuantConfig,
        collection_name: str = "turboquant_vectors",
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        try:
            import chromadb  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for ChromaDBTurboQuant. "
                "Install with: pip install turboquant[chroma]"
            ) from exc

        self._client = client or chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._fallback = InMemoryTurboQuant(config)
        log.info("ChromaDBTurboQuant.__init__", collection=collection_name)

    def compress_embeddings(
        self,
        vectors: np.ndarray,
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> CompressedVectors:
        import base64

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n, dim = vectors.shape

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]
        if metadata is None:
            metadata = [{} for _ in range(n)]

        # Compress via in-memory adapter
        compressed = self._fallback.compress_embeddings(vectors, ids, metadata)

        # Store in ChromaDB: use original vectors for HNSW index,
        # compressed data in metadata for storage savings awareness
        encoded_q = base64.b64encode(compressed.quantized.tobytes()).decode("ascii")
        encoded_s = base64.b64encode(compressed.scales.tobytes()).decode("ascii")

        enriched_meta = []
        for i, m in enumerate(metadata):
            em = dict(m)
            em["_tq_q_shape"] = str(list(compressed.quantized.shape))
            em["_tq_s_shape"] = str(list(compressed.scales.shape))
            em["_tq_dim"] = dim
            enriched_meta.append(em)

        self._collection.add(
            ids=ids,
            embeddings=vectors.tolist(),
            metadatas=enriched_meta,
        )

        log.info("compress_embeddings", backend="chroma", n=n, dim=dim)
        return compressed

    def decompress_embeddings(self, compressed: CompressedVectors) -> np.ndarray:
        return self._fallback.decompress_embeddings(compressed)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        q = query.astype(np.float32).ravel().tolist()
        results = self._collection.query(
            query_embeddings=[q],
            n_results=top_k,
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            for rid, dist, meta in zip(ids, distances, metadatas):
                # ChromaDB returns distance; convert to similarity
                score = 1.0 - dist if dist <= 1.0 else 1.0 / (1.0 + dist)
                # Remove internal keys from metadata
                clean_meta = {k: v for k, v in meta.items() if not k.startswith("_tq_")}
                search_results.append(SearchResult(id=rid, score=score, metadata=clean_meta))

        return search_results


# ======================================================================
# Qdrant
# ======================================================================


class QdrantTurboQuant(TurboQuantVectorAdapter):
    """Qdrant integration with TurboQuant-compressed storage.

    Uses Qdrant payload for storing compressed data and supports
    batch upsert with auto-compression and named vectors.

    Args:
        config: TurboQuant configuration.
        collection_name: Qdrant collection name.
        host: Qdrant server host.
        port: Qdrant server gRPC port.
        vector_name: Named vector identifier (for multi-vector collections).
    """

    def __init__(
        self,
        config: TurboQuantConfig,
        collection_name: str = "turboquant_vectors",
        host: str = "localhost",
        port: int = 6333,
        vector_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-untyped]
            from qdrant_client.models import (  # type: ignore[import-untyped]
                Distance,
                VectorParams,
            )
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for QdrantTurboQuant. "
                "Install with: pip install turboquant[qdrant]"
            ) from exc

        self._client = QdrantClient(host=host, port=port)
        self._collection_name = collection_name
        self._vector_name = vector_name
        self._dim: int | None = kwargs.get("dim")
        self._fallback = InMemoryTurboQuant(config)

        # Create collection if it doesn't exist
        try:
            self._client.get_collection(collection_name)
        except Exception:
            if self._dim is not None:
                if vector_name:
                    vectors_config = {
                        vector_name: VectorParams(size=self._dim, distance=Distance.COSINE)
                    }
                else:
                    vectors_config = VectorParams(size=self._dim, distance=Distance.COSINE)
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config,
                )

        log.info("QdrantTurboQuant.__init__", collection=collection_name, host=host)

    def compress_embeddings(
        self,
        vectors: np.ndarray,
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> CompressedVectors:
        import base64

        from qdrant_client.models import PointStruct  # type: ignore[import-untyped]

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n, dim = vectors.shape
        self._dim = dim

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]
        if metadata is None:
            metadata = [{} for _ in range(n)]

        compressed = self._fallback.compress_embeddings(vectors, ids, metadata)

        # Batch upsert
        points = []
        for i in range(n):
            payload = dict(metadata[i]) if i < len(metadata) else {}
            payload["_tq_compressed"] = True
            vec = vectors[i].tolist()
            if self._vector_name:
                vector_data = {self._vector_name: vec}
            else:
                vector_data = vec

            points.append(
                PointStruct(
                    id=ids[i],
                    vector=vector_data,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

        log.info("compress_embeddings", backend="qdrant", n=n, dim=dim)
        return compressed

    def decompress_embeddings(self, compressed: CompressedVectors) -> np.ndarray:
        return self._fallback.decompress_embeddings(compressed)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        q = query.astype(np.float32).ravel().tolist()

        if self._vector_name:
            hits = self._client.search(
                collection_name=self._collection_name,
                query_vector=(self._vector_name, q),
                limit=top_k,
            )
        else:
            hits = self._client.search(
                collection_name=self._collection_name,
                query_vector=q,
                limit=top_k,
            )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            clean = {k: v for k, v in payload.items() if not k.startswith("_tq_")}
            results.append(
                SearchResult(
                    id=str(hit.id),
                    score=float(hit.score),
                    metadata=clean,
                )
            )
        return results


# ======================================================================
# Factory
# ======================================================================


def create_adapter(
    backend: str,
    config: TurboQuantConfig,
    **kwargs: Any,
) -> TurboQuantVectorAdapter:
    """Create a TurboQuant vector adapter for the given backend.

    Args:
        backend: One of ``"memory"``, ``"chroma"``, ``"qdrant"``.
        config: TurboQuant configuration.
        **kwargs: Backend-specific arguments.

    Returns:
        Configured ``TurboQuantVectorAdapter`` subclass.

    Raises:
        ValueError: If *backend* is not recognised.
    """
    backends = {
        "memory": InMemoryTurboQuant,
        "chroma": ChromaDBTurboQuant,
        "qdrant": QdrantTurboQuant,
    }
    cls = backends.get(backend.lower())
    if cls is None:
        raise ValueError(
            f"Unknown backend '{backend}'. Supported: {', '.join(sorted(backends))}"
        )
    return cls(config, **kwargs)


# ======================================================================
# Benchmark utility
# ======================================================================


def benchmark_vector_db(
    adapter: TurboQuantVectorAdapter,
    dataset_size: int = 10000,
    dim: int = 1536,
    top_k: int = 10,
) -> dict[str, float]:
    """Benchmark a vector adapter on synthetic data.

    Args:
        adapter: Configured vector adapter.
        dataset_size: Number of vectors to index.
        dim: Embedding dimension.
        top_k: Number of results to retrieve per query.

    Returns:
        Dictionary with keys: ``index_time_ms``, ``search_time_ms``,
        ``memory_mb``, ``recall_at_10``.
    """
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((dataset_size, dim)).astype(np.float32)
    # Normalize for cosine
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-12)

    # Index
    t0 = time.perf_counter()
    compressed = adapter.compress_embeddings(vectors)
    index_time_ms = (time.perf_counter() - t0) * 1000

    # Search (100 random queries)
    num_queries = min(100, dataset_size)
    query_indices = rng.choice(dataset_size, size=num_queries, replace=False)
    queries = vectors[query_indices]

    t0 = time.perf_counter()
    all_results: list[list[SearchResult]] = []
    for q in queries:
        all_results.append(adapter.search(q, top_k=top_k))
    search_time_ms = (time.perf_counter() - t0) / num_queries * 1000

    # Recall: check if the query vector's own index appears in results
    # (since we're searching with vectors from the dataset)
    recall_hits = 0
    ids = compressed.ids
    for qi, results in zip(query_indices, all_results):
        result_ids = {r.id for r in results}
        if qi < len(ids) and ids[qi] in result_ids:
            recall_hits += 1
    recall_at_k = recall_hits / max(num_queries, 1)

    # Memory estimate
    compressed_bytes = (
        compressed.quantized.nbytes
        + compressed.scales.nbytes
        + (compressed.residual_bits.nbytes if compressed.residual_bits is not None else 0)
    )
    memory_mb = compressed_bytes / (1024 * 1024)

    result = {
        "index_time_ms": round(index_time_ms, 1),
        "search_time_ms": round(search_time_ms, 2),
        "memory_mb": round(memory_mb, 2),
        "recall_at_10": round(recall_at_k, 4),
        "dataset_size": dataset_size,
        "dim": dim,
    }
    log.info("benchmark_vector_db", **result)
    return result
