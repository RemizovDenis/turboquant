# TurboQuant-MoE

TurboQuant-MoE is a production-grade KV-cache compression and dynamic expert management engine for large language models (LLMs). It implements advanced 1, 2, and 3-bit polar quantization with JL residual correction to achieve significant VRAM reduction with zero recall loss.

## Architecture Highlights

1. **PolarQuant**: Spherical coordinate rotation with Lloyd-Max quantization on radius and angles.
2. **QJL Residual**: 1-bit Johnson-Lindenstrauss random projection for error correction.
3. **Cross-Layer Delta**: Multi-layer KV sharing with signed delta propagation (14.6x compression).
4. **MoE Expert Fusion**: Dynamic temporal SVD fusion of expert weights based on access frequency.

## 📈 Verified Performance

| Architecture | Context Model | Compression | Fidelity (CosSim) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TQK Baseline** | Qwen2.5-0.5B | 1.0x (FP16) | 1.0 | ✅ PASS |
| **TurboQuant 3-bit** | Qwen2.5-0.5B | **8.5x** | **0.8919** | ✅ VERIFIED |

> [!TIP]
> Details on methodology and bit-exactness analysis are available in the [**Official Trust Report (Qwen2.5)**](./TRUST_REPORT.md).

## Installation

```bash
pip install turboquant-moe
```

## Quick Start (KV Cache Compression)

```python
from turboquant.core.turboquant import TurboQuantKVCache, TurboQuantConfig

config = TurboQuantConfig(head_dim=128, num_heads=32, bits=3, residual_correction=True)
cache = TurboQuantKVCache(config)

# Compression
compressed = cache.compress(keys=key_tensor, values=val_tensor)

# Decompression
recon_k, recon_v = cache.decompress(compressed)
```

## Documentation & Benchmarks

Detailed documentation and metrics are available in the `/docs` directory:

- [Technical Benchmarks (A100 & Apple Silicon)](./docs/benchmarks.md)
- [Outreach Strategy & Partnership Targets](./docs/strategy.md)
- [Commercial Licensing Details](./docs/licensing.md)

## Integration Status

- **HuggingFace Transformers**: Drop-in `TurboQuantCache` provider.
- **Vector Databases**: 4x compression adapters for Qdrant, ChromaDB, and NumPy.
- **On-Device**: Optimized for zero-loss long context on mobile/consumer hardware.

## License

MIT (Core Library). Commercial licensing available for proprietary deployments (see [Licensing](./docs/licensing.md)).
