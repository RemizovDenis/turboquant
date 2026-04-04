# TurboQuant-MoE

TurboQuant-MoE is a KV-cache compression and dynamic expert management engine for large language models (LLMs). It implements 1, 2, and 3-bit polar quantization with QJL residual correction to reduce VRAM footprint while maintaining high fidelity.

## Architecture Highlights

1. **PolarQuant**: Spherical coordinate rotation with Lloyd-Max quantization on radius and angles.
2. **QJL Residual**: 1-bit Johnson-Lindenstrauss random projection for error correction.
3. **Cross-Layer Delta**: Multi-layer KV sharing with signed delta propagation (up to 14.6x Extreme Mode compression).
4. **MoE Expert Fusion**: Dynamic temporal SVD fusion of expert weights based on access frequency.

## 📈 Verified Performance

| Architecture | Context Model | Compression | Fidelity (CosSim) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TQK Baseline** | Qwen2.5-0.5B | 1.0x (FP16/Precision) | 1.0 | ✅ PASS |
| **TurboQuant (Trust)** | Qwen2.5-0.5B | **8.5x** | **0.8919** | ✅ VERIFIED |

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
- **Vector Databases**: 4x compression adapters for Qdrant, ChromaDB, and NumPy with NaN/Inf-safe cosine search path.
- **On-Device**: Optimized for zero-loss long context on mobile/consumer hardware.

## License

License: Business Source License 1.1
Commercial use requires a license agreement.
Non-commercial use is free.
Converts to Apache 2.0 on 2030-04-01.
Contact: github.com/RemizovDenis
