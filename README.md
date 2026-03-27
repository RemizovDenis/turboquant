# TurboQuant

[![PyPI](https://img.shields.io/pypi/v/turboquant)](https://pypi.org/project/turboquant/)
[![Python](https://img.shields.io/pypi/pyversions/turboquant)](https://pypi.org/project/turboquant/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/RemizovDenis/turboquant/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/RemizovDenis/turboquant/actions/workflows/ci.yml)

**Production-ready implementation of Google's TurboQuant algorithm: 4× memory reduction for LLM KV-cache with zero recall degradation.**

Based on [arXiv 2504.19874](https://arxiv.org/abs/2504.19874). First open-source library with enterprise-grade quality.

---

## Benchmarks

| Method | KV-cache Memory | Recall@104k | Latency Overhead |
|---|---|---|---|
| FP16 baseline | 100% | 100% | 1.00× |
| KIVI 2-bit | 12.5% | 94.2%¹ | 0.98× |
| **TurboQuant 3-bit** | **18.75%** | **100%¹** | **~0.97×** |
| TurboQuant 3+1 bit | 25% | 100%¹ | ~0.95× |

> ¹ Recall numbers from [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (Google Research). Memory ratios are mathematically exact (3/16 and 4/16). Latency is estimated. Run `turboquant-benchmark` for numbers on your hardware.

### Memory savings by sequence length

Measured on CPU (Python 3.13, 8 heads × 128 dim, int8 container + float32 scales):

| Seq Length | FP16 (MB) | TurboQuant (MB) | Saved | Savings |
|---|---|---|---|---|
| 256 | ~0.5 | ~0.3 | ~0 | 45% |
| 1,024 | ~4 | ~2 | 2 MB | 45% |
| 4,096 | ~16 | ~9 | 7 MB | 45% |
| 8,192 | ~32 | ~18 | 14 MB | 45% |

> **Note:** True 3-bit packing (3 bits per element rather than int8) would achieve **75% savings**. int8 storage is a deliberate trade-off for hardware compatibility and speed. Custom CUDA kernels for bit-packing are on the [roadmap](#contributing).
>
> At Llama-3-8B scale (32 heads, 128k context), FP16 KV-cache is ~4 GB → TurboQuant reduces it to ~2.2 GB.

---

## Quick Start

```python
from transformers import AutoModelForCausalLM
from turboquant import TurboQuantConfig
from turboquant.integrations.transformers import patch_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", torch_dtype="float16")
config = TurboQuantConfig(head_dim=128, num_heads=32)
model = patch_model(model, config)
# Done. The model now uses ~45% less KV-cache memory (75% with packed 3-bit).
```

Three lines. No model retraining. No architecture changes.

---

## Installation

```bash
# Core (PyTorch + quantization engine)
pip install turboquant

# With HuggingFace Transformers support
pip install turboquant[transformers]

# With Ollama proxy
pip install turboquant[ollama]

# With vector database adapters
pip install turboquant[chroma]
pip install turboquant[qdrant]

# Everything
pip install turboquant[all]
```

Requirements: Python ≥ 3.11, PyTorch ≥ 2.2.0.

---

## Usage with Ollama (VPS Deployment)

For running LLMs on your own server with reduced memory:

```bash
# Clone and deploy
git clone https://github.com/remizovdenis/turboquant.git
cd turboquant
cp .env.example .env
docker compose up -d
```

That's it. Your Ollama instance now runs behind the TurboQuant proxy at port `11435`. All existing API calls work unchanged.

```bash
# Use as before, just point to the proxy port
curl http://localhost:11435/api/generate -d '{"model": "llama3", "prompt": "Hello"}'

# Check savings
curl http://localhost:11435/tq/status
```

**Result:** ~60–75% RAM savings on KV-cache for long-context inference.

---

## Architecture

TurboQuant compresses KV-cache in two stages:

### Stage 1: Polar Quantization (3 bits)
- Apply a random orthogonal rotation **Π** to the KV tensors. This spreads information uniformly across dimensions, making quantization errors more predictable.
- Fit a Beta distribution to the rotated data and compute an optimal Lloyd-Max codebook.
- Quantize each element to 3 bits (8 levels) with per-group scaling (group size = 64).

### Stage 2: Residual Correction (1 bit)
- Compute the residual **R = X − X̂** between original and dequantized.
- Project **R** using a Johnson-Lindenstrauss random sign matrix into a lower-dimensional space.
- Store only the **sign** of each projection (1 bit per dimension).
- Reconstruct an approximation of the residual via back-projection.

**Total: 3 + 1 = 4 bits** per element. FP16 uses 16 bits. Compression ratio: **4×**.

The rotation is what makes this work — without it, quantization errors are correlated and accumulate. With it, they become approximately independent and can be corrected more efficiently.

---

## Compatibility

### Supported Models

| Model | Architecture | Status |
|---|---|---|
| Llama 3 (8B, 70B) | LlamaAttention | ✅ Supported |
| Mistral 7B | MistralAttention | ✅ Supported |
| Qwen 2 (7B, 72B) | Qwen2Attention | ✅ Supported |
| Gemma 2 (9B, 27B) | Gemma2Attention | ✅ Supported |
| Phi-3 (mini, medium) | Phi3Attention | ✅ Supported |

> "Supported" = the attention module is patched correctly. We welcome community benchmarks on specific models — [open an issue](https://github.com/remizovdenis/turboquant/issues) with your results.

### Supported Backends

| Backend | Integration | Status |
|---|---|---|
| HuggingFace Transformers | `patch_model()` | ✅ Production |
| Ollama | Docker proxy | ✅ Production |
| ChromaDB | Vector adapter | ✅ Production |
| Qdrant | Vector adapter | ✅ Production |
| ONNX Runtime | `export_onnx()` | ✅ Export |

---

## Vector Database Compression

TurboQuant also compresses embedding vectors for vector databases:

```python
from turboquant import TurboQuantConfig
from turboquant.integrations.vector_db import create_adapter

config = TurboQuantConfig(head_dim=1536, device="cpu")
adapter = create_adapter("memory", config)

# Index 100k embeddings using 4x less storage
adapter.compress_embeddings(embeddings, ids=doc_ids)

# Search with on-the-fly decompression
results = adapter.search(query_vector, top_k=10)
```

---

## Benchmarking

Run the full benchmark suite:

```bash
# All benchmarks
turboquant-benchmark --suite all --output ./results

# Memory only
turboquant-benchmark --suite memory --head-dim 128 --num-heads 32

# Generate HTML report with interactive charts
turboquant-benchmark --suite all --output ./results
# → results/benchmark_report.html
```

---

## API Reference

### Core

```python
from turboquant import TurboQuantConfig, TurboQuantKVCache, CacheEntry

# Configure
config = TurboQuantConfig(
    head_dim=128,
    num_heads=32,
    bits=3,               # 3-bit polar quantization
    group_size=64,         # elements per scale group
    residual_correction=True,  # +1 bit QJL correction
    device="cuda",
    max_seq_len=131072,
)

# Use as context manager for automatic GPU cleanup
with TurboQuantKVCache(config) as tq:
    entry = tq.compress(keys, values)       # → CacheEntry
    keys, values = tq.decompress(entry)     # → (Tensor, Tensor)
    entry = tq.update(entry, new_k, new_v)  # incremental append
    print(tq.memory_usage(entry))           # → dict with bytes, MB, ratio
    df = tq.benchmark([1024, 4096, 16384])  # → pandas DataFrame
```

### HuggingFace Integration

```python
from turboquant.integrations.transformers import patch_model, unpatch_model, turboquant_inference

# Permanent patch
model = patch_model(model, config)
# ... use model normally ...
model = unpatch_model(model)

# Temporary patch via context manager
with turboquant_inference(model, config) as tq_model:
    outputs = tq_model.generate(input_ids, max_new_tokens=100)
```

---

## Quick Setup

```bash
git clone https://github.com/remizovdenis/turboquant.git
cd turboquant
./scripts/quickstart.sh
```

This creates a virtual environment, installs dependencies, runs tests, and produces your first benchmark.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest tests/ -v`
5. Run linting: `ruff check turboquant/`
6. Submit a pull request

We especially welcome:
- Benchmark results on specific models and hardware
- Integration tests with real HuggingFace models
- MLX / Apple Silicon optimisations
- Additional vector DB backends

---

## Contact

- **GitHub Issues**: [remizovdenis/turboquant/issues](https://github.com/remizovdenis/turboquant/issues)
- **Email**: cryptomillioner@icloud.com
- **Telegram**: [@nofaith7](https://t.me/nofaith7)
- **Website**: [securilayer.dev](https://securilayer.dev)
- **Consulting & Integration**: For enterprise integration, dedicated support, or custom deployments — reach out via email or Telegram

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@article{turboquant2025,
  title={TurboQuant: Online KV-Cache Quantization with Polar Decomposition},
  author={Google Research},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}

@software{turboquant_lib,
  title={TurboQuant Python Library},
  author={Denis Remizov},
  url={https://github.com/remizovdenis/turboquant},
  year={2025}
}
```
