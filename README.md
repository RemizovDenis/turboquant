# TurboQuant-MoE

[![CI](https://github.com/RemizovDenis/turboquant/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/RemizovDenis/turboquant/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/RemizovDenis/turboquant)](https://github.com/RemizovDenis/turboquant/releases)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-orange.svg)](./LICENSE)

TurboQuant-MoE is a KV-cache compression and dynamic MoE expert management engine for LLM inference.

## Why it exists

Large-context and MoE inference is usually constrained by VRAM and memory bandwidth. TurboQuant targets this bottleneck with:

- 1/2/3-bit Polar quantization for KV tensors
- QJL residual correction for fidelity preservation
- Cross-layer KV sharing and delta-based compression
- MoE expert cache and prefetch primitives

## Installation

The project is currently distributed from source.

```bash
git clone https://github.com/RemizovDenis/turboquant.git
cd turboquant
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,transformers,benchmark]"
```

## Quick start

```python
from turboquant.core.turboquant import TurboQuantKVCache, TurboQuantConfig

config = TurboQuantConfig(
    head_dim=128,
    num_heads=32,
    bits=3,
    residual_correction=True,
)
cache = TurboQuantKVCache(config)

compressed = cache.compress(keys=key_tensor, values=val_tensor)
recon_k, recon_v = cache.decompress(compressed)
```

## Validate locally

```bash
ruff check turboquant tests
ruff format --check turboquant tests
mypy turboquant --strict
pytest tests/ -v --tb=short -x -k "not gpu and not cuda and not triton"
```

## Documentation

- [Benchmarks](./docs/benchmarks.md)
- [Applications](./docs/APPLICATIONS.md)
- [Licensing](./docs/licensing.md)
- [Trust report](./TRUST_REPORT.md)

## Integrations

- HuggingFace Transformers (`TurboQuantCache`)
- Vector databases (Qdrant, ChromaDB, NumPy adapter)
- Ollama/vLLM integration helpers

## Project standards

- [Contributing guide](./CONTRIBUTING.md)
- [Security policy](./SECURITY.md)
- [Code of conduct](./CODE_OF_CONDUCT.md)
- [Support](./SUPPORT.md)

## License

Business Source License 1.1 (BUSL-1.1).
Commercial use requires a commercial license.
Converts to Apache-2.0 on 2030-04-01.
