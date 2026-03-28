# TurboQuant-MoE

[![PyPI](https://img.shields.io/pypi/v/turboquant-moe.svg)](https://pypi.org/project/turboquant-moe/) [![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE) [![CI](https://img.shields.io/github/actions/workflow/status/RemizovDenis/turboquant/ci.yml)](https://github.com/RemizovDenis/turboquant/actions) [![arXiv](https://img.shields.io/badge/arXiv-2504.19874-b31b1b.svg)](https://arxiv.org/abs/2504.19874) [![Discord](https://img.shields.io/badge/discord-community-5865F2.svg)](https://discord.gg/)

> Production implementation of Google DeepMind's TurboQuant algorithm with dynamic MoE expert caching.
> Includes extension foundation for game-theoretic routing, speculative prefetch, dynamic VRAM control, semantic KV eviction, cross-layer sharing, and adaptive bitwidth quantization.

## Why TurboQuant-MoE?

Long-context inference and MoE serving are memory-bound: KV cache grows with sequence length, and MoE layers keep many expert weights resident even when only top-k experts are active each step. TurboQuant-MoE combines packed low-bit KV storage, adaptive per-token bitwidth control, dynamic expert offloading, and speculative prefetch.

Latest full benchmark snapshot (`results/benchmark_20260328_072610.json`, CPU fallback):

- KV compression: **8.52x average** (`seq_len=1k..16k`)
- Needle-in-haystack quality: **100% recall@1** (`1k..128k`)
- Retrieval degradation: **0.0%** on tested slices
- MoE cache hit rate: **88.44%**
- Prefetch readiness: **88.75%**
- Hidden IO ratio: **100%**
- Expert cache GPU memory saved: **5.34 GB**
- Predictor latency: **0.106 ms mean** (`p99 = 0.121 ms`)
- Latency stability (MoE step): **p99 = 42.19 ms**

Included extensions:

- `GameTheoreticRouter` (Nash-style routing)
- `MarkovTrajectoryPredictor` (speculative prefetch)
- `VRAM_PID_Controller` (dynamic GPU cache sizing)
- `SemanticKVEviction` (importance-aware token retention)
- `CrossLayerKVCache` (anchor + delta sharing)
- `AdaptiveBitwidthQuantizer` (dynamic bit assignment)

## Benchmarks

Numbers below come from `results/benchmark_20260328_072610.json` and `results/README_benchmark.md`.

### Memory (Mixtral-8x7B harness, CPU fallback)

| Method | KV MB @16k | Ratio | Compression |
|---|---:|---:|---:|
| FP16 baseline | 256.0 | 1.0000 | 1.00x |
| TurboQuant classic 3-bit path | 62.0 | 0.2422 | 4.13x |
| TurboQuant adaptive path | 30.1 | 0.1176 | 8.50x |

Measured adaptive KV points:

| Seq Len | FP16 MB | Adaptive KV MB | Ratio | Compression |
|---:|---:|---:|---:|
| 1024 | 16.0 | 1.859 | 0.1162 | 8.61x |
| 4096 | 64.0 | 7.564 | 0.1182 | 8.46x |
| 16384 | 256.0 | 30.118 | 0.1176 | 8.50x |

### Expert Cache Performance

| gpu_cache_size | Hit Rate | Prefetch Readiness | Avg Load (ms) | GPU Saved (GB) | Hidden IO |
|---:|---:|---:|---:|
| 28 | 0.884 | 0.887 | 3.30 | 5.344 | 100% |

Source: `results/benchmark_20260328_072610.json` (`moe_expert` suite).

### Inference Speed

| Seq Len | Decode Latency (ms) | Throughput (tokens/sec) | IO-Bound Speedup (x) |
|---:|---:|---:|---:|
| 1024 | 75.79 | 13.5k | 8.56x |
| 4096 | 365.95 | 11.2k | 8.47x |
| 16384 | 1710.55 | 9.6k | 8.49x |

Predictor metrics from the same run: rolling accuracy `0.92`, precision@k `0.951`, recall@k `0.97`, mean latency `0.106 ms`, p99 `0.121 ms`, memory overhead `1.055 MB`.

## Quick Start

```bash
pip install turboquant-moe[transformers]
```

```python
from transformers import AutoModelForCausalLM
from turboquant.integrations.transformers import patch_moe_model, auto_config

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    torch_dtype="auto",
    device_map="auto",
)
model = patch_moe_model(model, auto_config(model))
```

## Installation

```bash
pip install turboquant-moe                          # core
pip install turboquant-moe[transformers]            # HuggingFace
pip install turboquant-moe[vllm]                    # vLLM production
pip install turboquant-moe[benchmark]               # benchmarks
pip install turboquant-moe[all]                     # everything
```

## Usage

### With HuggingFace Transformers

```python
from turboquant.integrations.transformers import auto_config, patch_moe_model

cfg = auto_config(model, bits=3, gpu_cache_experts=4)
model = patch_moe_model(model, cfg)
```

### With vLLM (production servers)

```python
from turboquant.integrations.vllm import create_turboquant_llm
from turboquant.core.turboquant_moe import TurboQuantMoEConfig

cfg = TurboQuantMoEConfig.from_pretrained_config(type("C", (), {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "model_type": "mixtral",
})())

llm = create_turboquant_llm(
    model="mistralai/Mixtral-8x7B-v0.1",
    tq_moe_config=cfg,
    max_model_len=32768,
)
```

### With Ollama (VPS / local)

```bash
docker compose up -d
```

Proxy endpoints:
- `GET /tq/status`
- `GET /tq/metrics`
- `GET /tq/experts`
- `POST /tq/warmup`
- `GET /health`

### Expert Cache Tuning

- `gpu_cache_size`: start with `top_k * 2` to keep active + next-step candidates.
- `prefetch_depth`: `1-2` for low-latency online serving; increase for stable repetitive traffic.
- `prefetch_threshold`: raise if GPU churn is high; lower if miss spikes dominate latency.

## How It Works

KV-cache quantization applies an orthogonal transform before quantization to distribute error more evenly across dimensions, then packs values into compact representation with optional residual correction.

Expert caching keeps active experts on GPU and stores inactive experts on CPU, optionally compressed. Eviction policy (ARC/LRU/LFU) decides what leaves GPU when cache is full.

Expert prediction estimates which experts will be needed on upcoming steps and prefetches them in background. Correct predictions hide transfer latency behind compute.

## Supported Models

| Model | KV-Quant | Expert Cache | Expert Prediction | Status |
|---|---|---|---|---|
| Mixtral-8x7B | Yes | Yes | Yes | Implemented |
| Mixtral-8x22B | Yes | Yes | Yes | Experimental |
| DeepSeek-V2 | Yes | Yes | Yes | Experimental |
| DeepSeek-V3 | Yes | Yes | Yes | Experimental |
| Qwen1.5-MoE | Yes | Yes | Yes | Experimental |
| OLMoE | Yes | Yes | Yes | Experimental |
| Arctic | Yes | Yes | Yes | Experimental |
| Llama-3 | Yes | No | No | KV-only |

## Architecture

```text
Input tokens
  -> Attention KV creation
  -> TurboQuantKVCache.compress (3-bit + scales + optional residual)
  -> MoE router logits
  -> MoERouterOptimizer (prune, top-k, capacity)
  -> ExpertPredictor (optional, async prefetch targets)
  -> DynamicExpertCache (GPU hit / CPU load / evict)
  -> Layer output
```

## Contributing

1. Fork and create a feature branch.
2. Install dev dependencies: `pip install -e ".[dev]"`.
3. Run checks: `ruff check turboquant/`, `mypy turboquant/ --strict`, `pytest tests/ -v`.
4. Open PR with benchmark delta when performance-related.

## License

MIT. See [LICENSE](./LICENSE).

## Citation

```bibtex
@software{turboquant_moe_2026,
  author = {Remizov, Denis},
  title = {TurboQuant-MoE: Production KV-Cache Quantization with Dynamic Expert Caching},
  year = {2026},
  url = {https://github.com/RemizovDenis/turboquant-moe},
}
```

Based on:

```bibtex
@article{turboquant2025,
  title={TurboQuant},
  author={Google DeepMind},
  journal={arXiv:2504.19874},
  year={2025}
}
```
