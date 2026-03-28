# TurboQuant-MoE

[![PyPI](https://img.shields.io/pypi/v/turboquant-moe.svg)](https://pypi.org/project/turboquant-moe/) [![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE) [![CI](https://img.shields.io/github/actions/workflow/status/RemizovDenis/turboquant/ci.yml)](https://github.com/RemizovDenis/turboquant/actions) [![arXiv](https://img.shields.io/badge/arXiv-2504.19874-b31b1b.svg)](https://arxiv.org/abs/2504.19874) [![Discord](https://img.shields.io/badge/discord-community-5865F2.svg)](https://discord.gg/)

> Production implementation of Google DeepMind's TurboQuant algorithm with dynamic MoE expert caching.
> Includes extension foundation for game-theoretic routing, speculative prefetch, dynamic VRAM control, semantic KV eviction, cross-layer sharing, and adaptive bitwidth quantization.

## Why TurboQuant-MoE?

Long-context inference and MoE serving are memory-bound: KV cache grows with sequence length, and MoE layers keep many expert weights resident even when only top-k experts are active each step. TurboQuant-MoE combines true packed 3-bit KV compression, residual correction, CPU expert offloading, and prefetching. In the full benchmark run on March 28, 2026 (`results/benchmark_20260328_034636.json`, CPU fallback), the project reached 4.1x KV compression (24.22% of FP16 KV memory), recall@1 = 1.0 on tested quality slices (1k/4k/16k), and 2.625 GB equivalent GPU memory savings from expert caching.

## Extension Foundation (Patch v0.1.1)

This branch also includes the extension foundation integrated into the pipeline:

- `GameTheoreticRouter` (Nash-style routing with capacity-aware selection)
- `MarkovTrajectoryPredictor` (speculative expert prefetch)
- `VRAM_PID_Controller` (dynamic GPU cache sizing)
- `SemanticKVEviction` (importance-based KV token retention)
- `CrossLayerKVCache` (anchor + delta KV sharing)
- `AdaptiveBitwidthQuantizer` (per-token dynamic bitwidth)

Latest local synthetic CPU snapshot (`results/benchmark_20260328_051700.json`):

- predictor rolling accuracy: `1.00`
- predictor mean latency: `0.097 ms`
- markov accuracy@k: `0.828`
- markov hidden IO ratio: `44.48%`

These extensions are actively tuned toward the production targets (higher expert-cache hit rate, higher hidden IO ratio, and GPU-side throughput gains on long contexts).

## Benchmarks

Numbers below come from local benchmark outputs in `results/benchmark_20260328_034636.json` and `results/README_benchmark.md`.

### Memory (Mixtral-8x7B harness, seq_len up to 16k, CPU fallback)

| Method | GPU RAM (MB) | CPU RAM (MB) | Recall@64k | Tokens/sec |
|---|---:|---:|---:|---:|
| FP16 baseline (seq_len=16384) | 256.0 | 0.0 | n/a | n/a |
| KIVI 2bit (reference class) | n/a | n/a | n/a | n/a |
| TurboQuant KV-only (3bit) | 62.0 | 0.0 | 1.00 (tested up to 16k) | 2044.11 |
| TurboQuant-MoE (KV + expert cache) | 62.0 + expert offload | CPU expert tier | 1.00 (tested up to 16k) | 2044.11 |

Measured KV memory points (local run):

| Seq Len | FP16 MB | TurboQuant 3bit MB | Ratio |
|---:|---:|---:|---:|
| 1024 | 16.0 | 3.875 | 0.2422 |
| 4096 | 64.0 | 15.5 | 0.2422 |
| 16384 | 256.0 | 62.0 | 0.2422 |

### Expert Cache Performance

| gpu_cache_size | Hit Rate | Avg Load (ms) | GPU Saved (GB) |
|---:|---:|---:|---:|
| 4 | 0.10 | 152.75 | 2.625 |

Source: `results/benchmark_20260328_034636.json` (`moe_expert` suite).

### Inference Speed

| Seq Len | Prefill Latency (ms) | Decode Latency (ms) | Throughput (tokens/sec) |
|---:|---:|---:|---:|
| 1024 | 186.41 | 111.83 | 5493.33 |
| 4096 | 831.67 | 483.93 | 4925.03 |
| 16384 | 8015.24 | 4255.05 | 2044.11 |

Predictor metrics from the same run: rolling accuracy `0.04`, mean latency `0.266 ms`, p99 latency `0.434 ms`, memory overhead `1.055 MB`.

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
