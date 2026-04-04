# TurboQuant Benchmarks

This document consolidates performance metrics for TurboQuant-MoE v0.3.x across various architectures.

## Cumulative Metrics (A100 80GB)

| Metric | Score | Threshold | Status |
| :--- | :--- | :--- | :--- |
| Max Compression (Cross-Layer) | 14.63x | > 10.0x | Verified |
| KV Base Compression (3-bit) | 4.41x | > 4.0x | Verified |
| Expert Fusion VRAM Savings | 50.0% | > 30.0% | Verified |
| Quality (Cosine Similarity) | 0.885 | > 0.85 | Verified |

## Compression Performance (PolarQuant 3-bit)

| Seq Len | FP16 (MB) | Compressed (MB) | Ratio | CosSim |
| :--- | :--- | :--- | :--- | :--- |
| 1024 | 16.0 | 3.63 | 4.41x | 0.885 |
| 4096 | 64.0 | 14.5 | 4.41x | 0.885 |

## Apple Silicon Latency Projections (M4 Air / Ollama Proxy)

Results are based on initial proxy-benchmarking for v0.3.1-next.

### Reproducible local run (no cloud)

```bash
./scripts/run_local_field_suite.sh
```

For a stricter local matrix with Mixtral:

```bash
PROFILE=real MODELS="mixtral:latest mistral:latest llama3.1:latest" ./scripts/run_local_field_suite.sh
```

The command generates:

- `results/field_local/<timestamp>/benchmark_ultimate_m4.json`
- `results/field_local/<timestamp>/README_benchmark.md`

### mistral:latest (7B)
- **Baseline Average**: 13.69 tokens/s
- **TurboQuant Proxy**: 5.64 tokens/s (current overhead 2.4x)
- **RSS Delta**: -21.2 MB improvement

### llama3.1:latest (8B)
- **Baseline Average**: 4.96 tokens/s
- **TurboQuant Proxy**: 6.01 tokens/s (1.2x speedup)
- **RSS Delta**: 1601.7 MB VRAM reclaim

## Cross-Layer Delta Engine Details

- **Anchor Stride**: 4 Layers
- **Anchor Layer**: 3-bit PolarQuant
- **Delta Layer**: 1-bit Signed Delta
- **Aggregate Multi-Layer Compression**: 14.63x
- **Similarity Index**: 0.999

## MoE Expert Fusion (Temporal SVD)

- **Input experts**: 4
- **Fuseable**: 4
- **FP16 Weights**: 896.0 MB
- **Fused Weights**: 448.0 MB
- **Memory Reclaim**: 50%
