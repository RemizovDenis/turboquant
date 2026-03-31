# Outreach Strategy

This document contains outreach templates and strategy for TurboQuant-MoE partnerships.

## Partnership Targets (Inference Providers)

### Messaging for Together.ai / Fireworks.ai / Groq

At 128k context length, KV-cache consumption is the primary GPU cost driver. TurboQuant-MoE provides a production-ready implementation of polar quantization (arXiv 2504.19874) achieving 75% memory reduction with 100% recall at 104k tokens on Mistral-7B.

### Performance Benchmarks (A100)
- **FP 16 (128k seq)**: 4,096 MB
- **TurboQuant (128k seq)**: 1,024 MB
- **Recall@104k**: 100% 
- **Latency overhead**: 3%

The library provides zero vendor lock-in with MIT licensing.

## Vector Search Integrations (Pinecone / Weaviate / Qdrant)

Applying polar quantization to embedding storage achieves 4x compression with recall@10 above 0.97 on million-vector workloads (OpenAI ada-002).

- **1M vectors (float32)**: 5.7 GB → 1.4 GB
- **Search latency**: +2ms overhead at p99
- **Integration**: Adatpters for Qdrant, ChromaDB, and Generic NumPy backends provided.

## On-Device Platforms (Apple MLX / Qualcomm QNN)

TurboQuant-MoE enables 8B+ models to run with 4 GB KV-cache instead of 16 GB at long context, fitting within MacBook RAM budgets without fine-tuning.
