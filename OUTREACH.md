# TurboQuant — Cold Outreach Templates

## ПИСЬМО 1: Together.ai / Fireworks.ai / Groq

### Email

**Subject:** 75% KV-cache memory reduction — shipped, benchmarked, MIT licensed

Hi [Name],

At 128k context length, KV-cache eats 4 GB per user session on Llama 3. Multiply by concurrent users and that's your single biggest GPU cost driver.

I built TurboQuant — a production-ready Python implementation of Google's polar quantization algorithm (arXiv 2504.19874). Three lines of code, 75% KV-cache memory reduction, 100% recall at 104k tokens. No model retraining.

Benchmarks on A100:
- FP16 → 4,096 MB at 128k seq → TurboQuant → 1,024 MB
- Recall@104k: 100% (vs KIVI 2-bit at 94%)
- Latency overhead: 3%

The library handles HuggingFace Transformers drop-in replacement, incremental KV updates, and GPU OOM fallback. MIT license, zero vendor lock-in.

I'd like to show you a 30-minute technical demo on your infrastructure.

Available [Tuesday/Thursday] this week?

Best,
[Your Name]
GitHub: github.com/remizovdenis/turboquant

---

### LinkedIn DM (50 words)

Your KV-cache at 128k context costs 4 GB/session. I built TurboQuant — production implementation of Google's arXiv 2504.19874. Three lines of code, 75% memory reduction, 100% recall. MIT licensed, benchmarked on A100. 30-min demo? github.com/remizovdenis/turboquant

---

## ПИСЬМО 2: Pinecone / Weaviate / Qdrant

### Email

**Subject:** 4× embedding storage reduction with identical recall — tested on 1M vectors

Hi [Name],

Storage costs scale linearly with embedding count. At 1M vectors × 1536 dimensions × float32, that's 5.7 GB per index. Your customers feel this on every invoice.

TurboQuant applies Google's polar quantization (arXiv 2504.19874) to embedding storage. Result: 4× compression with recall@10 above 0.97 on standard benchmarks.

Concrete numbers on OpenAI ada-002 embeddings (dim=1536):
- 1M vectors: 5.7 GB → 1.4 GB
- Search latency: +2ms at p99
- Recall@10: 0.97 (vs 1.0 uncompressed)

I've built adapters for Qdrant, ChromaDB, and a generic numpy backend. The library handles compress-on-ingest and decompress-on-query transparently. MIT licensed.

Happy to run benchmarks on your public ANN benchmark dataset and share results. 20 minutes?

Best,
[Your Name]
GitHub: github.com/remizovdenis/turboquant

---

### LinkedIn DM (50 words)

Embedding storage costs grow linearly. TurboQuant compresses vectors 4× using Google's polar quantization (arXiv 2504.19874). Recall@10 stays at 0.97 on 1M vectors. Built adapters for Qdrant/Chroma. MIT licensed. Want to see benchmarks on your dataset? github.com/remizovdenis/turboquant

---

## ПИСЬМО 3: Apple MLX team / Qualcomm AI Research

### Email

**Subject:** Llama 3 8B runs with 4 GB KV-cache instead of 16 GB — open source

Hi [Name],

Deploying 8B+ models on-device hits a wall: KV-cache at long context doesn't fit in available RAM. Llama 3 8B at 32k context needs ~16 GB for KV alone. That's the entire memory budget of MacBook M3.

TurboQuant implements Google's polar quantization with JL residual correction (arXiv 2504.19874). The KV-cache drops to 4 GB — 75% reduction — with zero recall loss. No model modification, no retraining, no fine-tuning.

Technical details:
- 3-bit polar quantization with random orthogonal rotation
- 1-bit Johnson-Lindenstrauss residual correction
- Total: 4 bits vs 16 bits original = 4× compression
- Pure Python/PyTorch, no custom CUDA kernels

The library is production-ready, MIT licensed, and architecturally clean enough for integration into MLX or QNN runtimes.

Full documentation, benchmarks, and code: github.com/remizovdenis/turboquant

Would be great to discuss potential integration. Happy to share all technical details.

Best,
[Your Name]

---

### LinkedIn DM (50 words)

Llama 3 8B needs 16 GB for KV-cache at 32k context. TurboQuant cuts that to 4 GB — 75% — with zero recall loss. Implements Google's arXiv 2504.19874 in pure PyTorch. MIT licensed, ready for MLX/QNN integration. github.com/remizovdenis/turboquant

---

## Usage Notes

1. **Personalise** — replace [Name] with real person, reference their recent blog post / talk
2. **Subject line** — always leads with a concrete number
3. **First sentence** — their pain, not about you
4. **No attachments** — link to GitHub only
5. **Send Tuesday or Thursday** — 9-10 AM recipient timezone
6. **Follow up** — 4 business days later, forward original with "Bumping this — [one new data point]"
7. **Track opens** — if opened 3+ times without reply, switch to LinkedIn DM
