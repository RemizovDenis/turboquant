# TurboQuant Ultimate Benchmark (M4 Air + Ollama)

Generated: `2026-03-28T19:59:57`

## mistral:latest

| Mode | p50 latency (ms) | p95 latency (ms) | Avg TTFT (ms) | Avg tokens/s | RSS delta (MB) |
|---|---:|---:|---:|---:|---:|
| Baseline | 43057.6 | 47379.9 | 33447.3 | 13.69 | -40.7 |
| TurboQuant Proxy | 84749.1 | 86067.4 | 61997.9 | 5.64 | -21.2 |

- Speedup (latency avg): `0.508x`
- Memory delta improvement: `-19.5 MB`
- KV compression ratio (proxy monitor): `1.000x`

## llama3.1:latest

| Mode | p50 latency (ms) | p95 latency (ms) | Avg TTFT (ms) | Avg tokens/s | RSS delta (MB) |
|---|---:|---:|---:|---:|---:|
| Baseline | 98287.1 | 106121.9 | 72431.9 | 4.96 | 1598.0 |
| TurboQuant Proxy | 80208.9 | 80527.9 | 58865.4 | 6.01 | -3.7 |

- Speedup (latency avg): `1.225x`
- Memory delta improvement: `1601.7 MB`
- KV compression ratio (proxy monitor): `1.000x`
