# TurboQuant-MoE Benchmark

Generated: `20260328_074849`

## memory
|   seq_len |   baseline_fp16_mb |   kv_only_mb |   adaptive_kv_mb |   moe_mb |   actual_compression_ratio |   classic_compression_ratio |   adaptive_compression_x |   compression_gain_vs_classic_x |   theoretical_vs_actual_gap |
|----------:|-------------------:|-------------:|-----------------:|---------:|---------------------------:|----------------------------:|-------------------------:|--------------------------------:|----------------------------:|
|      1024 |                 16 |        3.875 |          1.88672 |      3.5 |                   0.11792  |                    0.242188 |                  8.48033 |                         2.05383 |                   0.0546875 |
|      4096 |                 64 |       15.5   |          7.52441 |     14   |                   0.117569 |                    0.242188 |                  8.50565 |                         2.05996 |                   0.0546875 |
|     16384 |                256 |       62     |         30.1572  |     56   |                   0.117802 |                    0.242188 |                  8.48884 |                         2.05589 |                   0.0546875 |

## speed
|   seq_len |   prefill_latency_ms |   decode_latency_ms |   baseline_prefill_latency_ms |   baseline_decode_latency_ms |   throughput_tokens_per_sec |   baseline_decode_tokens_per_sec |   projected_io_bound_decode_tokens_per_sec |   io_bound_speedup_x |   observed_prefill_speedup_x |   observed_decode_speedup_x |
|----------:|---------------------:|--------------------:|------------------------------:|-----------------------------:|----------------------------:|---------------------------------:|-------------------------------------------:|---------------------:|-----------------------------:|----------------------------:|
|      1024 |              88.4403 |             78.7817 |                      0.650742 |                     0.591583 |                    12997.9  |                      1.73095e+06 |                                1.47477e+07 |              8.52002 |                   0.00735797 |                  0.00750915 |
|      4096 |             391.276  |            362.733  |                      2.65884  |                     2.62385  |                    11292.1  |                      1.56107e+06 |                                1.32676e+07 |              8.49903 |                   0.00679531 |                  0.00723355 |
|     16384 |            1849.5    |           1820.04   |                     15.253    |                    14.9651   |                     9002.02 |                      1.09482e+06 |                                9.32604e+06 |              8.51836 |                   0.00824712 |                  0.0082224  |

## quality
|   seq_len |   recall_at_1 |   needle_similarity_drop_percent |   retrieval_degradation_percent |
|----------:|--------------:|---------------------------------:|--------------------------------:|
|      1000 |             1 |                          2.34375 |                               0 |
|      4000 |             1 |                          2.34375 |                               0 |
|     16000 |             1 |                          2.34375 |                               0 |
|     32000 |             1 |                          2.34375 |                               0 |
|     64000 |             1 |                          2.34375 |                               0 |
|    128000 |             1 |                          2.34375 |                               0 |

## moe_expert
```json
{
  "hit_rate": 0.9375,
  "avg_expert_load_latency_ms": 4.55528684775345,
  "prefetch_accuracy": 0.9425,
  "cache_prefetch_precision": 0.0,
  "gpu_memory_saved_gb": 6.424072265625,
  "markov_accuracy_at_k": 0.9650872817955112,
  "markov_accuracy_at_1": 0.9650872817955112,
  "markov_io_hidden_ms": 34969.084116672784,
  "markov_hidden_io_percent": 100.0,
  "latency_p95_ms": 8.988612436223773,
  "latency_p99_ms": 23.4794745314866
}
```

## moe_router
```json
{
  "dropped_tokens": 0,
  "imbalance_ratio": 1.0007348485581695,
  "expert_load_mean": 1024.0,
  "nash_imbalance_ratio": 1.004950490375089,
  "nash_dropped_tokens": 0,
  "nash_convergence_rate": 0.0,
  "nash_avg_iterations": 3.0,
  "nash_overhead_ms": 18.12973329797387,
  "imbalance_improvement_x": 0.9958051248720262
}
```

## predictor
```json
{
  "rolling_accuracy": 0.96,
  "precision_at_k": 0.9733333333333333,
  "recall_at_k": 0.98,
  "latency_ms_mean": 0.09709041332826018,
  "latency_ms_p99": 0.12236026814207455,
  "memory_overhead_mb": 1.055450439453125
}
```

## vps
```json
{
  "ram_mb": 0.0,
  "time_to_first_token_ms": 29.874208848923445,
  "monthly_cost_usd": {
    "p3.2xlarge": 2203.2,
    "p4d.24xlarge": 23594.4
  },
  "estimated_savings_usd_month": {
    "p3.2xlarge": 1211.76,
    "p4d.24xlarge": 12976.920000000002
  }
}
```
