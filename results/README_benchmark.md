# TurboQuant-MoE Benchmark

Generated: `20260328_080540`

## memory
|   seq_len |   baseline_fp16_mb |   kv_only_mb |   adaptive_kv_mb |   moe_mb |   actual_compression_ratio |   classic_compression_ratio |   adaptive_compression_x |   compression_gain_vs_classic_x |   theoretical_vs_actual_gap |
|----------:|-------------------:|-------------:|-----------------:|---------:|---------------------------:|----------------------------:|-------------------------:|--------------------------------:|----------------------------:|
|      1024 |                 16 |        3.875 |          1.87012 |      3.5 |                   0.116882 |                    0.242188 |                  8.55561 |                         2.07206 |                   0.0546875 |
|      4096 |                 64 |       15.5   |          7.53027 |     14   |                   0.117661 |                    0.242188 |                  8.49903 |                         2.05836 |                   0.0546875 |
|     16384 |                256 |       62     |         29.96    |     56   |                   0.117031 |                    0.242188 |                  8.54474 |                         2.06943 |                   0.0546875 |

## speed
|   seq_len |   prefill_latency_ms |   decode_latency_ms |   baseline_prefill_latency_ms |   baseline_decode_latency_ms |   throughput_tokens_per_sec |   baseline_decode_tokens_per_sec |   projected_io_bound_decode_tokens_per_sec |   io_bound_speedup_x |   observed_prefill_speedup_x |   observed_decode_speedup_x |
|----------:|---------------------:|--------------------:|------------------------------:|-----------------------------:|----------------------------:|---------------------------------:|-------------------------------------------:|---------------------:|-----------------------------:|----------------------------:|
|      1024 |               87.636 |             78.6156 |                      0.671504 |                     0.577012 |                    13025.4  |                      1.77466e+06 |                                1.49491e+07 |              8.42365 |                   0.00766243 |                  0.00733967 |
|      4096 |              395.045 |            364.496  |                      2.74354  |                     2.67255  |                    11237.4  |                      1.53262e+06 |                                1.30325e+07 |              8.50344 |                   0.00694486 |                  0.00733217 |
|     16384 |             1725.08  |           1697.43   |                     16.322    |                    11.9696   |                     9652.23 |                      1.3688e+06  |                                1.16535e+07 |              8.51366 |                   0.00946156 |                  0.00705158 |

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
  "hit_rate": 0.9675,
  "avg_expert_load_latency_ms": 0.6865889695473015,
  "prefetch_accuracy": 0.9675,
  "cache_prefetch_precision": 0.0,
  "gpu_memory_saved_gb": 6.424072265625,
  "markov_accuracy_at_k": 0.9650872817955112,
  "markov_accuracy_at_1": 0.9650872817955112,
  "markov_io_hidden_ms": 3591.821083604097,
  "markov_hidden_io_percent": 100.0,
  "latency_p95_ms": 3.6546708084642825,
  "latency_p99_ms": 8.917491734027825
}
```

## moe_router
```json
{
  "dropped_tokens": 0,
  "imbalance_ratio": 1.0007735620703893,
  "expert_load_mean": 1024.0,
  "nash_imbalance_ratio": 1.004950490375089,
  "nash_dropped_tokens": 0,
  "nash_convergence_rate": 0.0,
  "nash_avg_iterations": 3.0,
  "nash_overhead_ms": 18.310652091167867,
  "imbalance_improvement_x": 0.9958436476774685
}
```

## predictor
```json
{
  "rolling_accuracy": 0.94,
  "precision_at_k": 0.965,
  "recall_at_k": 0.96,
  "latency_ms_mean": 0.09918868541717529,
  "latency_ms_p99": 0.14188074041157997,
  "memory_overhead_mb": 1.055450439453125
}
```

## vps
```json
{
  "ram_mb": 0.0,
  "time_to_first_token_ms": 29.860167065635324,
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
