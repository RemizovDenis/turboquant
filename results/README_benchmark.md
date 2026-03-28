# TurboQuant-MoE Benchmark

Generated: `20260328_072610`

## memory
|   seq_len |   baseline_fp16_mb |   kv_only_mb |   adaptive_kv_mb |   moe_mb |   actual_compression_ratio |   classic_compression_ratio |   adaptive_compression_x |   compression_gain_vs_classic_x |   theoretical_vs_actual_gap |
|----------:|-------------------:|-------------:|-----------------:|---------:|---------------------------:|----------------------------:|-------------------------:|--------------------------------:|----------------------------:|
|      1024 |                 16 |        3.875 |          1.85938 |      3.5 |                   0.116211 |                    0.242188 |                  8.60504 |                         2.08403 |                   0.0546875 |
|      4096 |                 64 |       15.5   |          7.56445 |     14   |                   0.118195 |                    0.242188 |                  8.46062 |                         2.04906 |                   0.0546875 |
|     16384 |                256 |       62     |         30.1182  |     56   |                   0.117649 |                    0.242188 |                  8.49985 |                         2.05856 |                   0.0546875 |

## speed
|   seq_len |   prefill_latency_ms |   decode_latency_ms |   baseline_prefill_latency_ms |   baseline_decode_latency_ms |   throughput_tokens_per_sec |   baseline_decode_tokens_per_sec |   projected_io_bound_decode_tokens_per_sec |   io_bound_speedup_x |   observed_prefill_speedup_x |   observed_decode_speedup_x |
|----------:|---------------------:|--------------------:|------------------------------:|-----------------------------:|----------------------------:|---------------------------------:|-------------------------------------------:|---------------------:|-----------------------------:|----------------------------:|
|      1024 |              84.1597 |             75.7917 |                      0.642367 |                     0.733708 |                    13510.7  |                      1.39565e+06 |                                1.19406e+07 |              8.55561 |                   0.00763271 |                  0.00968059 |
|      4096 |             391.14   |            365.947  |                      2.7715   |                     2.6487   |                    11192.9  |                      1.54642e+06 |                                1.3104e+07  |              8.47375 |                   0.0070857  |                  0.00723794 |
|     16384 |            1710.31   |           1710.55   |                     10.7464   |                    10.5703   |                     9578.19 |                      1.55001e+06 |                                1.31621e+07 |              8.49159 |                   0.00628329 |                  0.00617944 |

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
  "hit_rate": 0.884375,
  "avg_expert_load_latency_ms": 3.2963908539386466,
  "prefetch_accuracy": 0.8875,
  "cache_prefetch_precision": 0.2608695652173913,
  "gpu_memory_saved_gb": 5.34375,
  "markov_accuracy_at_k": 0.895,
  "markov_accuracy_at_1": 0.855,
  "markov_io_hidden_ms": 4722.5566902311675,
  "markov_hidden_io_percent": 100.0,
  "latency_p95_ms": 24.117693491280075,
  "latency_p99_ms": 42.192060786765055
}
```

## moe_router
```json
{
  "dropped_tokens": 0,
  "imbalance_ratio": 1.0010249638418156,
  "expert_load_mean": 1024.0,
  "nash_imbalance_ratio": 1.004950490375089,
  "nash_dropped_tokens": 0,
  "nash_convergence_rate": 0.0,
  "nash_avg_iterations": 3.0,
  "nash_overhead_ms": 18.229187501128763,
  "imbalance_improvement_x": 0.9960938110176868
}
```

## predictor
```json
{
  "rolling_accuracy": 0.92,
  "precision_at_k": 0.9508333333333333,
  "recall_at_k": 0.97,
  "latency_ms_mean": 0.10615620762109756,
  "latency_ms_p99": 0.12086061062291273,
  "memory_overhead_mb": 1.055450439453125
}
```

## vps
```json
{
  "ram_mb": 0.0,
  "time_to_first_token_ms": 29.924457892775536,
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
