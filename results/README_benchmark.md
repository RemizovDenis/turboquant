# TurboQuant-MoE Benchmark

Generated: `20260328_034636`

## memory
|   seq_len |   baseline_fp16_mb |   kv_only_mb |   moe_mb |   actual_compression_ratio |   theoretical_vs_actual_gap |
|----------:|-------------------:|-------------:|---------:|---------------------------:|----------------------------:|
|      1024 |                 16 |        3.875 |    3.875 |                   0.242188 |                   0.0546875 |
|      4096 |                 64 |       15.5   |   15.5   |                   0.242188 |                   0.0546875 |
|     16384 |                256 |       62     |   62     |                   0.242188 |                   0.0546875 |

## speed
|   seq_len |   prefill_latency_ms |   decode_latency_ms |   throughput_tokens_per_sec |
|----------:|---------------------:|--------------------:|----------------------------:|
|      1024 |              186.408 |             111.826 |                     5493.33 |
|      4096 |              831.671 |             483.925 |                     4925.03 |
|     16384 |             8015.24  |            4255.05  |                     2044.11 |

## quality
|   seq_len |   recall_at_1 |
|----------:|--------------:|
|      1000 |             1 |
|      4000 |             1 |
|     16000 |             1 |

## moe_expert
```json
{
  "hit_rate": 0.1,
  "avg_expert_load_latency_ms": 152.7454065857455,
  "prefetch_accuracy": 0.0,
  "gpu_memory_saved_gb": 2.625
}
```

## moe_router
```json
{
  "dropped_tokens": 0,
  "imbalance_ratio": 1.0003867810351945,
  "expert_load_mean": 1024.0
}
```

## predictor
```json
{
  "rolling_accuracy": 0.04,
  "latency_ms_mean": 0.26609663385897875,
  "latency_ms_p99": 0.43443794595077945,
  "memory_overhead_mb": 1.055450439453125
}
```

## vps
```json
{
  "ram_mb": 0.0,
  "time_to_first_token_ms": 77.35787495039403,
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
