# Field Validation Protocol (TurboQuant vs vLLM/DeepSpeed)

## Goal

Run real-world, reproducible validation on identical hardware for:

- TurboQuant-MoE
- vLLM baseline
- DeepSpeed baseline

and collect:

- throughput (`tokens/sec`)
- latency (`p50/p95/p99`, TTFT)
- stability (no spikes > 50 ms over sustained load)
- OOM behavior
- quality guardrail (needle recall / degradation)
- soak reliability (error-rate, drift over time)

## Hardware Matrix (minimum)

Run each stack on the same model and prompt mix:

1. `1x L4 (24 GB)` cloud
2. `1x A10G (24 GB)` cloud
3. `1x A100 80GB`
4. `2x A100 80GB` (multi-node or multi-GPU deployment mode)

Optional extension:

5. `1x H100 80GB`
6. `4x A100/H100` cluster run

## Test Scenarios

Use the same dataset and request distribution for all stacks:

1. Short chat: prompt 256, output 128
2. Medium chat: prompt 1024, output 256
3. Long context: prompt 8192, output 256
4. Needle case: `1k..128k` (quality regression gate)
5. Mixed production profile:
   - 60% short
   - 30% medium
   - 10% long

Concurrency ladders:

1. `1, 2, 4, 8, 16, 32`
2. Stop when:
   - p99 > SLO
   - error-rate > 1%
   - OOM or throttling

Soak profile:

1. `6h` warm production load
2. `24h` sustained load
3. `48h` optional burn-in

## SLO / Pass Criteria

Target gates:

1. Decode throughput: `>= 1.8x` vs baseline at same quality constraints
2. Latency spikes: no sustained spikes `> 50 ms` at target concurrency
3. Error-rate: `< 0.5%` during soak
4. OOM: none in validated operating point
5. Quality:
   - recall@1 = `100%` on needle suite
   - degradation `< 0.5%`

## Execution Steps

1. Checkout exact commit on all targets.
2. Freeze environment:
   - GPU driver
   - CUDA/cuDNN
   - PyTorch
   - model revision
3. Start server stack:
   - TurboQuant endpoint
   - vLLM endpoint
   - DeepSpeed endpoint
4. Run common load harness (`scripts/load_probe.py`) against each endpoint.
5. Run internal benchmark suite for TurboQuant.
6. Save raw metrics JSON per run.
7. Build consolidated comparison table.

## Required Artifacts

For each run, keep:

1. stack name (`turboquant|vllm|deepspeed`)
2. hardware id / GPU model
3. commit SHA / container image tag
4. test profile id
5. full JSON metrics
6. server logs + OOM events
7. `nvidia-smi` snapshots

## Commands

### 1) TurboQuant internal suites

```bash
source .venv/bin/activate
turboquant-benchmark --suite memory,speed,quality,moe_expert,moe_router,predictor,vps --output ./results
```

### 2) Load probe (same for all stacks)

```bash
python scripts/load_probe.py \
  --base-url http://127.0.0.1:8000 \
  --model mistralai/Mixtral-8x7B-v0.1 \
  --concurrency 16 \
  --requests 500 \
  --max-tokens 128 \
  --timeout 120 \
  --output ./results/load_probe_turboquant.json
```

### 3) Soak (24h)

```bash
python scripts/load_probe.py \
  --base-url http://127.0.0.1:8000 \
  --model mistralai/Mixtral-8x7B-v0.1 \
  --concurrency 32 \
  --duration-sec 86400 \
  --max-tokens 128 \
  --timeout 120 \
  --output ./results/soak_24h_turboquant.json
```

## Final Report Template

Final report must include:

1. Executive summary
2. Hardware and software matrix
3. Throughput/latency tables by scenario and concurrency
4. Quality guardrail results
5. Soak stability and incident log
6. Cost-efficiency comparison
7. Recommended production operating point
