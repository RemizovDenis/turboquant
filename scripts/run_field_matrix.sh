#!/usr/bin/env bash
set -euo pipefail

# Field matrix runner for TurboQuant / vLLM / DeepSpeed
# Requires: python scripts/load_probe.py

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-mistralai/Mixtral-8x7B-v0.1}"
OUTDIR="${OUTDIR:-./results/field_runs_$(date +%Y%m%d_%H%M%S)}"
CONCURRENCY_SET="${CONCURRENCY_SET:-1 2 4 8 16 32}"
REQUESTS="${REQUESTS:-500}"
TIMEOUT="${TIMEOUT:-120}"
MAX_TOKENS_SHORT="${MAX_TOKENS_SHORT:-128}"
MAX_TOKENS_MEDIUM="${MAX_TOKENS_MEDIUM:-256}"
MAX_TOKENS_LONG="${MAX_TOKENS_LONG:-256}"
SOAK_SEC="${SOAK_SEC:-0}" # set 86400 for 24h

TQ_URL="${TQ_URL:-}"
VLLM_URL="${VLLM_URL:-}"
DS_URL="${DS_URL:-}"

if [[ -z "$TQ_URL" || -z "$VLLM_URL" || -z "$DS_URL" ]]; then
  echo "Set all endpoint URLs first:"
  echo "  export TQ_URL=http://<turboquant-host>:8000"
  echo "  export VLLM_URL=http://<vllm-host>:8000"
  echo "  export DS_URL=http://<deepspeed-host>:8000"
  exit 1
fi

mkdir -p "$OUTDIR"

run_case() {
  local stack="$1"
  local base_url="$2"
  local profile="$3"
  local prompt="$4"
  local max_tokens="$5"
  local c="$6"

  local out="$OUTDIR/${stack}__${profile}__c${c}.json"
  echo "[RUN] stack=$stack profile=$profile c=$c -> $out"
  python scripts/load_probe.py \
    --base-url "$base_url" \
    --model "$MODEL" \
    --prompt "$prompt" \
    --max-tokens "$max_tokens" \
    --concurrency "$c" \
    --requests "$REQUESTS" \
    --timeout "$TIMEOUT" \
    --output "$out"
}

run_soak() {
  local stack="$1"
  local base_url="$2"
  local c="$3"
  local out="$OUTDIR/${stack}__soak_${SOAK_SEC}s__c${c}.json"
  echo "[SOAK] stack=$stack duration=${SOAK_SEC}s c=$c -> $out"
  python scripts/load_probe.py \
    --base-url "$base_url" \
    --model "$MODEL" \
    --prompt "You are under soak test. Return a concise answer." \
    --max-tokens "$MAX_TOKENS_SHORT" \
    --concurrency "$c" \
    --duration-sec "$SOAK_SEC" \
    --timeout "$TIMEOUT" \
    --output "$out"
}

SHORT_PROMPT="Summarize TurboQuant in 5 bullet points."
MEDIUM_PROMPT="Given a long context serving setup, propose a latency-focused optimization plan with measurable KPIs."
LONG_PROMPT="$(python - <<'PY'
chunk = "TurboQuant long-context benchmark sentence. "
print(chunk * 800)
PY
)"

for c in $CONCURRENCY_SET; do
  run_case "turboquant" "$TQ_URL" "short" "$SHORT_PROMPT" "$MAX_TOKENS_SHORT" "$c"
  run_case "vllm" "$VLLM_URL" "short" "$SHORT_PROMPT" "$MAX_TOKENS_SHORT" "$c"
  run_case "deepspeed" "$DS_URL" "short" "$SHORT_PROMPT" "$MAX_TOKENS_SHORT" "$c"

  run_case "turboquant" "$TQ_URL" "medium" "$MEDIUM_PROMPT" "$MAX_TOKENS_MEDIUM" "$c"
  run_case "vllm" "$VLLM_URL" "medium" "$MEDIUM_PROMPT" "$MAX_TOKENS_MEDIUM" "$c"
  run_case "deepspeed" "$DS_URL" "medium" "$MEDIUM_PROMPT" "$MAX_TOKENS_MEDIUM" "$c"

  run_case "turboquant" "$TQ_URL" "long" "$LONG_PROMPT" "$MAX_TOKENS_LONG" "$c"
  run_case "vllm" "$VLLM_URL" "long" "$LONG_PROMPT" "$MAX_TOKENS_LONG" "$c"
  run_case "deepspeed" "$DS_URL" "long" "$LONG_PROMPT" "$MAX_TOKENS_LONG" "$c"
done

if [[ "$SOAK_SEC" -gt 0 ]]; then
  SOAK_CONCURRENCY="${SOAK_CONCURRENCY:-32}"
  run_soak "turboquant" "$TQ_URL" "$SOAK_CONCURRENCY"
  run_soak "vllm" "$VLLM_URL" "$SOAK_CONCURRENCY"
  run_soak "deepspeed" "$DS_URL" "$SOAK_CONCURRENCY"
fi

echo "Done. Artifacts: $OUTDIR"
