#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PYTHON_BIN:-}"
if [[ -z "${PY_BIN}" ]]; then
  for candidate in ".venv/bin/python" ".venv_final/bin/python"; do
    if [[ -x "${candidate}" ]]; then
      PY_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
if [[ "${AUTO_INSTALL_DEPS}" == "1" ]]; then
  "${PY_BIN}" - <<'PY'
import importlib.util
import subprocess
import sys

required = ["psutil", "requests"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(f"[INFO] Installing missing benchmark dependencies: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
PY
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-results/field_local/${STAMP}}"
PROFILE="${PROFILE:-real}"
MODELS_WAS_SET="${MODELS+x}"
RUNS_WAS_SET="${RUNS+x}"
SKIP_PULL_WAS_SET="${SKIP_PULL+x}"
MODELS="${MODELS:-mixtral:latest mistral:latest llama3.1:latest}"
RUNS="${RUNS:-10}"
SKIP_PULL="${SKIP_PULL:-0}"
NO_PROXY="${NO_PROXY:-0}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:11435}"

if [[ "${PROFILE}" == "quick" ]]; then
  if [[ -z "${MODELS_WAS_SET}" ]]; then
    MODELS="mistral:latest llama3.1:latest"
  fi
  if [[ -z "${RUNS_WAS_SET}" ]]; then
    RUNS="4"
  fi
  if [[ -z "${SKIP_PULL_WAS_SET}" ]]; then
    SKIP_PULL="1"
  fi
fi

ARGS=(
  --models ${MODELS}
  --runs "${RUNS}"
  --output-dir "${OUT_DIR}"
  --out-json benchmark_ultimate_m4.json
  --out-readme README_benchmark.md
  --ollama-url "${OLLAMA_URL}"
  --proxy-url "${PROXY_URL}"
)

if [[ "${SKIP_PULL}" == "1" ]]; then
  ARGS+=(--skip-pull)
fi
if [[ "${NO_PROXY}" == "1" ]]; then
  ARGS+=(--no-proxy)
fi

"${PY_BIN}" scripts/benchmark_ultimate_m4.py "${ARGS[@]}"

echo ""
echo "[OK] Local field benchmark finished"
echo "Python:  ${PY_BIN}"
echo "Profile: ${PROFILE}"
echo "Models:  ${MODELS}"
echo "JSON:   ${OUT_DIR}/benchmark_ultimate_m4.json"
echo "Report: ${OUT_DIR}/README_benchmark.md"
