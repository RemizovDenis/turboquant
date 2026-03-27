#!/usr/bin/env bash
# TurboQuant — Quick Setup & Validation
# Run this after cloning to verify everything works.

set -euo pipefail

echo "=================================="
echo "TurboQuant Quick Setup & Validation"
echo "=================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 11 ]); then
    echo "⚠️  Warning: Python >= 3.11 recommended (found $PYTHON_VERSION)"
    echo "    Some type hint features may not work on older versions."
fi

# Create virtual environment
echo ""
echo "1. Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install
echo ""
echo "2. Installing TurboQuant..."
pip install --upgrade pip -q
pip install -e ".[dev,benchmark]" -q

# Test
echo ""
echo "3. Running tests..."
pytest tests/ -v --tb=short

# Smoke tests
echo ""
echo "4. Running smoke tests..."
python3 -c "
from turboquant import TurboQuantConfig, TurboQuantKVCache
import torch

config = TurboQuantConfig(head_dim=128, num_heads=8, device='cpu')
with TurboQuantKVCache(config) as tq:
    k = torch.randn(1, 8, 64, 128, dtype=torch.float16)
    v = torch.randn(1, 8, 64, 128, dtype=torch.float16)
    entry = tq.compress(k, v)
    k_hat, v_hat = tq.decompress(entry)
    mem = tq.memory_usage(entry)
    print(f'  Compressed bytes: {mem[\"bytes\"]:,}')
    print(f'  MB:               {mem[\"mb\"]:.2f}')
    print(f'  Compression ratio: {mem[\"ratio\"]:.4f}')
    mse = ((k.float() - k_hat.float())**2).mean().item()
    print(f'  MSE keys:         {mse:.6f}')
print('✅ Core compression works!')
"

# Quick benchmark
echo ""
echo "5. Running quick benchmark..."
turboquant-benchmark --suite memory --output ./benchmark_results --head-dim 128 --num-heads 8

echo ""
echo "=================================="
echo "✅ All checks passed!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  - Check benchmark results: cat benchmark_results/benchmark_results.md"
echo "  - Run full suite: turboquant-benchmark --suite all --output ./benchmark_results"
echo "  - Start Ollama proxy: turboquant-proxy"
