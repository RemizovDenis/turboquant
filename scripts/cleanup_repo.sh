#!/bin/bash
# cleanup_repo.sh - Script for cleaning up git tracking for v0.3.0
# From ШАГ 00: ЧИСТКА РЕПОЗИТОРИЯ

echo "Starting TurboQuant-MoE repository cleanup..."

# Udal .venv iz git tracking (force if exists)
echo "Untracking .venv/..."
git rm -rf --cached .venv --force 2>/dev/null || true

# Udal HTML rezultaty
echo "Untracking results/*.html..."
git rm --cached "results/*.html" --force 2>/dev/null || true

# Udal benchmark HTML
echo "Untracking benchmark_results/*.html..."
git rm --cached "benchmark_results/*.html" --force 2>/dev/null || true

# Udal cache folders
echo "Untracking __pycache__, .mypy_cache, .ruff_cache..."
git rm -rf --cached "**/__pycache__" --force 2>/dev/null || true
git rm -rf --cached "**/.mypy_cache" --force 2>/dev/null || true
git rm -rf --cached "**/.ruff_cache" --force 2>/dev/null || true

echo "Cleanup complete. Status:"
git status

