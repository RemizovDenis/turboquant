# Contributing to TurboQuant-MoE

## Development setup

```bash
git clone https://github.com/RemizovDenis/turboquant.git
cd turboquant
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,transformers,benchmark]"
```

## Quality gates (required before PR)

```bash
ruff check turboquant tests
ruff format --check turboquant tests
mypy turboquant --strict
pytest tests/ -v --tb=short -x -k "not gpu and not cuda and not triton"
```

## Pull requests

1. Create a branch from `main`.
2. Keep PRs focused and small.
3. Use conventional commit prefixes (`feat:`, `fix:`, `docs:`, `chore:`).
4. Ensure CI is green.
5. Request review.

## Performance changes

If a PR modifies hot-path logic (quantization/cache/prefetch), include:

- benchmark command(s)
- hardware info
- before/after numbers
