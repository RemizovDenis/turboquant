## Summary

## What changed

## Validation
- [ ] ruff check turboquant tests
- [ ] ruff format --check turboquant tests
- [ ] mypy turboquant --strict
- [ ] pytest tests/ -v --tb=short -x -k "not gpu and not cuda and not triton"

## Risks

## Benchmarks (if performance path changed)
