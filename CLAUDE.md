# Unit Test Development

## Goal
Adding unit tests to the fdtdx project with the `pytest -m unit` marker.

## Setup
- Tests location: `tests/unit/` (mirrors `src/fdtdx/` structure)
- Run unit tests: `.venv/bin/python -m pytest tests/unit -m unit`
- Coverage report: `.venv/bin/python -m pytest tests/unit --cov=fdtdx --cov-report=html`
- Use `.venv` directly, not `uv` (uv reverts JAX fixes)

## Completed Unit Tests
| Module | Test File | Tests | Coverage |
|--------|-----------|-------|----------|
| colors.py | test_colors.py | 19 | 100% |
| config.py | test_config.py | 20 | 92% |
| constants.py | test_constants.py | 11 | 100% |
| conversion/json.py | conversion/test_json.py | 19 | 96% |
| conversion/stl.py | conversion/test_stl.py | 14 | 100% |
| core/grid.py | core/test_grid.py | 17 | 100% |
| core/linalg.py | core/test_linalg.py | 21 | 98% |

## Guidelines
- Avoid redundant tests - maintain coverage without duplication
- Use mocks when needed (e.g., JAX backend detection in config.py)
- Reuse existing test code and real classes (e.g., WaveCharacter) when possible
- Use `git mv` when moving test files to preserve history
