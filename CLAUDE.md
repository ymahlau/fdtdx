# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FDTDX is a JAX-based framework for GPU-accelerated finite-difference time-domain (FDTD) electromagnetic simulations, focused on inverse design of photonic devices. It supports automatic differentiation through simulations, enabling gradient-based optimization of nanostructures.

## Common Commands

### Install (development)
```bash
uv sync
```

### Run all tests
```bash
uv run python -m pytest tests
```

### Run tests by marker
```bash
uv run python -m pytest tests -m unit
uv run python -m pytest tests -m integration
uv run python -m pytest tests -m simulation
```

### Run a single test file or test
```bash
uv run python -m pytest tests/unit/test_example.py
uv run python -m pytest tests/unit/test_example.py::test_name
```

### Run tests with coverage (as CI does)
```bash
uv run python -m pytest tests -m "unit or integration" --cov --cov-branch --cov-config=pyproject.toml --cov-report=xml
```

### Lint and format
```bash
uv run pre-commit run -a
```

### Type checking
```bash
uvx ty check --error-on-warning
```

### Build docs
```bash
sh docs/scripts/sync_notebooks.sh && uv run sphinx-build -W --keep-going docs/source/ docs/build/
```

## Code Style

- **Ruff** for linting and formatting: target Python 3.12, line length 120, import sorting enabled
- Ruff excludes: `examples/`, `slurm/`, `checks/`, `docs/`
- Type checking via **ty** (run with `uvx ty check --error-on-warning`), excludes `tests/`

## Architecture

All source code is in `src/fdtdx/`. Public API is exported from `src/fdtdx/__init__.py`.

See `.claude/skills/fdtdx/SKILL.md` for detailed framework patterns, Yee grid conventions, field normalization, constraint system, gradient strategies, and common pitfalls.
