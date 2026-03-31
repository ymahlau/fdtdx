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

### Source layout
All source code is in `src/fdtdx/`. Public API is exported from `src/fdtdx/__init__.py`.

### Core abstractions

**TreeClass** (`core/jax/pytrees.py`): Base class wrapping `pytreeclass.TreeClass`. All config, container, and object classes inherit from it. Supports functional immutable updates via `.aset()`.

**SimulationConfig** (`config.py`): Holds simulation parameters (resolution, time steps, backend, dtype, courant factor). **GradientConfig** configures the gradient computation method.

**SimulationObject** (`objects/object.py`): Base class for everything placed in the simulation grid. Subclasses:
- **Static materials**: `SimulationVolume`, `UniformMaterialObject`, `Sphere`, `Cylinder`, `ExtrudedPolygon`
- **Devices**: Optimizable objects with parameter transformation pipelines
- **Sources**: `GaussianPlaneSource`, `UniformPlaneSource`, `ModePlaneSource`
- **Detectors**: `FieldDetector`, `EnergyDetector`, `PhasorDetector`, `PoyntingFluxDetector`, `ModeOverlapDetector`
- **Boundaries**: `PerfectlyMatchedLayer`, `PeriodicBoundary`

**Constraint system** (`objects/object.py`): Objects are positioned relative to each other via `PositionConstraint`, `SizeConstraint`, etc. `resolve_object_constraints()` in `fdtd/initialization.py` resolves these to grid slices.

### Simulation pipeline

1. **`place_objects()`** → creates `ObjectContainer`, `ArrayContainer`, `ParameterContainer`
2. **`apply_params()`** → updates device permittivity arrays from parameter values
3. **`run_fdtd()`** → dispatches to `reversible_fdtd()` or `checkpointed_fdtd()` → returns `SimulationState`

### FDTD engine (`fdtd/`)

- **`update.py`**: E and H field updates implementing Maxwell's curl equations on a Yee grid. Supports isotropic/anisotropic materials, PML, periodic boundaries, lossy materials.
- **`forward.py`**: Single time step (E update → H update → source injection → detector recording)
- **`fdtd.py`**: Two gradient strategies:
  - `reversible_fdtd()`: Exploits time-reversibility of Maxwell's equations — O(1) field memory, O(T) boundary memory
  - `checkpointed_fdtd()`: Standard checkpointing with configurable memory-time tradeoff
- **`container.py`**: `ArrayContainer` (E, H, PML fields, material arrays, detector states), `ObjectContainer` (typed access to simulation objects)

### Device optimization (`objects/device/parameters/`)

Parameter transformation pipeline applied in order: projection → smoothing → discretization → discrete post-processing → symmetry constraints. Each step is a `ParameterTransformation` subclass.

### Physics (`core/physics/`)

- `curl.py`: Discrete curl operators for E and H on the Yee grid
- `modes.py`: Guided mode solver (eigenvalue problem) and mode overlap computation
- `metrics.py`: Energy density and Poynting flux calculations

### Material system (`materials.py`)

`Material` class supports isotropic (scalar), diagonal anisotropic (3 components), and fully anisotropic (3×3 tensor) electromagnetic properties (permittivity, permeability, electric/magnetic conductivity).
