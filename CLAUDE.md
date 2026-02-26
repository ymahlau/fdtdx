# Unit Test Development

## Goal
Adding unit tests to the fdtdx project with the `pytest -m unit` marker.

## Setup
- Tests location: `tests/unit/` (mirrors `src/fdtdx/` structure)
- Run unit tests: `.venv/bin/python -m pytest tests/unit -m unit`
- Coverage report: `.venv/bin/python -m pytest tests/unit --cov=fdtdx --cov-report=html`
- Coverage must run on the full unit suite (not individual files) to avoid JAX crash
- Use `.venv` directly, not `uv` (uv reverts JAX fixes)

## Completed Unit Tests
| Module | Test File | Tests | Coverage |
|--------|-----------|-------|----------|
| objects/boundaries/boundary.py | objects/boundaries/test_boundary.py | 15 | ~90% |
| objects/detectors/detector.py | objects/detectors/test_detector.py | 16 | ~85% |
| objects/detectors/diffractive.py | objects/detectors/test_diffractive.py | 22 | ~90% |
| objects/detectors/energy.py | objects/detectors/test_energy.py | 17 | ~95% |
| objects/detectors/field.py | objects/detectors/test_field.py | 16 | ~95% |
| objects/detectors/mode.py | objects/detectors/test_mode.py | 18 | ~85% |
| objects/detectors/phasor.py | objects/detectors/test_phasor.py | 14 | ~90% |
| objects/detectors/plotting/line_plot.py | objects/detectors/plotting/test_line_plot.py | 15 | ~95% |
| objects/detectors/plotting/plot2d.py | objects/detectors/plotting/test_plot2d.py | 16 | ~95% |
| objects/detectors/plotting/video.py | objects/detectors/plotting/test_video.py | 12 | ~90% |
| objects/detectors/poynting_flux.py | objects/detectors/test_poynting_flux.py | 15 | ~95% |
| objects/boundaries/initialization.py | objects/boundaries/test_initialization.py | 44 | ~95% |
| objects/boundaries/perfectly_matched_layer.py | objects/boundaries/test_perfectly_matched_layer.py | 35 | ~90% |
| objects/boundaries/periodic.py | objects/boundaries/test_periodic.py | 17 | ~95% |
| objects/boundaries/utils.py | objects/boundaries/test_boundary_utils.py | 50 | ~100% |
| colors.py | test_colors.py | 19 | 100% |
| config.py | test_config.py | 20 | 92% |
| constants.py | test_constants.py | 11 | 100% |
| conversion/json.py | conversion/test_json.py | 19 | 96% |
| conversion/stl.py | conversion/test_stl.py | 14 | 100% |
| conversion/vti.py | conversion/test_vti.py | 29 | 100% |
| core/grid.py | core/test_grid.py | 17 | 100% |
| core/jax/pytrees.py | core/jax/test_pytrees.py | 44 | 96% |
| core/jax/sharding.py | core/jax/test_sharding.py | 26 | 100% |
| core/jax/ste.py | core/jax/test_ste.py | 8 | 100% |
| core/jax/utils.py | core/jax/test_utils.py | 15 | 100% |
| core/linalg.py | core/test_linalg.py | 21 | 98% |
| core/misc.py | core/test_misc.py | 88 | 98% |
| core/physics/curl.py | core/physics/test_curl.py | 14 | 100% |
| core/physics/losses.py | core/physics/test_losses.py | 8 | 100% |
| core/physics/metrics.py | core/physics/test_metrics.py | 18 | 100% |
| core/physics/modes.py | core/physics/test_modes.py | 21 | 99% |
| core/plotting/debug.py | core/plotting/test_debug.py | 19 | 100% |
| core/plotting/device_permittivity_index_utils.py | core/plotting/test_device_permittivity_index_utils.py | 8 | 100% |
| core/plotting/utils.py | core/plotting/test_plotting_utils.py | 10 | 96% |
| core/switch.py | core/test_switch.py | 21 | 93% |
| fdtd/backward.py | fdtd/test_backward.py | 11 | 100% |
| fdtd/container.py | fdtd/test_container.py | 45 | 100% |
| fdtd/fdtd.py | fdtd/test_fdtd.py | 21 | 71% |
| fdtd/forward.py | fdtd/test_forward.py | 11 | 100% |
| fdtd/initialization.py | fdtd/test_initialization.py | 57 | 70% |
| fdtd/misc.py | fdtd/test_fdtd_misc.py | 17 | 100% |
| fdtd/stop_conditions.py | fdtd/test_stop_conditions.py | 34 | 97% |
| fdtd/update.py | fdtd/test_update.py | 51 | 100% |
| fdtd/wrapper.py | fdtd/test_wrapper.py | 8 | 100% |
| interfaces/modules.py | interfaces/test_modules.py | 24 | 95% |
| interfaces/recorder.py | interfaces/test_recorder.py | 15 | ~85% |
| interfaces/state.py | interfaces/test_state.py | 16 | 100% |
| interfaces/time_filter.py | interfaces/test_time_filter.py | 22 | ~90% |
| utils/logger.py | utils/test_logger.py | 17 | ~90% |
| utils/plot_field_slice.py | utils/test_plot_field_slice.py | 35 | ~95% |
| utils/plot_material.py | utils/test_plot_material.py | 20 | ~90% |
| utils/plot_setup.py | utils/test_plot_setup.py | 15 | ~90% |

## Integration Tests
- Tests location: `tests/integration/` (mirrors `src/fdtdx/` structure)
- Run integration tests: `.venv/bin/python -m pytest tests/integration -m integration`
- Use `@pytest.mark.integration` marker
- For tests that require full simulation setup (e.g., `place_objects`, `apply_params`)
- When splitting existing tests from `tests/`, move integration-style tests here

| Module | Test File | Tests |
|--------|-----------|-------|
| conversion/vti.py | conversion/test_vti.py | 1 |
| fdtd/initialization.py | fdtd/test_initialization.py | 11 |
| fdtd/stop_conditions.py | fdtd/test_stop_conditions.py | 5 |
| utils/plot_material.py | utils/test_plot_material.py | 14 |
| utils/plot_setup.py | utils/test_plot_setup.py | 6 |

## Simulation Tests
- Tests location: `tests/simulation/` (mirrors `src/fdtdx/` structure)
- Run simulation tests: `.venv/bin/python -m pytest tests/simulation -m simulation`
- Use `@pytest.mark.simulation` marker (auto-applied via conftest.py)
- For tests that require `jax.grad`/`jax.value_and_grad` through full simulation passes

| Module | Test File | Tests |
|--------|-----------|-------|
| fdtd/fdtd.py | fdtd/test_fdtd.py | 3 |

## Guidelines
- Avoid redundant tests - maintain coverage without duplication
- Use mocks when needed (e.g., JAX backend detection in config.py, ArrayContainer in vti.py)
- Reuse existing test code and real classes (e.g., WaveCharacter) when possible
- Use `git mv` when moving test files to preserve history
- When existing tests in `tests/` contain both unit and integration tests, split them:
  move unit tests to `tests/unit/`, integration tests to `tests/integration/`
