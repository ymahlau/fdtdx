"""TFSF source normalization must be independent of the grid metric (regression for #392).

A plane wave injected through a fixed physical area carries the same power regardless of how the
domain is discretized. On a quasi-uniform grid with dz != dx, the bulk curl update is metric-aware,
so the TFSF correction must carry the same local metric factor; without it, the source over-injects
by roughly (dz / dx)^2 in power.
"""

import jax
import numpy as np

import fdtdx
from fdtdx.core.grid import QuasiUniformGrid, UniformGrid

_DX = 25e-9
_LXY = 0.8e-6
_LZ = 1.6e-6
_WAVELENGTH = 1.0e-6
_SIM_TIME = 40e-15
_PML_CELLS = 8


def _run_flux(grid) -> float:
    config = fdtdx.SimulationConfig(time=_SIM_TIME, grid=grid, backend="cpu", gradient_config=None)
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_real_shape=(_LXY, _LXY, _LZ))
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        boundary_type="pml",
        override_types={f: "periodic" for f in ("min_x", "max_x", "min_y", "max_y")},
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    source = fdtdx.UniformPlaneSource(
        name="src",
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
        amplitude=1.0,
    )
    constraints.append(source.same_size(volume, axes=(0, 1)))
    constraints.append(
        source.place_relative_to(volume, axes=(2,), own_positions=(-1,), other_positions=(-1,), margins=(0.5e-6,))
    )
    objects.append(source)

    det = fdtdx.PoyntingFluxDetector(name="flux", partial_grid_shape=(None, None, 1), direction="+", plot=False)
    constraints.append(det.same_size(volume, axes=(0, 1)))
    constraints.append(
        det.place_relative_to(volume, axes=(2,), own_positions=(1,), other_positions=(1,), margins=(-0.5e-6,))
    )
    objects.append(det)

    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
    _, out = fdtdx.run_fdtd(arrays=arrays, objects=obj, config=config, key=key, show_progress=False)

    flux = np.asarray(out.detector_states["flux"]["poynting_flux"][:, 0])
    dt = config.time_step_duration
    steps_per_period = round(_WAVELENGTH / (fdtdx.constants.c * dt))
    return float(np.mean(flux[-3 * steps_per_period :]))


def test_tfsf_injection_is_grid_metric_independent():
    """Injected plane-wave power matches between uniform and quasi-uniform (dz = 2dx) grids."""
    flux_uniform = _run_flux(UniformGrid(spacing=_DX))
    flux_quasi = _run_flux(QuasiUniformGrid(dx=_DX, dy=_DX, dz=2 * _DX))
    ratio = flux_quasi / flux_uniform
    # Without the metric factor on the TFSF correction, the ratio is ~3 (see #392).
    assert 0.95 < ratio < 1.05, f"TFSF injected power depends on grid metric: ratio={ratio:.3f}"
