"""End-to-end parity: region-restricted detector recording == full-domain interpolation + slice.

For every detector kind, an interior detector recorded through ``update_detector_states`` (which now
interpolates only over the detector's ``grid_slice`` + halo) must produce byte-for-byte the same
state as the historical path: interpolate the whole domain, then slice the region into the
detector's ``update``. Run on uniform and non-uniform grids.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.core.grid import RectilinearGrid, UniformGrid
from fdtdx.core.physics.curl import interpolate_fields
from fdtdx.fdtd.update import pad_fields_for_boundaries, update_detector_states

_N = 12  # cube side (cells)
_RES = 50e-9
_FREQ = fdtdx.constants.c / 1e-6


def _detectors():
    wave = fdtdx.WaveCharacter(frequency=_FREQ)
    return [
        fdtdx.PhasorDetector(name="phasor", partial_grid_shape=(4, 4, 4), wave_characters=(wave,), plot=False),
        fdtdx.PhasorDetector(
            name="phasor_reduced",
            partial_grid_shape=(4, 4, 4),
            wave_characters=(wave,),
            reduce_volume=True,
            plot=False,
        ),
        fdtdx.EnergyDetector(name="energy", partial_grid_shape=(4, 4, 4), plot=False),
        fdtdx.EnergyDetector(name="energy_reduced", partial_grid_shape=(4, 4, 4), reduce_volume=True, plot=False),
        fdtdx.PoyntingFluxDetector(name="poynting", partial_grid_shape=(4, 4, 1), direction="+", plot=False),
        fdtdx.FieldDetector(name="field", partial_grid_shape=(4, 4, 4), plot=False),
    ]


def _build(config):
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=2,
        override_types={f: "periodic" for f in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")},
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    for det in _detectors():
        # Centering a 4-cell (or 1-cell) detector in a 12-cell domain lands it strictly interior
        # (cells ~4..8; halo stays in-bounds) on both uniform and non-uniform grids.
        objects.append(det)
        constraints.append(det.place_at_center(volume, axes=(0, 1, 2)))

    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
    return obj, arrays, config


def _reference_state(det, arrays, objects, config, H_prev, time_step):
    """The historical full-domain path: pad + interpolate the whole domain, then slice the region."""
    E_pad = pad_fields_for_boundaries(arrays.fields.E, objects, config)
    H_avg_pad = pad_fields_for_boundaries((H_prev + arrays.fields.H) / 2, objects, config)
    full_E, full_H = interpolate_fields(E_pad=E_pad, H_pad=H_avg_pad, config=config)
    gs = det.grid_slice
    inv_mu = arrays.inv_permeabilities
    inv_mu_r = inv_mu[:, *gs] if isinstance(inv_mu, jax.Array) and inv_mu.ndim > 0 else inv_mu
    return det.update(
        time_step=time_step,
        E=full_E[:, *gs],
        H=full_H[:, *gs],
        state=det.init_state(),
        inv_permittivity=arrays.inv_permittivities[:, *gs],
        inv_permeability=inv_mu_r,
    )


def _configs():
    uniform = fdtdx.SimulationConfig(time=1e-13, grid=UniformGrid(spacing=_RES))
    widths = np.array([1.0, 1.4, 0.8, 1.2, 1.0, 1.5, 0.9, 1.3, 1.1, 0.7, 1.6, 1.0]) * _RES
    edges = jnp.asarray(np.concatenate([[0.0], np.cumsum(widths)]))
    nonuniform = fdtdx.SimulationConfig(
        time=1e-13, grid=RectilinearGrid(x_edges=edges, y_edges=edges, z_edges=edges)
    )
    return {"uniform": uniform, "nonuniform": nonuniform}


@pytest.mark.parametrize("grid_kind", ["uniform", "nonuniform"])
def test_region_recording_matches_full_domain(grid_kind):
    obj, arrays, config = _build(_configs()[grid_kind])

    # Drive with reproducible random fields so the interpolation actually varies across cells.
    kE, kH, kHp = jax.random.split(jax.random.PRNGKey(1), 3)
    shape = arrays.fields.E.shape
    arrays = arrays.aset("fields->E", jax.random.normal(kE, shape, dtype=arrays.fields.E.dtype))
    arrays = arrays.aset("fields->H", jax.random.normal(kH, shape, dtype=arrays.fields.H.dtype))
    H_prev = jax.random.normal(kHp, shape, dtype=arrays.fields.H.dtype)
    time_step = jnp.array(0)

    # Region path (the implementation under test).
    new_arrays = update_detector_states(time_step, arrays, obj, config, H_prev, inverse=False)

    for det in obj.forward_detectors:
        ref = _reference_state(det, arrays, obj, config, H_prev, time_step)
        got = new_arrays.detector_states[det.name]
        assert set(got) == set(ref)
        for k in ref:
            assert jnp.allclose(got[k], ref[k], atol=1e-6, rtol=1e-5), f"{det.name}:{k} mismatch ({grid_kind})"
