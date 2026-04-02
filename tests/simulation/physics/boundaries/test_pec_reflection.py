"""Physics simulation test: PEC total reflection.

A plane wave (z-propagating, x-polarized) hits a PEC wall at z-max.
PML at z-min absorbs the source, PEC at z-max reflects all energy.
Periodic boundaries in x and y.

Domain layout (50 nm resolution = 20 cells/λ, z is propagation axis):
  80 cells total in z (4 µm):
    cells  0– 9 : PML (0.5 µm)
    cells 10–79 : active region (3.5 µm), PEC terminates at z-max

  Source      : z-index 12 (2 cells into active region)
  Detector    : z-index 40 (midway in active region)

Metric:
  In steady state, a PEC wall produces a perfect standing wave.
  The time-averaged Poynting flux in a standing wave is zero.
  Using two-run normalization (reference = PML everywhere):
    T = S_standing / S_reference ≈ 0
    R = 1 − T ≈ 1.0

Tolerance: 5 % relative error on R.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
_RESOLUTION = 50e-9  # 20 cells/λ in vacuum
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION  # 3 cells, periodic
_DOMAIN_Z = 4e-6  # 80 cells

_SOURCE_Z = _PML_CELLS + 2  # = 12
_DET_Z = 40  # detector in middle of active region

_SIM_TIME = 120e-15  # 120 fs ≈ 36 optical periods
_TOLERANCE = 0.05

# Time-averaging
_DT_APPROX = 0.99 * _RESOLUTION / (3e8 * np.sqrt(3))
_STEPS_PER_PERIOD = int(round(_WAVELENGTH / (3e8 * _DT_APPROX)))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_base(z_max_type="pml"):
    """Build domain with periodic xy, PML z-min, and configurable z-max."""
    config = fdtdx.SimulationConfig(
        resolution=_RESOLUTION,
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_x": "periodic",
            "max_x": "periodic",
            "min_y": "periodic",
            "max_y": "periodic",
            "max_z": z_max_type,
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)

    return objects, constraints, config, volume, wave


def _add_flux_det(name, z_idx, volume, objects, constraints):
    det = fdtdx.PoyntingFluxDetector(
        name=name,
        partial_grid_shape=(None, None, 1),
        direction="+",
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0, 1)),
            det.place_at_center(volume, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(z_idx,)),
        ]
    )
    objects.append(det)


def _run(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)
    return arrays


def _mean_flux(arrays, name):
    flux = np.array(arrays.detector_states[name]["poynting_flux"][:, 0])
    return float(np.mean(flux[-_N_AVG_STEPS:]))


# ── Tests ────────────────────────────────────────────────────────────────────


def test_pec_total_reflection():
    """PEC wall reflects 100 % of incident power (R ≈ 1.0 within 5 %).

    Two-run normalization:
      Reference (PML everywhere): S₀ = forward flux through detector
      PEC run (PEC at z-max):     S  = net flux (≈ 0 due to standing wave)
      T = S / S₀ ≈ 0  →  R = 1 − T ≈ 1.0
    """
    # Reference: PML on all z-faces
    obj_ref, con_ref, cfg_ref, vol_ref, _ = _build_base(z_max_type="pml")
    _add_flux_det("flux", _DET_Z, vol_ref, obj_ref, con_ref)
    S_ref = _mean_flux(_run(obj_ref, con_ref, cfg_ref), "flux")

    # PEC run
    obj_pec, con_pec, cfg_pec, vol_pec, _ = _build_base(z_max_type="pec")
    _add_flux_det("flux", _DET_Z, vol_pec, obj_pec, con_pec)
    S_pec = _mean_flux(_run(obj_pec, con_pec, cfg_pec), "flux")

    assert S_ref > 0, f"Reference flux is zero/negative: {S_ref}"

    T_measured = S_pec / S_ref
    R_measured = 1.0 - T_measured
    R_analytic = 1.0

    rel_err = abs(R_measured - R_analytic) / R_analytic
    assert rel_err < _TOLERANCE, (
        f"PEC reflection: R_measured={R_measured:.4f}, R_analytic={R_analytic:.4f}, "
        f"T_measured={T_measured:.4f}, relative error={rel_err:.3f}"
    )
