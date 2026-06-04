"""Physics test: automatic config.symmetry reproduces the full-domain result after unfolding.

This is the automated counterpart of ``test_quarter_domain_symmetry.py``. There, the quarter
domain (PEC min_y, PMC min_z) and its unfolding were built by hand. Here we build the *full*
model once, run it both with ``config.symmetry=(0, 0, 0)`` (reference) and with
``config.symmetry=(0, -1, 1)`` (PEC y-plane, PMC z-plane), and check that
``fdtdx.unfold_detector_states`` of the reduced run reproduces the reference:

* the unfolded Ey phasor field matches the full-domain field element-wise, and
* the unfolded scalar Poynting flux matches the full-domain total power (the x4 quarter-domain
  scaling is applied automatically).

An Ey-polarized +x plane wave is even about both the PEC y-plane (Ey normal) and the PMC z-plane
(Ey tangential), so both walls act as perfect mirrors for this polarization.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

_WAVELENGTH = 1e-6
_RESOLUTION = 25e-9  # 40 cells / λ
_DOMAIN_YZ = 1e-6  # full transverse extent (40 cells, even -> clean halving)
_DOMAIN_X = 5e-6
_PML_CELLS = 10
_SOURCE_X = _PML_CELLS + 2
_DET_X = _SOURCE_X + 20
_SIM_TIME = 80e-15

_DT = 0.99 * _RESOLUTION / (3e8 * np.sqrt(3))
_STEPS_PER_PERIOD = max(1, round(_WAVELENGTH / (3e8 * _DT)))
_N_AVG = 10 * _STEPS_PER_PERIOD


def _build(symmetry):
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=_SIM_TIME,
        dtype=jnp.float32,
        symmetry=symmetry,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(partial_real_shape=(_DOMAIN_X, _DOMAIN_YZ, _DOMAIN_YZ))
    objects.append(volume)

    # Periodic in y,z (infinite uniform slab); PML in x. Under symmetry the min_y/min_z periodic
    # boundaries are replaced by the PEC/PMC walls, leaving periodic only on the far side — exactly
    # the validated quarter-domain configuration.
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_y": "periodic",
            "max_y": "periodic",
            "min_z": "periodic",
            "max_z": "periodic",
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(1, None, None),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(0, 1, 0),
        normalize_by_energy=False,
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(1, 2)),
            source.place_at_center(volume, axes=(1, 2)),
            source.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_SOURCE_X,)),
        ]
    )
    objects.append(source)

    ey = fdtdx.PhasorDetector(
        name="ey",
        partial_grid_shape=(1, None, None),
        wave_characters=(wave,),
        components=("Ey",),
        reduce_volume=False,
        plot=False,
    )
    constraints.extend(
        [
            ey.same_size(volume, axes=(1, 2)),
            ey.place_at_center(volume, axes=(1, 2)),
            ey.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_DET_X,)),
        ]
    )
    objects.append(ey)

    flux = fdtdx.PoyntingFluxDetector(
        name="flux",
        partial_grid_shape=(1, None, None),
        direction="+",
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            flux.same_size(volume, axes=(1, 2)),
            flux.place_at_center(volume, axes=(1, 2)),
            flux.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_DET_X,)),
        ]
    )
    objects.append(flux)

    return objects, constraints, config


def _run(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)
    return obj_container, arrays, config


def test_unfolded_phasor_matches_full_domain():
    _oc_full, arrays_full, _cfg_full = _run(*_build((0, 0, 0)))
    full_ey = np.asarray(arrays_full.detector_states["ey"]["phasor"][0, 0, 0, 0, :, :])

    oc_sym, arrays_sym, cfg_sym = _run(*_build((0, -1, 1)))
    unfolded = fdtdx.unfold_detector_states(arrays_sym, oc_sym, cfg_sym)
    unf_ey = np.asarray(unfolded.detector_states["ey"]["phasor"][0, 0, 0, 0, :, :])

    assert unf_ey.shape == full_ey.shape, f"{unf_ey.shape} != {full_ey.shape}"
    assert np.max(np.abs(full_ey)) > 1e-30, "reference Ey is zero — wave not launched"

    ref_amp = float(np.mean(np.abs(full_ey)))
    rel_err = float(np.mean(np.abs(unf_ey - full_ey))) / ref_amp
    assert rel_err < 0.03, f"unfolded vs full Ey mismatch: mean|Δ|/mean|full| = {rel_err:.4f}"


def test_unfolded_flux_recovers_full_power():
    _oc_full, arrays_full, _cfg_full = _run(*_build((0, 0, 0)))
    full_flux = float(np.mean(np.asarray(arrays_full.detector_states["flux"]["poynting_flux"][-_N_AVG:, 0])))

    oc_sym, arrays_sym, cfg_sym = _run(*_build((0, -1, 1)))
    unfolded = fdtdx.unfold_detector_states(arrays_sym, oc_sym, cfg_sym)
    unf_flux = float(np.mean(np.asarray(unfolded.detector_states["flux"]["poynting_flux"][-_N_AVG:, 0])))

    assert abs(full_flux) > 1e-30, "reference flux is zero — wave not launched"
    rel_err = abs(unf_flux - full_flux) / abs(full_flux)
    assert rel_err < 0.05, (
        f"unfolded quarter-domain flux x4 = {unf_flux:.4e} vs full {full_flux:.4e} (rel_err={rel_err:.3f})"
    )
