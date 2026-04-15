"""Physics simulation tests: magnetic dipole source.

  3a. Magnetic dipole radiates positive power in free space.
  3b. Electric and magnetic dipoles have comparable total power and
      consistent radiation patterns (E↔H duality).

Domain: 60×60×60 cells, PML on all faces.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
_RESOLUTION = 50e-9
_PML_CELLS = 10
_DOMAIN = 60 * _RESOLUTION
_SIM_TIME = 120e-15

_CENTER = 30
_DET_OFFSET = 10

_DT_APPROX = 0.99 * _RESOLUTION / (3e8 * np.sqrt(3))
_STEPS_PER_PERIOD = int(round(_WAVELENGTH / (3e8 * _DT_APPROX)))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_dipole(source_type="electric", polarization=2):
    config = fdtdx.SimulationConfig(
        resolution=_RESOLUTION,
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(_DOMAIN, _DOMAIN, _DOMAIN),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    source = fdtdx.PointDipoleSource(
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        polarization=polarization,
        source_type=source_type,
        amplitude=1.0,
    )
    constraints.append(source.place_at_center(volume, axes=(0, 1, 2)))
    objects.append(source)

    return objects, constraints, config, volume


def _add_flux_det(name, volume, objects, constraints, axis, side):
    grid_shapes = [None, None, None]
    grid_shapes[axis] = 1
    det = fdtdx.PoyntingFluxDetector(
        name=name,
        partial_grid_shape=tuple(grid_shapes),
        direction=side,
        fixed_propagation_axis=axis,
        reduce_volume=True,
        plot=False,
    )
    other_axes = [a for a in range(3) if a != axis]
    constraints.extend(
        [
            det.same_size(volume, axes=tuple(other_axes)),
            det.place_at_center(volume, axes=tuple(other_axes)),
        ]
    )
    coord = _CENTER + _DET_OFFSET if side == "+" else _CENTER - _DET_OFFSET
    constraints.append(
        det.set_grid_coordinates(axes=(axis,), sides=("-",), coordinates=(coord,)),
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


def _total_outward_flux(arrays):
    total = 0.0
    for axis in range(3):
        for side in ("+", "-"):
            name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
            total += _mean_flux(arrays, name)
    return total


def _run_with_all_detectors(source_type, polarization):
    objects, constraints, config, volume = _build_dipole(source_type, polarization)
    for axis in range(3):
        for side in ("+", "-"):
            name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
            _add_flux_det(name, volume, objects, constraints, axis, side)
    arrays = _run(objects, constraints, config)
    return arrays


# ── Tests ────────────────────────────────────────────────────────────────────


def test_magnetic_dipole_radiates():
    """A z-polarized magnetic dipole radiates positive total power."""
    arrays = _run_with_all_detectors("magnetic", polarization=2)
    P = _total_outward_flux(arrays)
    assert P > 0, f"Magnetic dipole total power should be positive, got {P}"


def test_magnetic_dipole_symmetry():
    """Z-polarized magnetic dipole has azimuthal symmetry like electric dipole."""
    arrays = _run_with_all_detectors("magnetic", polarization=2)

    S_xp = _mean_flux(arrays, "flux_x_+")
    S_xm = _mean_flux(arrays, "flux_x_-")
    S_yp = _mean_flux(arrays, "flux_y_+")
    S_zp = _mean_flux(arrays, "flux_z_+")

    # Forward/backward symmetry
    tol = 0.05
    assert abs(S_xp - S_xm) / max(abs(S_xp), 1e-30) < tol, (
        f"Magnetic dipole x-symmetry broken: S_+x={S_xp:.4e}, S_-x={S_xm:.4e}"
    )

    # Azimuthal symmetry
    assert abs(S_xp - S_yp) / max(abs(S_xp), 1e-30) < tol, (
        f"Magnetic dipole azimuthal symmetry broken: S_+x={S_xp:.4e}, S_+y={S_yp:.4e}"
    )

    # Less radiation along dipole axis
    assert abs(S_zp) < abs(S_xp), f"Expected less axial radiation: S_+z={S_zp:.4e} < S_+x={S_xp:.4e}"


def test_magnetic_vs_electric_duality():
    """Electric and magnetic dipoles both radiate with matching pattern structure.

    Both z-polarized: both should have max radiation in xy-plane and
    min along z. Total powers can differ but should be same order of magnitude.
    """
    arrays_e = _run_with_all_detectors("electric", polarization=2)
    arrays_m = _run_with_all_detectors("magnetic", polarization=2)

    P_e = _total_outward_flux(arrays_e)
    P_m = _total_outward_flux(arrays_m)

    assert P_e > 0
    assert P_m > 0

    # Same order of magnitude (within 10×)
    ratio = max(P_e, P_m) / min(P_e, P_m)
    assert ratio < 10, f"Electric/magnetic power ratio too large: P_e={P_e:.4e}, P_m={P_m:.4e}, ratio={ratio:.1f}"

    # Both should have min flux along dipole axis (z)
    for label, arrays in [("electric", arrays_e), ("magnetic", arrays_m)]:
        face_fluxes = []
        for axis in range(3):
            axis_flux = sum(abs(_mean_flux(arrays, f"flux_{['x', 'y', 'z'][axis]}_{s}")) for s in ("+", "-"))
            face_fluxes.append(axis_flux)
        min_axis = int(np.argmin(face_fluxes))
        assert min_axis == 2, f"{label} dipole: min flux on axis {min_axis}, expected 2 (z)"
