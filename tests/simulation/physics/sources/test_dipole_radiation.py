"""Physics simulation tests: dipole radiation in free space.

Tests that the PointDipoleSource radiates correctly:
  1a. Total radiated power is positive and steady-state.
  1b. Radiation pattern has correct symmetry for a z-polarized dipole.
  1d. Tilted dipole (azimuth_angle) rotates the radiation pattern.
  1c. Rotating polarization axis rotates the radiation pattern.

Domain layout (50 nm resolution, PML on all 6 faces):
  60×60×60 cells total (3 µm cube):
    cells  0–9  and 50–59 : PML (0.5 µm each side)
    cells 10–49 : active region (2 µm)
    Dipole at center: cell (30, 30, 30)

Detectors: PoyntingFluxDetectors on 6 faces of a box around the dipole,
  each 10 cells from the source (well outside the reactive near field).
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
_RESOLUTION = 50e-9  # 20 cells/λ
_PML_CELLS = 10
_DOMAIN = 60 * _RESOLUTION  # 60 cells = 3 µm
_SIM_TIME = 120e-15  # ~36 optical periods

_CENTER = 30  # grid index of domain center
_DET_OFFSET = 10  # detector distance from dipole (in cells)

_TOLERANCE = 0.10  # 10% for symmetry tests (grid dispersion + near-field effects)

# Time-averaging
_DT_APPROX = 0.99 * _RESOLUTION / (3e8 * np.sqrt(3))
_STEPS_PER_PERIOD = int(round(_WAVELENGTH / (3e8 * _DT_APPROX)))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_dipole_domain(polarization=2, azimuth_angle=0.0, elevation_angle=0.0):
    """Build domain with a point dipole at center and PML on all faces."""
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
        azimuth_angle=azimuth_angle,
        elevation_angle=elevation_angle,
        amplitude=1.0,
    )
    constraints.extend(
        [
            source.place_at_center(volume, axes=(0, 1, 2)),
        ]
    )
    objects.append(source)

    return objects, constraints, config, volume


def _add_flux_det(name, volume, objects, constraints, axis, side):
    """Add a PoyntingFluxDetector on one face of the detection box.

    axis: 0, 1, or 2  (x, y, z)
    side: "+" or "-"
    """
    # The detector is a plane perpendicular to `axis`.
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

    # Size: fill the other two axes, position along `axis`
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


# ── Tests ────────────────────────────────────────────────────────────────────


def test_dipole_total_radiated_power():
    """A z-polarized electric dipole radiates positive, steady-state power."""
    objects, constraints, config, volume = _build_dipole_domain(polarization=2)

    # Add 6 flux detectors forming a box around the dipole
    for axis in range(3):
        for side in ("+", "-"):
            name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
            _add_flux_det(name, volume, objects, constraints, axis, side)

    arrays = _run(objects, constraints, config)

    total_flux = 0.0
    for axis in range(3):
        for side in ("+", "-"):
            name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
            f = _mean_flux(arrays, name)
            total_flux += f

    assert total_flux > 0, f"Total radiated power should be positive, got {total_flux}"

    # Check steady-state: std/mean of total flux over averaging window should be small
    # We check individual detector stability instead
    for axis in range(3):
        name = f"flux_{['x', 'y', 'z'][axis]}_+"
        flux_arr = np.array(arrays.detector_states[name]["poynting_flux"][:, 0])
        last = flux_arr[-_N_AVG_STEPS:]
        mean = np.mean(np.abs(last))
        if mean > 1e-30:
            variation = np.std(last) / mean
            assert variation < 0.5, f"Flux through {name} is not steady-state: std/mean={variation:.3f}"


def test_dipole_radiation_pattern_symmetry():
    """Z-polarized dipole has azimuthal symmetry: S_+x ≈ S_-x ≈ S_+y ≈ S_-y.

    Flux through ±z faces (along dipole axis) should be smaller than through
    the transverse faces.
    """
    objects, constraints, config, volume = _build_dipole_domain(polarization=2)

    for axis in range(3):
        for side in ("+", "-"):
            name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
            _add_flux_det(name, volume, objects, constraints, axis, side)

    arrays = _run(objects, constraints, config)

    fluxes = {}
    for axis in range(3):
        for side in ("+", "-"):
            name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
            fluxes[name] = _mean_flux(arrays, name)

    S_xp = fluxes["flux_x_+"]
    S_xm = fluxes["flux_x_-"]
    S_yp = fluxes["flux_y_+"]
    S_zp = fluxes["flux_z_+"]

    # Forward/backward symmetry in x
    assert abs(S_xp - S_xm) / max(abs(S_xp), 1e-30) < _TOLERANCE, f"x-symmetry broken: S_+x={S_xp:.4e}, S_-x={S_xm:.4e}"

    # Azimuthal symmetry: x and y faces should be equal
    assert abs(S_xp - S_yp) / max(abs(S_xp), 1e-30) < _TOLERANCE, (
        f"Azimuthal symmetry broken: S_+x={S_xp:.4e}, S_+y={S_yp:.4e}"
    )

    # Less radiation along dipole axis (z) than perpendicular (x)
    assert abs(S_zp) < abs(S_xp), f"Expected less axial radiation: S_+z={S_zp:.4e} should be < S_+x={S_xp:.4e}"


def test_dipole_polarization_axes():
    """Rotating dipole polarization rotates the radiation pattern.

    Total power should be the same for x, y, z polarized dipoles.
    The minimum-flux face should be along the dipole axis.
    """
    total_powers = {}
    min_flux_axis = {}

    for pol in range(3):
        objects, constraints, config, volume = _build_dipole_domain(polarization=pol)

        for axis in range(3):
            for side in ("+", "-"):
                name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
                _add_flux_det(name, volume, objects, constraints, axis, side)

        arrays = _run(objects, constraints, config)

        face_fluxes = []
        total = 0.0
        for axis in range(3):
            axis_flux = 0.0
            for side in ("+", "-"):
                name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
                f = _mean_flux(arrays, name)
                total += f
                axis_flux += abs(f)
            face_fluxes.append(axis_flux)

        total_powers[pol] = total
        min_flux_axis[pol] = int(np.argmin(face_fluxes))

    # Total power should be similar across all three polarizations
    powers = list(total_powers.values())
    mean_power = np.mean(powers)
    for pol, p in total_powers.items():
        rel_err = abs(p - mean_power) / max(abs(mean_power), 1e-30)
        assert rel_err < _TOLERANCE, (
            f"Polarization {pol}: power={p:.4e} differs from mean={mean_power:.4e} by {rel_err:.2%}"
        )

    # Minimum flux face should be along the dipole axis
    for pol in range(3):
        assert min_flux_axis[pol] == pol, f"Polarization {pol}: min flux on axis {min_flux_axis[pol]}, expected {pol}"


def test_dipole_tilted_radiation_pattern():
    """A z-polarized dipole tilted 45° via azimuth rotates the radiation pattern.

    With polarization=2 and azimuth_angle=45°, the dipole orientation lies in
    the xz-plane at 45° from both axes.  Physics predictions:
      - ±y faces see the highest flux (fully in the equatorial plane).
      - ±x and ±z faces see equal flux (both at 45° from the dipole axis).
      - Total radiated power matches an axis-aligned dipole.
    """
    # ── Tilted dipole ────────────────────────────────────────────────────────
    objects, constraints, config, volume = _build_dipole_domain(
        polarization=2,
        azimuth_angle=45.0,
    )
    for axis in range(3):
        for side in ("+", "-"):
            name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
            _add_flux_det(name, volume, objects, constraints, axis, side)

    arrays_tilted = _run(objects, constraints, config)

    tilted_fluxes = {}
    for axis in range(3):
        for side in ("+", "-"):
            name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
            tilted_fluxes[name] = _mean_flux(arrays_tilted, name)

    # Sum flux per axis pair (|+| + |-|)
    flux_x = abs(tilted_fluxes["flux_x_+"]) + abs(tilted_fluxes["flux_x_-"])
    flux_y = abs(tilted_fluxes["flux_y_+"]) + abs(tilted_fluxes["flux_y_-"])
    flux_z = abs(tilted_fluxes["flux_z_+"]) + abs(tilted_fluxes["flux_z_-"])

    # ±x and ±z faces should receive roughly equal flux (both at 45° from dipole)
    assert abs(flux_x - flux_z) / max(flux_x, 1e-30) < _TOLERANCE, (
        f"Tilted dipole: x/z symmetry broken: flux_x={flux_x:.4e}, flux_z={flux_z:.4e}"
    )

    # ±y faces should see more flux than ±x (y is equatorial, x is at 45°)
    assert flux_y > flux_x, f"Tilted dipole: expected flux_y > flux_x, got flux_y={flux_y:.4e}, flux_x={flux_x:.4e}"

    # ── Compare total power to axis-aligned dipole ───────────────────────────
    objects_ref, constraints_ref, config_ref, volume_ref = _build_dipole_domain(polarization=2)
    for axis in range(3):
        for side in ("+", "-"):
            name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
            _add_flux_det(name, volume_ref, objects_ref, constraints_ref, axis, side)

    arrays_ref = _run(objects_ref, constraints_ref, config_ref)

    total_tilted = sum(tilted_fluxes.values())
    total_ref = sum(_mean_flux(arrays_ref, f"flux_{['x', 'y', 'z'][a]}_{s}") for a in range(3) for s in ("+", "-"))

    rel_err = abs(total_tilted - total_ref) / max(abs(total_ref), 1e-30)
    assert rel_err < _TOLERANCE, (
        f"Total power differs: tilted={total_tilted:.4e}, ref={total_ref:.4e}, rel_err={rel_err:.2%}"
    )
