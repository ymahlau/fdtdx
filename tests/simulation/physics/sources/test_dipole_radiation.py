"""Physics simulation tests: dipole radiation in free space.

Tests that the PointDipoleSource radiates correctly:
  1a. Total radiated power is positive and steady-state.
  1b. Radiation pattern has correct symmetry for a z-polarized dipole.
  1d. Tilted dipole (azimuth_angle) rotates the radiation pattern.
  1c. Rotating polarization axis rotates the radiation pattern.

Domain layout (50 nm resolution, PML on all 6 faces):
  60x60x60 cells total (3 µm cube):
    cells  0-9  and 50-59 : PML (0.5 µm each side)
    cells 10-49 : active region (2 µm)
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
_STEPS_PER_PERIOD = round(_WAVELENGTH / (3e8 * _DT_APPROX))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_dipole_domain(polarization=2, azimuth_angle=0.0, elevation_angle=0.0):
    """Build domain with a point dipole at center and PML on all faces."""
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
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


# ── Analytic face-flux pattern (Hertzian dipole) ─────────────────────────────
#
# A Hertzian dipole's *time-averaged* radial Poynting flux is exactly
# C·sin²ψ/r² (ψ = angle from the dipole axis); the reactive near field carries
# no net real power, so this holds even at the modest kr of these detectors.
# We integrate that pattern over the **actual detector geometry**: each face is
# a plane perpendicular to its axis at perpendicular distance _DET_OFFSET cells,
# spanning ±_CENTER cells (the full domain half-width, since the detectors use
# `same_size(volume)`).  The constant C cancels in every ratio below.


def _face_flux_sin2(d_hat, normal_axis, d_cells=_DET_OFFSET, w_cells=_CENTER, n=601):
    """Analytic flux through one detector face for a dipole oriented ``d_hat``."""
    t = np.linspace(-float(w_cells), float(w_cells), n)
    A, B = np.meshgrid(t, t, indexing="ij")
    coords = [None, None, None]
    other = [a for a in range(3) if a != normal_axis]
    coords[normal_axis] = np.full_like(A, float(d_cells))
    coords[other[0]] = A
    coords[other[1]] = B
    X, Y, Z = coords
    r = np.sqrt(X**2 + Y**2 + Z**2)
    cos_psi = (X * d_hat[0] + Y * d_hat[1] + Z * d_hat[2]) / r
    sin2 = 1.0 - cos_psi**2
    integrand = sin2 / r**2 * (d_cells / r)  # (r_hat · n_hat) = d_cells / r
    return float(np.sum(integrand)) * (t[1] - t[0]) ** 2


def _axis_pair_flux(d_hat, axis):
    """Flux for one axis pair (both faces) = 2x one face."""
    return 2.0 * _face_flux_sin2(d_hat, axis)


def _perp_over_axial(d_hat, axial_axis):
    """Derived (perpendicular-pair-sum) / (axial-pair) face-flux ratio."""
    perp = sum(_axis_pair_flux(d_hat, a) for a in range(3) if a != axial_axis)
    return perp / _axis_pair_flux(d_hat, axial_axis)


# Derived ratio for an axis-aligned dipole's sin²θ pattern over this geometry.
# (≈ 2.68 here: d_cells=10, w_cells=30.  A true cube w=d would give ≈ 5.7;
# infinite faces → larger — the value depends on the actual face extent.)
_EQ_AXIAL_RATIO = _perp_over_axial((0.0, 0.0, 1.0), axial_axis=2)
_PATTERN_TOL = 0.15  # 15 %: grid dispersion + PML inset shift the measured value up ~7 %


def _add_closed_cube_det(name, volume, objects, constraints, axis, side, half_side):
    """One face of a TRUE closed cube (half-side ``half_side`` cells) about the dipole.

    Unlike the full-domain `_add_flux_det` planes, these faces span only
    ±half_side in the transverse directions, so the 6-face sum is a genuine
    closed-surface integral — the conserved radiated power (independent of
    ``half_side`` in a lossless interior).
    """
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
    for oa in (a for a in range(3) if a != axis):
        constraints.append(
            det.set_grid_coordinates(
                axes=(oa, oa),
                sides=("-", "+"),
                coordinates=(_CENTER - half_side, _CENTER + half_side),
            )
        )
    coord = _CENTER + half_side if side == "+" else _CENTER - half_side
    constraints.append(det.set_grid_coordinates(axes=(axis,), sides=("-",), coordinates=(coord,)))
    objects.append(det)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_dipole_total_radiated_power():
    """A z-polarized electric dipole radiates positive, steady-state power.

    Two independent checks back up the magnitude:
      * Steady state — the *cycle-averaged* total flux (averaged over each
        optical period) is flat to < 5 % across an integer number of periods.
      * Surface independence — the power through two concentric **true closed
        cubes** of different radii agrees to < 15 %.  A genuine radiated power
        is independent of the enclosing-surface radius (energy conservation),
        so this confirms the measured number is real, not coincidental.
    """
    objects, constraints, config, volume = _build_dipole_domain(polarization=2)

    # 6 full-domain flux planes (the primary total-power surface)
    for axis in range(3):
        for side in ("+", "-"):
            name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
            _add_flux_det(name, volume, objects, constraints, axis, side)

    # Two concentric true closed cubes for the surface-independence check.
    for hs in (8, 12):
        for axis in range(3):
            for side in ("+", "-"):
                _add_closed_cube_det(
                    f"cube{hs}_{['x', 'y', 'z'][axis]}_{side}",
                    volume,
                    objects,
                    constraints,
                    axis,
                    side,
                    hs,
                )

    arrays = _run(objects, constraints, config)

    total_flux = 0.0
    for axis in range(3):
        for side in ("+", "-"):
            name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
            total_flux += _mean_flux(arrays, name)

    assert total_flux > 0, f"Total radiated power should be positive, got {total_flux}"

    # --- Steady state: cycle-averaged total flux over an integer # of periods ---
    # The instantaneous Poynting flux oscillates at 2ω, so we average over each
    # optical period first (steps_per_period ≈ 35.0 here, so period chunks are
    # essentially exact), then require the per-period means to be flat.
    total_series = sum(
        np.array(arrays.detector_states[f"flux_{['x', 'y', 'z'][axis]}_{side}"]["poynting_flux"][:, 0])
        for axis in range(3)
        for side in ("+", "-")
    )
    n_periods = 10
    period_block = total_series[-n_periods * _STEPS_PER_PERIOD :].reshape(n_periods, _STEPS_PER_PERIOD)
    cycle_means = period_block.mean(axis=1)
    variation = np.std(cycle_means) / np.abs(np.mean(cycle_means))
    assert variation < 0.05, (
        f"Total radiated power is not steady-state: cycle-averaged std/mean={variation:.4f} "
        f"over {n_periods} optical periods"
    )

    # --- Magnitude: radius independence of the closed-cube radiated power ---
    cube_powers = {}
    for hs in (8, 12):
        cube_powers[hs] = sum(
            _mean_flux(arrays, f"cube{hs}_{['x', 'y', 'z'][axis]}_{side}") for axis in range(3) for side in ("+", "-")
        )
    p_small, p_large = cube_powers[8], cube_powers[12]
    assert p_small > 0 and p_large > 0, f"Closed-cube powers must be positive: {cube_powers}"
    radius_mismatch = abs(p_small - p_large) / max(abs(p_small), abs(p_large))
    assert radius_mismatch < 0.15, (
        f"Radiated power is not surface-independent: closed cube (half-side 8) "
        f"P={p_small:.4e} vs (half-side 12) P={p_large:.4e}, mismatch={radius_mismatch:.2%}"
    )


def test_dipole_radiation_pattern_symmetry():
    """Z-polarized dipole: azimuthal symmetry + quantitative sin²θ flux ratio.

    Besides the azimuthal/forward-backward symmetry, the equatorial (±x, ±y)
    vs axial (±z) face-flux ratio is compared against the value derived by
    integrating the dipole's exact sin²θ pattern over the actual detector-box
    geometry (``_EQ_AXIAL_RATIO`` ≈ 2.68).
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

    # Forward/backward symmetry in x
    assert abs(S_xp - S_xm) / max(abs(S_xp), 1e-30) < _TOLERANCE, f"x-symmetry broken: S_+x={S_xp:.4e}, S_-x={S_xm:.4e}"

    # Azimuthal symmetry: x and y faces should be equal
    assert abs(S_xp - S_yp) / max(abs(S_xp), 1e-30) < _TOLERANCE, (
        f"Azimuthal symmetry broken: S_+x={S_xp:.4e}, S_+y={S_yp:.4e}"
    )

    # Quantitative equatorial/axial flux ratio vs the derived sin²θ value.
    S_eq = sum(abs(fluxes[f"flux_{['x', 'y', 'z'][a]}_{s}"]) for a in (0, 1) for s in ("+", "-"))
    S_ax = sum(abs(fluxes[f"flux_z_{s}"]) for s in ("+", "-"))
    ratio = S_eq / S_ax
    rel_err = abs(ratio - _EQ_AXIAL_RATIO) / _EQ_AXIAL_RATIO
    assert rel_err < _PATTERN_TOL, (
        f"Equatorial/axial flux ratio S_eq/S_ax={ratio:.3f} differs from the derived "
        f"sin²θ value {_EQ_AXIAL_RATIO:.3f} by {rel_err:.2%}"
    )


def test_dipole_polarization_axes():
    """Rotating dipole polarization rotates the radiation pattern.

    Total power should be the same for x, y, z polarized dipoles, and for each
    polarization the perpendicular/axial face-flux ratio should match the
    derived sin²θ value (``_EQ_AXIAL_RATIO``) — a quantitative replacement for
    the old "min-flux face is along the dipole axis" check.
    """
    total_powers = {}
    perp_axial_ratio = {}

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
        perp = sum(face_fluxes[a] for a in range(3) if a != pol)
        perp_axial_ratio[pol] = perp / face_fluxes[pol]

    # Total power should be similar across all three polarizations
    powers = list(total_powers.values())
    mean_power = np.mean(powers)
    for pol, p in total_powers.items():
        rel_err = abs(p - mean_power) / max(abs(mean_power), 1e-30)
        assert rel_err < _TOLERANCE, (
            f"Polarization {pol}: power={p:.4e} differs from mean={mean_power:.4e} by {rel_err:.2%}"
        )

    # Perpendicular/axial flux ratio should match the derived sin²θ value for
    # every polarization (the pattern rotates rigidly with the dipole axis).
    for pol in range(3):
        rel_err = abs(perp_axial_ratio[pol] - _EQ_AXIAL_RATIO) / _EQ_AXIAL_RATIO
        assert rel_err < _PATTERN_TOL, (
            f"Polarization {pol}: perpendicular/axial flux ratio={perp_axial_ratio[pol]:.3f} "
            f"differs from derived sin²θ value {_EQ_AXIAL_RATIO:.3f} by {rel_err:.2%}"
        )


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

    # Quantitative flux_y/flux_x: derive it for the 45°-tilted dipole axis
    # (x+z)/√2 integrated over the actual detector geometry (≈ 1.14 here).
    d_hat_tilt = (1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0))
    expected_yx = _axis_pair_flux(d_hat_tilt, 1) / _axis_pair_flux(d_hat_tilt, 0)
    measured_yx = flux_y / flux_x
    rel_err = abs(measured_yx - expected_yx) / expected_yx
    assert rel_err < _PATTERN_TOL, (
        f"Tilted dipole flux_y/flux_x={measured_yx:.3f} differs from the derived "
        f"value {expected_yx:.3f} (45° tilt) by {rel_err:.2%}"
    )

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
