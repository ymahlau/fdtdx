"""Physics simulation tests: magnetic dipole source.

  3a. Magnetic dipole radiates the same total power as the electric dipole
      (within a few %), as vacuum E↔H duality requires.
  3b. Electric and magnetic dipoles are E↔H duals: equal total power (< 1.1x),
      matching sin²θ pattern structure, and swapped near-field roles
      (electric → E-dominant, magnetic → H-dominant on the dipole axis).

Domain: 60x60x60 cells, PML on all faces.
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
_STEPS_PER_PERIOD = round(_WAVELENGTH / (3e8 * _DT_APPROX))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_dipole(source_type="electric", polarization=2):
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


def _add_point_phasor_det(name, volume, objects, constraints, axis, coord):
    """Single-cell phasor detector recording all 6 field components."""
    det = fdtdx.PhasorDetector(
        name=name,
        partial_grid_shape=(1, 1, 1),
        wave_characters=(fdtdx.WaveCharacter(wavelength=_WAVELENGTH),),
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        reduce_volume=True,
        plot=False,
    )
    other = [a for a in range(3) if a != axis]
    constraints.append(det.place_at_center(volume, axes=tuple(other)))
    constraints.append(det.set_grid_coordinates(axes=(axis,), sides=("-",), coordinates=(coord,)))
    objects.append(det)


def _axial_field_roles(source_type, offset=6):
    """Return (|E|, |H|) at a near-field point on the dipole axis (kr ≈ 1.9).

    On the dipole axis the *radiation* field vanishes (∝ sinθ → 0), so the
    measurement isolates the near-field role: an electric dipole is strongly
    E-dominant there, a magnetic dipole strongly H-dominant — the defining
    E↔H duality signature.
    """
    objects, constraints, config, volume = _build_dipole(source_type, polarization=2)
    _add_point_phasor_det("axial", volume, objects, constraints, axis=2, coord=_CENTER + offset)
    arrays = _run(objects, constraints, config)
    ph = arrays.detector_states["axial"]["phasor"]  # (1, num_freqs, 6)
    E = float(jnp.sum(jnp.abs(ph[0, 0, :3]) ** 2) ** 0.5)
    H = float(jnp.sum(jnp.abs(ph[0, 0, 3:]) ** 2) ** 0.5)
    return E, H


# ── Analytic sin²θ face-flux ratio over the actual detector geometry ─────────
#
# A Hertzian dipole's time-averaged radial flux is exactly C·sin²ψ/r².
# Integrating it over the detector faces (perpendicular distance _DET_OFFSET
# cells, spanning ±_CENTER cells) gives the equatorial/axial face-flux ratio
# (≈ 2.68 here).  See test_dipole_radiation.py for the full derivation.


def _face_flux_sin2(d_hat, normal_axis, d_cells=_DET_OFFSET, w_cells=_CENTER, n=601):
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
    integrand = sin2 / r**2 * (d_cells / r)
    return float(np.sum(integrand)) * (t[1] - t[0]) ** 2


def _axis_pair_flux(d_hat, axis):
    return 2.0 * _face_flux_sin2(d_hat, axis)


_EQ_AXIAL_RATIO = (_axis_pair_flux((0.0, 0.0, 1.0), 0) + _axis_pair_flux((0.0, 0.0, 1.0), 1)) / _axis_pair_flux(
    (0.0, 0.0, 1.0), 2
)
_PATTERN_TOL = 0.15  # 15 %: grid dispersion + PML inset shift the measured value up ~5 %


# ── Tests ────────────────────────────────────────────────────────────────────


def test_magnetic_dipole_radiates():
    """A z-polarized magnetic dipole radiates the same total power as the
    electric dipole in vacuum (E↔H duality), to within a few percent."""
    P_m = _total_outward_flux(_run_with_all_detectors("magnetic", polarization=2))
    P_e = _total_outward_flux(_run_with_all_detectors("electric", polarization=2))
    assert P_m > 0, f"Magnetic dipole total power should be positive, got {P_m}"
    rel_err = abs(P_m - P_e) / P_e
    assert rel_err < 0.05, (
        f"Magnetic vs electric radiated power differ by {rel_err:.2%} "
        f"(P_m={P_m:.4e}, P_e={P_e:.4e}); vacuum duality predicts equality"
    )


def test_magnetic_dipole_symmetry():
    """Z-polarized magnetic dipole: azimuthal symmetry + quantitative sin²θ ratio."""
    arrays = _run_with_all_detectors("magnetic", polarization=2)

    fluxes = {
        f"flux_{['x', 'y', 'z'][a]}_{s}": _mean_flux(arrays, f"flux_{['x', 'y', 'z'][a]}_{s}")
        for a in range(3)
        for s in ("+", "-")
    }
    S_xp = fluxes["flux_x_+"]
    S_xm = fluxes["flux_x_-"]
    S_yp = fluxes["flux_y_+"]

    # Forward/backward symmetry
    tol = 0.05
    assert abs(S_xp - S_xm) / max(abs(S_xp), 1e-30) < tol, (
        f"Magnetic dipole x-symmetry broken: S_+x={S_xp:.4e}, S_-x={S_xm:.4e}"
    )

    # Azimuthal symmetry
    assert abs(S_xp - S_yp) / max(abs(S_xp), 1e-30) < tol, (
        f"Magnetic dipole azimuthal symmetry broken: S_+x={S_xp:.4e}, S_+y={S_yp:.4e}"
    )

    # Quantitative equatorial/axial flux ratio vs the derived sin²θ value.
    S_eq = sum(abs(fluxes[f"flux_{['x', 'y', 'z'][a]}_{s}"]) for a in (0, 1) for s in ("+", "-"))
    S_ax = sum(abs(fluxes[f"flux_z_{s}"]) for s in ("+", "-"))
    ratio = S_eq / S_ax
    rel_err = abs(ratio - _EQ_AXIAL_RATIO) / _EQ_AXIAL_RATIO
    assert rel_err < _PATTERN_TOL, (
        f"Magnetic dipole equatorial/axial flux ratio S_eq/S_ax={ratio:.3f} differs from "
        f"the derived sin²θ value {_EQ_AXIAL_RATIO:.3f} by {rel_err:.2%}"
    )


def test_magnetic_vs_electric_duality():
    """Electric and magnetic dipoles are E↔H duals in vacuum.

    With inv_eps = inv_mu = 1 (η₀-normalized vacuum) the two radiate
    essentially identical *total power* and identical pattern structure (max in
    the xy-plane, min along z), but with swapped near-field roles: on the dipole
    axis the electric dipole is E-dominant and the magnetic dipole H-dominant.
    """
    arrays_e = _run_with_all_detectors("electric", polarization=2)
    arrays_m = _run_with_all_detectors("magnetic", polarization=2)

    P_e = _total_outward_flux(arrays_e)
    P_m = _total_outward_flux(arrays_m)

    assert P_e > 0
    assert P_m > 0

    # Vacuum duality: total radiated powers are essentially equal.
    ratio = max(P_e, P_m) / min(P_e, P_m)
    assert ratio < 1.1, f"Electric/magnetic power ratio too large: P_e={P_e:.4e}, P_m={P_m:.4e}, ratio={ratio:.3f}"

    # Both should have min flux along dipole axis (z)
    for label, arrays in [("electric", arrays_e), ("magnetic", arrays_m)]:
        face_fluxes = []
        for axis in range(3):
            axis_flux = sum(abs(_mean_flux(arrays, f"flux_{['x', 'y', 'z'][axis]}_{s}")) for s in ("+", "-"))
            face_fluxes.append(axis_flux)
        min_axis = int(np.argmin(face_fluxes))
        assert min_axis == 2, f"{label} dipole: min flux on axis {min_axis}, expected 2 (z)"

    # Field-role duality: on the dipole axis the electric dipole is E-dominant
    # while the magnetic dipole is H-dominant (the radiation field ∝ sinθ → 0
    # there, isolating the source's intrinsic near-field role).
    E_e, H_e = _axial_field_roles("electric")
    E_m, H_m = _axial_field_roles("magnetic")
    assert E_e > H_e, f"Electric dipole should be E-dominant on axis: |E|={E_e:.3e}, |H|={H_e:.3e}"
    assert H_m > E_m, f"Magnetic dipole should be H-dominant on axis: |E|={E_m:.3e}, |H|={H_m:.3e}"
