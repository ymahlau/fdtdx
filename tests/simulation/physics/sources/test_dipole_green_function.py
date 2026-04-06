"""Physics simulation tests: dipole in homogeneous dielectric.

Validates that the dipole source radiates correctly in a dielectric medium:
  2a. Radiated power scales with refractive index (P_diel / P_vac ≈ n).
  2b. Phase velocity in the dielectric matches c/n.

The physical domain size is fixed at 3 μm so that detector distances stay
constant in physical units when _RESOLUTION is changed.  The cell count
scales as _DOMAIN / _RESOLUTION (e.g. 60 at 50 nm, 120 at 25 nm).
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
_RESOLUTION = 50e-9
_PML_CELLS = 10
_DOMAIN = 3e-6  # fixed physical size — resolution-independent
_SIM_TIME = 120e-15

_CENTER = round(_DOMAIN / (2 * _RESOLUTION))  # grid cell at the centre
_DET_OFFSET = 10

_TOLERANCE = 0.10  # 10% for decay tests (grid dispersion + near-field effects)

_DT_APPROX = 0.99 * _RESOLUTION / (3e8 * np.sqrt(3))
_STEPS_PER_PERIOD = int(round(_WAVELENGTH / (3e8 * _DT_APPROX)))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD

_EPS_R = 2.25  # n = 1.5
_N = np.sqrt(_EPS_R)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_dipole_in_medium(eps_r=1.0):
    """Build domain with dipole in a uniform medium."""
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

    if eps_r != 1.0:
        medium = fdtdx.UniformMaterialObject(
            name="medium",
            partial_real_shape=(None, None, None),
            material=fdtdx.Material(permittivity=eps_r),
        )
        constraints.extend(medium.same_position_and_size(volume))
        objects.append(medium)

    source = fdtdx.PointDipoleSource(
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        polarization=2,  # z-polarized
        amplitude=1.0,
    )
    constraints.append(source.place_at_center(volume, axes=(0, 1, 2)))
    objects.append(source)

    return objects, constraints, config, volume


def _add_flux_det(name, volume, objects, constraints, axis, side, offset=_DET_OFFSET):
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
    coord = _CENTER + offset if side == "+" else _CENTER - offset
    constraints.append(
        det.set_grid_coordinates(axes=(axis,), sides=("-",), coordinates=(coord,)),
    )
    objects.append(det)


def _add_point_phasor_det(name, volume, objects, constraints, axis, coord):
    """Add a single-cell PhasorDetector on the axis at the domain centre.

    Unlike _add_phasor_det this spans only one cell in all three axes so
    the measured phasor can be directly compared with the analytical
    Hertzian-dipole on-axis formula.
    """
    det = fdtdx.PhasorDetector(
        name=name,
        partial_grid_shape=(1, 1, 1),
        wave_characters=(fdtdx.WaveCharacter(wavelength=_WAVELENGTH),),
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        reduce_volume=True,
        plot=False,
    )
    other_axes = [a for a in range(3) if a != axis]
    constraints.append(det.place_at_center(volume, axes=tuple(other_axes)))
    constraints.append(
        det.set_grid_coordinates(axes=(axis,), sides=("-",), coordinates=(coord,)),
    )
    objects.append(det)


def _hertzian_Ez_amplitude(r: float, k: float) -> float:
    """Expected |E_z| amplitude for a z-dipole measured on the x-axis (θ=π/2).

    From the exact Hertzian-dipole field in the equatorial plane::

        E_θ ∝ (k²/r) · |1 − 1/(kr)² + i/(kr)|

    At θ=π/2 the only non-zero Cartesian E-component is E_z = −E_θ, so its
    amplitude is the same expression (up to a global constant that cancels in
    ratios).
    """
    kr = k * r
    return (k**2 / r) * np.sqrt((1 - 1 / kr**2) ** 2 + (1 / kr) ** 2)


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


# ── Tests ────────────────────────────────────────────────────────────────────


def test_dipole_in_dielectric_power_scaling():
    """Radiated power in dielectric scales as P_diel / P_vac ≈ n.

    A Hertzian dipole in a homogeneous medium with index n radiates n times
    the power of the same dipole in vacuum (Purcell effect in bulk).
    """
    # Vacuum run
    obj_vac, con_vac, cfg_vac, vol_vac = _build_dipole_in_medium(eps_r=1.0)
    for axis in range(3):
        for side in ("+", "-"):
            name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
            _add_flux_det(name, vol_vac, obj_vac, con_vac, axis, side)
    arrays_vac = _run(obj_vac, con_vac, cfg_vac)
    P_vac = _total_outward_flux(arrays_vac)

    # Dielectric run
    obj_diel, con_diel, cfg_diel, vol_diel = _build_dipole_in_medium(eps_r=_EPS_R)
    for axis in range(3):
        for side in ("+", "-"):
            name = f"flux_{['x', 'y', 'z'][axis]}_{side}"
            _add_flux_det(name, vol_diel, obj_diel, con_diel, axis, side)
    arrays_diel = _run(obj_diel, con_diel, cfg_diel)
    P_diel = _total_outward_flux(arrays_diel)

    assert P_vac > 0, f"Vacuum power should be positive: {P_vac}"
    assert P_diel > 0, f"Dielectric power should be positive: {P_diel}"

    ratio = P_diel / P_vac
    expected = _N  # 1.5
    rel_err = abs(ratio - expected) / expected
    assert rel_err < 0.05, (
        f"Power ratio P_diel/P_vac={ratio:.3f}, expected {expected:.3f}, relative error={rel_err:.2%}"
    )


def test_dipole_in_dielectric_field_decay():
    """Dipole field decay matches the exact Hertzian-dipole formula.

        A z-polarised point dipole sits at the centre of a dielectric slab
        (n = 1.5).  Two single-cell phasor detectors are placed on the x-axis
        at fixed physical distances r₁ = 400 nm and r₂ = 650 nm (intermediate-
        field regime, kr₁ ≈ 3.8, kr₂ ≈ 6.1 at λ = 1 μm, n = 1.5).  The
        amplitude ratio amp_near / amp_far is compared with the prediction of the
        exact Hertzian-dipole formula:

            |E_z(r)| ∝ (k²/r) · sqrt((1 − 1/(kr)²)² + (1/(kr))²)
    .
    """
    obj, con, cfg, vol = _build_dipole_in_medium(eps_r=_EPS_R)

    # Physical distances chosen so kr ≈ 3.8 and 6.1 (intermediate-field
    # regime).
    r1 = 400e-9
    r2 = 650e-9
    r1_cells = round(r1 / _RESOLUTION)
    r2_cells = round(r2 / _RESOLUTION)
    _add_point_phasor_det("phasor_near", vol, obj, con, axis=0, coord=_CENTER + r1_cells)
    _add_point_phasor_det("phasor_far", vol, obj, con, axis=0, coord=_CENTER + r2_cells)

    arrays = _run(obj, con, cfg)

    # |E| = sqrt(|Ex|² + |Ey|² + |Ez|²); index [0] selects the single wavelength
    phasor_near = arrays.detector_states["phasor_near"]["phasor"]
    phasor_far = arrays.detector_states["phasor_far"]["phasor"]
    amp_near = float(jnp.sum(jnp.abs(phasor_near[0, :3].ravel()) ** 2) ** 0.5)
    amp_far = float(jnp.sum(jnp.abs(phasor_far[0, :3].ravel()) ** 2) ** 0.5)

    assert amp_near > 1e-30, "Near phasor amplitude is zero"
    assert amp_far > 1e-30, "Far phasor amplitude is zero"

    # --- compare to analytical prediction ---
    k = 2 * np.pi * _N / _WAVELENGTH  # wavenumber in medium
    expected_ratio = _hertzian_Ez_amplitude(r1, k) / _hertzian_Ez_amplitude(r2, k)
    measured_ratio = amp_near / amp_far

    rel_err = abs(measured_ratio - expected_ratio) / expected_ratio
    assert rel_err < _TOLERANCE, (
        f"Field decay ratio amp_near/amp_far={measured_ratio:.3f}, "
        f"expected {expected_ratio:.3f} (Hertzian dipole, kr₁={k * r1:.2f}, kr₂={k * r2:.2f}), "
        f"relative error={rel_err:.2%}"
    )
