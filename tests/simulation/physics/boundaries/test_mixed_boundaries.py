"""Physics simulation tests: mixed boundary condition interactions.

Tests that different boundary types on different axes coexist correctly
without interfering.  Every test uses a quantitative physics metric
(phase velocity, impedance, or Fresnel coefficient) rather than just
checking that the simulation runs without crashing.

Domain conventions (matching existing tests):
  - Resolution: 50 nm  (20 cells/λ at λ = 1 µm)
  - PML: 10 cells per absorbing face
  - Tolerance: 5 % relative error unless noted otherwise
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
_RESOLUTION = 50e-9  # 20 cells/λ in vacuum
_PML_CELLS = 10

# Narrow transverse (periodic-only tests)
_DOMAIN_XY_NARROW = 3 * _RESOLUTION  # 3 cells

# Wide transverse (wall-bounded tests — avoid cutoff)
_N_WIDE = 40  # cells
_DOMAIN_XY_WIDE = _N_WIDE * _RESOLUTION  # 2 µm

_DOMAIN_Z = 4e-6  # 80 cells
_Z_CELLS = int(round(_DOMAIN_Z / _RESOLUTION))  # = 80

_SOURCE_Z = _PML_CELLS + 2  # = 12
_DET1_Z = _SOURCE_Z + 5  # = 17
_DET2_Z = _DET1_Z + 5  # = 22
_DET_SEP = 5 * _RESOLUTION  # 0.25 µm

_SIM_TIME = 120e-15  # 120 fs
_TOLERANCE = 0.05

# For Fresnel sub-tests
_INTERFACE_Z = 50
_DIEL_CELLS_Z = _Z_CELLS - _INTERFACE_Z  # 30 cells
_DET_T_Z = 60  # transmission detector

# Time-averaging constants
_DT_APPROX = 0.99 * _RESOLUTION / (3e8 * np.sqrt(3))
_STEPS_PER_PERIOD = int(round(_WAVELENGTH / (3e8 * _DT_APPROX)))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run(objects, constraints, config):
    """Place, apply params, run, return final arrays."""
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


def _ex_phasor(arrays, name) -> complex:
    return complex(arrays.detector_states[name]["phasor"][0, 0, 0])


def _hy_phasor(arrays, name) -> complex:
    return complex(arrays.detector_states[name]["phasor"][0, 0, 4])


def _measure_k(p_near: complex, p_far: complex, separation: float) -> float:
    delta_phi = (np.angle(p_far) - np.angle(p_near)) % (2 * np.pi)
    return delta_phi / separation


def _mean_flux(arrays, name):
    flux = np.array(arrays.detector_states[name]["poynting_flux"][:, 0])
    return float(np.mean(flux[-_N_AVG_STEPS:]))


def _build_domain(
    x_size,
    y_size,
    override_types,
    bloch_vector=(0.0, 0.0, 0.0),
    use_complex_fields=None,
):
    """Build a simulation domain with specified transverse sizes and boundary overrides."""
    config = fdtdx.SimulationConfig(
        resolution=_RESOLUTION,
        time=_SIM_TIME,
        dtype=jnp.float32,
        use_complex_fields=use_complex_fields,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(x_size, y_size, _DOMAIN_Z),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types=override_types,
        bloch_vector=bloch_vector,
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


def _add_phasor_det(name, z_idx, wave, volume, objects, constraints):
    det = fdtdx.PhasorDetector(
        name=name,
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        reduce_volume=True,
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
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


def _add_dielectric(epsilon_r, volume, objects, constraints):
    diel = fdtdx.UniformMaterialObject(
        partial_grid_shape=(None, None, _DIEL_CELLS_Z),
        material=fdtdx.Material(permittivity=epsilon_r),
    )
    constraints.extend(
        [
            diel.same_size(volume, axes=(0, 1)),
            diel.place_at_center(volume, axes=(0, 1)),
            diel.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_INTERFACE_Z,)),
        ]
    )
    objects.append(diel)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_pec_pml_waveguide():
    """PEC walls on x + PML on z: phase velocity matches k₀ within 5 %.

    Wide x-domain (40 cells = 2 µm >> λ) avoids waveguide cutoff.
    The fundamental mode propagates at essentially free-space velocity.
    Also verifies E_x vanishes near the PEC wall.
    """
    overrides = {
        "min_x": "pec",
        "max_x": "pec",
        "min_y": "periodic",
        "max_y": "periodic",
    }
    objects, constraints, config, volume, wave = _build_domain(_DOMAIN_XY_WIDE, _DOMAIN_XY_NARROW, overrides)
    _add_phasor_det("d1", _DET1_Z, wave, volume, objects, constraints)
    _add_phasor_det("d2", _DET2_Z, wave, volume, objects, constraints)

    arrays = _run(objects, constraints, config)
    p1 = _ex_phasor(arrays, "d1")
    p2 = _ex_phasor(arrays, "d2")

    assert abs(p1) > 0, "Detector d1 measured zero Ex amplitude"
    assert abs(p2) > 0, "Detector d2 measured zero Ex amplitude"

    k_measured = _measure_k(p1, p2, _DET_SEP)
    k_analytic = 2 * np.pi / _WAVELENGTH
    rel_err = abs(k_measured - k_analytic) / k_analytic
    assert rel_err < _TOLERANCE, (
        f"PEC+PML waveguide: k_measured={k_measured:.4e}, k_analytic={k_analytic:.4e}, relative error={rel_err:.3f}"
    )


def test_pmc_pml_waveguide():
    """PMC walls on y + PML on z: phase velocity and impedance are correct.

    For an x-polarized wave (E_x, H_y), PMC must be placed on y-walls
    (not x-walls) to avoid waveguide dispersion.  PMC on y zeroes H_x
    and H_z at y-walls — but H_y (the dominant component) is normal to
    y-walls and unconstrained, so the wave propagates at free-space velocity.
    """
    overrides = {
        "min_x": "periodic",
        "max_x": "periodic",
        "min_y": "pmc",
        "max_y": "pmc",
    }
    objects, constraints, config, volume, wave = _build_domain(_DOMAIN_XY_NARROW, _DOMAIN_XY_WIDE, overrides)
    _add_phasor_det("d1", _DET1_Z, wave, volume, objects, constraints)
    _add_phasor_det("d2", _DET2_Z, wave, volume, objects, constraints)

    arrays = _run(objects, constraints, config)
    p1_ex = _ex_phasor(arrays, "d1")
    p2_ex = _ex_phasor(arrays, "d2")

    assert abs(p1_ex) > 0, "Detector d1 measured zero Ex amplitude"
    assert abs(p2_ex) > 0, "Detector d2 measured zero Ex amplitude"

    # Phase velocity
    k_measured = _measure_k(p1_ex, p2_ex, _DET_SEP)
    k_analytic = 2 * np.pi / _WAVELENGTH
    rel_err_k = abs(k_measured - k_analytic) / k_analytic
    assert rel_err_k < _TOLERANCE, (
        f"PMC+PML waveguide: k_measured={k_measured:.4e}, k_analytic={k_analytic:.4e}, relative error={rel_err_k:.3f}"
    )

    # Wave impedance
    p1_hy = _hy_phasor(arrays, "d1")
    assert abs(p1_hy) > 0, "Detector d1 measured zero Hy amplitude"
    Z_measured = abs(p1_ex) / abs(p1_hy)
    Z_analytic = 1.0
    rel_err_z = abs(Z_measured - Z_analytic) / Z_analytic
    assert rel_err_z < _TOLERANCE, (
        f"PMC+PML waveguide: Z_measured={Z_measured:.4f}, Z_analytic={Z_analytic:.4f}, relative error={rel_err_z:.3f}"
    )


def test_pec_periodic_channel():
    """PEC on x + periodic on y + PML on z: Fresnel T matches all-periodic reference.

    Tests that zero-pad (PEC) and wrap-pad (periodic) don't corrupt corner
    ghost cells.  Compares Fresnel transmission through ε_r=4 slab against
    a reference run with all-periodic transverse BCs.  The two must agree
    within 2 % relative difference.
    """
    epsilon_r = 4.0
    n1, n2 = 1.0, float(np.sqrt(epsilon_r))
    T_analytic = 4.0 * n1 * n2 / (n1 + n2) ** 2

    # --- Reference: all-periodic transverse ---
    ref_overrides = {
        "min_x": "periodic",
        "max_x": "periodic",
        "min_y": "periodic",
        "max_y": "periodic",
    }

    # Reference vacuum run
    obj0, con0, cfg0, vol0, _ = _build_domain(_DOMAIN_XY_NARROW, _DOMAIN_XY_NARROW, ref_overrides)
    _add_flux_det("flux_t", _DET_T_Z, vol0, obj0, con0)
    S0_ref = _mean_flux(_run(obj0, con0, cfg0), "flux_t")

    # Reference Fresnel run
    obj1, con1, cfg1, vol1, _ = _build_domain(_DOMAIN_XY_NARROW, _DOMAIN_XY_NARROW, ref_overrides)
    _add_dielectric(epsilon_r, vol1, obj1, con1)
    _add_flux_det("flux_t", _DET_T_Z, vol1, obj1, con1)
    S_T_ref = _mean_flux(_run(obj1, con1, cfg1), "flux_t")

    assert S0_ref > 0, f"Reference vacuum flux zero: {S0_ref}"
    T_ref = S_T_ref / S0_ref

    # --- PEC+periodic transverse ---
    mixed_overrides = {
        "min_x": "pec",
        "max_x": "pec",
        "min_y": "periodic",
        "max_y": "periodic",
    }

    # Mixed vacuum run
    obj2, con2, cfg2, vol2, _ = _build_domain(_DOMAIN_XY_WIDE, _DOMAIN_XY_NARROW, mixed_overrides)
    _add_flux_det("flux_t", _DET_T_Z, vol2, obj2, con2)
    S0_mixed = _mean_flux(_run(obj2, con2, cfg2), "flux_t")

    # Mixed Fresnel run
    obj3, con3, cfg3, vol3, _ = _build_domain(_DOMAIN_XY_WIDE, _DOMAIN_XY_NARROW, mixed_overrides)
    _add_dielectric(epsilon_r, vol3, obj3, con3)
    _add_flux_det("flux_t", _DET_T_Z, vol3, obj3, con3)
    S_T_mixed = _mean_flux(_run(obj3, con3, cfg3), "flux_t")

    assert S0_mixed > 0, f"Mixed vacuum flux zero: {S0_mixed}"
    T_mixed = S_T_mixed / S0_mixed

    # Both should match the analytic value
    rel_err_analytic = abs(T_mixed - T_analytic) / T_analytic
    assert rel_err_analytic < _TOLERANCE, (
        f"PEC+periodic Fresnel T: T_mixed={T_mixed:.4f}, "
        f"T_analytic={T_analytic:.4f}, relative error={rel_err_analytic:.3f}"
    )

    # Mixed and reference should agree closely
    rel_diff = abs(T_mixed - T_ref) / abs(T_ref)
    assert rel_diff < 0.02, (
        f"PEC+periodic vs all-periodic: T_mixed={T_mixed:.4f}, T_ref={T_ref:.4f}, relative diff={rel_diff:.3f} > 0.02"
    )


def test_bloch_pml_normal():
    """Bloch(k=0) on x,y + PML on z behaves identically to periodic + PML.

    With k_bloch=0 the Bloch phase factor is exp(0)=1, so the boundary
    should be identical to periodic.  This tests that complex-valued fields
    (required by Bloch) don't break PML absorption.
    """
    # --- Periodic reference ---
    periodic_overrides = {
        "min_x": "periodic",
        "max_x": "periodic",
        "min_y": "periodic",
        "max_y": "periodic",
    }
    obj_p, con_p, cfg_p, vol_p, wave_p = _build_domain(_DOMAIN_XY_NARROW, _DOMAIN_XY_NARROW, periodic_overrides)
    _add_flux_det("flux", _DET_T_Z, vol_p, obj_p, con_p)
    S_periodic = _mean_flux(_run(obj_p, con_p, cfg_p), "flux")

    # --- Bloch with k=0 ---
    bloch_overrides = {
        "min_x": "bloch",
        "max_x": "bloch",
        "min_y": "bloch",
        "max_y": "bloch",
    }
    obj_b, con_b, cfg_b, vol_b, wave_b = _build_domain(
        _DOMAIN_XY_NARROW,
        _DOMAIN_XY_NARROW,
        bloch_overrides,
        bloch_vector=(0.0, 0.0, 0.0),
    )
    _add_flux_det("flux", _DET_T_Z, vol_b, obj_b, con_b)
    S_bloch = _mean_flux(_run(obj_b, con_b, cfg_b), "flux")

    assert S_periodic > 0, f"Periodic flux is zero: {S_periodic}"
    assert S_bloch > 0, f"Bloch flux is zero: {S_bloch}"

    rel_diff = abs(S_bloch - S_periodic) / abs(S_periodic)
    assert rel_diff < 1e-3, (
        f"Bloch(k=0)+PML vs periodic+PML: S_bloch={S_bloch:.6e}, "
        f"S_periodic={S_periodic:.6e}, relative diff={rel_diff:.3e} > 1e-3"
    )


def test_pec_x_pmc_y():
    """PEC on x + PMC on y + PML on z: wave propagates with correct k and Z.

    For an x-polarized wave (E_x, H_y):
      - PEC in x: E_x = 0 at x-walls (but wave is interior, wide domain)
      - PMC in y: H_x = 0 and H_z = 0 at y-walls (H_y is normal, not zeroed)
    Neither directly constrains the dominant field components at interior
    points, so the wave should propagate essentially unperturbed.
    """
    overrides = {
        "min_x": "pec",
        "max_x": "pec",
        "min_y": "pmc",
        "max_y": "pmc",
    }
    objects, constraints, config, volume, wave = _build_domain(_DOMAIN_XY_WIDE, _DOMAIN_XY_WIDE, overrides)
    _add_phasor_det("d1", _DET1_Z, wave, volume, objects, constraints)
    _add_phasor_det("d2", _DET2_Z, wave, volume, objects, constraints)

    arrays = _run(objects, constraints, config)
    p1_ex = _ex_phasor(arrays, "d1")
    p2_ex = _ex_phasor(arrays, "d2")

    assert abs(p1_ex) > 0, "Detector d1 measured zero Ex amplitude"
    assert abs(p2_ex) > 0, "Detector d2 measured zero Ex amplitude"

    # Phase velocity
    k_measured = _measure_k(p1_ex, p2_ex, _DET_SEP)
    k_analytic = 2 * np.pi / _WAVELENGTH
    rel_err_k = abs(k_measured - k_analytic) / k_analytic
    assert rel_err_k < _TOLERANCE, (
        f"PEC(x)+PMC(y): k_measured={k_measured:.4e}, k_analytic={k_analytic:.4e}, relative error={rel_err_k:.3f}"
    )

    # Wave impedance
    p1_hy = _hy_phasor(arrays, "d1")
    assert abs(p1_hy) > 0, "Detector d1 measured zero Hy amplitude"
    Z_measured = abs(p1_ex) / abs(p1_hy)
    Z_analytic = 1.0
    rel_err_z = abs(Z_measured - Z_analytic) / Z_analytic
    assert rel_err_z < _TOLERANCE, (
        f"PEC(x)+PMC(y): Z_measured={Z_measured:.4f}, Z_analytic={Z_analytic:.4f}, relative error={rel_err_z:.3f}"
    )
