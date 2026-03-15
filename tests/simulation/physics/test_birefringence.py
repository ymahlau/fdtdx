"""Physics simulation tests: birefringence in a diagonally anisotropic dielectric.

Tests that x- and y-polarized plane waves propagating in the same anisotropic
medium travel at different phase velocities (different wave vectors) and exhibit
the correspondingly different wave impedances.

Material: diagonal permittivity tensor ε = diag(ε_o, ε_e, 1) with
  ε_o = 2.25  →  n_o = 1.5  (ordinary axis, x-polarization)
  ε_e = 4.00  →  n_e = 2.0  (extraordinary axis, y-polarization)

For a +z propagating wave:
  Ordinary (x-pol):   k_o = 2π n_o / λ,  Z_o (normalized) = 1/n_o = 1/1.5 ≈ 0.667
  Extraordinary (y-pol): k_e = 2π n_e / λ,  Z_e (normalized) = 1/n_e = 1/2.0 = 0.500

fdtdx field normalization: E_stored = E_SI / η₀, H_stored = H_SI.
  Wave impedance in normalized units: |E_stored|/|H_stored| = Z_SI/η₀ = 1/n.

Domain layout (50 nm resolution = 20 cells/λ in vacuum, z is propagation axis):
  80 cells total in z (4 µm):
    cells  0– 9 : PML (0.5 µm)
    cells 10–69 : active region (3 µm)
    cells 70–79 : PML (0.5 µm)

  Source     : z-index 12 (x-pol or y-pol, +z direction)
  Detector D1: z-index 17
  Detector D2: z-index 20  (3-cell = 150 nm separation)

The entire domain is filled with the birefringent material.

Detector separation d = 150 nm:
  Δφ_o = k_o · d = 2π·1.5·0.15 ≈ 1.41 rad  < 2π  (no phase wrapping)
  Δφ_e = k_e · d = 2π·2.0·0.15 ≈ 1.88 rad  < 2π

The four tests:
  1. test_ordinary_wave_vector       : Δφ(Ex) / d = k_o ± 5 %
  2. test_extraordinary_wave_vector  : Δφ(Ey) / d = k_e ± 5 %
  3. test_ordinary_impedance         : |Ex|/|Hy| = 1/n_o = 0.667 ± 5 %
  4. test_extraordinary_impedance    : |Ey|/|Hx| = 1/n_e = 0.500 ± 5 %

Impedance residual bias: with exact_interpolation=True, E_x acquires a
cos(k_z·Δz/2) factor from the necessary z-forward half-step, while H_y requires
only an x-backward shift (no z factor for k_x=0 wave).  This gives a ~2.8 %
bias for the ordinary axis (n_o=1.5) and ~4.9 % for the extraordinary (n_e=2.0),
both within the 5 % tolerance at 50 nm resolution.

Tolerance: 5 % relative error.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ──────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6  # free-space wavelength (m)
_RESOLUTION = 50e-9  # 20 cells/λ in vacuum
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION  # 3 cells, periodic
_DOMAIN_Z = 4e-6  # 80 cells total
_Z_CELLS = int(round(_DOMAIN_Z / _RESOLUTION))  # = 80

_SOURCE_Z = _PML_CELLS + 2  # = 12  (2 cells into active region)
_BIRE_START_Z = 14  # birefringent material starts here (2 cells after source)
_BIRE_CELLS_Z = _Z_CELLS - _BIRE_START_Z  # = 66 cells (extends through right PML)
_DET1_Z = _SOURCE_Z + 5  # = 17  (3 cells into birefringent region)
_DET2_Z = _DET1_Z + 3  # = 20  (6 cells into birefringent region)
_DET_SEP = 3 * _RESOLUTION  # = 150 nm

# ── Material ──────────────────────────────────────────────────────────────────
_EPS_ORDINARY = 2.25  # ordinary permittivity (x-axis)
_EPS_EXTRAORDINARY = 4.0  # extraordinary permittivity (y-axis)
_N_ORDINARY = float(np.sqrt(_EPS_ORDINARY))  # = 1.5
_N_EXTRAORDINARY = float(np.sqrt(_EPS_EXTRAORDINARY))  # = 2.0

_SIM_TIME = 120e-15  # 120 fs ≈ 36 optical periods
_TOLERANCE = 0.05  # 5 % relative error


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_base(polarization_vector):
    """Birefringent domain with +z source at given polarization.

    The anisotropic material starts at _BIRE_START_Z (2 cells after the source)
    and extends through the right PML, so the source is in vacuum.
    """
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
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    # Fill domain from _BIRE_START_Z (after source) through right PML.
    # For z-propagating waves: x-pol sees ε_o, y-pol sees ε_e.
    diel = fdtdx.UniformMaterialObject(
        partial_grid_shape=(None, None, _BIRE_CELLS_Z),
        material=fdtdx.Material(
            permittivity=(_EPS_ORDINARY, _EPS_EXTRAORDINARY, 1.0),
        ),
    )
    constraints.extend(
        [
            diel.same_size(volume, axes=(0, 1)),
            diel.place_at_center(volume, axes=(0, 1)),
            diel.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_BIRE_START_Z,)),
        ]
    )
    objects.append(diel)

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=polarization_vector,
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
    """1-cell-thick PhasorDetector (all six components) at z_idx."""
    det = fdtdx.PhasorDetector(
        name=name,
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        reduce_volume=True,
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        exact_interpolation=True,
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


def _ex_phasor(arrays, name) -> complex:
    """Ex phasor; state['phasor'] shape (1, num_freqs, num_components)."""
    return complex(arrays.detector_states[name]["phasor"][0, 0, 0])


def _ey_phasor(arrays, name) -> complex:
    """Ey phasor (component index 1)."""
    return complex(arrays.detector_states[name]["phasor"][0, 0, 1])


def _hx_phasor(arrays, name) -> complex:
    """Hx phasor (component index 3)."""
    return complex(arrays.detector_states[name]["phasor"][0, 0, 3])


def _hy_phasor(arrays, name) -> complex:
    """Hy phasor (component index 4)."""
    return complex(arrays.detector_states[name]["phasor"][0, 0, 4])


def _measure_k(p_near: complex, p_far: complex, separation: float) -> float:
    """Wave vector k from two phasors separated by `separation` in +z.

    Phase increases with z for a +z propagating wave, so the modulo maps
    Δφ = (angle_far − angle_near) % 2π unambiguously into [0, 2π) provided
    k·separation < 2π.
    """
    delta_phi = (np.angle(p_far) - np.angle(p_near)) % (2 * np.pi)
    return delta_phi / separation


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_ordinary_wave_vector():
    """Ordinary-axis wave vector k_o = 2π n_o / λ within 5 %.

    x-polarized wave propagating in +z through the anisotropic medium.
    n_o = √ε_o = 1.5; k_o = 2π·1.5/1 µm ≈ 9.42×10⁶ m⁻¹.
    Detector separation 3 cells = 150 nm; Δφ ≈ 1.41 rad, well below 2π.
    """
    objects, constraints, config, volume, wave = _build_base((1, 0, 0))
    _add_phasor_det("d1", _DET1_Z, wave, volume, objects, constraints)
    _add_phasor_det("d2", _DET2_Z, wave, volume, objects, constraints)

    arrays = _run(objects, constraints, config)
    p1 = _ex_phasor(arrays, "d1")
    p2 = _ex_phasor(arrays, "d2")

    assert abs(p1) > 0, "Detector d1 measured zero Ex amplitude"
    assert abs(p2) > 0, "Detector d2 measured zero Ex amplitude"

    k_measured = _measure_k(p1, p2, _DET_SEP)
    k_analytic = 2 * np.pi * _N_ORDINARY / _WAVELENGTH
    rel_err = abs(k_measured - k_analytic) / k_analytic
    assert rel_err < _TOLERANCE, (
        f"k_o_measured={k_measured:.4e} m⁻¹, k_o_analytic={k_analytic:.4e} m⁻¹, "
        f"relative error={rel_err:.3f} > {_TOLERANCE}"
    )


def test_extraordinary_wave_vector():
    """Extraordinary-axis wave vector k_e = 2π n_e / λ within 5 %.

    y-polarized wave propagating in +z through the anisotropic medium.
    n_e = √ε_e = 2.0; k_e = 2π·2.0/1 µm ≈ 12.57×10⁶ m⁻¹.
    Detector separation 3 cells = 150 nm; Δφ ≈ 1.88 rad, well below 2π.
    """
    objects, constraints, config, volume, wave = _build_base((0, 1, 0))
    _add_phasor_det("d1", _DET1_Z, wave, volume, objects, constraints)
    _add_phasor_det("d2", _DET2_Z, wave, volume, objects, constraints)

    arrays = _run(objects, constraints, config)
    p1 = _ey_phasor(arrays, "d1")
    p2 = _ey_phasor(arrays, "d2")

    assert abs(p1) > 0, "Detector d1 measured zero Ey amplitude"
    assert abs(p2) > 0, "Detector d2 measured zero Ey amplitude"

    k_measured = _measure_k(p1, p2, _DET_SEP)
    k_analytic = 2 * np.pi * _N_EXTRAORDINARY / _WAVELENGTH
    rel_err = abs(k_measured - k_analytic) / k_analytic
    assert rel_err < _TOLERANCE, (
        f"k_e_measured={k_measured:.4e} m⁻¹, k_e_analytic={k_analytic:.4e} m⁻¹, "
        f"relative error={rel_err:.3f} > {_TOLERANCE}"
    )


def test_ordinary_impedance():
    """Ordinary-axis wave impedance |Ex|/|Hy| = 1/n_o = 0.667 (normalized) ± 5 %.

    fdtdx stores E_stored = E_SI/η₀, so Z_normalized = Z_SI/η₀ = 1/n_o.
    For n_o = 1.5: Z_o = 1/1.5 ≈ 0.667.

    Residual bias: with exact_interpolation=True, E_x acquires a factor
    cos(k_o·Δz/2) ≈ cos(1.5π/20) ≈ 0.972 from its z-forward half-step,
    while H_y (needing only an x-backward shift) has no z factor.
    This gives a ~2.8 % systematic reduction in the measured impedance,
    within the 5 % tolerance.
    """
    Z_analytic = 1.0 / _N_ORDINARY  # ≈ 0.6667 in fdtdx normalized units

    objects, constraints, config, volume, wave = _build_base((1, 0, 0))
    _add_phasor_det("d1", _DET1_Z, wave, volume, objects, constraints)

    arrays = _run(objects, constraints, config)
    p_ex = _ex_phasor(arrays, "d1")
    p_hy = _hy_phasor(arrays, "d1")

    assert abs(p_ex) > 0, "Detector d1 measured zero Ex amplitude"
    assert abs(p_hy) > 0, "Detector d1 measured zero Hy amplitude"

    Z_measured = abs(p_ex) / abs(p_hy)
    rel_err = abs(Z_measured - Z_analytic) / Z_analytic
    assert rel_err < _TOLERANCE, (
        f"Z_o measured={Z_measured:.4f} (normalized), Z_o analytic={Z_analytic:.4f}, "
        f"relative error={rel_err:.3f} > {_TOLERANCE}"
    )


def test_extraordinary_impedance():
    """Extraordinary-axis wave impedance |Ey|/|Hx| = 1/n_e = 0.500 (normalized) ± 5 %.

    For a +z propagating y-polarized wave: H = (−H_x) x̂, so the impedance
    ratio is |Ey| / |Hx| = Z_e_normalized = 1/n_e = 0.5.

    Residual bias: cos(k_e·Δz/2) ≈ cos(π/10) ≈ 0.951 for n_e = 2 at 50 nm
    resolution gives a ~4.9 % systematic reduction — within the 5 % tolerance.
    This matches the identical measurement in test_wave_impedance_dielectric
    from test_plane_wave.py (same parameters).
    """
    Z_analytic = 1.0 / _N_EXTRAORDINARY  # = 0.500 in fdtdx normalized units

    objects, constraints, config, volume, wave = _build_base((0, 1, 0))
    _add_phasor_det("d1", _DET1_Z, wave, volume, objects, constraints)

    arrays = _run(objects, constraints, config)
    p_ey = _ey_phasor(arrays, "d1")
    p_hx = _hx_phasor(arrays, "d1")

    assert abs(p_ey) > 0, "Detector d1 measured zero Ey amplitude"
    assert abs(p_hx) > 0, "Detector d1 measured zero Hx amplitude"

    Z_measured = abs(p_ey) / abs(p_hx)
    rel_err = abs(Z_measured - Z_analytic) / Z_analytic
    assert rel_err < _TOLERANCE, (
        f"Z_e measured={Z_measured:.4f} (normalized), Z_e analytic={Z_analytic:.4f}, "
        f"relative error={rel_err:.3f} > {_TOLERANCE}"
    )
