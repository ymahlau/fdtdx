"""Physics simulation tests: plane wave propagation.

Tests phase velocity and wave impedance in vacuum and in a dielectric.
Phase velocity: measured from the phasor phase difference between two detectors,
compared to the analytic k = 2πn/λ.
Wave impedance: measured as |Ex_phasor|/|Hy_phasor| from a single detector,
compared to Z₀/n (Z₀ ≈ 376.73 Ω, n = √ε_r).

Domain layout (50 nm resolution = 20 cells/λ, z is propagation axis):
  80 cells total in z (4 µm):
    cells  0– 9 : PML (0.5 µm)
    cells 10–69 : active region (3 µm)
    cells 70–79 : PML (0.5 µm)

  Source     : 1 cell thick, left face at z-index 12
  Detector 1 : 1 cell thick, left face at z-index 17
  Detector 2 : 1 cell thick, left face at z-index 22 (vacuum, 5-cell = 0.25 µm gap)
             or z-index 20 (dielectric, 3-cell = 0.15 µm gap – avoids 2π ambiguity)

Transverse (x, y): 3 cells each with periodic boundaries.

Tolerance: 5 % relative error. At 20 cells/λ FDTD dispersion is ~0.6% for n=2.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ──────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6  # free-space wavelength (m)
_RESOLUTION = 50e-9  # grid resolution (m) → 20 cells/λ in vacuum
_PML_CELLS = 10  # PML thickness in cells
_DOMAIN_XY = 3 * _RESOLUTION  # transverse extent (3 cells, periodic BCs)
_DOMAIN_Z = 4e-6  # total z-extent including PML (40 cells)

# Grid indices (left face of each object)
_SOURCE_Z = _PML_CELLS + 2  # = 12  (2 cells into active region)
_DET1_Z = _SOURCE_Z + 5  # = 17  (5 cells from source)
_DET2_VAC_Z = _DET1_Z + 5  # = 22  (5 cells from D1 → 0.25 µm, Δφ = π/2)
_DET2_DIEL_Z = _DET1_Z + 3  # = 20  (3 cells from D1 → 0.15 µm, Δφ ≈ 1.88 rad)

_DET_SEP_VAC = 5 * _RESOLUTION  # 0.25 µm
_DET_SEP_DIEL = 3 * _RESOLUTION  # 0.15 µm

_SIM_TIME = 120e-15  # 120 fs ≈ 36 optical periods at λ = 1 µm
_TOLERANCE = 0.05  # 5 % relative tolerance


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_1d_base():
    """Build the common 1D-like simulation domain.

    Returns:
        objects, constraints, config, volume, wave
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

    # Periodic in x and y, PML in z only
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

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),  # x-polarized, propagation in +z
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
    """Add a 1-cell-thick PhasorDetector at the given z-index, spanning full xy."""
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
    """Return the accumulated Ex phasor (scalar) from a reduce_volume detector.

    state["phasor"] shape: (1, num_freqs, num_components).
    Components order: Ex=0, Ey=1, Ez=2, Hx=3, Hy=4, Hz=5.
    """
    return complex(arrays.detector_states[name]["phasor"][0, 0, 0])


def _hy_phasor(arrays, name) -> complex:
    """Return the accumulated Hy phasor (scalar) from a reduce_volume detector.

    Components order: Ex=0, Ey=1, Ez=2, Hx=3, Hy=4, Hz=5.
    """
    return complex(arrays.detector_states[name]["phasor"][0, 0, 4])


def _measure_k(p_near: complex, p_far: complex, separation: float) -> float:
    """Derive wave vector k from two phasors.

    For a +z propagating wave the PhasorDetector phasor phase *increases*
    with z: angle(p_far) > angle(p_near) for p_far at larger z.
    delta_phi = (angle(p_far) - angle(p_near)) % 2π = k * separation.
    Provided k*separation < 2π (separation < one medium wavelength) the
    modulo operation maps the result unambiguously into [0, 2π).
    """
    delta_phi = (np.angle(p_far) - np.angle(p_near)) % (2 * np.pi)
    return delta_phi / separation


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_phase_velocity_vacuum():
    """Wave vector in vacuum matches k = 2π/λ within 5 %.

    Detector separation: 5 cells = 0.25 µm < λ, so no phase-wrapping
    ambiguity.  Expected Δφ = k·d = (2π/1 µm)·0.25 µm = π/2.
    """
    objects, constraints, config, volume, wave = _build_1d_base()
    _add_phasor_det("d1", _DET1_Z, wave, volume, objects, constraints)
    _add_phasor_det("d2", _DET2_VAC_Z, wave, volume, objects, constraints)

    arrays = _run(objects, constraints, config)
    p1 = _ex_phasor(arrays, "d1")
    p2 = _ex_phasor(arrays, "d2")

    # Both detectors must receive non-zero signal
    assert abs(p1) > 0, "Detector d1 measured zero Ex amplitude"
    assert abs(p2) > 0, "Detector d2 measured zero Ex amplitude"

    # Amplitudes should be equal (no attenuation in lossless vacuum)
    rel_amp_diff = abs(abs(p1) - abs(p2)) / abs(p1)
    assert rel_amp_diff < _TOLERANCE, (
        f"Amplitude mismatch: |d1|={abs(p1):.4e}, |d2|={abs(p2):.4e}, relative diff={rel_amp_diff:.3f} > {_TOLERANCE}"
    )

    # Wave vector must match analytic value
    k_measured = _measure_k(p1, p2, _DET_SEP_VAC)
    k_analytic = 2 * np.pi / _WAVELENGTH
    rel_err = abs(k_measured - k_analytic) / k_analytic
    assert rel_err < _TOLERANCE, (
        f"k_measured={k_measured:.4e} m⁻¹, k_analytic={k_analytic:.4e} m⁻¹, relative error={rel_err:.3f} > {_TOLERANCE}"
    )


def test_phase_velocity_dielectric():
    """Wave vector in ε_r=4 (n=2) dielectric matches k = 2πn/λ within 5 %.

    The entire domain is filled with the dielectric so there is no
    interface.  Detector separation: 3 cells = 0.15 µm.  With n=2 and
    λ=1 µm, the medium wavelength is 0.5 µm, so 0.15 µm < 0.5 µm avoids
    the 2π ambiguity.  Expected Δφ = 2π·2·0.15/1 ≈ 1.88 rad.
    At 20 cells/λ_vacuum (10 cells/λ_medium) FDTD dispersion is ~0.6%.
    """
    epsilon_r = 4.0
    n = np.sqrt(epsilon_r)  # = 2.0

    objects, constraints, config, volume, wave = _build_1d_base()

    # Fill entire domain with dielectric
    diel = fdtdx.UniformMaterialObject(
        partial_real_shape=(None, None, None),
        material=fdtdx.Material(permittivity=epsilon_r),
    )
    pos_c, size_c = diel.same_position_and_size(volume)
    constraints.extend([pos_c, size_c])
    objects.append(diel)

    # Use 3-cell separation to avoid 2π ambiguity in the denser medium
    _add_phasor_det("d1", _DET1_Z, wave, volume, objects, constraints)
    _add_phasor_det("d2", _DET2_DIEL_Z, wave, volume, objects, constraints)

    arrays = _run(objects, constraints, config)
    p1 = _ex_phasor(arrays, "d1")
    p2 = _ex_phasor(arrays, "d2")

    assert abs(p1) > 0, "Detector d1 measured zero Ex amplitude"
    assert abs(p2) > 0, "Detector d2 measured zero Ex amplitude"

    k_measured = _measure_k(p1, p2, _DET_SEP_DIEL)
    k_analytic = 2 * np.pi * n / _WAVELENGTH
    rel_err = abs(k_measured - k_analytic) / k_analytic
    assert rel_err < _TOLERANCE, (
        f"ε_r={epsilon_r} (n={n}): k_measured={k_measured:.4e} m⁻¹, "
        f"k_analytic={k_analytic:.4e} m⁻¹, relative error={rel_err:.3f} > {_TOLERANCE}"
    )


def test_wave_impedance_vacuum():
    """Wave impedance in vacuum: |Ex_phasor|/|Hy_phasor| = 1 (normalized) within 5 %.

    fdtdx stores E_stored = E_SI / η₀ and H_stored = H_SI, so
    |E_stored|/|H_stored| = Z_SI / η₀ = 1 for vacuum.
    """
    Z_analytic = 1.0  # = Z₀/η₀ in fdtdx normalized units

    objects, constraints, config, volume, wave = _build_1d_base()
    _add_phasor_det("d1", _DET1_Z, wave, volume, objects, constraints)

    arrays = _run(objects, constraints, config)
    p_ex = _ex_phasor(arrays, "d1")
    p_hy = _hy_phasor(arrays, "d1")

    assert abs(p_ex) > 0, "Detector d1 measured zero Ex amplitude"
    assert abs(p_hy) > 0, "Detector d1 measured zero Hy amplitude"

    Z_measured = abs(p_ex) / abs(p_hy)
    rel_err = abs(Z_measured - Z_analytic) / Z_analytic
    assert rel_err < _TOLERANCE, (
        f"Z_measured={Z_measured:.4f} (normalized), Z_analytic={Z_analytic:.4f}, "
        f"relative error={rel_err:.3f} > {_TOLERANCE}"
    )


def test_wave_impedance_dielectric():
    """Wave impedance in ε_r=4 (n=2) dielectric: |Ex|/|Hy| = 1/n within 5 %.

    In fdtdx normalized units: Z_analytic = Z_SI / η₀ = (Z₀/n) / Z₀ = 1/n.
    The entire domain is filled with the dielectric (no interface) to avoid
    standing-wave artifacts.
    """
    epsilon_r = 4.0
    n = float(np.sqrt(epsilon_r))  # = 2.0
    Z_analytic = 1.0 / n  # = 0.5 in fdtdx normalized units

    objects, constraints, config, volume, wave = _build_1d_base()

    diel = fdtdx.UniformMaterialObject(
        partial_real_shape=(None, None, None),
        material=fdtdx.Material(permittivity=epsilon_r),
    )
    pos_c, size_c = diel.same_position_and_size(volume)
    constraints.extend([pos_c, size_c])
    objects.append(diel)

    _add_phasor_det("d1", _DET1_Z, wave, volume, objects, constraints)

    arrays = _run(objects, constraints, config)
    p_ex = _ex_phasor(arrays, "d1")
    p_hy = _hy_phasor(arrays, "d1")

    assert abs(p_ex) > 0, "Detector d1 measured zero Ex amplitude"
    assert abs(p_hy) > 0, "Detector d1 measured zero Hy amplitude"

    Z_measured = abs(p_ex) / abs(p_hy)
    rel_err = abs(Z_measured - Z_analytic) / Z_analytic
    assert rel_err < _TOLERANCE, (
        f"ε_r={epsilon_r} (n={n}): Z_measured={Z_measured:.4f} (normalized), "
        f"Z_analytic={Z_analytic:.4f}, relative error={rel_err:.3f} > {_TOLERANCE}"
    )
