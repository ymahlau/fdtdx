"""Physics simulation tests: skin depth in a lossy conductor.

Tests that a plane wave decays exponentially when propagating through a medium
with nonzero electric conductivity. The attenuation coefficient α = 1/δ is
measured from the amplitude ratio of Ex phasors at two positions inside the
conductor, and compared to the analytic value from the exact complex dispersion
relation for a lossy dielectric.

Analytic complex wave vector (exact, no approximation):
  k = (ω/c₀) × √(ε_r − iσ/(ωε₀))
  α = Im(k)   (attenuation coefficient; skin depth δ = 1/α)
  β = Re(k)   (phase constant)

Domain layout (50 nm resolution, z is propagation axis):
  90 cells total in z (4.5 µm):
    cells  0– 9 : PML (0.5 µm)
    cells 10–89 : active region (4.0 µm)
    cells 80–89 : PML (0.5 µm, right side)

  Source       : z-index 12 (vacuum, x-polarized, +z direction)
  Conductor    : z-index 30–79 (50 cells = 2.5 µm, σ = 1×10⁴ S/m, ε_r = 1)
  Detector D1  : z-index 35 (5 cells = 250 nm into conductor)
  Detector D2  : z-index 55 (25 cells = 1250 nm into conductor)
  Separation d : 20 cells = 1 µm

At σ = 1×10⁴ S/m, λ = 1 µm (ε_r = 1):
  σ/(ωε₀) ≈ 0.60  (moderate loss — exact formula used, not good-conductor approx.)
  α_analytic ≈ 1.81×10⁶ m⁻¹  →  δ_analytic ≈ 552 nm ≈ 11 cells
  |p₂|/|p₁| = exp(−d/δ) ≈ exp(−1.81) ≈ 0.163  (clearly measurable)

The conductor extends to the right PML so there is no conductor→vacuum back-
reflection that would corrupt the steady-state profile.

Measurement: PhasorDetector amplitude ratio — why it is bias-free:
  With exact_interpolation=True, E_x is spatially shifted z-forward by Δz/2,
  acquiring a bias factor |cos(k_z·Δz/2)|.  Because both D1 and D2 are in the
  same lossy medium (same complex k_z, same Δz), this factor is identical at
  both detectors and cancels exactly in the ratio |p₂|/|p₁|.  fdtdx field
  normalization (E_stored = E_SI/η₀) also cancels in the ratio.

Tolerance: 10 % relative error (FDTD numerical dispersion in lossy media is
larger than in lossless media and this tolerance is intentionally generous).
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Physical constants ────────────────────────────────────────────────────────
_C0 = 3e8  # speed of light (m/s)
_EPS0 = 8.854187817e-12  # vacuum permittivity (F/m)

# ── Domain constants ──────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6  # free-space wavelength (m)
_RESOLUTION = 50e-9  # grid resolution (m) → 20 cells/λ in vacuum
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION  # 3 cells, periodic
_DOMAIN_Z = 4.5e-6  # 90 cells total
_Z_CELLS = int(round(_DOMAIN_Z / _RESOLUTION))  # = 90

_SOURCE_Z = _PML_CELLS + 2  # = 12 (2 cells into active region)
_CONDUCTOR_START_Z = 30  # first conductor cell (18 vacuum cells from source)
_CONDUCTOR_CELLS_Z = _Z_CELLS - _PML_CELLS - _CONDUCTOR_START_Z  # = 50 cells
_DET1_Z = 35  # 5 cells (250 nm) into conductor
_DET2_Z = 55  # 25 cells (1250 nm) into conductor
_DET_SEP = (_DET2_Z - _DET1_Z) * _RESOLUTION  # = 1.0 µm

# ── Conductor parameters ──────────────────────────────────────────────────────
_EPS_R = 1.0  # relative permittivity of conductor
_SIGMA = 1.0e4  # electric conductivity (S/m)

# ── Simulation timing ─────────────────────────────────────────────────────────
_SIM_TIME = 80e-15  # 80 fs ≈ 24 optical periods
_DT_APPROX = 0.99 * _RESOLUTION / (_C0 * np.sqrt(3))  # ≈ 9.53e-17 s
_STEPS_PER_PERIOD = int(round(_WAVELENGTH / (_C0 * _DT_APPROX)))  # ≈ 35
_N_AVG_STEPS = 5 * _STEPS_PER_PERIOD  # last ~5 optical periods for steady state

_TOLERANCE = 0.10  # 10 % — generous for lossy-medium FDTD dispersion


# ── Analytic reference ────────────────────────────────────────────────────────


def _analytic_alpha() -> float:
    """Exact attenuation coefficient from the complex dispersion relation.

    k = (ω/c₀) × √(ε_r − iσ/(ωε₀))
    For a passive lossy medium propagating in +z, numpy's principal sqrt gives
    Im(k) < 0 (decaying wave). The physical attenuation coefficient is
    α = −Im(k) = |Im(k)| > 0.
    """
    omega = 2 * np.pi * _C0 / _WAVELENGTH
    eps_complex = _EPS_R - 1j * _SIGMA / (omega * _EPS0)
    k_complex = (omega / _C0) * np.sqrt(eps_complex)
    return abs(float(np.imag(k_complex)))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_base():
    """Vacuum domain with periodic xy BCs, PML in z, and an x-polarized +z source."""
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


def _add_conductor(volume, objects, constraints):
    """Fill cells [_CONDUCTOR_START_Z, _Z_CELLS − _PML_CELLS) with the lossy conductor."""
    cond = fdtdx.UniformMaterialObject(
        partial_grid_shape=(None, None, _CONDUCTOR_CELLS_Z),
        material=fdtdx.Material(
            permittivity=_EPS_R,
            electric_conductivity=_SIGMA,
        ),
    )
    constraints.extend(
        [
            cond.same_size(volume, axes=(0, 1)),
            cond.place_at_center(volume, axes=(0, 1)),
            cond.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_CONDUCTOR_START_Z,)),
        ]
    )
    objects.append(cond)


def _add_phasor_det(name, z_idx, wave, volume, objects, constraints):
    """Add a 1-cell-thick PhasorDetector (Ex only) at z_idx."""
    det = fdtdx.PhasorDetector(
        name=name,
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        reduce_volume=True,
        components=("Ex",),
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
    """Ex phasor from a reduce_volume PhasorDetector with components=('Ex',).

    state['phasor'] shape: (1, num_freqs=1, num_components=1).
    """
    return complex(arrays.detector_states[name]["phasor"][0, 0, 0])


# ── Test ──────────────────────────────────────────────────────────────────────


def test_skin_depth_attenuation():
    """Attenuation coefficient α matches Im(k) from exact dispersion within 10 %.

    Measurement: α_measured = −ln(|p₂|/|p₁|) / d, where d = 1 µm is the
    detector separation.  Both detectors are in the same lossy medium, so the
    interpolation bias factor |cos(k_z·Δz/2)| and the fdtdx η₀ normalization
    cancel exactly in the amplitude ratio.
    """
    alpha_analytic = _analytic_alpha()
    delta_analytic = 1.0 / alpha_analytic

    objects, constraints, config, volume, wave = _build_base()
    _add_conductor(volume, objects, constraints)
    _add_phasor_det("d1", _DET1_Z, wave, volume, objects, constraints)
    _add_phasor_det("d2", _DET2_Z, wave, volume, objects, constraints)

    arrays = _run(objects, constraints, config)
    p1 = _ex_phasor(arrays, "d1")
    p2 = _ex_phasor(arrays, "d2")

    assert abs(p1) > 0, "Detector d1 measured zero Ex amplitude"
    assert abs(p2) > 0, "Detector d2 measured zero Ex amplitude"
    assert abs(p2) < abs(p1), f"Wave is not attenuating: |p2|={abs(p2):.4e} >= |p1|={abs(p1):.4e}"

    alpha_measured = -np.log(abs(p2) / abs(p1)) / _DET_SEP
    delta_measured = 1.0 / alpha_measured

    rel_err = abs(alpha_measured - alpha_analytic) / abs(alpha_analytic)
    assert rel_err < _TOLERANCE, (
        f"α_measured={alpha_measured:.4e} m⁻¹, α_analytic={alpha_analytic:.4e} m⁻¹, "
        f"δ_measured={delta_measured * 1e9:.1f} nm, δ_analytic={delta_analytic * 1e9:.1f} nm, "
        f"relative error={rel_err:.3f} > {_TOLERANCE}"
    )
