"""Physics simulation test: Bloch boundary with oblique incidence.

Validates Fresnel transmission at oblique incidence using Bloch BCs.
A plane wave is launched at angle θ = 60° from normal in the xz-plane
using the TFSF source's azimuth_angle parameter.  Bloch BCs on x
provide the correct transverse phase condition k_x = k₀ sin(θ).

The angle is deliberately large (60°).  At 30° the TE transmittance
(T = 0.854) sits only ~4 % below the normal-incidence value
(T = 0.889), so a 5 % tolerance could not tell the oblique result
apart from a normal-incidence bug.  At 60° the TE transmittance drops
to T = 0.680 — ~24 % below normal incidence — so the 5 % tolerance now
genuinely discriminates the angular Fresnel physics.

TE polarization (s-pol): E_y perpendicular to the plane of incidence (xz).
Interface: vacuum (n₁=1) → dielectric ε_r=4 (n₂=2) at z = z_interface.

Analytic Fresnel coefficients for TE at θ_i = 60°:
  θ_t = arcsin(sin(60°)/2) = 25.66°
  r_s = (n₁ cos θ_i - n₂ cos θ_t) / (n₁ cos θ_i + n₂ cos θ_t) = -0.420
  t_s = 2 n₁ cos θ_i / (n₁ cos θ_i + n₂ cos θ_t) = 0.580
  T_power = (n₂ cos θ_t) / (n₁ cos θ_i) x |t_s|² = 0.680

A transverse phase-ramp check supplements the flux test: a PhasorDetector
samples E_y across x at fixed z and the measured phase gradient must match
the imposed transverse wave number k_x = k₀ sin(θ).

Two-run normalization:
  Reference: oblique wave in vacuum (no interface), flux detector at z_det
  Fresnel:   oblique wave with dielectric half-space, same detector position
  T_measured = S_Fresnel / S_reference

Domain: Bloch on x (20 cells), periodic on y (3 cells), PML on z (160 cells).
Resolution: 25 nm (40 cells/λ).
Tolerance: 5 %.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
_RESOLUTION = 25e-9  # 40 cells/λ
_PML_CELLS = 10
_N_X = 20  # cells in x (Bloch direction)
_DOMAIN_X = _N_X * _RESOLUTION  # 0.5 µm
_DOMAIN_Y = 3 * _RESOLUTION
_DOMAIN_Z = 4e-6  # 160 cells

_Z_CELLS = round(_DOMAIN_Z / _RESOLUTION)  # = 160

_THETA_DEG = 60.0  # incidence angle (degrees) — large angle so TE T differs >>5% from normal
_THETA = np.deg2rad(_THETA_DEG)
_K0 = 2 * np.pi / _WAVELENGTH
_KX = _K0 * np.sin(_THETA)

_SOURCE_Z = _PML_CELLS + 4  # = 14
_INTERFACE_Z = 80  # midpoint
_DIEL_CELLS_Z = _Z_CELLS - _INTERFACE_Z  # = 80
_DET_T_Z = 120  # well into dielectric

_SIM_TIME = 120e-15
_TOLERANCE = 0.05

# Time averaging
_DT_APPROX = 0.99 * _RESOLUTION / (3e8 * np.sqrt(3))
_STEPS_PER_PERIOD = round(_WAVELENGTH / (3e8 * _DT_APPROX))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD

# Analytic Fresnel TE coefficients
_N1, _N2 = 1.0, 2.0  # n = sqrt(ε_r=4)
_THETA_T = np.arcsin(np.sin(_THETA) / _N2)
_COS_I, _COS_T = np.cos(_THETA), np.cos(_THETA_T)
_T_ANALYTIC = (_N2 * _COS_T) / (_N1 * _COS_I) * (2 * _N1 * _COS_I / (_N1 * _COS_I + _N2 * _COS_T)) ** 2


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_base():
    """Domain with Bloch x, periodic y, PML z, oblique TE source."""
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(_DOMAIN_X, _DOMAIN_Y, _DOMAIN_Z),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_x": "bloch",
            "max_x": "bloch",
            "min_y": "periodic",
            "max_y": "periodic",
        },
        bloch_vector=(_KX, 0.0, 0.0),
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    # TE polarization: E_y perpendicular to xz plane of incidence
    # azimuth_angle tilts wave vector in the horizontal(x)-propagation(z) plane
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(0, 1, 0),  # TE: E along y
        azimuth_angle=_THETA_DEG,
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)

    return objects, constraints, config, volume


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


def _add_dielectric(volume, objects, constraints):
    diel = fdtdx.UniformMaterialObject(
        partial_grid_shape=(None, None, _DIEL_CELLS_Z),
        material=fdtdx.Material(permittivity=_N2**2),
    )
    constraints.extend(
        [
            diel.same_size(volume, axes=(0, 1)),
            diel.place_at_center(volume, axes=(0, 1)),
            diel.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_INTERFACE_Z,)),
        ]
    )
    objects.append(diel)


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


def _add_xline_phasor(name, z_idx, volume, objects, constraints):
    """PhasorDetector spanning the full Bloch (x) extent at one y, z (Ey)."""
    det = fdtdx.PhasorDetector(
        name=name,
        partial_grid_shape=(None, 1, 1),  # full x, single y, single z
        wave_characters=(fdtdx.WaveCharacter(wavelength=_WAVELENGTH),),
        reduce_volume=False,
        components=("Ey",),
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0,)),
            det.place_at_center(volume, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(z_idx,)),
        ]
    )
    objects.append(det)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_bloch_oblique_fresnel_te():
    """Fresnel TE transmission at 60° matches analytic T within 5 %.

    At 60° the analytic TE transmittance (T = 0.680) lies ~24 % below the
    normal-incidence value (T = 0.889), so the 5 % tolerance genuinely
    discriminates the angular Fresnel physics (at 30° the two differ by
    only ~4 %, inside the tolerance).

    Two-run normalization eliminates absolute calibration:
      Reference: oblique wave in vacuum → S₀
      Fresnel:   oblique wave + dielectric half-space → S_T
      T_measured = S_T / S₀
    """
    # Reference: vacuum, oblique wave
    obj_ref, con_ref, cfg_ref, vol_ref = _build_base()
    _add_flux_det("flux_t", _DET_T_Z, vol_ref, obj_ref, con_ref)
    S_ref = _mean_flux(_run(obj_ref, con_ref, cfg_ref), "flux_t")

    # Fresnel: dielectric from interface to z-max
    obj_fr, con_fr, cfg_fr, vol_fr = _build_base()
    _add_dielectric(vol_fr, obj_fr, con_fr)
    _add_flux_det("flux_t", _DET_T_Z, vol_fr, obj_fr, con_fr)
    S_T = _mean_flux(_run(obj_fr, con_fr, cfg_fr), "flux_t")

    assert S_ref > 0, f"Reference flux is zero/negative: {S_ref}"
    assert S_T > 0, f"Transmitted flux is zero/negative: {S_T}"

    T_measured = S_T / S_ref
    rel_err = abs(T_measured - _T_ANALYTIC) / _T_ANALYTIC
    assert rel_err < _TOLERANCE, (
        f"Bloch oblique Fresnel TE (θ={_THETA_DEG}°): "
        f"T_measured={T_measured:.4f}, T_analytic={_T_ANALYTIC:.4f}, "
        f"relative error={rel_err:.3f}"
    )


def test_bloch_oblique_transverse_phase_ramp():
    """Transverse phase gradient of Ey across x equals k_x = k₀ sin(θ).

    The oblique source + Bloch BCs impose a transverse plane-wave phase
    exp(i k_x x).  A line PhasorDetector sampling Ey across the full
    Bloch (x) extent in vacuum must therefore show a linear phase ramp
    whose slope is k_x = k₀ sin(60°).  This directly confirms the
    transverse wave number (independent of the flux/Fresnel test).

    Tolerance: 5 % on the measured-vs-imposed k_x (matches the file's
    Fresnel tolerance); a discrete λ/Δx = 40 cells-per-wavelength grid
    resolves the per-cell phase step k_x·Δx = 0.34 rad well below π.
    """
    # Detector well past the source, before the (absent) interface — pure vacuum.
    z_probe = (_SOURCE_Z + _INTERFACE_Z) // 2
    obj, con, cfg, vol = _build_base()
    _add_xline_phasor("xline", z_probe, vol, obj, con)
    arrays = _run(obj, con, cfg)

    # phasor shape: (1, num_freqs=1, num_components=1, Nx, 1, 1)
    phasor = np.array(arrays.detector_states["xline"]["phasor"][0, 0, 0, :, 0, 0])
    assert np.all(np.abs(phasor) > 0), "Phasor line has zero-amplitude cells"

    # Unwrapped phase along x; least-squares slope gives the per-cell phase step.
    phase = np.unwrap(np.angle(phasor))
    x_idx = np.arange(phase.size)
    slope = np.polyfit(x_idx, phase, 1)[0]  # rad per cell
    kx_measured = abs(slope) / _RESOLUTION  # rad/m

    rel_err = abs(kx_measured - _KX) / _KX
    assert rel_err < _TOLERANCE, (
        f"Transverse phase ramp (θ={_THETA_DEG}°): "
        f"kx_measured={kx_measured:.4e}, kx_imposed={_KX:.4e} (rad/m), "
        f"per-cell step measured={abs(slope):.4f} vs imposed={_KX * _RESOLUTION:.4f} rad, "
        f"relative error={rel_err:.3f}"
    )
