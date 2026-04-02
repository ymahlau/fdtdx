"""Physics simulation test: Bloch boundary with oblique incidence.

Validates Fresnel transmission at oblique incidence using Bloch BCs.
A plane wave is launched at angle θ = 30° from normal in the xz-plane
using the TFSF source's azimuth_angle parameter.  Bloch BCs on x
provide the correct transverse phase condition k_x = k₀ sin(θ).

TE polarization (s-pol): E_y perpendicular to the plane of incidence (xz).
Interface: vacuum (n₁=1) → dielectric ε_r=4 (n₂=2) at z = z_interface.

Analytic Fresnel coefficients for TE at θ_i = 30°:
  θ_t = arcsin(sin(30°)/2) = 14.48°
  r_s = (n₁ cos θ_i − n₂ cos θ_t) / (n₁ cos θ_i + n₂ cos θ_t) = −0.382
  t_s = 2 n₁ cos θ_i / (n₁ cos θ_i + n₂ cos θ_t) = 0.618
  T_power = (n₂ cos θ_t) / (n₁ cos θ_i) × |t_s|² = 0.854

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

_Z_CELLS = int(round(_DOMAIN_Z / _RESOLUTION))  # = 160

_THETA_DEG = 30.0  # incidence angle (degrees)
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
_STEPS_PER_PERIOD = int(round(_WAVELENGTH / (3e8 * _DT_APPROX)))
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
        resolution=_RESOLUTION,
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


# ── Tests ────────────────────────────────────────────────────────────────────


def test_bloch_oblique_fresnel_te():
    """Fresnel TE transmission at 30° matches analytic T within 5 %.

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
