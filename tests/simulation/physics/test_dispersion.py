"""Physics simulation tests for dispersive (Lorentz / Drude) materials.

Normal-incidence transmission/reflection through a semi-infinite dispersive
half-space at a single frequency. The dispersive ADE update must reproduce
the analytic Fresnel coefficient computed from the DispersionModel's
susceptibility at the test frequency.

Layout mirrors ``test_fresnel.py`` — 3×3 periodic transverse, PMLs in z,
``UniformPlaneSource`` in +z, one transmission-side Poynting flux detector.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx
from fdtdx.constants import c as c0

_WAVELENGTH = 1e-6
_OMEGA = 2.0 * np.pi * c0 / _WAVELENGTH  # ≈ 1.884e15 rad/s
_RESOLUTION = 25e-9  # 40 cells/λ in vacuum
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION
_DOMAIN_Z = 5e-6
_Z_CELLS = int(round(_DOMAIN_Z / _RESOLUTION))  # = 200

_SOURCE_Z = _PML_CELLS + 2  # = 12
_INTERFACE_Z = 100
_DET_T_Z = 140

_DIEL_CELLS_Z = _Z_CELLS - _INTERFACE_Z  # = 100 cells

_SIM_TIME = 120e-15
_TOLERANCE = 0.05

_DT_APPROX = 0.99 * _RESOLUTION / (c0 * np.sqrt(3))
_STEPS_PER_PERIOD = int(round(_WAVELENGTH / (c0 * _DT_APPROX)))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD


# ---------------------------------------------------------------------------
# Dispersion models used by the tests
# ---------------------------------------------------------------------------


def _lorentz_model():
    """A Lorentz pole whose resonance is well above the test frequency, so
    Im(ε) is tiny at ω and the medium behaves like a low-loss dielectric."""
    # omega_0 = 2*omega, Δε chosen so that ε_inf + Re(χ) ≈ 4 (n ≈ 2)
    omega_0 = 2.0 * _OMEGA
    gamma = 1e13
    # At ω = omega_0/2, Re(χ) = Δε·ω₀²/(ω₀² - ω²) = Δε · 4/3
    # For ε_inf=1 and target ε=4, we want Re(χ) = 3 → Δε = 9/4 = 2.25
    delta_eps = 2.25
    return fdtdx.DispersionModel(
        poles=(fdtdx.LorentzPole(resonance_frequency=omega_0, damping=gamma, delta_epsilon=delta_eps),)
    )


def _drude_model():
    """A Drude pole with ω_p ≫ ω, damping small compared to ω — gives ε with
    a large negative real part, i.e. a highly reflective metallic response."""
    omega_p = 5.0 * _OMEGA
    gamma = 0.05 * _OMEGA
    return fdtdx.DispersionModel(poles=(fdtdx.DrudePole(plasma_frequency=omega_p, damping=gamma),))


def _fresnel_transmission_semi_infinite(eps_complex: complex) -> float:
    """Power transmission coefficient from vacuum into a semi-infinite
    medium with complex permittivity ``eps_complex`` at normal incidence.

    T = Re(n2) · |t|^2 with t = 2 / (1 + n2) and n2 = sqrt(eps_complex).
    """
    n2 = np.sqrt(eps_complex)
    t = 2.0 / (1.0 + n2)
    return float(np.real(n2) * np.abs(t) ** 2)


# ---------------------------------------------------------------------------
# Scene builders — mirror test_fresnel.py
# ---------------------------------------------------------------------------


def _build_base():
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


def _add_half_space(material, volume, objects, constraints):
    """Fill cells [_INTERFACE_Z, _Z_CELLS) with ``material``."""
    slab = fdtdx.UniformMaterialObject(
        partial_grid_shape=(None, None, _DIEL_CELLS_Z),
        material=material,
    )
    constraints.extend(
        [
            slab.same_size(volume, axes=(0, 1)),
            slab.place_at_center(volume, axes=(0, 1)),
            slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_INTERFACE_Z,)),
        ]
    )
    objects.append(slab)


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


# ---------------------------------------------------------------------------
# Lorentz tests
# ---------------------------------------------------------------------------


def test_lorentz_transmission_matches_fresnel():
    """Semi-infinite Lorentz dielectric transmits as predicted by Fresnel.

    Two-run normalization: vacuum reference establishes S0, Lorentz run gives
    S_T, and T_measured = S_T / S0 is compared to the analytic Fresnel
    transmission evaluated from the model's own susceptibility at the source
    frequency.
    """
    model = _lorentz_model()
    eps_inf = 1.0
    eps_omega = eps_inf + complex(model.susceptibility(_OMEGA))
    T_analytic = _fresnel_transmission_semi_infinite(eps_omega)

    # Reference run: vacuum everywhere
    obj0, con0, cfg0, vol0 = _build_base()
    _add_flux_det("flux_t", _DET_T_Z, vol0, obj0, con0)
    S0 = _mean_flux(_run(obj0, con0, cfg0), "flux_t")

    # Dispersive run
    obj1, con1, cfg1, vol1 = _build_base()
    material = fdtdx.Material(permittivity=eps_inf, dispersion=model)
    _add_half_space(material, vol1, obj1, con1)
    _add_flux_det("flux_t", _DET_T_Z, vol1, obj1, con1)
    S_T = _mean_flux(_run(obj1, con1, cfg1), "flux_t")

    assert S0 > 0, f"Reference flux zero: {S0}"
    assert S_T > 0, f"Dispersive transmitted flux zero: {S_T}"

    T_measured = S_T / S0
    rel_err = abs(T_measured - T_analytic) / T_analytic
    assert rel_err < _TOLERANCE, (
        f"Lorentz T_measured={T_measured:.4f}, "
        f"T_analytic={T_analytic:.4f} (eps={eps_omega}), "
        f"rel_err={rel_err:.3f} > {_TOLERANCE}"
    )


def test_lorentz_permittivity_sanity():
    """Quick unit-level sanity check on the Lorentz model itself so failures
    in the simulation test are easier to attribute."""
    model = _lorentz_model()
    eps_inf = 1.0
    eps = eps_inf + model.susceptibility(_OMEGA)
    # Target was Re(ε) ≈ 4, Im(ε) small
    assert abs(eps.real - 4.0) < 0.05
    assert eps.imag > 0  # causal absorption sign
    assert eps.imag < 0.01 * eps.real


# ---------------------------------------------------------------------------
# Drude test
# ---------------------------------------------------------------------------


def test_drude_metal_is_highly_reflective():
    """A Drude half-space with ω_p ≫ ω reflects ≳ 90 % of incident power.

    Measures transmitted flux and checks it is a small fraction of the
    vacuum-reference flux, matching the Fresnel prediction for the complex
    permittivity at the source frequency within 5 %.
    """
    model = _drude_model()
    eps_inf = 1.0
    eps_omega = eps_inf + complex(model.susceptibility(_OMEGA))
    T_analytic = _fresnel_transmission_semi_infinite(eps_omega)
    # Sanity: Drude above plasma limit should reflect strongly → T small
    assert T_analytic < 0.2, f"Test premise wrong: Drude T_analytic={T_analytic:.3f} not small"

    obj0, con0, cfg0, vol0 = _build_base()
    _add_flux_det("flux_t", _DET_T_Z, vol0, obj0, con0)
    S0 = _mean_flux(_run(obj0, con0, cfg0), "flux_t")

    obj1, con1, cfg1, vol1 = _build_base()
    material = fdtdx.Material(permittivity=eps_inf, dispersion=model)
    _add_half_space(material, vol1, obj1, con1)
    _add_flux_det("flux_t", _DET_T_Z, vol1, obj1, con1)
    S_T = _mean_flux(_run(obj1, con1, cfg1), "flux_t")

    assert S0 > 0, f"Reference flux zero: {S0}"

    T_measured = S_T / S0
    # Absolute rather than relative tolerance because T_analytic is close to 0
    assert abs(T_measured - T_analytic) < _TOLERANCE, (
        f"Drude T_measured={T_measured:.4f}, T_analytic={T_analytic:.4f}, "
        f"|diff|={abs(T_measured - T_analytic):.3f} > {_TOLERANCE}"
    )


# ---------------------------------------------------------------------------
# Reversible-gradient sanity: the algebraically-inverted polarization
# recurrence in update_E_reverse must produce a finite gradient for a loss
# that depends on a dispersive simulation run.
# ---------------------------------------------------------------------------


def _build_lorentz_scene_for_grad():
    """Same Lorentz half-space scene, returning objects/constraints/config
    (without the Recorder wiring, which the caller sets per test)."""
    obj, con, cfg, vol = _build_base()
    material = fdtdx.Material(permittivity=1.0, dispersion=_lorentz_model())
    _add_half_space(material, vol, obj, con)
    _add_flux_det("flux_t", _DET_T_Z, vol, obj, con)
    return obj, con, cfg


def test_lorentz_reversible_forward_runs():
    """Reversible gradient config forward-only run works with dispersive
    materials — exercises the per-step primal plumbing in reversible_fdtd.
    """
    recorder = fdtdx.Recorder(modules=[fdtdx.DtypeConversion(dtype=jnp.bfloat16)])
    gradient_config = fdtdx.GradientConfig(method="reversible", recorder=recorder)

    objects, constraints, config = _build_lorentz_scene_for_grad()
    config = config.aset("gradient_config", gradient_config)
    S_T = _mean_flux(_run(objects, constraints, config), "flux_t")
    assert S_T > 0, f"Reversible Lorentz flux zero: {S_T}"
    assert np.isfinite(S_T)
