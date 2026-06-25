"""Physics simulation tests for dispersive (Lorentz / Drude) materials.

Normal-incidence transmission/reflection through a semi-infinite dispersive
half-space at a single frequency. The dispersive ADE update must reproduce
the analytic Fresnel coefficient computed from the DispersionModel's
susceptibility at the test frequency.

Layout mirrors ``test_fresnel.py`` — 3x3 periodic transverse, PMLs in z,
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
_Z_CELLS = round(_DOMAIN_Z / _RESOLUTION)  # = 200

_SOURCE_Z = _PML_CELLS + 2  # = 12
_INTERFACE_Z = 100
_DET_T_Z = 140
# Vacuum-side detector just outside the interface. For a strongly absorbing metal the
# transmitted wave is gone within a skin depth (~1 cell), so a deep detector reads ~0;
# the net forward flux here equals the transmitted/absorbed power (1 - R) by energy
# conservation and is what we compare to the Fresnel power transmission.
_DET_R_Z = 99

# Layout for tests with a source fully embedded in a uniform dispersive medium.
_UNIFORM_SOURCE_Z = 60
_UNIFORM_FWD_Z = 90
_UNIFORM_BWD_Z = 30

# Two forward phasor detectors for measuring k(omega) = Re(n(omega)) * omega / c inside
# a uniform dispersive medium (3-cell separation, below one medium wavelength).
_PHASE_D1_Z = 70
_PHASE_D2_Z = 73
_PHASE_SEP = (_PHASE_D2_Z - _PHASE_D1_Z) * _RESOLUTION

_DIEL_CELLS_Z = _Z_CELLS - _INTERFACE_Z  # = 100 cells

_SIM_TIME = 120e-15
_TOLERANCE = 0.05  # Lorentz transmission (single frequency)
_DRUDE_REL_TOL = 0.10  # Drude transmission relative tolerance
_DRUDE_T_FLOOR = 0.01  # nonzero floor so "transmits nothing" (T=0) fails
_PHASE_TOL = 0.03  # Lorentz n(omega) phase-velocity tolerance (achieved ≈ 1-2 %)

_DT_APPROX = 0.99 * _RESOLUTION / (c0 * np.sqrt(3))
_STEPS_PER_PERIOD = round(_WAVELENGTH / (c0 * _DT_APPROX))
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
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
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


def _fill_domain(material, volume, objects, constraints):
    """Fill the entire z-extent of the domain with ``material``."""
    slab = fdtdx.UniformMaterialObject(
        partial_grid_shape=(None, None, _Z_CELLS),
        material=material,
    )
    constraints.extend(
        [
            slab.same_size(volume, axes=(0, 1)),
            slab.place_at_center(volume, axes=(0, 1)),
            slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(0,)),
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


def _build_uniform_lorentz_phase(wavelength: float, model):
    """Uniform Lorentz-filled domain with an embedded +z source and two forward
    PhasorDetectors, for measuring k(omega) = Re(n(omega)) * omega / c in the medium.

    Measuring the phase velocity inside a uniform medium avoids the interface and
    Poynting time-staggering biases of the transmission tests, so it stays accurate
    as omega approaches the Lorentz resonance (where Re(eps) departs strongly from
    its static value).
    """
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
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

    wave = fdtdx.WaveCharacter(wavelength=wavelength)
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
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_UNIFORM_SOURCE_Z,)),
        ]
    )
    objects.append(source)

    material = fdtdx.Material(permittivity=1.0, dispersion=model)
    _fill_domain(material, volume, objects, constraints)

    for name, z in (("d1", _PHASE_D1_Z), ("d2", _PHASE_D2_Z)):
        det = fdtdx.PhasorDetector(
            name=name,
            partial_grid_shape=(None, None, 1),
            wave_characters=(wave,),
            components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
            reduce_volume=True,
            plot=False,
        )
        constraints.extend(
            [
                det.same_size(volume, axes=(0, 1)),
                det.place_at_center(volume, axes=(0, 1)),
                det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(z,)),
            ]
        )
        objects.append(det)

    return objects, constraints, config


def test_lorentz_phase_velocity_matches_dispersion_at_multiple_frequencies():
    """The Lorentz phase velocity tracks Re(n(omega)) where eps(omega) departs
    materially from its static (degenerate) value.

    test_lorentz_transmission_matches_fresnel probes only OMEGA, where eps ≈ 4 is
    numerically indistinguishable from a static eps=4 slab. Here we measure k(omega)
    inside a uniform Lorentz medium at two higher frequencies nearer the resonance
    (omega_0 = 2*OMEGA), where Re(eps) grows to ≈ 4.9 and ≈ 7.3, and compare each to
    the closed-form k = Re(sqrt(eps_inf + chi(omega))) * omega / c from
    ``dispersion.py``'s own susceptibility. A static-eps=4 material would fail here.
    """
    model = _lorentz_model()
    eps_inf = 1.0
    # factor = omega / OMEGA; both frequencies move materially away from the static
    # eps ≈ 4 case (factor 1.0). Errors stay ≈ 1-2 % up to factor 1.6.
    for factor in (1.3, 1.6):
        wavelength = _WAVELENGTH / factor
        omega = 2.0 * np.pi * c0 / wavelength
        eps = eps_inf + complex(model.susceptibility(omega))
        # Confirm the test frequency is materially different from a static eps=4 slab.
        assert abs(eps.real - 4.0) > 0.5, f"factor={factor}: eps(omega)={eps} too close to static 4"
        n_real = float(np.sqrt(eps).real)
        k_analytic = omega * n_real / c0
        # Detector separation must stay below one medium wavelength (no 2*pi wrap).
        assert k_analytic * _PHASE_SEP < 2.0 * np.pi

        objects, constraints, config = _build_uniform_lorentz_phase(wavelength, model)
        arrays = _run(objects, constraints, config)
        p1 = complex(arrays.detector_states["d1"]["phasor"][0, 0, 0])
        p2 = complex(arrays.detector_states["d2"]["phasor"][0, 0, 0])
        assert abs(p1) > 0 and abs(p2) > 0, f"factor={factor}: zero Ex phasor (p1={p1}, p2={p2})"

        delta_phi = (np.angle(p2) - np.angle(p1)) % (2.0 * np.pi)
        k_measured = delta_phi / _PHASE_SEP
        rel_err = abs(k_measured - k_analytic) / k_analytic
        assert rel_err < _PHASE_TOL, (
            f"factor={factor} (eps={eps:.3f}, n={n_real:.3f}): k_measured={k_measured:.4e}, "
            f"k_analytic={k_analytic:.4e}, relative error={rel_err:.3f} > {_PHASE_TOL}"
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


def _build_embedded_source(source_z: int):
    """Same as ``_build_base`` but with the plane source placed at an arbitrary
    z-coordinate so it can be embedded inside a uniform medium."""
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
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
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(source_z,)),
        ]
    )
    objects.append(source)

    return objects, constraints, config, volume


def test_plane_source_inside_lorentz_medium_has_correct_impedance():
    """A TFSF source embedded in a uniform Lorentz medium must inject the
    impedance of the medium at the carrier frequency, not of vacuum. If the
    impedance is matched, the backward-scattered flux is close to zero; if
    the source used ``eps_inf`` (the pre-fix behavior) the impedance
    mismatch would reflect ~10 % of the injected power into the backward
    half-space.
    """
    model = _lorentz_model()
    eps_inf = 1.0
    eps_omega = eps_inf + complex(model.susceptibility(_OMEGA))
    # Sanity: the medium must be a meaningfully different impedance from vacuum
    assert abs(np.sqrt(eps_omega.real) - 1.0) > 0.5, "Test premise weak: Lorentz is too close to vacuum"

    obj, con, cfg, vol = _build_embedded_source(_UNIFORM_SOURCE_Z)
    material = fdtdx.Material(permittivity=eps_inf, dispersion=model)
    _fill_domain(material, vol, obj, con)
    _add_flux_det("flux_fwd", _UNIFORM_FWD_Z, vol, obj, con)
    _add_flux_det("flux_bwd", _UNIFORM_BWD_Z, vol, obj, con)

    arrays = _run(obj, con, cfg)
    S_fwd = _mean_flux(arrays, "flux_fwd")
    S_bwd = _mean_flux(arrays, "flux_bwd")

    assert S_fwd > 0, f"Forward flux should be positive, got {S_fwd}"
    # Both detectors have direction='+'; a backward wave registers as a
    # negative flux on the '-' side of the source. Take the magnitude.
    ratio = abs(S_bwd) / abs(S_fwd)
    assert ratio < 0.02, (
        f"Backward/forward flux ratio {ratio:.4f} exceeds 2% — the source "
        "impedance is not matched to the dispersive medium."
    )


def test_drude_metal_is_highly_reflective():
    """A Drude half-space with ω_p ≫ ω reflects ≳ 90 % of incident power.

    The Fresnel power transmission into the metal is small but NONZERO
    (T_analytic ≈ 0.0202). Because the metal absorbs the transmitted wave within a
    skin depth (~1 cell), a deep detector reads ~0 — an absolute tolerance of 0.05
    there would let a spurious "transmits nothing" (T=0) pass. Instead we measure the
    net forward Poynting flux just outside the interface (z=99), which by energy
    conservation equals the transmitted/absorbed power (1 - R), and assert a RELATIVE
    tolerance plus a nonzero floor so T=0 genuinely fails.
    """
    model = _drude_model()
    eps_inf = 1.0
    eps_omega = eps_inf + complex(model.susceptibility(_OMEGA))
    T_analytic = _fresnel_transmission_semi_infinite(eps_omega)
    # Sanity: Drude above plasma limit should reflect strongly → T small but nonzero.
    assert _DRUDE_T_FLOOR < T_analytic < 0.05, f"Test premise wrong: Drude T_analytic={T_analytic:.3f}"

    obj0, con0, cfg0, vol0 = _build_base()
    _add_flux_det("flux_r", _DET_R_Z, vol0, obj0, con0)
    S0 = _mean_flux(_run(obj0, con0, cfg0), "flux_r")

    obj1, con1, cfg1, vol1 = _build_base()
    material = fdtdx.Material(permittivity=eps_inf, dispersion=model)
    _add_half_space(material, vol1, obj1, con1)
    _add_flux_det("flux_r", _DET_R_Z, vol1, obj1, con1)
    S_T = _mean_flux(_run(obj1, con1, cfg1), "flux_r")

    assert S0 > 0, f"Reference flux zero: {S0}"

    # Net forward flux normalized by the vacuum incident flux = 1 - R = transmitted power.
    T_measured = S_T / S0
    # Nonzero floor: a metal that "transmits nothing" (T=0) must fail this test.
    assert T_measured > _DRUDE_T_FLOOR, (
        f"Drude T_measured={T_measured:.4f} ≤ floor {_DRUDE_T_FLOOR}: appears to transmit nothing"
    )
    rel_err = abs(T_measured - T_analytic) / T_analytic
    assert rel_err < _DRUDE_REL_TOL, (
        f"Drude T_measured={T_measured:.4f}, T_analytic={T_analytic:.4f}, "
        f"relative error={rel_err:.3f} > {_DRUDE_REL_TOL}"
    )
