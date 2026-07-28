"""Physics simulation tests for oriented (off-diagonal) dispersion.

An oriented pole is a 1D Lorentz oscillator along a unit vector u,
contributing chi(omega) u u^T to the susceptibility tensor. These tests
validate the fully anisotropic ADE update against the (independently
validated) diagonal per-axis path:

* a pole oriented along x must reproduce a per-axis pole (de, 0, 0),
* a crystal rotated 45 degrees about the propagation axis must transmit
  the same power as the grid-aligned crystal probed with a 45-degree
  polarized source (same physics, different code path),
* a monoclinic-style pair of non-orthogonal in-plane oscillators must
  run stably (passivity of the K u u^T coupling),
* gradients flow through the checkpointed path; the reversible path
  rejects oriented dispersion.

Layout mirrors ``test_anisotropic_dispersion.py`` — 3x3 periodic
transverse, PMLs in z, ``UniformPlaneSource`` in +z, one transmission-side
Poynting flux detector.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.constants import c as c0

_WAVELENGTH = 1e-6
_OMEGA = 2.0 * np.pi * c0 / _WAVELENGTH
_RESOLUTION = 25e-9
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION
_DOMAIN_Z = 5e-6
_Z_CELLS = round(_DOMAIN_Z / _RESOLUTION)

_SOURCE_Z = _PML_CELLS + 2
_INTERFACE_Z = 100
_DET_T_Z = 140
_DIEL_CELLS_Z = _Z_CELLS - _INTERFACE_Z

_SIM_TIME = 120e-15
_TOLERANCE = 0.05

_DT_APPROX = 0.99 * _RESOLUTION / (c0 * np.sqrt(3))
_STEPS_PER_PERIOD = round(_WAVELENGTH / (c0 * _DT_APPROX))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD

_LORENTZ_W0 = 2.0 * _OMEGA
_LORENTZ_GAMMA = 1e13
_DE_A = 2.25  # strong axis
_DE_B = 0.75  # weak axes

_SQRT_HALF = float(1.0 / np.sqrt(2.0))


def _per_axis_model():
    return fdtdx.DispersionModel(
        poles=(
            fdtdx.LorentzPole(
                resonance_frequency=_LORENTZ_W0,
                damping=_LORENTZ_GAMMA,
                delta_epsilon=(_DE_A, _DE_B, _DE_B),
            ),
        )
    )


def _build_base(polarization, sim_time=_SIM_TIME):
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=sim_time,
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
        fixed_E_polarization_vector=polarization,
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


def _transmitted_flux(material, polarization, sim_time=_SIM_TIME):
    obj, con, cfg, vol = _build_base(polarization, sim_time=sim_time)
    if material is not None:
        _add_half_space(material, vol, obj, con)
    _add_flux_det("flux_t", _DET_T_Z, vol, obj, con)
    return _mean_flux(_run(obj, con, cfg), "flux_t")


def test_oriented_along_x_matches_per_axis():
    """A pole oriented along x is the same physics as a per-axis pole
    (de, 0, 0) — but runs through the fully anisotropic kernel instead of the
    diagonal one. Transmitted flux must agree closely (the update algebra is
    identical up to the matrix-solve formulation; float32 op order differs)."""
    oriented = fdtdx.Material(
        permittivity=1.0,
        dispersion=fdtdx.DispersionModel(
            poles=(
                fdtdx.LorentzPole(
                    resonance_frequency=_LORENTZ_W0,
                    damping=_LORENTZ_GAMMA,
                    delta_epsilon=_DE_A,
                    orientation=(1.0, 0.0, 0.0),
                ),
            )
        ),
    )
    per_axis = fdtdx.Material(
        permittivity=1.0,
        dispersion=fdtdx.DispersionModel(
            poles=(
                fdtdx.LorentzPole(
                    resonance_frequency=_LORENTZ_W0,
                    damping=_LORENTZ_GAMMA,
                    delta_epsilon=(_DE_A, 0.0, 0.0),
                ),
            )
        ),
    )
    s_oriented = _transmitted_flux(oriented, (1, 0, 0))
    s_per_axis = _transmitted_flux(per_axis, (1, 0, 0))
    assert s_per_axis > 0
    rel = abs(s_oriented - s_per_axis) / abs(s_per_axis)
    assert rel < 1e-2, (
        f"oriented-along-x diverged from per-axis reference: "
        f"S_oriented={s_oriented:.6e}, S_per_axis={s_per_axis:.6e}, rel={rel:.2e}"
    )


def test_rotated_crystal_matches_grid_aligned():
    """A dispersive uniaxial crystal rotated 45 degrees about the propagation
    axis, probed with an x-polarized wave, transmits the same power as the
    grid-aligned crystal probed at 45-degree polarization."""
    model = _per_axis_model()
    aligned = fdtdx.Material(permittivity=1.0, dispersion=model)
    rotated = fdtdx.Material(permittivity=1.0, dispersion=model.rotated((0.0, 0.0, float(np.pi / 4))))
    assert rotated.dispersion is not None and rotated.dispersion.has_off_diagonal_coupling

    s_aligned = _transmitted_flux(aligned, (_SQRT_HALF, _SQRT_HALF, 0.0))
    s_rotated = _transmitted_flux(rotated, (1.0, 0.0, 0.0))
    assert s_aligned > 0, f"grid-aligned transmitted flux vanished: {s_aligned}"
    assert s_rotated > 0, f"rotated-crystal transmitted flux vanished: {s_rotated}"

    rel = abs(s_rotated - s_aligned) / s_aligned
    assert rel < _TOLERANCE, (
        f"rotated dispersive crystal S={s_rotated:.4e} disagrees with grid-aligned S={s_aligned:.4e}, "
        f"rel={rel:.3f} > {_TOLERANCE}"
    )


def test_monoclinic_oscillator_pair_is_stable():
    """Two non-orthogonal in-plane oscillators (30 degrees apart) — the
    monoclinic / shear-polariton configuration. The K u u^T couplings are
    positive semi-definite, so a long CW run must stay finite (no blow-up)
    and absorb/transmit a sane fraction of the incident power."""
    ang = np.pi / 6
    model = fdtdx.DispersionModel(
        poles=(
            fdtdx.LorentzPole(
                resonance_frequency=_LORENTZ_W0,
                damping=_LORENTZ_GAMMA,
                delta_epsilon=_DE_A,
                orientation=(1.0, 0.0, 0.0),
            ),
            fdtdx.LorentzPole(
                resonance_frequency=1.5 * _OMEGA,
                damping=2e13,
                delta_epsilon=_DE_B,
                orientation=(float(np.cos(ang)), float(np.sin(ang)), 0.0),
            ),
        )
    )
    material = fdtdx.Material(permittivity=1.0, dispersion=model)

    obj, con, cfg, vol = _build_base((_SQRT_HALF, _SQRT_HALF, 0.0))
    _add_half_space(material, vol, obj, con)
    _add_flux_det("flux_t", _DET_T_Z, vol, obj, con)
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, cfg, _ = fdtdx.place_objects(object_list=obj, config=cfg, constraints=con, key=key)
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=cfg, key=key)

    assert bool(jnp.all(jnp.isfinite(arrays.fields.E))), "fields blew up in monoclinic medium"
    assert bool(jnp.all(jnp.isfinite(arrays.fields.dispersive_P_curr)))
    s_vac = _transmitted_flux(None, (_SQRT_HALF, _SQRT_HALF, 0.0))
    s_t = _mean_flux(arrays, "flux_t")
    assert 0.0 < s_t < s_vac, f"unphysical transmission through lossy monoclinic slab: {s_t} vs vacuum {s_vac}"


def test_reversible_gradient_rejected_at_runtime():
    """run_fdtd must reject the reversible gradient method for oriented
    dispersion even when the gradient config is attached after placement."""
    material = fdtdx.Material(
        permittivity=1.0,
        dispersion=fdtdx.DispersionModel(
            poles=(
                fdtdx.LorentzPole(
                    resonance_frequency=_LORENTZ_W0,
                    damping=_LORENTZ_GAMMA,
                    delta_epsilon=_DE_A,
                    orientation=(1.0, 1.0, 0.0),
                ),
            )
        ),
    )
    obj, con, cfg, vol = _build_base((1, 0, 0))
    _add_half_space(material, vol, obj, con)
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, cfg, _ = fdtdx.place_objects(object_list=obj, config=cfg, constraints=con, key=key)
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    grad_cfg = fdtdx.GradientConfig(method="reversible", recorder=fdtdx.Recorder(modules=[]))
    cfg = cfg.aset("gradient_config", grad_cfg)
    with pytest.raises(NotImplementedError, match="checkpointed"):
        fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=cfg, key=key)


def _tilted_uniaxial_material(inclination_rad: float):
    """Dispersive uniaxial crystal with the optic axis in the xz plane,
    inclined ``inclination_rad`` above the surface (xy) plane — the tilted-cut
    geometry of ghost-polariton experiments. Returns (material, eps_tensor_fn)
    where eps_tensor_fn(omega) is the analytic complex permittivity tensor."""
    eps_inf_perp, eps_inf_par = 2.5, 2.0
    # perpendicular (E normal to c) resonance below the drive -> eps_perp < 0 in-band
    perp_pole = fdtdx.LorentzPole(
        resonance_frequency=0.75 * _OMEGA,
        damping=0.01 * _OMEGA,
        delta_epsilon=(3.0, 3.0, 0.0),
    )
    # parallel resonance far above the drive -> eps_par > 0, weakly dispersive
    par_pole = fdtdx.LorentzPole(
        resonance_frequency=2.5 * _OMEGA,
        damping=0.01 * _OMEGA,
        delta_epsilon=(0.0, 0.0, 1.0),
    )
    crystal_model = fdtdx.DispersionModel(poles=(perp_pole, par_pole))

    tilt = float(np.pi / 2 - inclination_rad)
    ct, st = float(np.cos(tilt)), float(np.sin(tilt))
    r_mat = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    eps_inf = r_mat @ np.diag([eps_inf_perp, eps_inf_perp, eps_inf_par]) @ r_mat.T
    model = crystal_model.rotated(tuple(tuple(float(v) for v in row) for row in r_mat))
    material = fdtdx.Material(
        permittivity=tuple(tuple(float(v) for v in row) for row in eps_inf),
        dispersion=model,
    )

    def eps_tensor(omega: float) -> np.ndarray:
        return model.permittivity_tensor(omega, eps_inf=tuple(tuple(float(v) for v in row) for row in eps_inf))

    return material, eps_tensor


def _analytic_flux_fraction(n_eff_sq: complex, distance: float) -> float:
    """Poynting flux fraction ``distance`` into a half-space with effective
    index squared ``n_eff_sq`` at normal incidence: Fresnel interface
    transmission times bulk attenuation."""
    n_eff = np.sqrt(n_eff_sq)
    if n_eff.real < 0:
        n_eff = -n_eff
    t = 2.0 / (1.0 + n_eff)
    interface = float(np.real(n_eff) * np.abs(t) ** 2)
    return interface * float(np.exp(-2.0 * n_eff.imag * (_OMEGA / c0) * distance))


class TestTiltedHalfspaceNormalIncidence:
    """The decisive quantitative audit of the off-diagonal (eps_xz) coupling.

    At normal incidence (k along z) on a homogeneous half-space with
    eps_xy = eps_yz = 0, Maxwell's equations give exact effective indices:
    an x-polarized wave propagates with n^2 = eps_xx - eps_xz^2 / eps_zz
    (the off-diagonal coupling enters the observable directly, through the
    longitudinal Ez it drives), a y-polarized wave with n^2 = eps_yy. Both
    yield analytic complex Fresnel transmission. A sign or index error in the
    oriented-dispersion coupling tensor or its Yee-averaged application would
    shift the measured transmission away from these values."""

    def test_transmission_matches_effective_index_both_polarizations(self):
        material, eps_tensor = _tilted_uniaxial_material(np.deg2rad(23.0))
        eps = eps_tensor(_OMEGA)
        # premise: hyperbolic in-plane (eps_yy < 0) with substantial eps_xz
        assert eps[1, 1].real < 0
        assert abs(eps[0, 2].real) > 0.5
        distance = (_DET_T_Z - _INTERFACE_Z) * _RESOLUTION

        n_sq_x = eps[0, 0] - eps[0, 2] ** 2 / eps[2, 2]
        n_sq_y = eps[1, 1]

        # The strong pole just below the drive makes the medium heavily
        # dispersive (group velocity several times slower than phase velocity):
        # the deep detector needs ~3x the standard simulation time to reach
        # steady state.
        sim_time = 3 * _SIM_TIME
        s_vac_x = _transmitted_flux(None, (1, 0, 0), sim_time=sim_time)
        s_x = _transmitted_flux(material, (1, 0, 0), sim_time=sim_time)
        t_x = s_x / s_vac_x
        t_x_analytic = _analytic_flux_fraction(complex(n_sq_x), distance)
        assert t_x_analytic > 0.1, f"premise: x-pol should transmit measurably, analytic={t_x_analytic:.3f}"
        assert abs(t_x - t_x_analytic) / t_x_analytic < _TOLERANCE, (
            f"x-pol (eps_xz-coupled): T_measured={t_x:.4f}, T_analytic={t_x_analytic:.4f} "
            f"(n_eff^2={n_sq_x:.3f}) — off-diagonal coupling logic error?"
        )

        s_vac_y = _transmitted_flux(None, (0, 1, 0), sim_time=sim_time)
        s_y = _transmitted_flux(material, (0, 1, 0), sim_time=sim_time)
        t_y = s_y / s_vac_y
        t_y_analytic = _analytic_flux_fraction(complex(n_sq_y), distance)
        assert t_y_analytic < 0.05, f"premise: y-pol sees metallic eps_yy, analytic={t_y_analytic:.3f}"
        assert abs(t_y - t_y_analytic) < _TOLERANCE, (
            f"y-pol (metallic): T_measured={t_y:.4f}, T_analytic={t_y_analytic:.4f} (eps_yy={n_sq_y:.3f})"
        )

    def test_effective_index_differs_from_ignoring_off_diagonal(self):
        """Guard that the test is sharp: the eps_xz term must change the
        analytic answer substantially, otherwise passing would not validate
        the off-diagonal coupling at all."""
        _, eps_tensor = _tilted_uniaxial_material(np.deg2rad(23.0))
        eps = eps_tensor(_OMEGA)
        distance = (_DET_T_Z - _INTERFACE_Z) * _RESOLUTION
        with_xz = _analytic_flux_fraction(complex(eps[0, 0] - eps[0, 2] ** 2 / eps[2, 2]), distance)
        without_xz = _analytic_flux_fraction(complex(eps[0, 0]), distance)
        assert abs(with_xz - without_xz) > 0.1, (
            f"test premise too weak: T with eps_xz {with_xz:.3f} vs without {without_xz:.3f}"
        )
