"""Physics simulation tests for per-axis (diagonally anisotropic) dispersion.

A per-axis dispersive medium applies different pole parameters to the x, y and
z field components. For a normally incident plane wave propagating along z,
an x-polarized wave sees only chi_x and a y-polarized wave only chi_y — so
the per-axis run must match (i) an equivalent isotropic run per polarization
and (ii) the analytic Fresnel transmission from ``susceptibility_axes``.

The hyperbolic test drives a per-axis Drude medium in the band where
Re(eps_x) < 0 < eps_y: the x polarization is reflected like a metal while the
y polarization passes through — the defining anisotropy of a hyperbolic
(indefinite) medium. This is the physics that motivates per-axis dispersion:
a *static* negative permittivity is unconditionally unstable in FDTD, so
hyperbolic behavior must come from poles.

Layout mirrors ``test_dispersion.py`` — 3x3 periodic transverse, PMLs in z,
``UniformPlaneSource`` in +z, one transmission-side Poynting flux detector.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.constants import c as c0

_WAVELENGTH = 1e-6
_OMEGA = 2.0 * np.pi * c0 / _WAVELENGTH
_RESOLUTION = 25e-9  # 40 cells/wavelength in vacuum
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION
_DOMAIN_Z = 5e-6
_Z_CELLS = round(_DOMAIN_Z / _RESOLUTION)  # = 200

_SOURCE_Z = _PML_CELLS + 2
_INTERFACE_Z = 100
_DET_T_Z = 140
_DIEL_CELLS_Z = _Z_CELLS - _INTERFACE_Z

_SIM_TIME = 120e-15
_TOLERANCE = 0.05

_DT_APPROX = 0.99 * _RESOLUTION / (c0 * np.sqrt(3))
_STEPS_PER_PERIOD = round(_WAVELENGTH / (c0 * _DT_APPROX))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD

# Per-axis Lorentz strengths: x sees a strong resonance, y/z a weak one.
_DE_X = 2.25  # eps_x ~ 4 at _OMEGA (resonance at 2*_OMEGA)
_DE_Y = 0.75  # eps_y ~ 2 at _OMEGA
_LORENTZ_W0 = 2.0 * _OMEGA
_LORENTZ_GAMMA = 1e13


def _per_axis_lorentz_model():
    return fdtdx.DispersionModel(
        poles=(
            fdtdx.LorentzPole(
                resonance_frequency=_LORENTZ_W0,
                damping=_LORENTZ_GAMMA,
                delta_epsilon=(_DE_X, _DE_Y, _DE_Y),
            ),
        )
    )


def _isotropic_lorentz_model(delta_eps):
    return fdtdx.DispersionModel(
        poles=(
            fdtdx.LorentzPole(
                resonance_frequency=_LORENTZ_W0,
                damping=_LORENTZ_GAMMA,
                delta_epsilon=delta_eps,
            ),
        )
    )


def _fresnel_transmission_semi_infinite(eps_complex: complex) -> float:
    n2 = np.sqrt(eps_complex)
    t = 2.0 / (1.0 + n2)
    return float(np.real(n2) * np.abs(t) ** 2)


def _build_base(polarization):
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


def _transmitted_flux(material, polarization):
    obj, con, cfg, vol = _build_base(polarization)
    if material is not None:
        _add_half_space(material, vol, obj, con)
    _add_flux_det("flux_t", _DET_T_Z, vol, obj, con)
    return _mean_flux(_run(obj, con, cfg), "flux_t")


_X_POL = (1, 0, 0)
_Y_POL = (0, 1, 0)


@pytest.fixture(scope="module")
def vacuum_flux():
    """Vacuum reference flux per polarization (normalization)."""
    return {
        _X_POL: _transmitted_flux(None, _X_POL),
        _Y_POL: _transmitted_flux(None, _Y_POL),
    }


class TestPerAxisLorentz:
    def test_per_axis_equals_isotropic_run_per_polarization(self):
        """The airtight equivalence check: for each polarization, the per-axis
        run must match an isotropic run built from that axis' pole parameters.
        The underlying arithmetic is identical (the coefficient arrays merely
        stop broadcasting), so agreement is expected to near float32 precision.
        """
        eps_inf = 1.0
        per_axis_mat = fdtdx.Material(permittivity=eps_inf, dispersion=_per_axis_lorentz_model())

        for pol, de in ((_X_POL, _DE_X), (_Y_POL, _DE_Y)):
            iso_mat = fdtdx.Material(permittivity=eps_inf, dispersion=_isotropic_lorentz_model(de))
            s_per_axis = _transmitted_flux(per_axis_mat, pol)
            s_iso = _transmitted_flux(iso_mat, pol)
            assert s_iso > 0
            rel = abs(s_per_axis - s_iso) / abs(s_iso)
            assert rel < 1e-4, (
                f"per-axis run diverged from equivalent isotropic run for polarization {pol}: "
                f"S_per_axis={s_per_axis:.6e}, S_iso={s_iso:.6e}, rel={rel:.2e}"
            )

    def test_transmission_matches_fresnel_per_polarization(self, vacuum_flux):
        """x- and y-polarized transmissions match the analytic Fresnel values
        computed from ``susceptibility_axes`` — and differ from each other."""
        eps_inf = 1.0
        model = _per_axis_lorentz_model()
        material = fdtdx.Material(permittivity=eps_inf, dispersion=model)
        chi = model.susceptibility_axes(_OMEGA)

        t_measured = {}
        for pol, chi_ax in ((_X_POL, chi[0]), (_Y_POL, chi[1])):
            t_analytic = _fresnel_transmission_semi_infinite(eps_inf + chi_ax)
            s_t = _transmitted_flux(material, pol)
            t_meas = s_t / vacuum_flux[pol]
            t_measured[pol] = t_meas
            rel_err = abs(t_meas - t_analytic) / t_analytic
            assert rel_err < _TOLERANCE, (
                f"polarization {pol}: T_measured={t_meas:.4f}, T_analytic={t_analytic:.4f} "
                f"(eps={eps_inf + chi_ax}), rel_err={rel_err:.3f} > {_TOLERANCE}"
            )
        # The two polarizations must see genuinely different media.
        assert abs(t_measured[_X_POL] - t_measured[_Y_POL]) > 0.02


class TestHyperbolicDrude:
    def test_hyperbolic_medium_is_polarization_selective(self, vacuum_flux):
        """Per-axis Drude with a plasma frequency only on x: at _OMEGA the
        permittivity tensor is indefinite (Re eps_x < 0 < eps_y = eps_z) — a
        hyperbolic medium. The x polarization reflects like a metal while the
        y polarization transmits per Fresnel."""
        eps_inf = 1.0
        omega_p = 5.0 * _OMEGA
        gamma = 0.05 * _OMEGA
        model = fdtdx.DispersionModel(poles=(fdtdx.DrudePole(plasma_frequency=(omega_p, 0.0, 0.0), damping=gamma),))
        chi = model.susceptibility_axes(_OMEGA)
        eps_x = eps_inf + chi[0]
        eps_y = eps_inf + chi[1]
        # Premise: the tensor is indefinite at the test frequency.
        assert eps_x.real < 0 < eps_y.real, f"premise broken: eps_x={eps_x}, eps_y={eps_y}"

        material = fdtdx.Material(permittivity=eps_inf, dispersion=model)

        t_x_analytic = _fresnel_transmission_semi_infinite(eps_x)
        assert t_x_analytic < 0.05, f"premise: metallic x response should barely transmit, got {t_x_analytic}"
        s_x = _transmitted_flux(material, _X_POL)
        t_x = s_x / vacuum_flux[_X_POL]
        assert abs(t_x - t_x_analytic) < _TOLERANCE, (
            f"x-pol (metallic axis): T_measured={t_x:.4f}, T_analytic={t_x_analytic:.4f}"
        )

        # y polarization sees eps_y = eps_inf = 1 (vacuum): full transmission.
        s_y = _transmitted_flux(material, _Y_POL)
        t_y = s_y / vacuum_flux[_Y_POL]
        t_y_analytic = _fresnel_transmission_semi_infinite(eps_y)
        assert abs(t_y - t_y_analytic) < _TOLERANCE, (
            f"y-pol (dielectric axis): T_measured={t_y:.4f}, T_analytic={t_y_analytic:.4f}"
        )
        # The defining hyperbolic signature: strong polarization selectivity.
        assert t_y - t_x > 0.8
