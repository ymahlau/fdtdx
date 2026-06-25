"""Physics validation of the analytic source power and transmission methods.

The central check is that the analytic ``Source.injected_power_spectrum`` (computed from the
source profile + temporal signal, no run) matches the *measured* source-plane Poynting flux
(``PhasorDetector.flux_spectrum``) in a homogeneous medium — i.e. the analytic source power is
correct to discretization accuracy.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

_RES = 50e-9
_WL = 1e-6
_PML = 8
_SIM_TIME = 60e-15


def _build_base(domain_z=3e-6):
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=_SIM_TIME, dtype=jnp.float32)
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_real_shape=(1.0e-6, 1.0e-6, domain_z))
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML,
        override_types={"min_x": "periodic", "max_x": "periodic", "min_y": "periodic", "max_y": "periodic"},
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())
    wave = fdtdx.WaveCharacter(wavelength=_WL)
    # A broadband pulse fully transits every plane, so the recorded spectrum (hence the
    # transmitted fraction) is position-independent — the right choice for spectral power.
    profile = fdtdx.GaussianPulseProfile(
        center_wave=wave,
        spectral_width=fdtdx.WaveCharacter(wavelength=3e-6),
    )
    source = fdtdx.UniformPlaneSource(
        name="source",
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        temporal_profile=profile,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(14,)),
        ]
    )
    objects.append(source)
    return objects, constraints, config, volume, wave


def _add_plane_phasor(name, z_idx, wave, volume, objects, constraints):
    det = fdtdx.PhasorDetector(
        name=name,
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        reduce_volume=False,
        scaling_mode="pulse",
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
    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key)
    return oc, arrays


def test_analytic_injected_power_matches_measured_flux():
    """Analytic injected_power_spectrum == measured source-plane flux (homogeneous medium)."""
    objects, constraints, config, volume, wave = _build_base()
    _add_plane_phasor("src_plane", 18, wave, volume, objects, constraints)
    oc, arrays = _run(objects, constraints, config)

    source = oc["source"]
    detector = oc["src_plane"]
    freqs = jnp.array([wave.get_frequency()])
    analytic = source.injected_power_spectrum(freqs)
    measured = detector.flux_spectrum(arrays)

    assert float(analytic[0]) > 0
    rel_err = abs(float(measured[0]) - float(analytic[0])) / float(analytic[0])
    # Analytic source power must reproduce the measured plane flux to discretization accuracy.
    assert rel_err < 0.05, f"analytic={float(analytic[0]):.3e} measured={float(measured[0]):.3e} rel_err={rel_err:.3f}"


def test_transmission_unity_in_homogeneous_medium():
    """Lossless homogeneous medium -> transmitted fraction ~ 1 (energy conservation)."""
    objects, constraints, config, volume, wave = _build_base()
    _add_plane_phasor("out", 40, wave, volume, objects, constraints)
    oc, arrays = _run(objects, constraints, config)
    t = oc["out"].transmission(arrays, oc["source"])
    assert abs(float(t[0]) - 1.0) < 0.05, f"transmission={float(t[0]):.3f}"


def test_transmission_matches_fresnel():
    """Dielectric half-space -> transmitted fraction matches the Fresnel power coefficient."""
    eps_r = 4.0
    n1, n2 = 1.0, float(np.sqrt(eps_r))
    t_analytic = 4.0 * n1 * n2 / (n1 + n2) ** 2  # = 8/9

    objects, constraints, config, volume, wave = _build_base()
    # dielectric fills cells [28, 60): the +z half-space past the source.
    interface_z = 28
    diel = fdtdx.UniformMaterialObject(
        name="slab",
        partial_grid_shape=(None, None, 60 - interface_z),
        material=fdtdx.Material(permittivity=eps_r),
    )
    constraints.extend(
        [
            diel.same_size(volume, axes=(0, 1)),
            diel.place_at_center(volume, axes=(0, 1)),
            diel.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(interface_z,)),
        ]
    )
    objects.append(diel)
    _add_plane_phasor("out", 40, wave, volume, objects, constraints)
    oc, arrays = _run(objects, constraints, config)

    t = oc["out"].transmission(arrays, oc["source"])
    rel_err = abs(float(t[0]) - t_analytic) / t_analytic
    # The effect under test is the 11% Fresnel dip from 1.0 to 8/9. A 5% band excludes the
    # no-interface T=1 case, and the explicit t<0.95 guard makes the dip mandatory: the test
    # cannot pass if the dielectric half-space is silently ignored.
    assert float(t[0]) < 0.95, f"no Fresnel dip seen: T_measured={float(t[0]):.3f}"
    assert rel_err < 0.05, f"T_measured={float(t[0]):.3f} T_analytic={t_analytic:.3f} rel_err={rel_err:.3f}"
