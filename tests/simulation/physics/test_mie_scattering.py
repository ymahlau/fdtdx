"""2D Mie scattering validation: a dielectric cylinder vs. the analytical series.

An infinite dielectric cylinder (modelled as a 2D circle with a periodic, 2-cell
z axis) is illuminated by a plane wave whose E field is parallel to the cylinder
axis (TM polarization). The plane wave is launched with a :class:`TFSFPlaneSourceRegion`
box that confines the total field to its interior and leaves only the scattered
field outside; the z axis is excluded from the box and closed by periodic
boundaries. PML terminates x and y.

Two closed-surface Poynting detectors measure power:
  * ``outside`` -- in the scattered-field region (outside the TFSF box): the net
    outward flux is the scattered power, giving the scattering cross section.
  * ``inside``  -- in the total-field region (around the cylinder): the net
    inward flux is the absorbed power, which must be ~0 for a lossless dielectric.

The scattering efficiency ``Q_sca = sigma_sca / (2a)`` is compared against the
analytical 2D Mie series. The incident irradiance used to normalize the FDTD
cross section is measured with a plane Poynting detector in a reference run
without the cylinder (which also confirms the TFSF cancellation).

This is a deliberately small/coarse simulation (staircased circle, ~7-cell
radius), so the agreement tolerance is loose (~15%). Calibrated result:
Q_sca_fdtd/Q_sca_analytical ~ 1.08, absorbed/scattered ~ 0.01.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import h1vp, hankel1, jv, jvp

import fdtdx

_SPACING = 25e-9
_WAVELENGTH = 1000e-9
_EPS = 4.0  # dielectric, n = 2
_RADIUS = 175e-9
_DOMAIN = 1600e-9
_LZ = 2 * _SPACING
_PML = 8
_SIM_TIME = 55e-15
_AVG_PERIODS = 6


def _a_n(n: int, x: float, m: float) -> complex:
    """TM (E parallel to axis) 2D scattering coefficient for a dielectric cylinder."""
    j_x, jp_x = jv(n, x), jvp(n, x)
    j_mx, jp_mx = jv(n, m * x), jvp(n, m * x)
    h_x, hp_x = hankel1(n, x), h1vp(n, x)
    num = m * jp_mx * j_x - jp_x * j_mx
    den = m * jp_mx * h_x - hp_x * j_mx
    return num / den


def _analytical_q_sca_tm(x: float, m: float, nmax: int = 10) -> float:
    """Analytical 2D scattering efficiency Q_sca for TM polarization."""
    total = abs(_a_n(0, x, m)) ** 2
    for n in range(1, nmax + 1):
        total += 2 * abs(_a_n(n, x, m)) ** 2
    return (2.0 / x) * total


def _build(with_cylinder: bool):
    config = fdtdx.SimulationConfig(
        time=_SIM_TIME,
        grid=fdtdx.UniformGrid(spacing=_SPACING),
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=None,
    )
    volume = fdtdx.SimulationVolume(partial_real_shape=(_DOMAIN, _DOMAIN, _LZ))
    objects = [volume]
    constraints = []

    # TFSF box: plane wave propagating +x, E along z (TM), z axis periodic/wrap.
    source = fdtdx.TFSFPlaneSourceRegion(
        name="src",
        partial_real_shape=(700e-9, 700e-9, None),
        propagation_axis=0,
        direction="+",
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        fixed_E_polarization_vector=(0, 0, 1),
        periodic_axes=(2,),
    )
    constraints += [source.place_at_center(volume), source.same_size(volume, axes=(2,))]
    objects.append(source)

    # Absorbed power (total-field region, encloses the cylinder, inside the TFSF box).
    inside = fdtdx.ClosedSurfacePoyntingFluxDetector(
        name="inside", partial_real_shape=(500e-9, 500e-9, None), orientation="inward", axes=(0, 1), plot=False
    )
    constraints += [inside.place_at_center(volume), inside.same_size(volume, axes=(2,))]
    objects.append(inside)

    # Scattered power (scattered-field region, encloses the TFSF box).
    outside = fdtdx.ClosedSurfacePoyntingFluxDetector(
        name="outside", partial_real_shape=(900e-9, 900e-9, None), orientation="outward", axes=(0, 1), plot=False
    )
    constraints += [outside.place_at_center(volume), outside.same_size(volume, axes=(2,))]
    objects.append(outside)

    # Incident irradiance probe: a +x flux plane inside the TFSF box (pure incident
    # field in the reference run).
    inc = fdtdx.PoyntingFluxDetector(
        name="inc",
        direction="+",
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, 600e-9, None),
        reduce_volume=True,
        plot=False,
    )
    constraints += [
        inc.place_at_center(volume, axes=(1, 2)),
        inc.same_size(volume, axes=(2,)),
        inc.place_relative_to(volume, axes=(0,), own_positions=(0,), other_positions=(0,), margins=(-250e-9,)),
    ]
    objects.append(inc)

    if with_cylinder:
        cylinder = fdtdx.Cylinder(
            name="cyl",
            radius=_RADIUS,
            axis=2,
            materials={"vac": fdtdx.Material(permittivity=1.0), "diel": fdtdx.Material(permittivity=_EPS)},
            material_name="diel",
        )
        constraints += [cylinder.place_at_center(volume), cylinder.same_size(volume, axes=(2,))]
        objects.append(cylinder)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        boundary_type="pml", thickness=_PML, override_types={"min_z": "periodic", "max_z": "periodic"}
    )
    bd, bc = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints += bc
    objects += list(bd.values())
    return objects, constraints, config


def _run(with_cylinder: bool) -> dict[str, float]:
    objects, constraints, config = _build(with_cylinder)
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, key = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)

    dt = config.time_step_duration
    steps_per_period = round(_WAVELENGTH / (fdtdx.constants.c * dt))
    n_avg = _AVG_PERIODS * steps_per_period

    def steady(name: str) -> float:
        series = np.array(arrays.detector_states[name]["poynting_flux"])[:, 0]
        return float(np.mean(series[-n_avg:]))

    return {"inside": steady("inside"), "outside": steady("outside"), "inc": steady("inc")}


def test_mie_scattering_dielectric_cylinder_2d():
    x = 2 * np.pi * _RADIUS / _WAVELENGTH
    m = np.sqrt(_EPS)
    q_sca_analytical = _analytical_q_sca_tm(x, m)

    reference = _run(with_cylinder=False)
    scene = _run(with_cylinder=True)

    # Incident irradiance from the reference run (pure incident field in the TFSF box).
    incident_irradiance = reference["inc"] / (600e-9 * _LZ)
    assert incident_irradiance > 0

    # Without a scatterer the scattered-field region must be ~0 (TFSF cancellation).
    p_scat = scene["outside"]
    assert abs(reference["outside"]) < 0.05 * abs(p_scat)

    # Lossless dielectric: absorbed power (inward net through the total-field box) ~ 0.
    p_abs = scene["inside"]
    assert abs(p_abs) < 0.1 * abs(p_scat)

    # Scattering efficiency vs. the analytical 2D Mie series (per unit z-length; the
    # z-extent cancels between the box power and the incident irradiance).
    sigma_sca = (p_scat / _LZ) / incident_irradiance
    q_sca_fdtd = sigma_sca / (2 * _RADIUS)
    assert q_sca_fdtd == pytest.approx(q_sca_analytical, rel=0.15)
