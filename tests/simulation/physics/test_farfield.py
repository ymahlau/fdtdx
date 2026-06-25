"""Physics validation of the far-field projectors.

Planar projector: a normally-incident plane wave on a finite aperture radiates a forward
lobe peaked at theta=0, and the FFT k-space path agrees with the direct spherical DFT.
Box projector (dipole): the radiation pattern follows sin^2(theta) and the integrated
far-field power matches the near-field surface flux (radiated_power).
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx
from fdtdx.core.physics.farfield import far_field_power_density
from fdtdx.objects.detectors.farfield import PlanarFarFieldProjector, far_field_box

_RES = 50e-9
_WL = 1e-6
_PML = 8
_SIM_TIME = 60e-15


def _build_planar():
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=_SIM_TIME, dtype=jnp.float32)
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_real_shape=(1.5e-6, 1.5e-6, 3e-6))
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML,
        override_types={"min_x": "periodic", "max_x": "periodic", "min_y": "periodic", "max_y": "periodic"},
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())
    wave = fdtdx.WaveCharacter(wavelength=_WL)
    profile = fdtdx.GaussianPulseProfile(center_wave=wave, spectral_width=fdtdx.WaveCharacter(wavelength=3e-6))
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

    projector = PlanarFarFieldProjector(
        name="ff",
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        direction="+",
        background_index=1.0,
        scaling_mode="pulse",
    )
    constraints.extend(
        [
            projector.same_size(volume, axes=(0, 1)),
            projector.place_at_center(volume, axes=(0, 1)),
            projector.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(40,)),
        ]
    )
    objects.append(projector)
    return objects, constraints, config


def _run(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key)
    return oc, arrays


def test_planar_forward_lobe_and_shapes():
    objects, constraints, config = _build_planar()
    oc, arrays = _run(objects, constraints, config)
    proj = oc["ff"]

    # near field shapes: (num_freqs, 3, nx, ny, 1)
    E_near, H_near = proj.near_field(arrays)
    assert E_near.shape[0] == 1 and E_near.shape[1] == 3
    assert H_near.shape == E_near.shape

    theta = jnp.linspace(0.0, jnp.radians(70.0), 36)
    phi = jnp.zeros_like(theta)
    e_theta, e_phi = proj.spherical(arrays, theta, phi)
    assert e_theta.shape == (1, 36)
    intensity = np.asarray(jnp.abs(e_theta[0]) ** 2 + jnp.abs(e_phi[0]) ** 2)
    # normal incidence -> forward lobe dominates
    assert int(np.argmax(intensity)) == 0
    assert intensity[0] > 10 * intensity[-1]


def test_planar_kspace_matches_spherical():
    objects, constraints, config = _build_planar()
    oc, arrays = _run(objects, constraints, config)
    proj = oc["ff"]

    ux, uy, e_theta_k, e_phi_k = proj.kspace(arrays)
    assert ux.shape[0] == 1  # one frequency
    ux0, uy0 = np.asarray(ux[0]), np.asarray(uy[0])
    mag_k = np.asarray(jnp.abs(e_theta_k[0]) ** 2 + jnp.abs(e_phi_k[0]) ** 2)

    # Sample a few propagating bins near the axis and compare to the direct spherical DFT.
    nu, nv = ux0.shape
    cu, cv = nu // 2, nv // 2  # DC after fftshift
    for du, dv in [(0, 0), (2, 0), (0, 2)]:
        i, j = cu + du, cv + dv
        ut, vt = float(ux0[i, j]), float(uy0[i, j])
        s = ut**2 + vt**2
        if s >= 0.9:  # skip near grazing/evanescent
            continue
        theta = jnp.asarray(np.arcsin(np.sqrt(s)))
        phi = jnp.asarray(np.arctan2(vt, ut))
        et_s, ep_s = proj.spherical(arrays, theta.reshape(1), phi.reshape(1))
        mag_s = float(jnp.abs(et_s[0, 0]) ** 2 + jnp.abs(ep_s[0, 0]) ** 2)
        if mag_s == 0.0 and mag_k[i, j] == 0.0:
            continue
        rel = abs(mag_k[i, j] - mag_s) / max(mag_s, 1e-30)
        assert rel < 0.05, f"kspace vs spherical mismatch at ({du},{dv}): {mag_k[i, j]:.3e} vs {mag_s:.3e}"


def test_planar_directivity_shape_and_positive():
    objects, constraints, config = _build_planar()
    oc, arrays = _run(objects, constraints, config)
    proj = oc["ff"]
    theta = jnp.linspace(0.0, jnp.pi, 49)
    phi = jnp.linspace(0.0, 2 * jnp.pi, 13)
    d = proj.directivity(arrays, theta, phi)
    assert d.shape == (1, 49, 13)
    assert float(jnp.min(d)) >= 0.0

    # Physical magnitude check: a uniformly-illuminated aperture of area A radiates with on-axis
    # directivity D0 ~ 4*pi*A/lambda^2. The normally-incident plane wave on the full transverse
    # unit cell (1.5um x 1.5um) approximates exactly that, so the measured theta=0 directivity must
    # land within a factor of a few of D0 (not merely exceed the isotropic value 1.0).
    aperture_area = 1.5e-6 * 1.5e-6
    lam_medium = _WL / 1.0  # background_index = 1.0
    d0_uniform = 4.0 * np.pi * aperture_area / lam_medium**2
    d_on_axis = float(jnp.mean(d[0, 0, :]))  # theta=0 pole: phi-degenerate, average over phi
    assert 0.3 * d0_uniform <= d_on_axis <= 3.0 * d0_uniform, (
        f"on-axis directivity {d_on_axis:.3f} not within 0.3-3x of uniform-aperture D0={d0_uniform:.3f}"
    )
    # ...and it is strongly forward-directed: on-axis >> broadside (theta=90 deg).
    d_broadside = float(jnp.max(d[0, 24, :]))  # theta index 24 of 49 -> pi/2
    assert d_on_axis > 5.0 * d_broadside, (
        f"aperture not forward-directed: on-axis={d_on_axis:.3f} broadside={d_broadside:.3f}"
    )


def test_diffraction_orders_normal_incidence():
    """Normal-incidence uniform plane wave: 0th order carries the flux; order angles obey the
    grating equation sin(theta_m) = m*lambda/Lambda (Lambda = transverse unit-cell extent)."""
    objects, constraints, config = _build_planar()
    oc, arrays = _run(objects, constraints, config)
    proj = oc["ff"]

    res = proj.diffraction_orders(arrays, orders=[(0, 0), (1, 0), (0, 1), (-1, 0)])
    # 0th order is along the normal and carries essentially all of the plane flux.
    assert float(res["theta"][0, 0]) < np.radians(2)
    flux = float(proj.flux_spectrum(arrays)[0])
    p0 = float(res["power"][0, 0])
    assert abs(p0 - flux) / flux < 0.05, f"0th order power {p0:.3e} vs flux {flux:.3e}"
    # +/-1 order angles match the grating equation for the 1.5 um transverse unit cell.
    Lambda = 1.5e-6
    expected = float(np.arcsin(_WL / Lambda))
    assert bool(res["propagating"][0, 1])
    assert np.isclose(float(res["theta"][0, 1]), expected, atol=np.radians(2))
    assert np.isclose(float(res["theta"][0, 3]), expected, atol=np.radians(2))


def _build_dipole_box(domain=2.5e-6, pol=2):
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=_SIM_TIME, dtype=jnp.float32)
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_real_shape=(domain, domain, domain))
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML)  # all PML: isolated radiator
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())
    wave = fdtdx.WaveCharacter(wavelength=_WL)
    profile = fdtdx.GaussianPulseProfile(center_wave=wave, spectral_width=fdtdx.WaveCharacter(wavelength=3e-6))
    dipole = fdtdx.PointDipoleSource(
        name="dip",
        partial_grid_shape=(1, 1, 1),
        wave_character=wave,
        temporal_profile=profile,
        polarization=pol,
    )
    constraints.append(dipole.place_at_center(volume))
    objects.append(dipole)
    box, box_constraints = far_field_box(
        volume, bound_cfg, (wave,), config, margin=2, name="ff_box", background_index=1.0, scaling_mode="pulse"
    )
    objects.append(box)
    constraints.extend(box_constraints)
    return objects, constraints, config


def test_box_dipole_radiation_pattern():
    """z-polarized dipole -> |E_far|^2 ~ sin^2(theta) (toroidal pattern, nulls on the axis)."""
    objects, constraints, config = _build_dipole_box(pol=2)
    oc, arrays = _run(objects, constraints, config)
    box = oc["ff_box"]

    theta = jnp.linspace(jnp.radians(5.0), jnp.radians(175.0), 35)
    phi = jnp.zeros_like(theta)
    e_theta, e_phi = box.spherical(arrays, theta, phi)
    intensity = np.asarray(jnp.abs(e_theta[0]) ** 2 + jnp.abs(e_phi[0]) ** 2)
    intensity = intensity / intensity.max()
    expected = np.sin(np.asarray(theta)) ** 2
    # peak near theta = 90 deg
    assert abs(float(theta[int(np.argmax(intensity))]) - np.pi / 2) < np.radians(10)
    # matches sin^2 across the cut
    assert np.max(np.abs(intensity - expected)) < 0.15


def test_box_power_conservation():
    """Integrated far-field power equals the near-field surface flux (radiated_power).

    Self-consistency check of the NTFF transform: ``p_far`` (sphere integral of
    ``box.spherical``) and ``p_surface`` (``box.radiated_power``) derive from the *same*
    recorded face phasors and share the same eta0-normalized phasor convention, so they are
    directly commensurate (a common global normalization error would cancel and go unseen).
    What the tight tolerance does check is the angular-spectrum projection itself: that the
    surface-equivalence radiation integral conserves power.
    """
    objects, constraints, config = _build_dipole_box(pol=2)
    oc, arrays = _run(objects, constraints, config)
    box = oc["ff_box"]

    p_surface = float(box.radiated_power(arrays)[0])
    # Proper spherical quadrature. The earlier ``sum * dth * dph`` over phi = linspace(0, 2*pi, nph)
    # double-counted the phi=0==2*pi endpoint (a ~1/(nph-1) ~ 2% bias). Sampling phi on [0, 2*pi)
    # with endpoint=False makes the uniform Riemann sum the exact periodic trapezoid rule, removing
    # that bias instead of hiding it in a loose tolerance. The theta endpoints carry sin(theta)=0,
    # so the Riemann sum there already coincides with the trapezoid.
    nth, nph = 121, 96
    theta = jnp.linspace(0.0, jnp.pi, nth)
    phi = jnp.linspace(0.0, 2 * jnp.pi, nph + 1)[:-1]  # [0, 2*pi), no duplicated endpoint
    TH, PH = jnp.meshgrid(theta, phi, indexing="ij")
    e_theta, e_phi = box.spherical(arrays, TH, PH)
    u = np.asarray(far_field_power_density(e_theta[0], e_phi[0], 1.0))  # r=1 -> radiant intensity
    sin_theta = np.sin(np.asarray(theta))[:, None]
    dth = float(theta[1] - theta[0])
    dph = 2.0 * np.pi / nph
    p_far = float(np.sum(u * sin_theta) * dth * dph)

    assert p_surface > 0 and p_far > 0
    rel = abs(p_far - p_surface) / p_surface
    assert rel < 0.05, f"p_far={p_far:.3e} p_surface={p_surface:.3e} rel_err={rel:.3f}"
