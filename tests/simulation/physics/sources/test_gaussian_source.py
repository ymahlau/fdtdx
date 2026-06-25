"""Physics simulation tests: GaussianPlaneSource validation.

  5a. Spatial profile is approximately Gaussian at a detector plane.
  5b. Gaussian forward flux / Uniform forward flux equals the analytic area fraction.

Both tests use the same paraxial beam (σ₀ = 500 nm, σ₀/λ = 0.5), chosen so that the
evanescent fraction stays below ~0.4 %:
  - Profile test: the Fresnel propagation formula for the beam width is accurate.
  - Power test: with the amplitude normalization turned off, both sources inject their raw
    aperture amplitude, so the forward-flux ratio reduces to the ratio of the integrated
    intensities — i.e. the analytic Gaussian-vs-uniform area fraction. (A narrow,
    sub-wavelength beam would carry large high-angle/evanescent content that does not
    propagate, breaking that identity, so the paraxial beam is required for the quantitative
    comparison.)
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1e-6
_RESOLUTION = 50e-9
_PML_CELLS = 10
_DOMAIN_Z = 4e-6

_SOURCE_Z = _PML_CELLS + 2
_DET_Z = _SOURCE_Z + 4  # 4 cells downstream (200 nm)

_SIM_TIME = 120e-15

# ── Per-test beam geometry ────────────────────────────────────────────────────
# Profile test: beam chosen so the waist σ₀ = radius/√2 = 500 nm >> λ/(2π) — Fresnel valid
_PROFILE_DOMAIN_XY = 5e-6  # 100 cells at 50 nm
_PROFILE_BEAM_RADIUS = 0.5e-6 * np.sqrt(2)  # radius/√2 = 500 nm waist

# Power test: same paraxial waist (σ₀ = radius/√2 = 500 nm, σ₀/λ = 0.5) so the beam is
# (almost) fully propagating and the forward-flux ratio equals the analytic area fraction.
_POWER_DOMAIN_XY = 4e-6  # 80 cells at 50 nm
_POWER_BEAM_RADIUS = 0.5e-6 * np.sqrt(2)  # σ₀ = radius/√2 = 500 nm

_DT_APPROX = 0.99 * _RESOLUTION / (3e8 * np.sqrt(3))
_STEPS_PER_PERIOD = round(_WAVELENGTH / (3e8 * _DT_APPROX))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_gaussian(beam_radius: float, domain_xy: float, normalize_by_energy: bool = True):
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(domain_xy, domain_xy, _DOMAIN_Z),
    )
    objects.append(volume)

    # PML on all faces so the Gaussian beam doesn't wrap around
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.GaussianPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
        radius=beam_radius,
        normalize_by_energy=normalize_by_energy,
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


def _build_uniform(domain_xy: float = _POWER_DOMAIN_XY, normalize_by_energy: bool = True):
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(domain_xy, domain_xy, _DOMAIN_Z),
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
        normalize_by_energy=normalize_by_energy,
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


# ── Tests ────────────────────────────────────────────────────────────────────


def test_gaussian_source_spatial_profile():
    """Transverse |E| profile matches the propagated Gaussian beam width.

    GaussianPlaneSource injects a Gaussian-weighted field with uniform phase::

        |E(r)| ∝ exp(-r² / radius²) = exp(-r² / (2 σ₀²))   (smooth, no aperture)

    where σ₀ = radius/√2 = 500 nm (the unified gaussian_amplitude convention).

    With σ₀/λ = 0.5, only ~0.3 % of the source's k-space power is evanescent,
    so the paraxial Fresnel propagation formula is accurate::

        σ(z) = sqrt(σ₀² + (λ z / (2π σ₀))²)

    The RMS width of the measured intensity profile (= σ(z)/√2 for a Gaussian)
    is compared with this prediction (tolerance 10 %).
    """
    objects, constraints, config, volume, wave = _build_gaussian(
        beam_radius=_PROFILE_BEAM_RADIUS, domain_xy=_PROFILE_DOMAIN_XY
    )

    det = fdtdx.PhasorDetector(
        name="phasor",
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        components=("Ex",),
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0, 1)),
            det.place_at_center(volume, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_DET_Z,)),
        ]
    )
    objects.append(det)

    arrays = _run(objects, constraints, config)

    phasor = np.array(arrays.detector_states["phasor"]["phasor"])
    ex_raw = np.abs(phasor[0, 0])
    ex_amplitude = np.squeeze(ex_raw)  # (Nx, Ny)

    center_x = ex_amplitude.shape[0] // 2
    center_y = ex_amplitude.shape[1] // 2
    center_amp = float(np.asarray(ex_amplitude[center_x, center_y]).item())
    assert center_amp > 0, "Center amplitude should be positive"

    # --- RMS beam width vs. propagated Gaussian formula ---
    # Unified convention: |E| ∝ exp(-r²/radius²) = exp(-r²/(2σ₀²))  →  σ₀ = radius/√2
    sigma0 = _PROFILE_BEAM_RADIUS / np.sqrt(2)

    z_det = (_DET_Z - _SOURCE_Z) * _RESOLUTION  # physical propagation distance
    # Free-space second-moment propagation for a Gaussian aperture source
    sigma_amp_expected = np.sqrt(sigma0**2 + (_WAVELENGTH * z_det / (2 * np.pi * sigma0)) ** 2)
    # RMS width of intensity profile = sigma_amp / sqrt(2) for a Gaussian
    rms_expected = sigma_amp_expected / np.sqrt(2)

    # Measured RMS width from the 1D intensity slice along x through the beam centre
    profile_1d = ex_amplitude[:, center_y]
    x_phys = (np.arange(len(profile_1d)) - center_x) * _RESOLUTION
    intensity_1d = profile_1d**2
    rms_measured = float(np.sqrt(np.sum(x_phys**2 * intensity_1d) / np.sum(intensity_1d)))

    rel_err = abs(rms_measured - rms_expected) / rms_expected
    assert rel_err < 0.10, (
        f"Gaussian beam RMS width: measured={rms_measured * 1e9:.1f} nm, "
        f"expected={rms_expected * 1e9:.1f} nm "
        f"(σ₀={sigma0 * 1e9:.0f} nm, z_det={z_det * 1e9:.0f} nm), "
        f"relative error={rel_err:.2%}"
    )


def test_gaussian_vs_uniform_total_power():
    """Gaussian / Uniform forward-flux ratio equals the analytic area fraction.

    With ``normalize_by_energy=False`` both sources inject their raw aperture amplitude
    (uniform → 1 everywhere, Gaussian → exp(-r²/radius²)). For a paraxial beam essentially
    all of that amplitude propagates forward, so the ratio of the total forward Poynting flux
    reduces to the ratio of integrated intensities::

        mean_g / mean_u ≈ (Σ_plane exp(-2 r²/radius²)) / N_cells
                        = ⟨exp(-2 r²/radius²)⟩  over the source plane,

    i.e. the analytic Gaussian-vs-uniform area fraction (intensity ∝ exp(-2 r²/radius²) since
    the amplitude is exp(-r²/radius²)). The measured ratio matches this to better than ~1 %;
    the assertion allows ~15 % to absorb the short diffraction gap and grid discretization.
    This both guarantees mean_g < mean_u (the fraction is < 1) and pins the value quantitatively
    (the old ``mean_g < mean_u`` was geometrically trivial).
    """
    # Gaussian run — un-normalized paraxial beam so the forward flux ∝ ∫ exp(-2 r²/radius²)
    obj_g, con_g, cfg_g, vol_g, _ = _build_gaussian(
        beam_radius=_POWER_BEAM_RADIUS, domain_xy=_POWER_DOMAIN_XY, normalize_by_energy=False
    )
    det_g = fdtdx.PoyntingFluxDetector(
        name="flux",
        partial_grid_shape=(None, None, 1),
        direction="+",
        reduce_volume=True,
        plot=False,
    )
    con_g.extend(
        [
            det_g.same_size(vol_g, axes=(0, 1)),
            det_g.place_at_center(vol_g, axes=(0, 1)),
            det_g.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_DET_Z,)),
        ]
    )
    obj_g.append(det_g)
    arrays_g = _run(obj_g, con_g, cfg_g)

    # Uniform run
    obj_u, con_u, cfg_u, vol_u, _ = _build_uniform(normalize_by_energy=False)
    det_u = fdtdx.PoyntingFluxDetector(
        name="flux",
        partial_grid_shape=(None, None, 1),
        direction="+",
        reduce_volume=True,
        plot=False,
    )
    con_u.extend(
        [
            det_u.same_size(vol_u, axes=(0, 1)),
            det_u.place_at_center(vol_u, axes=(0, 1)),
            det_u.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_DET_Z,)),
        ]
    )
    obj_u.append(det_u)
    arrays_u = _run(obj_u, con_u, cfg_u)

    flux_g = np.array(arrays_g.detector_states["flux"]["poynting_flux"][:, 0])
    flux_u = np.array(arrays_u.detector_states["flux"]["poynting_flux"][:, 0])

    mean_g = float(np.mean(flux_g[-_N_AVG_STEPS:]))
    mean_u = float(np.mean(flux_u[-_N_AVG_STEPS:]))

    assert mean_g > 0, f"Gaussian flux should be positive: {mean_g}"
    assert mean_u > 0, f"Uniform flux should be positive: {mean_u}"

    # Analytic area fraction: mean intensity exp(-2 r²/radius²) over the source-plane cell centers
    # (center-origin convention). For the paraxial beam this equals the forward-flux ratio.
    n_cells = round(_POWER_DOMAIN_XY / _RESOLUTION)
    axis_coords = (np.arange(n_cells) - (n_cells - 1) / 2) * _RESOLUTION
    xx, yy = np.meshgrid(axis_coords, axis_coords, indexing="ij")
    area_fraction = float(np.mean(np.exp(-2 * (xx**2 + yy**2) / _POWER_BEAM_RADIUS**2)))

    ratio = mean_g / mean_u
    assert ratio < 1.0, f"Gaussian flux ({mean_g:.4e}) should be less than Uniform flux ({mean_u:.4e})"
    rel_err = abs(ratio - area_fraction) / area_fraction
    assert rel_err < 0.15, (
        f"Gaussian/Uniform forward-flux ratio={ratio:.4e}, analytic area fraction={area_fraction:.4e} "
        f"(σ₀={_POWER_BEAM_RADIUS / np.sqrt(2) * 1e9:.0f} nm), relative error={rel_err:.2%}"
    )
