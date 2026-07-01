"""Physics simulation tests: ModePlaneSource in a waveguide.

  6a. Mode propagates along waveguide with correct effective index.
  6b. Mode is confined to the waveguide core.

Domain: Silicon waveguide (300nm wide, eps=12.25) in SiO2 cladding (eps=2.25).
  ModePlaneSource exciting fundamental TE mode, propagating +x.
  PML on x (absorb), periodic on y, PML on z.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Domain constants ─────────────────────────────────────────────────────────
_WAVELENGTH = 1.55e-6
_RESOLUTION = 50e-9
_PML_CELLS = 10
_DOMAIN_Y = 3 * _RESOLUTION  # periodic, minimal (slab waveguide — infinite in y)
_WG_HEIGHT = 220e-9  # ~4 cells (slab thickness in z)
_DOMAIN_Z = 3e-6  # enough for mode + cladding + PML
_DOMAIN_X = 5e-6  # propagation direction

_SOURCE_X = _PML_CELLS + 2
_DET1_X = _SOURCE_X + 10
_DET2_X = _DET1_X + 4  # small separation to avoid phase aliasing

_SIM_TIME = 200e-15

_EPS_SI = fdtdx.constants.relative_permittivity_silicon  # 12.25
_EPS_SIO2 = fdtdx.constants.relative_permittivity_silica  # 2.25


# ── Helpers ──────────────────────────────────────────────────────────────────


def _slab_neff_te(n_core: float, n_clad: float, wavelength: float, thickness: float) -> float:
    """Effective index of the fundamental TE mode in a symmetric slab waveguide.

    Solves the characteristic equation for the even (fundamental) TE mode::

        κ tan(κd) = γ,   u² + w² = V²

    where d = thickness/2, u = κd, w = γd, V = k₀d√(n_core² - n_clad²).
    Uses bisection on u ∈ (0, min(V, π/2)).
    """
    k0 = 2 * np.pi / wavelength
    d = thickness / 2
    V = k0 * d * np.sqrt(n_core**2 - n_clad**2)

    u_max = min(V, np.pi / 2 - 1e-10)  # avoid tan singularity

    def f(u: float) -> float:
        return u * np.tan(u) - np.sqrt(max(V**2 - u**2, 0.0))

    lo, hi = 1e-10, u_max
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) >= 0:
            hi = mid
        else:
            lo = mid
    u = (lo + hi) / 2

    gamma = np.sqrt(max(V**2 - u**2, 0.0)) / d
    beta = np.sqrt((k0 * n_clad) ** 2 + gamma**2)
    return float(beta / k0)


def _build_waveguide_domain(core_conductivity: float = 0.0):
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
            "min_y": "periodic",
            "max_y": "periodic",
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    # SiO2 cladding fills the volume
    cladding = fdtdx.UniformMaterialObject(
        name="cladding",
        partial_real_shape=(None, None, None),
        material=fdtdx.Material(permittivity=_EPS_SIO2),
    )
    constraints.extend(cladding.same_position_and_size(volume))
    objects.append(cladding)

    # Silicon slab waveguide (infinite in y via periodic BC, finite height in z)
    core = fdtdx.UniformMaterialObject(
        name="core",
        partial_real_shape=(None, None, _WG_HEIGHT),
        material=fdtdx.Material(permittivity=_EPS_SI, electric_conductivity=core_conductivity),
    )
    constraints.extend(
        [
            core.same_size(volume, axes=(0, 1)),
            core.place_at_center(volume, axes=(0, 1, 2)),
        ]
    )
    objects.append(core)

    # Mode source exciting TE fundamental
    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.ModePlaneSource(
        partial_grid_shape=(1, None, None),
        wave_character=wave,
        direction="+",
        mode_index=0,
        filter_pol="te",
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(1, 2)),
            source.place_at_center(volume, axes=(1, 2)),
            source.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_SOURCE_X,)),
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


def test_mode_source_waveguide_propagation():
    """Mode propagates with the effective index predicted for the TE slab mode.

    Two phasor detectors at different x positions give the phase difference,
    from which n_eff = k_measured x λ / 2π is extracted.  The result is
    compared with the analytical solution of the symmetric-slab characteristic
    equation κ tan(κd) = γ (tolerance 10 %; the 50 nm resolution corresponds
    to ~4 cells across the 220 nm core so some discretisation error is expected).
    """
    objects, constraints, config, volume, wave = _build_waveguide_domain()

    for name, x_coord in [("phasor_near", _DET1_X), ("phasor_far", _DET2_X)]:
        det = fdtdx.PhasorDetector(
            name=name,
            partial_grid_shape=(1, None, None),
            wave_characters=(wave,),
            components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
            reduce_volume=True,
            plot=False,
        )
        constraints.extend(
            [
                det.same_size(volume, axes=(1, 2)),
                det.place_at_center(volume, axes=(1, 2)),
                det.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(x_coord,)),
            ]
        )
        objects.append(det)

    arrays = _run(objects, constraints, config)

    # Use Ey for TE mode (E perpendicular to propagation in slab waveguide)
    # Shape: (latent_time_steps=1, num_frequencies=1, num_components=6)
    p_near = complex(arrays.detector_states["phasor_near"]["phasor"][0, 0, 1])
    p_far = complex(arrays.detector_states["phasor_far"]["phasor"][0, 0, 1])

    assert abs(p_near) > 1e-30, "Near phasor is zero — mode not launched"
    assert abs(p_far) > 1e-30, "Far phasor is zero — mode not propagating"

    delta_phi = (np.angle(p_far) - np.angle(p_near)) % (2 * np.pi)
    if delta_phi > np.pi:
        delta_phi -= 2 * np.pi
    separation = (_DET2_X - _DET1_X) * _RESOLUTION
    k_measured = abs(delta_phi) / separation
    n_eff_measured = k_measured * _WAVELENGTH / (2 * np.pi)

    n_clad = np.sqrt(_EPS_SIO2)  # 1.5
    n_core = np.sqrt(_EPS_SI)  # 3.5

    # Sanity bound
    assert n_clad < n_eff_measured < n_core, f"n_eff_measured={n_eff_measured:.3f} outside [{n_clad:.2f}, {n_core:.2f}]"

    # Quantitative comparison against the analytical TE slab mode
    n_eff_analytical = _slab_neff_te(n_core, n_clad, _WAVELENGTH, _WG_HEIGHT)
    rel_err = abs(n_eff_measured - n_eff_analytical) / n_eff_analytical
    assert rel_err < 0.10, (
        f"n_eff_measured={n_eff_measured:.3f}, "
        f"analytical={n_eff_analytical:.3f} (TE slab, h={_WG_HEIGHT * 1e9:.0f} nm), "
        f"relative error={rel_err:.2%}"
    )


def test_mode_source_confinement():
    """More than 30% of mode energy is confined within the waveguide core."""
    objects, constraints, config, volume, wave = _build_waveguide_domain()

    det = fdtdx.PhasorDetector(
        name="cross_section",
        partial_grid_shape=(1, None, None),
        wave_characters=(wave,),
        components=("Ey",),
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(1, 2)),
            det.place_at_center(volume, axes=(1, 2)),
            det.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_DET1_X,)),
        ]
    )
    objects.append(det)

    arrays = _run(objects, constraints, config)

    # phasor shape: (num_freq, num_components, 1, Ny, Nz)
    phasor = np.array(arrays.detector_states["cross_section"]["phasor"])
    ey_amp = np.abs(phasor[0, 0, 0, :, :])  # (Ny, Nz)

    total_energy = np.sum(ey_amp**2)
    assert total_energy > 0, "Total energy is zero — mode not launched"

    # Find core region indices. The waveguide is centered.
    nz = ey_amp.shape[1]
    # Core spans WG_HEIGHT / RESOLUTION cells centered in z
    wg_cells_z = round(_WG_HEIGHT / _RESOLUTION)
    z_start = (nz - wg_cells_z) // 2
    z_end = z_start + wg_cells_z

    # y is periodic with 3 cells, core fills all of it
    core_energy = np.sum(ey_amp[:, z_start:z_end] ** 2)

    confinement = core_energy / total_energy
    assert confinement > 0.3, f"Mode confinement={confinement:.2%}, expected > 30% in core"


# Conductive core so the guided mode acquires loss. sigma = 1e4 S/m gives
# eps'' = sigma/(omega*eps0) ~ 0.9 at 1.55 um — a clearly measurable modal loss
# without destabilising the solve.
_CORE_SIGMA = 1.0e4


def _place_and_apply(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    return obj_container


def test_mode_source_lossy_core_yields_complex_neff():
    """A conductive waveguide core makes the solved effective index complex.

    Verifies the step-1b fix: electric conductivity is threaded into the mode
    solver (via the complex effective permittivity), so the source's stored
    effective index gains a nonzero imaginary part (modal attenuation). The
    lossless control stays essentially real, and the real part is barely
    perturbed by the moderate loss.
    """
    # Lossless control
    objects, constraints, config, _, _ = _build_waveguide_domain()
    src_lossless = _place_and_apply(objects, constraints, config).sources[0]
    neff_lossless = complex(np.asarray(src_lossless._neff))

    # Lossy core
    objects, constraints, config, _, _ = _build_waveguide_domain(core_conductivity=_CORE_SIGMA)
    src_lossy = _place_and_apply(objects, constraints, config).sources[0]
    neff_lossy = complex(np.asarray(src_lossy._neff))

    # Lossless mode index is essentially real.
    assert abs(neff_lossless.imag) < 1e-3 * abs(neff_lossless.real), (
        f"Lossless neff should be ~real, got {neff_lossless}"
    )
    # Conductive core ⇒ clearly nonzero imaginary part (attenuation).
    assert abs(neff_lossy.imag) > 1e-2 * abs(neff_lossy.real), (
        f"Lossy neff should have a nonzero imaginary part, got {neff_lossy}"
    )
    # The guided mode is the same one — loss perturbs the real part only weakly.
    rel = abs(neff_lossy.real - neff_lossless.real) / neff_lossless.real
    assert rel < 0.20, f"Real(neff) shifted too much under loss: {neff_lossless.real} -> {neff_lossy.real}"


def test_mode_overlap_detector_lossy_core_yields_complex_neff():
    """ModeOverlapDetector solves its reference mode against the complex permittivity.

    Companion to the source test: with a conductive core the detector's stored
    reference effective index is complex, confirming the conductivity reaches the
    detector's mode solve too.
    """
    objects, constraints, config, volume, wave = _build_waveguide_domain(core_conductivity=_CORE_SIGMA)
    det = fdtdx.ModeOverlapDetector(
        name="overlap",
        partial_grid_shape=(1, None, None),
        wave_characters=(wave,),
        direction="+",
        mode_index=0,
        filter_pol="te",
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(1, 2)),
            det.place_at_center(volume, axes=(1, 2)),
            det.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(_DET1_X,)),
        ]
    )
    objects.append(det)

    obj_container = _place_and_apply(objects, constraints, config)
    placed_det = next(o for o in obj_container.objects if o.name == "overlap")
    neff = complex(np.asarray(placed_det._mode_neff).reshape(-1)[0])

    assert abs(neff.imag) > 1e-2 * abs(neff.real), f"Detector reference neff should be complex under loss, got {neff}"


def test_mode_source_injects_complex_transverse_phase():
    """The lossy eigenmode's transverse phase is carried through to injection.

    Reviewer concern: solving against complex permittivity but projecting the
    profile to ``jnp.real`` would launch a different field than ``compute_mode``
    returned, and a ``_neff``-only assertion would not catch it. Here we
    (1) confirm the stored modal fields stay complex with a non-negligible
    imaginary part, and (2) confirm the injected field actually depends on that
    imaginary part — discarding it changes the launched field.
    """
    objects, constraints, config, _volume, wave = _build_waveguide_domain(core_conductivity=_CORE_SIGMA)
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    source = obj_container.sources[0]

    # (1) the transverse phase of the lossy mode is retained (not projected to real)
    assert jnp.iscomplexobj(source._E) and jnp.iscomplexobj(source._H)
    assert float(jnp.max(jnp.abs(jnp.imag(source._E)))) > 1e-4 * float(jnp.max(jnp.abs(source._E)))
    assert float(jnp.max(jnp.abs(jnp.imag(source._H)))) > 1e-4 * float(jnp.max(jnp.abs(source._H)))

    # (2) the injected field depends on that imaginary part. Inject once with the
    # full complex mode and once with its real projection; a mid-run time step
    # (past the 4-period startup ramp) gives a nonzero carrier amplitude.
    steps_per_period = max(1, round(wave.get_period() / config.time_step_duration))
    time_step = jnp.asarray(6 * steps_per_period)
    E0 = jnp.zeros_like(arrays.fields.E)

    def _inject(src):
        return src.update_E(
            E=E0,
            inv_permittivities=arrays.inv_permittivities,
            inv_permeabilities=arrays.inv_permeabilities,
            time_step=time_step,
            inverse=False,
        )

    E_complex = _inject(source)
    source_real = source.aset("_E", jnp.real(source._E)).aset("_H", jnp.real(source._H))
    E_real = _inject(source_real)

    diff = float(jnp.max(jnp.abs(E_complex - E_real)))
    scale = float(jnp.max(jnp.abs(E_complex)))
    assert scale > 0, "Injected field is zero — cannot assess phase"
    assert diff > 1e-6 * scale, "Injected field does not depend on the modal transverse phase"
