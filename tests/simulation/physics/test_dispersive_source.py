"""Physics test for broadband impedance correction at a dispersive TFSF source.

A Gaussian pulse source embedded in a uniform Lorentz medium is fired twice:

1. **Corrected run** — the default path. ``apply()`` populates
   ``_temporal_H_filter`` with the precomputed FIR-filtered H profile that
   bakes in the frequency-dependent impedance
   ``G(ω) = √(ε(ω)/ε(ω_c))``.

2. **Uncorrected run** — same scene, same ω_c impedance rescale baked into
   ``self._H``, but ``_temporal_H_filter`` is reset to ``None`` so the
   inner update loop falls back to the scalar ``η(ω_c)`` injection.

In a dispersive medium the off-center frequency components of the pulse
see a mismatched impedance in the uncorrected case and radiate spurious
power backward through the TFSF boundary. The broadband filter must
materially reduce that backward leakage.

Also included: a smoke test that ``jax.value_and_grad`` through a loss
defined on the corrected run returns finite gradients.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx
from fdtdx.constants import c as c0
from fdtdx.objects.sources.tfsf import TFSFPlaneSource

_WAVELENGTH = 1e-6
_OMEGA = 2.0 * np.pi * c0 / _WAVELENGTH
_RESOLUTION = 25e-9
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION
_DOMAIN_Z = 5e-6
_Z_CELLS = round(_DOMAIN_Z / _RESOLUTION)

_SOURCE_Z = 60
_FWD_Z = 90
_BWD_Z = 30

_SIM_TIME = 180e-15

_DT_APPROX = 0.99 * _RESOLUTION / (c0 * np.sqrt(3))


def _wide_lorentz_model():
    """A Lorentz pole whose resonance sits near the upper edge of the
    test pulse band so that ε(ω) varies appreciably across the pulse
    bandwidth. Damping is small so the medium stays low-loss.
    """
    omega_0 = 1.4 * _OMEGA
    gamma = 5e12
    delta_eps = 1.5
    return fdtdx.DispersionModel(
        poles=(fdtdx.LorentzPole(resonance_frequency=omega_0, damping=gamma, delta_epsilon=delta_eps),)
    )


def _build_scene():
    """Uniform Lorentz medium, pulsed plane source in the middle, two
    Poynting flux detectors straddling the source."""
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

    # Gaussian pulse centered at ω_c with ~15% bandwidth. The spectral width
    # is chosen wide enough that √(ε(ω)/ε(ω_c)) differs by several percent
    # across the pulse band, so a scalar η(ω_c) correction leaves a
    # measurable broadband mismatch.
    center_freq = c0 / _WAVELENGTH
    bandwidth_hz = 0.15 * center_freq
    temporal_profile = fdtdx.GaussianPulseProfile(
        spectral_width=fdtdx.WaveCharacter(frequency=bandwidth_hz),
        center_wave=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
    )

    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
        temporal_profile=temporal_profile,
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)

    material = fdtdx.Material(permittivity=1.0, dispersion=_wide_lorentz_model())
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

    for name, z in (("flux_fwd", _FWD_Z), ("flux_bwd", _BWD_Z)):
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
                det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(z,)),
            ]
        )
        objects.append(det)

    return objects, constraints, config


def _disable_filter_on_sources(obj_container):
    """Clone ``obj_container`` with ``_temporal_H_filter`` reset to ``None``
    on every TFSF source, so the inner update loop falls back to the
    per-step scalar-impedance call path.
    """
    new_sources = []
    for src in obj_container.sources:
        if isinstance(src, TFSFPlaneSource) and src._temporal_H_filter is not None:
            src = src.aset("_temporal_H_filter", None)
        new_sources.append(src)
    return obj_container.replace_sources(new_sources)


def _run(disable_filter: bool):
    key = jax.random.PRNGKey(0)
    objects, constraints, config = _build_scene()
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)

    # Sanity: in a dispersive scene apply() should have populated the filter.
    has_filter = any(isinstance(s, TFSFPlaneSource) and s._temporal_H_filter is not None for s in obj.sources)
    assert has_filter, "Dispersive source should have a precomputed H filter after apply_params"

    if disable_filter:
        obj = _disable_filter_on_sources(obj)

    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj, config=config, key=key)
    return arrays


def _total_unsigned_flux(arrays, name):
    """Time-integrated |flux| through a detector — a robust proxy for
    pulse energy that ignores sign oscillations."""
    flux = np.array(arrays.detector_states[name]["poynting_flux"][:, 0])
    return float(np.sum(np.abs(flux)))


def _impedance_filter_response(frequencies_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Measured vs analytic broadband impedance shaping ``G(ω)`` at given frequencies.

    Rebuilds the (corrected) dispersive source via place + apply (no FDTD run) and compares the
    spectrum of its precomputed H-side temporal profile (``_temporal_H_filter``) to the raw E-side
    profile. That ratio is the impedance-matching filter the source bakes in; analytically it must
    equal ``G(ω) = √(ε(ω)/ε(ω_c))`` with ``ε(ω)`` from the medium's Lorentz dispersion model
    (``eps_inf = 1.0``). Returns ``(|G_measured|, |G_analytic|)`` at ``frequencies_hz``.
    """
    key = jax.random.PRNGKey(0)
    objects, constraints, config = _build_scene()
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
    src = next(s for s in obj.sources if isinstance(s, TFSFPlaneSource))

    dt = config.time_step_duration
    n = config.time_steps_total
    period = src.wave_character.get_period()
    phase_shift = src.wave_character.phase_shift
    times = np.arange(n) * dt
    s_E = np.asarray(src.temporal_profile.get_amplitude(jnp.asarray(times), period, phase_shift), dtype=np.float64)
    s_H = np.asarray(src._temporal_H_filter, dtype=np.float64)
    # The Gaussian pulse is fully contained in the window, so the zero-padded linear-convolution
    # FFT used to build the filter is reproduced here exactly (same M as _build_dispersive_H_filter).
    m = 1
    while m < 2 * n:
        m *= 2
    omegas = 2.0 * np.pi * np.fft.rfftfreq(m, d=dt)
    G_meas = np.fft.rfft(s_H, n=m) / np.fft.rfft(s_E, n=m)

    model = _wide_lorentz_model()
    eps_c = complex(model.permittivity(_OMEGA, eps_inf=1.0))
    g_meas_at = np.empty(len(frequencies_hz))
    g_an_at = np.empty(len(frequencies_hz))
    for i, f in enumerate(frequencies_hz):
        w = 2.0 * np.pi * f
        eps_w = complex(model.permittivity(w, eps_inf=1.0))
        k = int(np.argmin(np.abs(omegas - w)))
        g_meas_at[i] = abs(G_meas[k])
        g_an_at[i] = abs(np.sqrt(eps_w / eps_c))
    return g_meas_at, g_an_at


def test_broadband_correction_reduces_backward_reflection():
    """The FIR-filtered H profile must reduce the pulse-integrated backward flux relative to the
    scalar ω_c-only correction, AND shape the injected spectrum by the analytic impedance ratio.

    Two complementary claims:

    1. **Comparative** (end-to-end): off-center spectral components passing through a TFSF boundary
       in a dispersive medium leak a little backward power due to numerical discretization; adding
       the convolution filter must leave materially LESS backward energy than the single-frequency
       correction. (We do not require near-zero absolute backward flux.)

    2. **Absolute** (the mechanism): the injected H-side spectrum follows the analytic impedance
       shaping ``G(ω) = √(ε(ω)/ε(ω_c))`` at off-carrier frequencies to within a few percent, with
       ``ε(ω)`` taken from the source's Lorentz dispersion model — confirming the leakage drop comes
       from correct frequency-dependent impedance matching, not an unrelated amplitude change.
    """
    # --- Absolute impedance-shaping check on the injected H-side spectrum ---
    center_freq = c0 / _WAVELENGTH
    # Off-carrier frequencies spanning ±10% of the carrier, inside the pulse band, where the
    # analytic G ranges ~0.93..1.10 (a non-trivial shape, not just ~1).
    probe_freqs = center_freq * np.array([0.90, 0.95, 1.05, 1.10])
    g_meas, g_an = _impedance_filter_response(probe_freqs)
    for f, gm, ga in zip(probe_freqs, g_meas, g_an):
        rel = abs(gm - ga) / ga
        assert rel < 0.03, (
            f"injected H-side impedance shaping off analytic √(ε(ω)/ε(ω_c)) at f={f:.3e} Hz: "
            f"|G_meas|={gm:.4f} |G_analytic|={ga:.4f} rel_err={rel:.4f}"
        )

    arrays_corrected = _run(disable_filter=False)
    arrays_uncorrected = _run(disable_filter=True)

    bwd_corrected = _total_unsigned_flux(arrays_corrected, "flux_bwd")
    bwd_uncorrected = _total_unsigned_flux(arrays_uncorrected, "flux_bwd")
    fwd_corrected = _total_unsigned_flux(arrays_corrected, "flux_fwd")
    fwd_uncorrected = _total_unsigned_flux(arrays_uncorrected, "flux_fwd")

    assert fwd_corrected > 0.0, "Corrected run produced no forward flux"
    assert fwd_uncorrected > 0.0, "Uncorrected run produced no forward flux"
    assert bwd_uncorrected > 0.0, "Reference (uncorrected) run produced no backward flux"

    # Compare normalized leakage (bwd/fwd) so the assertion stays meaningful
    # even when the two runs happen to inject different total energies.
    leakage_corrected = bwd_corrected / fwd_corrected
    leakage_uncorrected = bwd_uncorrected / fwd_uncorrected
    ratio = leakage_corrected / leakage_uncorrected
    # Tightened from the original 0.85 toward the observed reduction (measured ratio ~ 0.07, i.e.
    # the broadband filter removes ~93% of the normalized backward leakage). 0.2 keeps a robust
    # margin over the measured value while being >4x tighter than the near-vacuous 0.85 bound.
    assert ratio < 0.2, (
        f"Broadband correction did not sufficiently reduce normalized backward leakage: "
        f"leakage_corrected={leakage_corrected:.4e}, "
        f"leakage_uncorrected={leakage_uncorrected:.4e}, "
        f"ratio={ratio:.3f} (expected < 0.2)"
    )


def test_broadband_correction_gradient_is_finite():
    """``jax.value_and_grad`` through a loss defined on the corrected-run
    forward flux must return a finite value and a finite gradient — the
    new FIR filter code path must not break reverse-mode AD.

    The differentiable quantity is the forward flux evaluated against a
    scalar scale factor applied to the source's static amplitude. We
    don't care about the numerical value of the gradient — only that AD
    survives the filter lookup inside ``update_E``.
    """

    def build_and_run(scale: jax.Array):
        key = jax.random.PRNGKey(0)
        objects, constraints, config = _build_scene()
        obj, arrays, params, config, _ = fdtdx.place_objects(
            object_list=objects,
            config=config,
            constraints=constraints,
            key=key,
        )
        arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)

        # Scale the source amplitude via its per-source static factor. We
        # update every TFSFPlaneSource's static_amplitude_factor to scale * 1.
        new_sources = []
        for src in obj.sources:
            if isinstance(src, TFSFPlaneSource):
                src = src.aset("static_amplitude_factor", scale * src.static_amplitude_factor)
            new_sources.append(src)
        obj = obj.replace_sources(new_sources)

        _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj, config=config, key=key)
        flux = arrays.detector_states["flux_fwd"]["poynting_flux"][:, 0]
        return jnp.sum(jnp.abs(flux))

    scale0 = jnp.asarray(1.0, dtype=jnp.float32)
    value, grad = jax.value_and_grad(build_and_run)(scale0)
    assert jnp.isfinite(value), f"Forward flux is not finite: {value}"
    assert jnp.isfinite(grad), f"Gradient through broadband filter is not finite: {grad}"
