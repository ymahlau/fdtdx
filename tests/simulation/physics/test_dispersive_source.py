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


def test_broadband_correction_reduces_backward_reflection():
    """The FIR-filtered H profile must reduce the pulse-integrated
    backward flux relative to the scalar ω_c-only correction.

    We do not require an absolute near-zero backward flux — off-center
    spectral components passing through a TFSF boundary in a dispersive
    medium always leak a little due to numerical discretization. The
    claim under test is comparative: adding the convolution filter
    should leave LESS backward energy than the legacy single-frequency
    correction.
    """
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
    assert ratio < 0.85, (
        f"Broadband correction did not reduce normalized backward leakage: "
        f"leakage_corrected={leakage_corrected:.4e}, "
        f"leakage_uncorrected={leakage_uncorrected:.4e}, "
        f"ratio={ratio:.3f} (expected < 0.85)"
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
