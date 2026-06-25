"""Frequency-domain power helpers: injected source power, flux, transmission.

These give per-frequency injected power, transmitted fraction,
measured radiated power. ``flux_spectrum`` and
``injected_power_spectrum`` share the same raw windowed-DFT convention (a ``PhasorDetector``
with ``scaling_mode="pulse"``, or any scaling un-scaled internally), so their ratio
``transmission`` is a true per-frequency power fraction in which the source pulse spectrum
and a matching apodization window cancel.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from fdtdx.core.physics.metrics import compute_poynting_flux
from fdtdx.core.temporal.profile import TemporalProfile
from fdtdx.fdtd.container import ArrayContainer
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.sources.source import Source


def _detector_static_scale(detector: PhasorDetector) -> float:
    """Reconstruct the per-step scale baked into the detector's stored phasors."""
    if detector.scaling_mode == "continuous":
        return 2.0 / detector._window_sum
    return 1.0


def _plane_normal_axis(detector: PhasorDetector) -> int:
    grid_shape = detector.grid_shape
    if sum(s == 1 for s in grid_shape) != 1:
        raise ValueError(f"flux_spectrum expects a plane detector (one singleton axis); got {grid_shape}")
    return grid_shape.index(1)


def _face_area(detector: PhasorDetector, axis: int) -> jax.Array:
    grid = detector._config.resolved_grid
    if grid is not None:
        return grid.face_area(axis=axis, slice_tuple=detector.grid_slice_tuple)
    spacing = detector._config.uniform_spacing()
    return jnp.ones(detector.grid_shape, dtype=jnp.float32) * spacing * spacing


def flux_spectrum(detector: PhasorDetector, arrays: ArrayContainer) -> jax.Array:
    """Net time-averaged Poynting flux through a plane detector, per frequency.

    ``½ Re ∮ (E x H*)·n̂ dA`` evaluated from the detector's recorded phasors (un-scaled to
    the raw windowed-DFT convention), for every frequency in ``detector.wave_characters``.
    Requires a plane :class:`~fdtdx.PhasorDetector` (one singleton axis) recording all six
    components in the default order (``Ex, Ey, Ez, Hx, Hy, Hz``) with ``reduce_volume=False``.

    Returns:
        Real ``jax.Array`` of shape ``(num_freqs,)`` — net flux along the +normal axis.
    """
    if detector.reduce_volume:
        raise ValueError("flux_spectrum requires reduce_volume=False (it integrates over the plane).")
    phasor = arrays.detector_states[detector.name]["phasor"]  # (1, num_freqs, num_components, *spatial)
    if phasor.shape[2] != 6:
        raise ValueError("flux_spectrum requires the detector to record all 6 components (Ex..Hz).")

    axis = _plane_normal_axis(detector)
    area = _face_area(detector, axis)
    static_scale = _detector_static_scale(detector)
    num_freqs = phasor.shape[1]

    def flux_at(freq_index: jax.Array) -> jax.Array:
        e_field = phasor[0, freq_index, :3] / static_scale
        h_field = phasor[0, freq_index, 3:] / static_scale
        poynting = compute_poynting_flux(e_field, h_field, axis=0)[axis]
        return 0.5 * jnp.real(jnp.sum(poynting * area))

    return jax.vmap(flux_at)(jnp.arange(num_freqs))


def injected_power_spectrum(
    source: Source,
    frequencies: jax.Array,
    *,
    apodization: TemporalProfile | None = None,
) -> jax.Array:
    """Analytic power injected by ``source`` per frequency (free-function wrapper).

    Delegates to ``source.injected_power_spectrum`` — analytic for plane / mode sources and
    the nominal free-space value for a dipole. For a *measured* value (Purcell-aware) use
    :func:`radiated_power_spectrum`. See ``Source.injected_power_spectrum``.
    """
    return source.injected_power_spectrum(jnp.asarray(frequencies), apodization=apodization)


def transmission(
    out_detector: PhasorDetector,
    arrays: ArrayContainer,
    source: Source,
    *,
    frequencies: jax.Array | None = None,
) -> jax.Array:
    """Transmitted power fraction ``flux_spectrum(out) / injected_power_spectrum(source)``.

    A per-frequency power fraction. The output detector's
    ``apodization`` is forwarded to the source's injected-power computation so the window and
    the pulse spectrum cancel in the ratio. ``frequencies`` defaults to the detector's
    wave-character frequencies (which is what the phasors were recorded at).
    """
    if frequencies is None:
        frequencies = jnp.asarray([wc.get_frequency() for wc in out_detector.wave_characters])
    flux = flux_spectrum(out_detector, arrays)
    injected = source.injected_power_spectrum(jnp.asarray(frequencies), apodization=out_detector.apodization)
    return flux / injected


def radiated_power_spectrum(
    face_detectors: Sequence[tuple[PhasorDetector, float]],
    arrays: ArrayContainer,
) -> jax.Array:
    """Measured net power radiated out of a closed box, per frequency.

    Sums signed :func:`flux_spectrum` over the box faces, each given as
    ``(detector, outward_sign)`` where ``outward_sign`` is ``+1`` if the detector's +normal
    points out of the box and ``-1`` otherwise. Works for *any* source type and any number
    of sources (interference included), and is environment-dependent — so it captures the
    actual (Purcell-affected) radiated power, unlike the nominal
    :func:`injected_power_spectrum`. Also the validation oracle for the analytic path.
    """
    total: jax.Array | None = None
    for detector, outward_sign in face_detectors:
        contribution = outward_sign * flux_spectrum(detector, arrays)
        total = contribution if total is None else total + contribution
    if total is None:
        raise ValueError("face_detectors must contain at least one (detector, outward_sign) pair.")
    return total
