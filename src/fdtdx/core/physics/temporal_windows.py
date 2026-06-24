"""Reusable temporal envelope / window shapes and a windowed DFT.

These pure functions are the single source of truth for the time-domain envelopes used by
the source temporal profiles (``GaussianPulseProfile`` Gaussian envelope,
``SingleFrequencyProfile`` linear ramp) *and* by detector apodization windows and the
analytic source power spectrum.  Keeping them here avoids duplicating the window math across
``objects/sources/profile.py`` and the spectral utilities.
"""

import jax
import jax.numpy as jnp


def gaussian_envelope(
    time: jax.Array,
    center: float | jax.Array,
    sigma: float | jax.Array,
) -> jax.Array:
    """Gaussian envelope ``exp(-(t - center)^2 / (2 sigma^2))``."""
    return jnp.exp(-((time - center) ** 2) / (2.0 * sigma**2))


def linear_rampup(
    time: jax.Array,
    ramp_duration: float | jax.Array,
) -> jax.Array:
    """Linear ramp from 0 to 1 over ``[0, ramp_duration]``, clamped to ``[0, 1]``."""
    return jnp.clip(time / ramp_duration, 0.0, 1.0)


def tukey_envelope(
    time: jax.Array,
    start: float | jax.Array,
    end: float | jax.Array,
    alpha: float = 0.5,
) -> jax.Array:
    """Tukey (tapered-cosine) window over ``[start, end]``.

    A flat top of value 1 with cosine-tapered edges occupying a fraction ``alpha`` of the
    window (``alpha/2`` at each end).  ``alpha=0`` is a rectangular window over
    ``[start, end]``; ``alpha=1`` is a Hann window.  Zero outside ``[start, end]``.
    """
    duration = end - start
    x = (time - start) / duration
    in_range = (x >= 0.0) & (x <= 1.0)
    if alpha <= 0.0:
        return jnp.where(in_range, 1.0, 0.0)

    half = alpha / 2.0
    left = 0.5 * (1.0 + jnp.cos(jnp.pi * (x / half - 1.0)))
    right = 0.5 * (1.0 + jnp.cos(jnp.pi * ((x - 1.0) / half + 1.0)))
    window = jnp.where(x < half, left, jnp.where(x > 1.0 - half, right, 1.0))
    return jnp.where(in_range, window, 0.0)


def windowed_dft(
    signal: jax.Array,
    window: jax.Array,
    frequencies: jax.Array,
    dt: float,
) -> jax.Array:
    """Complex windowed DFT ``S(f) = sum_n window[n] * signal[n] * exp(i w_f n dt)``.

    Evaluated at arbitrary (not necessarily FFT-bin) ``frequencies`` in Hz.

    Args:
        signal: Real or complex time signal, shape ``(num_time_steps,)``.
        window: Per-step weights of the same shape (use all-ones for no window).
        frequencies: Frequencies in Hz, shape ``(num_freqs,)``.
        dt: Time-step duration in seconds.

    Returns:
        Complex array of shape ``(num_freqs,)``.
    """
    n = jnp.arange(signal.shape[0])
    omega = 2.0 * jnp.pi * jnp.asarray(frequencies)  # (num_freqs,)
    phase = jnp.exp(1j * omega[:, None] * (n[None, :] * dt))  # (num_freqs, num_time_steps)
    weighted = window * signal  # (num_time_steps,)
    return jnp.sum(phase * weighted[None, :], axis=1)
