"""Temporal envelope shapes and detector apodization windows.

The envelope functions are the single source of truth for the time-domain shapes used by
the source temporal profiles (``GaussianPulseProfile``'s Gaussian envelope,
``SingleFrequencyProfile``'s linear ramp) and by the detector apodization windows below.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field


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
    window (``alpha/2`` at each end). ``alpha=0`` is a rectangular window over
    ``[start, end]``; ``alpha=1`` is a Hann window. Zero outside ``[start, end]``.
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


class TemporalWindow(TreeClass, ABC):
    """Base class for carrier-free temporal windows used as detector apodization."""

    @abstractmethod
    def get_window(self, time: jax.Array) -> jax.Array:
        """Evaluate the window at the given time points.

        Args:
            time (jax.Array): Time points in seconds.

        Returns:
            jax.Array: Non-negative window weights, same shape as ``time``.
        """
        raise NotImplementedError()


@autoinit
class GaussianWindow(TemporalWindow):
    """Gaussian window ``exp(-(t - center_time)^2 / (2 sigma_time^2))``.

    Shares its shape with :class:`~fdtdx.GaussianPulseProfile` via
    :func:`gaussian_envelope`, but carries no oscillating carrier.
    """

    #: Center of the Gaussian window in seconds.
    center_time: float = frozen_field()

    #: Standard deviation of the Gaussian window in seconds. Must be positive.
    sigma_time: float = frozen_field()

    def __post_init__(self):
        if not self.sigma_time > 0:
            raise ValueError(f"sigma_time must be positive, got {self.sigma_time}")

    def get_window(self, time: jax.Array) -> jax.Array:
        return gaussian_envelope(time, self.center_time, self.sigma_time)


@autoinit
class TukeyWindow(TemporalWindow):
    """Tukey (tapered-cosine) window over ``[start_time, end_time]``.

    Flat top of value 1 with cosine-tapered edges occupying a fraction ``alpha`` of the
    window (``alpha=1`` is a Hann window, ``alpha=0`` a rectangular one).
    """

    #: Start of the window in seconds.
    start_time: float = frozen_field()

    #: End of the window in seconds.
    end_time: float = frozen_field()

    #: Fraction of the window occupied by the cosine tapers (0 = rectangular, 1 = Hann).
    alpha: float = frozen_field(default=0.5)

    def __post_init__(self):
        if not self.end_time > self.start_time:
            raise ValueError(f"end_time must exceed start_time, got {self.start_time} and {self.end_time}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {self.alpha}")

    def get_window(self, time: jax.Array) -> jax.Array:
        return tukey_envelope(time, self.start_time, self.end_time, self.alpha)
