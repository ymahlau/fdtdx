import math
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field


@autoinit
class TemporalProfile(TreeClass, ABC):
    """Base class for temporal profiles of sources.

    This class defines how the source amplitude varies in time.
    """

    @abstractmethod
    def get_amplitude(
        self,
        time: jax.Array,
        period: float,
        phase_shift: float = 0.0,
    ) -> jax.Array:
        """Calculate the temporal amplitude at given time points.

        Args:
            time (jax.Array): Time points to evaluate amplitude at
            period (float): Period of the carrier wave (1/frequency)
            phase_shift (float): Phase shift of the carrier wave

        Returns:
            jax.Array: Amplitude values at the given time points
        """
        raise NotImplementedError()


@autoinit
class SingleFrequencyProfile(TemporalProfile):
    """Simple sinusoidal temporal profile at a single frequency."""

    phase_shift: float = frozen_field(default=math.pi)
    num_startup_periods: int = 2

    def get_amplitude(
        self,
        time: jax.Array,
        period: float,
        phase_shift: float = 0.0,
    ) -> jax.Array:
        time_phase = 2 * jnp.pi * time / period + phase_shift + self.phase_shift
        raw_amplitude = jnp.real(jnp.exp(-1j * time_phase))
        startup_time = self.num_startup_periods * period
        factor = jnp.clip(time / startup_time, 0, 1)
        return factor * raw_amplitude


@autoinit
class GaussianPulseProfile(TemporalProfile):
    """Gaussian pulse temporal profile with carrier wave."""

    spectral_width: float = frozen_field()  # Width of the Gaussian envelope in frequency domain
    center_frequency: float = frozen_field()  # Center frequency of the pulse

    def get_amplitude(
        self,
        time: jax.Array,
        period: float,
        phase_shift: float = 0.0,
    ) -> jax.Array:
        del period
        # Calculate envelope parameters
        sigma_t = 1.0 / (2 * jnp.pi * self.spectral_width)
        t0 = 6 * sigma_t  # Offset peak to avoid discontinuity at t=0

        # Gaussian envelope
        envelope = jnp.exp(-((time - t0) ** 2) / (2 * sigma_t**2))

        # Carrier wave
        carrier_phase = 2 * jnp.pi * self.center_frequency * time + phase_shift
        carrier = jnp.real(jnp.exp(-1j * carrier_phase))

        return envelope * carrier
