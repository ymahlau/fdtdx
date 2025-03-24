from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import extended_autoinit


@extended_autoinit
class TemporalProfile(ABC):
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
            time: Time points to evaluate amplitude at
            period: Period of the carrier wave (1/frequency)
            phase_shift: Phase shift of the carrier wave

        Returns:
            Amplitude values at the given time points
        """
        raise NotImplementedError()


@extended_autoinit
class SingleFrequencyProfile(TemporalProfile):
    """Simple sinusoidal temporal profile at a single frequency."""

    def get_amplitude(
        self,
        time: jax.Array,
        period: float,
        phase_shift: float = 0.0,
    ) -> jax.Array:
        time_phase = 2 * jnp.pi * time / period + phase_shift
        return jnp.cos(time_phase)


@extended_autoinit
class GaussianPulseProfile(TemporalProfile):
    """Gaussian pulse temporal profile with carrier wave."""

    spectral_width: float  # Width of the Gaussian envelope in frequency domain
    center_frequency: float  # Center frequency of the pulse

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
        carrier = jnp.cos(carrier_phase)

        return envelope * carrier
