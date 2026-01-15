import math
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field
from fdtdx.core.wavelength import WaveCharacter


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

    #: Phase shift of the carrier wave
    phase_shift: float = frozen_field(default=math.pi)

    #: number of periods between start
    num_startup_periods: int = 4

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
    """Gaussian pulse temporal profile with carrier wave.

    The pulse envelope is characterized by its spectral width, which determines
    the temporal width of the pulse. The carrier wave can be specified using
    any of the WaveCharacter parameters (wavelength, frequency, or period).
    """

    #: Spectral width of the Gaussian envelope (can specify via wavelength, frequency, or period)
    spectral_width: WaveCharacter = frozen_field()

    #: Center frequency/wavelength of the carrier wave
    center_wave: WaveCharacter = frozen_field()

    def __post_init__(self):
        if self.spectral_width.phase_shift != 0.0:
            raise ValueError(
                "spectral_width should not have a phase_shift. Phase shifts should only be applied to center_wave."
            )

    def get_amplitude(
        self,
        time: jax.Array,
        period: float,
        phase_shift: float = 0.0,
    ) -> jax.Array:
        del period

        # Get frequency values from WaveCharacter objects
        spectral_width_hz = self.spectral_width.get_frequency()
        center_frequency_hz = self.center_wave.get_frequency()

        # Calculate envelope parameters
        sigma_t = 1.0 / (2 * jnp.pi * spectral_width_hz)
        t0 = 6 * sigma_t  # Offset peak to avoid discontinuity at t=0

        # Gaussian envelope
        envelope = jnp.exp(-((time - t0) ** 2) / (2 * sigma_t**2))

        # Carrier wave (including phase shift from center_wave)
        carrier_phase = 2 * jnp.pi * center_frequency_hz * time + phase_shift + self.center_wave.phase_shift
        carrier = jnp.real(jnp.exp(-1j * carrier_phase))

        return envelope * carrier
