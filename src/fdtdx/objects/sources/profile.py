import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field
from fdtdx.core.wavelength import WaveCharacter


def _unit_scale(max_abs_value: float, units: tuple[tuple[str, float], ...]) -> tuple[str, float]:
    for label, scale in units:
        if max_abs_value * scale >= 0.1:
            return label, scale
    return units[-1]


def _auto_range(
    x: np.ndarray,
    y: np.ndarray,
    *,
    relative_threshold: float,
    pad_fraction: float,
    center: float | None = None,
) -> tuple[float, float]:
    if x.size < 2:
        return (float(x[0]), float(x[0])) if x.size == 1 else (0.0, 1.0)

    y_abs = np.abs(y)
    max_y = float(np.max(y_abs)) if y_abs.size else 0.0
    if max_y == 0.0 or not np.isfinite(max_y):
        return float(x[0]), float(x[-1])

    idx = np.flatnonzero(y_abs >= relative_threshold * max_y)
    if idx.size == 0:
        return float(x[0]), float(x[-1])

    x_min = float(x[idx[0]])
    x_max = float(x[idx[-1]])
    if center is not None:
        x_min = min(x_min, center)
        x_max = max(x_max, center)

    spacing = float(np.median(np.diff(x)))
    if x_max <= x_min:
        half_width = max(20 * spacing, abs(x_min) * 0.05, spacing)
        x_min -= half_width
        x_max += half_width
    else:
        padding = pad_fraction * (x_max - x_min)
        x_min -= padding
        x_max += padding

    return max(float(x[0]), x_min), min(float(x[-1]), x_max)


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

    def get_reference_frequency(self, period: float) -> float:
        """Return the frequency expected to dominate the plotted spectrum."""
        return 1.0 / period

    def get_time_plot_range(self, period: float, total_time: float) -> tuple[float, float] | None:
        del period, total_time
        return None

    def get_frequency_plot_range(
        self,
        period: float,
        frequencies: np.ndarray,
        spectrum: np.ndarray,
    ) -> tuple[float, float] | None:
        del period, frequencies, spectrum
        return None

    def sample_time_signal(
        self,
        *,
        period: float,
        time_step_duration: float,
        num_time_steps: int,
        phase_shift: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample this temporal profile at the same cadence as an FDTD simulation."""
        if num_time_steps < 2:
            raise ValueError("num_time_steps must be at least 2 to sample a time signal")
        if time_step_duration <= 0:
            raise ValueError("time_step_duration must be positive")

        time = jnp.arange(num_time_steps) * time_step_duration
        signal = self.get_amplitude(time=time, period=period, phase_shift=phase_shift)
        return np.asarray(time, dtype=float), np.asarray(signal)

    def frequency_spectrum(
        self,
        *,
        period: float,
        time_step_duration: float,
        num_time_steps: int,
        phase_shift: float = 0.0,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the one-sided FFT magnitude of the sampled source signal."""
        _, signal = self.sample_time_signal(
            period=period,
            time_step_duration=time_step_duration,
            num_time_steps=num_time_steps,
            phase_shift=phase_shift,
        )
        frequencies = np.fft.rfftfreq(num_time_steps, d=time_step_duration)
        spectrum = np.abs(np.fft.rfft(signal))
        if normalize:
            max_spectrum = float(np.max(spectrum)) if spectrum.size else 0.0
            if max_spectrum > 0:
                spectrum = spectrum / max_spectrum
        return frequencies, spectrum

    def plot_time_signal_and_spectrum(
        self,
        *,
        period: float,
        time_step_duration: float,
        num_time_steps: int,
        phase_shift: float = 0.0,
        axs: Any | None = None,
        filename: str | Path | None = None,
        time_range: tuple[float, float] | Literal["auto", "full"] = "auto",
        frequency_range: tuple[float, float] | Literal["auto", "full"] = "auto",
        relative_threshold: float = 0.01,
        normalize_spectrum: bool = True,
    ):
        """Plot the sampled source time signal and its one-sided frequency spectrum."""
        from matplotlib import pyplot as plt

        time, signal = self.sample_time_signal(
            period=period,
            time_step_duration=time_step_duration,
            num_time_steps=num_time_steps,
            phase_shift=phase_shift,
        )
        frequencies, spectrum = self.frequency_spectrum(
            period=period,
            time_step_duration=time_step_duration,
            num_time_steps=num_time_steps,
            phase_shift=phase_shift,
            normalize=normalize_spectrum,
        )

        if axs is None:
            fig, _axs = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        else:
            _axs = axs
            fig = axs[0].figure

        total_time = float(time[-1])
        if time_range == "auto":
            suggested_time_range = self.get_time_plot_range(period=period, total_time=total_time)
            if suggested_time_range is None:
                suggested_time_range = _auto_range(
                    time,
                    signal,
                    relative_threshold=relative_threshold,
                    pad_fraction=0.05,
                )
            time_range = suggested_time_range
        elif time_range == "full":
            time_range = float(time[0]), float(time[-1])

        if frequency_range == "auto":
            suggested_frequency_range = self.get_frequency_plot_range(
                period=period,
                frequencies=frequencies,
                spectrum=spectrum,
            )
            if suggested_frequency_range is None:
                suggested_frequency_range = _auto_range(
                    frequencies,
                    spectrum,
                    relative_threshold=relative_threshold,
                    pad_fraction=0.15,
                    center=self.get_reference_frequency(period),
                )
            frequency_range = suggested_frequency_range
        elif frequency_range == "full":
            frequency_range = float(frequencies[0]), float(frequencies[-1])

        time_label, time_scale = _unit_scale(
            max(abs(time_range[0]), abs(time_range[1])),
            (("fs", 1e15), ("ps", 1e12), ("ns", 1e9), ("s", 1.0)),
        )
        frequency_label, frequency_scale = _unit_scale(
            max(abs(frequency_range[0]), abs(frequency_range[1])),
            (("THz", 1e-12), ("GHz", 1e-9), ("MHz", 1e-6), ("Hz", 1.0)),
        )

        signal_to_plot = np.real(signal) if np.iscomplexobj(signal) else signal
        _axs[0].plot(time * time_scale, signal_to_plot, color="#1f77b4", linewidth=1.6)
        _axs[0].set_xlim(time_range[0] * time_scale, time_range[1] * time_scale)
        _axs[0].set_xlabel(f"Time ({time_label})")
        _axs[0].set_ylabel("Amplitude")
        _axs[0].grid(True, alpha=0.25)

        _axs[1].plot(frequencies * frequency_scale, spectrum, color="#d55e00", linewidth=1.6)
        _axs[1].set_xlim(frequency_range[0] * frequency_scale, frequency_range[1] * frequency_scale)
        _axs[1].set_xlabel(f"Frequency ({frequency_label})")
        _axs[1].set_ylabel("Normalized |FFT|" if normalize_spectrum else "|FFT|")
        _axs[1].grid(True, alpha=0.25)

        if filename is not None:
            fig.savefig(filename, dpi=300, bbox_inches="tight")
        return fig


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

    def get_reference_frequency(self, period: float) -> float:
        del period
        return self.center_wave.get_frequency()

    def get_time_plot_range(self, period: float, total_time: float) -> tuple[float, float] | None:
        del period
        sigma_t = 1.0 / (2 * math.pi * self.spectral_width.get_frequency())
        t0 = 6 * sigma_t
        return 0.0, min(total_time, t0 + 5 * sigma_t)

    def get_frequency_plot_range(
        self,
        period: float,
        frequencies: np.ndarray,
        spectrum: np.ndarray,
    ) -> tuple[float, float] | None:
        del period, spectrum
        center_frequency_hz = self.center_wave.get_frequency()
        spectral_width_hz = self.spectral_width.get_frequency()
        lower = max(float(frequencies[0]), center_frequency_hz - 5 * spectral_width_hz)
        upper = min(float(frequencies[-1]), center_frequency_hz + 5 * spectral_width_hz)
        if upper <= lower:
            return None
        return lower, upper

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


@autoinit
class CustomTimeSignalProfile(TemporalProfile):
    """Sampled waveform temporal profile for arbitrary time signals.

    Stores the source waveform as a pre-computed JAX array and interpolates it
    at each time step inside the FDTD loop.  All signal shaping is done *outside*
    JIT before constructing this object.

    The sampled ``signal`` fully defines the injected time waveform, so the FDTD
    loop does not apply an additional source-level ``phase_shift``.

    Unlike :class:`GaussianPulseProfile`, this profile does **not** accept a
    ``center_wave`` or spectral-width parameter.  For an arbitrary waveform the
    spectral centre is not a free parameter — it is an emergent property of the
    signal.  Letting the user specify it separately would create a silent
    inconsistency risk (e.g. labelling a broadband sinc pulse as if it were a
    narrowband Gaussian).  Instead, :meth:`get_reference_frequency` computes the
    power-weighted spectral centroid directly from ``signal``, and the frequency
    axis of any plot is determined automatically from the actual spectrum content
    via :func:`_auto_range`.
    """

    #: Pre-sampled waveform, shape ``(N,)``.  Lives in the pytree so JAX can
    #: differentiate through the interpolation if needed.
    signal: jax.Array = field(on_setattr=[jnp.asarray])

    #: Duration of a single simulation time step (seconds).
    time_step_duration: float = frozen_field()

    #: Simulation time at which ``signal[0]`` was sampled (seconds).
    start_time: float = frozen_field(default=0.0)

    #: Interpolation mode: ``"linear"`` (default) or ``"nearest"``.
    interpolation: Literal["linear", "nearest"] = frozen_field(default="linear")

    #: Value returned for times outside the sampled window.
    outside_value: float = frozen_field(default=0.0)

    def __post_init__(self):
        if self.signal.ndim != 1:
            raise ValueError(f"signal must be one-dimensional, got shape {self.signal.shape}")
        if self.signal.shape[0] < 2:
            raise ValueError("signal must contain at least two samples")
        if self.time_step_duration <= 0:
            raise ValueError("time_step_duration must be positive")
        if self.interpolation not in ("linear", "nearest"):
            raise ValueError(f"interpolation must be 'linear' or 'nearest', got {self.interpolation!r}")

    def get_reference_frequency(self, period: float) -> float:
        """Return the power-weighted spectral centroid of the stored signal.

        The centroid is well-defined for any spectrum — it equals the peak
        frequency for unimodal signals and the power-weighted average for
        multi-band ones.  It is used by :func:`_auto_range` to anchor the
        frequency axis of spectrum plots.
        """
        del period
        frequencies = np.fft.rfftfreq(len(self.signal), d=self.time_step_duration)
        spectrum = np.abs(np.fft.rfft(np.asarray(self.signal)))
        total = float(np.sum(spectrum))
        if total == 0.0:
            return 0.0
        return float(np.sum(frequencies * spectrum) / total)

    def get_amplitude(
        self,
        time: jax.Array,
        period: float,
        phase_shift: float = 0.0,
    ) -> jax.Array:
        del period, phase_shift

        idx = (time - self.start_time) / self.time_step_duration
        floor_idx = jnp.floor(idx)
        idx0 = floor_idx.astype(jnp.int32)
        frac = idx - floor_idx

        n = self.signal.shape[0]
        valid = (idx0 >= 0) & (idx0 < n)

        idx0_clip = jnp.clip(idx0, 0, n - 1)
        idx1_clip = jnp.clip(idx0_clip + 1, 0, n - 1)

        y0 = self.signal[idx0_clip]
        y1 = self.signal[idx1_clip]

        if self.interpolation == "nearest":
            y = jnp.where(frac < 0.5, y0, y1)
        else:
            y = (1.0 - frac) * y0 + frac * y1

        return jnp.where(valid, y, self.outside_value)
