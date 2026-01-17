import jax.numpy as jnp
import pytest

from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.profile import GaussianPulseProfile


def test_gaussian_pulse_initialization_with_wavelength():
    """Test that GaussianPulseProfile can be initialized with wavelength."""
    pulse = GaussianPulseProfile(
        spectral_width=WaveCharacter(wavelength=100e-9), center_wave=WaveCharacter(wavelength=800e-9)
    )
    assert pulse.spectral_width.get_wavelength() == 100e-9
    assert pulse.center_wave.get_wavelength() == 800e-9


def test_gaussian_pulse_initialization_with_frequency():
    """Test that GaussianPulseProfile can be initialized with frequency."""
    pulse = GaussianPulseProfile(
        spectral_width=WaveCharacter(frequency=1e12), center_wave=WaveCharacter(frequency=3.75e14)
    )
    assert pulse.spectral_width.get_frequency() == 1e12
    assert pulse.center_wave.get_frequency() == 3.75e14


def test_gaussian_pulse_initialization_with_period():
    """Test that GaussianPulseProfile can be initialized with period."""
    pulse = GaussianPulseProfile(spectral_width=WaveCharacter(period=1e-12), center_wave=WaveCharacter(period=2.67e-15))
    assert pulse.spectral_width.get_period() == 1e-12
    assert pulse.center_wave.get_period() == 2.67e-15


def test_gaussian_pulse_get_amplitude_runs():
    """Test that get_amplitude executes without error."""
    pulse = GaussianPulseProfile(
        spectral_width=WaveCharacter(wavelength=100e-9), center_wave=WaveCharacter(wavelength=800e-9)
    )

    # Create time array
    time = jnp.linspace(0, 1e-14, 100)
    period = 2.67e-15  # ~800 nm period

    # Get amplitude
    amplitude = pulse.get_amplitude(time, period)

    # Check that we get an array of the right shape
    assert amplitude.shape == time.shape
    assert jnp.all(jnp.isfinite(amplitude))


def test_gaussian_pulse_get_amplitude_with_phase_shift():
    """Test that get_amplitude works with phase shift."""
    pulse = GaussianPulseProfile(
        spectral_width=WaveCharacter(frequency=1e12),
        center_wave=WaveCharacter(frequency=3.75e14, phase_shift=jnp.pi / 4),
    )

    time = jnp.linspace(0, 1e-14, 100)
    period = 1.0 / 3.75e14
    phase_shift = jnp.pi / 2

    amplitude = pulse.get_amplitude(time, period, phase_shift=phase_shift)

    assert amplitude.shape == time.shape
    assert jnp.all(jnp.isfinite(amplitude))


def test_gaussian_pulse_amplitude_properties():
    """Test basic properties of the Gaussian pulse amplitude."""
    pulse = GaussianPulseProfile(
        spectral_width=WaveCharacter(wavelength=100e-9), center_wave=WaveCharacter(wavelength=800e-9)
    )

    # Use a time range that captures the pulse peak and decay
    time = jnp.linspace(0, 2e-13, 1000)
    period = 2.67e-15

    amplitude = pulse.get_amplitude(time, period)

    # Amplitude should be bounded between -1 and 1
    assert jnp.max(jnp.abs(amplitude)) <= 1.0

    # Find the peak amplitude
    max_amplitude = jnp.max(jnp.abs(amplitude))

    # Pulse should have significant amplitude (not all zeros)
    assert max_amplitude > 0.1

    # Amplitude at the end should be much smaller than the peak
    assert jnp.abs(amplitude[-1]) < 0.1 * max_amplitude


def test_gaussian_pulse_spectral_width_with_phase_shift_raises_error():
    """Test that initializing with phase shift in spectral_width raises ValueError."""
    with pytest.raises(ValueError, match="spectral_width should not have a phase_shift"):
        GaussianPulseProfile(
            spectral_width=WaveCharacter(wavelength=100e-9, phase_shift=jnp.pi / 4),
            center_wave=WaveCharacter(wavelength=800e-9),
        )


def test_gaussian_pulse_center_wave_with_phase_shift_allowed():
    """Test that phase shift in center_wave is allowed."""
    # This should not raise an error
    pulse = GaussianPulseProfile(
        spectral_width=WaveCharacter(wavelength=100e-9),
        center_wave=WaveCharacter(wavelength=800e-9, phase_shift=jnp.pi / 2),
    )

    assert pulse.center_wave.phase_shift == jnp.pi / 2

    # And get_amplitude should still work
    time = jnp.linspace(0, 1e-14, 100)
    amplitude = pulse.get_amplitude(time, period=2.67e-15)
    assert jnp.all(jnp.isfinite(amplitude))
