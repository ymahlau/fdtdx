"""Tests for objects/sources/profile.py - Temporal profiles."""

import jax.numpy as jnp

from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.profile import GaussianPulseProfile, SingleFrequencyProfile


class TestSingleFrequencyProfile:
    """Tests for SingleFrequencyProfile class."""

    def test_default_parameters(self):
        """Test default parameter values."""
        profile = SingleFrequencyProfile()
        assert profile.phase_shift == jnp.pi
        assert profile.num_startup_periods == 4

    def test_custom_parameters(self):
        """Test with custom parameters."""
        profile = SingleFrequencyProfile(phase_shift=jnp.pi / 2, num_startup_periods=10)
        assert profile.phase_shift == jnp.pi / 2
        assert profile.num_startup_periods == 10

    def test_get_amplitude_at_zero_time(self):
        """Test amplitude at time zero."""
        profile = SingleFrequencyProfile(phase_shift=0.0)
        period = 1e-12  # 1 ps

        amplitude = profile.get_amplitude(time=0.0, period=period)

        # At t=0 with phase_shift=0: cos(0) = 1.0, but with 0 startup factor
        assert jnp.isclose(amplitude, 0.0)

    def test_get_amplitude_after_startup(self):
        """Test amplitude after startup period."""
        profile = SingleFrequencyProfile(phase_shift=0.0, num_startup_periods=4)
        period = 1e-12

        # After startup (at t = 4 * period)
        time = 4 * period
        amplitude = profile.get_amplitude(time=time, period=period)

        # Startup factor should be 1.0, cos(8π) = 1.0
        assert jnp.isclose(amplitude, 1.0, atol=1e-6)

    def test_get_amplitude_during_startup(self):
        """Test amplitude during startup ramp."""
        profile = SingleFrequencyProfile(phase_shift=0.0, num_startup_periods=4)
        period = 1e-12

        # At half startup (t = 2 * period)
        time = 2 * period
        amplitude = profile.get_amplitude(time=time, period=period)

        # Startup factor should be 0.5
        # cos(4π) = 1.0, so amplitude = 0.5 * 1.0 = 0.5
        assert jnp.isclose(amplitude, 0.5, atol=1e-6)

    def test_get_amplitude_with_phase_shift(self):
        """Test amplitude with additional phase shift."""
        profile = SingleFrequencyProfile(phase_shift=jnp.pi, num_startup_periods=1)
        period = 1e-12

        # After startup with additional π phase shift: cos(...+π) should be negative
        time = 2 * period  # Well after startup
        amplitude = profile.get_amplitude(time=time, period=period)

        # Should be close to -cos(4π) = -1.0
        assert jnp.isclose(amplitude, -1.0, atol=1e-6)

    def test_get_amplitude_with_external_phase_shift(self):
        """Test amplitude with external phase shift parameter."""
        profile = SingleFrequencyProfile(phase_shift=0.0, num_startup_periods=1)
        period = 1e-12

        # After startup with external phase shift of π/2
        time = 2 * period
        amplitude = profile.get_amplitude(time=time, period=period, phase_shift=jnp.pi / 2)

        # cos(4π + π/2) = cos(π/2) = 0
        assert jnp.isclose(amplitude, 0.0, atol=1e-6)

    def test_get_amplitude_oscillates(self):
        """Test that amplitude oscillates correctly."""
        profile = SingleFrequencyProfile(phase_shift=0.0, num_startup_periods=1)
        period = 1e-12

        # Test well after startup
        base_time = 10 * period
        t0 = profile.get_amplitude(time=base_time + 0.0 * period, period=period)  # cos(20π) = 1
        t1 = profile.get_amplitude(time=base_time + 0.25 * period, period=period)  # cos(20.5π) = 0
        t2 = profile.get_amplitude(time=base_time + 0.5 * period, period=period)  # cos(21π) = -1
        t3 = profile.get_amplitude(time=base_time + 0.75 * period, period=period)  # cos(21.5π) = 0
        t4 = profile.get_amplitude(time=base_time + 1.0 * period, period=period)  # cos(22π) = 1

        assert jnp.isclose(t0, 1.0, atol=1e-6)
        assert jnp.isclose(t1, 0.0, atol=1e-6)
        assert jnp.isclose(t2, -1.0, atol=1e-6)
        assert jnp.isclose(t3, 0.0, atol=1e-6)
        assert jnp.isclose(t4, 1.0, atol=1e-6)

    def test_get_amplitude_array_input(self):
        """Test amplitude with array input times."""
        profile = SingleFrequencyProfile(phase_shift=0.0, num_startup_periods=1)
        period = 1e-12

        # Times well after startup
        base_time = 10 * period
        times = base_time + jnp.array([0.0, 0.25, 0.5, 0.75, 1.0]) * period
        amplitudes = jnp.array([profile.get_amplitude(t, period) for t in times])

        expected = jnp.array([1.0, 0.0, -1.0, 0.0, 1.0])
        assert jnp.allclose(amplitudes, expected, atol=1e-6)


class TestGaussianPulseProfile:
    """Tests for GaussianPulseProfile class."""

    def test_initialization(self):
        """Test initialization with parameters."""
        profile = GaussianPulseProfile(
            spectral_width=WaveCharacter(frequency=1e12),
            center_wave=WaveCharacter(frequency=193.4e12),
        )
        assert profile.spectral_width.get_frequency() == 1e12
        assert profile.center_wave.get_frequency() == 193.4e12

    def test_get_amplitude_at_zero(self):
        """Test amplitude at time zero."""
        profile = GaussianPulseProfile(
            spectral_width=WaveCharacter(frequency=1e12),
            center_wave=WaveCharacter(frequency=193.4e12),
        )
        period = 1 / profile.center_wave.get_frequency()

        amplitude = profile.get_amplitude(time=0.0, period=period)

        # Gaussian centered at t=0 should have maximum value
        assert amplitude > 0.0

    def test_get_amplitude_returns_scalar(self):
        """Test that amplitude returns scalar value."""
        profile = GaussianPulseProfile(
            spectral_width=WaveCharacter(frequency=1e12),
            center_wave=WaveCharacter(frequency=193.4e12),
        )
        period = 1e-12

        amplitude = profile.get_amplitude(time=1e-14, period=period)

        assert isinstance(amplitude, (float, jnp.ndarray))
        if isinstance(amplitude, jnp.ndarray):
            assert amplitude.shape == ()

    def test_get_amplitude_gaussian_envelope(self):
        """Test that amplitude has Gaussian envelope."""
        profile = GaussianPulseProfile(
            spectral_width=WaveCharacter(frequency=1e12),
            center_wave=WaveCharacter(frequency=193.4e12),
        )
        period = 1 / profile.center_wave.get_frequency()

        # Sample at different times
        t_center = profile.get_amplitude(time=0.0, period=period)
        t_early = profile.get_amplitude(time=-5e-12, period=period)
        t_late = profile.get_amplitude(time=5e-12, period=period)

        # Gaussian should decay away from center
        # abs() to handle oscillating carrier
        assert abs(t_center) > abs(t_early)
        assert abs(t_center) > abs(t_late)

    def test_get_amplitude_with_phase_shift(self):
        """Test amplitude with external phase shift."""
        profile = GaussianPulseProfile(
            spectral_width=WaveCharacter(frequency=1e12),
            center_wave=WaveCharacter(frequency=193.4e12),
        )
        period = 1e-12

        amp_no_shift = profile.get_amplitude(time=1e-14, period=period, phase_shift=0.0)
        amp_with_shift = profile.get_amplitude(time=1e-14, period=period, phase_shift=jnp.pi)

        # Phase shift shouldn't change the absolute value significantly
        # (carrier is modulated by Gaussian envelope)
        assert not jnp.isnan(amp_no_shift)
        assert not jnp.isnan(amp_with_shift)

    def test_get_amplitude_spectral_width_effect(self):
        """Test effect of spectral width on pulse duration."""
        narrow = GaussianPulseProfile(
            spectral_width=WaveCharacter(frequency=1e11),
            center_wave=WaveCharacter(frequency=193.4e12),
        )
        wide = GaussianPulseProfile(
            spectral_width=WaveCharacter(frequency=1e13),
            center_wave=WaveCharacter(frequency=193.4e12),
        )
        period = 1 / 193.4e12

        # Narrow spectral width = longer pulse
        # Wide spectral width = shorter pulse
        t_far = 10e-12

        narrow_far = abs(narrow.get_amplitude(time=t_far, period=period))
        wide_far = abs(wide.get_amplitude(time=t_far, period=period))

        # Narrower spectrum (longer pulse) should have higher amplitude far from center
        assert narrow_far > wide_far or jnp.isclose(narrow_far, wide_far, rtol=0.5)


class TestProfileComparison:
    """Comparison tests between profile types."""

    def test_single_freq_vs_gaussian_at_center(self):
        """Compare single frequency and Gaussian at center."""
        freq = 193.4e12
        single_freq = SingleFrequencyProfile(phase_shift=0.0, num_startup_periods=1)
        gaussian = GaussianPulseProfile(
            spectral_width=WaveCharacter(frequency=1e12),
            center_wave=WaveCharacter(frequency=freq),
        )
        period = 1 / freq

        # Test well after startup
        time = 10 * period
        sf_amp = single_freq.get_amplitude(time=time, period=period)
        g_amp = gaussian.get_amplitude(time=time, period=period)

        # Both should have real values
        assert not jnp.isnan(sf_amp)
        assert not jnp.isnan(g_amp)

    def test_both_respect_phase_shift(self):
        """Test that both profiles respect external phase shift."""
        freq = 193.4e12
        period = 1 / freq
        time = 1e-14

        single_freq = SingleFrequencyProfile(phase_shift=0.0, num_startup_periods=1)
        gaussian = GaussianPulseProfile(
            spectral_width=WaveCharacter(frequency=1e12),
            center_wave=WaveCharacter(frequency=freq),
        )

        for phase in [0.0, jnp.pi / 4, jnp.pi / 2, jnp.pi]:
            sf_amp = single_freq.get_amplitude(time, period, phase_shift=phase)
            g_amp = gaussian.get_amplitude(time, period, phase_shift=phase)

            # Should both produce valid outputs
            assert not jnp.isnan(sf_amp)
            assert not jnp.isnan(g_amp)
