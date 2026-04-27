"""Tests for objects/sources/profile.py - Temporal profiles."""

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.profile import CustomTimeSignalProfile, GaussianPulseProfile, SingleFrequencyProfile


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


class TestCustomTimeSignalProfile:
    """Tests for CustomTimeSignalProfile class."""

    def test_default_parameters(self):
        """Test default parameter values."""
        signal = jnp.array([0.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
        profile = CustomTimeSignalProfile(signal=signal, time_step_duration=0.25)

        assert jnp.allclose(profile.signal, signal)
        assert profile.time_step_duration == 0.25
        assert profile.start_time == 0.0
        assert profile.center_wave is None
        assert profile.fwidth is None
        assert profile.interpolation == "linear"
        assert profile.outside_value == 0.0

    def test_exact_sample_times(self):
        """Test that exact sample times return exact signal values."""
        signal = jnp.array([0.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
        profile = CustomTimeSignalProfile(signal=signal, time_step_duration=0.25)

        times = jnp.array([0.0, 0.25, 0.5, 0.75], dtype=jnp.float32)
        amplitudes = profile.get_amplitude(time=times, period=1.0)

        assert jnp.allclose(amplitudes, signal)

    def test_linear_interpolation_between_samples(self):
        """Test linear interpolation between adjacent signal samples."""
        signal = jnp.array([0.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
        profile = CustomTimeSignalProfile(signal=signal, time_step_duration=0.25)

        times = jnp.array([0.125, 0.375, 0.625], dtype=jnp.float32)
        amplitudes = profile.get_amplitude(time=times, period=1.0)

        expected = jnp.array([1.0, 3.0, 6.0], dtype=jnp.float32)
        assert jnp.allclose(amplitudes, expected)

    def test_nearest_interpolation(self):
        """Test nearest-neighbor interpolation around the half-step boundary."""
        signal = jnp.array([0.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
        profile = CustomTimeSignalProfile(
            signal=signal,
            time_step_duration=0.25,
            interpolation="nearest",
        )

        times = jnp.array([0.10, 0.15, 0.35, 0.40], dtype=jnp.float32)
        amplitudes = profile.get_amplitude(time=times, period=1.0)

        expected = jnp.array([0.0, 2.0, 2.0, 4.0], dtype=jnp.float32)
        assert jnp.allclose(amplitudes, expected)

    def test_outside_sample_window_returns_outside_value(self):
        """Test outside_value before the first sample and after the last sample."""
        signal = jnp.array([0.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
        profile = CustomTimeSignalProfile(
            signal=signal,
            time_step_duration=0.25,
            start_time=1.0,
            outside_value=-3.0,
        )

        times = jnp.array([0.75, 1.0, 1.75, 2.0], dtype=jnp.float32)
        amplitudes = profile.get_amplitude(time=times, period=1.0)

        expected = jnp.array([-3.0, 0.0, 8.0, -3.0], dtype=jnp.float32)
        assert jnp.allclose(amplitudes, expected)

    def test_phase_shift_is_ignored(self):
        """Test that pre-sampled custom signals ignore phase_shift."""
        signal = jnp.array([0.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
        profile = CustomTimeSignalProfile(signal=signal, time_step_duration=0.25)

        amp_no_shift = profile.get_amplitude(time=0.375, period=1.0, phase_shift=0.0)
        amp_with_shift = profile.get_amplitude(time=0.375, period=1.0, phase_shift=jnp.pi)

        assert jnp.isclose(amp_no_shift, amp_with_shift)

    def test_array_time_input(self):
        """Test amplitude with vector input times."""
        signal = jnp.array([0.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
        profile = CustomTimeSignalProfile(signal=signal, time_step_duration=0.25)

        times = jnp.array([-0.25, 0.0, 0.125, 0.75, 1.0], dtype=jnp.float32)
        amplitudes = profile.get_amplitude(time=times, period=1.0)

        expected = jnp.array([0.0, 0.0, 1.0, 8.0, 0.0], dtype=jnp.float32)
        assert jnp.allclose(amplitudes, expected)

    def test_reference_frequency_uses_center_wave(self):
        """Test reference frequency metadata fallback and center_wave override."""
        signal = jnp.array([0.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
        profile_without_center = CustomTimeSignalProfile(signal=signal, time_step_duration=0.25)
        profile_with_center = CustomTimeSignalProfile(
            signal=signal,
            time_step_duration=0.25,
            center_wave=WaveCharacter(frequency=193.4e12),
        )

        assert profile_without_center.get_reference_frequency(period=2e-12) == 0.5e12
        assert profile_with_center.get_reference_frequency(period=2e-12) == 193.4e12

    def test_frequency_plot_range_uses_center_and_fwidth(self):
        """Test frequency plot range from center_wave and fwidth metadata."""
        signal = jnp.array([0.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
        profile = CustomTimeSignalProfile(
            signal=signal,
            time_step_duration=0.25,
            center_wave=WaveCharacter(frequency=5.0),
            fwidth=WaveCharacter(frequency=2.0),
        )
        frequencies = jnp.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=jnp.float32)
        spectrum = jnp.ones_like(frequencies)

        frequency_range = profile.get_frequency_plot_range(
            period=1.0,
            frequencies=frequencies,
            spectrum=spectrum,
        )

        assert frequency_range == (0.0, 10.0)

    def test_plot_time_signal_and_spectrum_custom_writes_file(self, tmp_path):
        """Test that custom time-signal plots can be written to disk."""
        signal = jnp.array([0.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
        profile = CustomTimeSignalProfile(signal=signal, time_step_duration=0.25)
        filename = tmp_path / "custom_time_signal.png"

        fig = profile.plot_time_signal_and_spectrum(
            period=1.0,
            time_step_duration=0.25,
            num_time_steps=signal.shape[0],
            filename=filename,
        )

        assert fig is not None
        assert filename.exists()
        plt.close("all")

    def test_plot_time_signal_and_spectrum_accepts_external_axes(self):
        """Test plotting into caller-provided axes."""
        signal = jnp.array([0.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
        profile = CustomTimeSignalProfile(signal=signal, time_step_duration=0.25)
        fig, axs = plt.subplots(1, 2)

        result_fig = profile.plot_time_signal_and_spectrum(
            period=1.0,
            time_step_duration=0.25,
            num_time_steps=signal.shape[0],
            axs=axs,
        )

        assert result_fig is fig
        assert len(axs[0].lines) == 1
        assert len(axs[1].lines) == 1
        plt.close("all")


class TestTemporalProfilePlotting:
    """Tests for shared temporal profile plotting helpers."""

    def test_plot_time_signal_and_spectrum_gaussian_writes_file(self, tmp_path):
        """Test that Gaussian pulse plots can be written to disk."""
        center_wave = WaveCharacter(frequency=193.4e12)
        profile = GaussianPulseProfile(
            spectral_width=WaveCharacter(frequency=10e12),
            center_wave=center_wave,
        )
        filename = tmp_path / "gaussian_time_signal.png"

        fig = profile.plot_time_signal_and_spectrum(
            period=center_wave.get_period(),
            time_step_duration=1e-15,
            num_time_steps=64,
            filename=filename,
        )

        assert fig is not None
        assert filename.exists()
        plt.close("all")


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
