"""Tests for the shared temporal-window util and the window TemporalProfiles."""

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.temporal.profile import (
    GaussianPulseProfile,
    GaussianWindowProfile,
    SingleFrequencyProfile,
    TukeyWindowProfile,
)
from fdtdx.core.temporal.window import (
    gaussian_envelope,
    linear_rampup,
    tukey_envelope,
    windowed_dft,
)
from fdtdx.core.wavelength import WaveCharacter


class TestWindowShapes:
    def test_gaussian_envelope(self):
        t = jnp.linspace(0, 10, 101)
        w = gaussian_envelope(t, center=5.0, sigma=1.0)
        assert float(w[50]) == pytest.approx(1.0, abs=1e-6)  # peak at center
        # value one sigma away == exp(-0.5)
        assert float(gaussian_envelope(jnp.array(6.0), 5.0, 1.0)) == pytest.approx(np.exp(-0.5), rel=1e-6)
        np.testing.assert_allclose(np.array(w), np.array(w)[::-1], atol=1e-6)  # symmetric

    def test_linear_rampup(self):
        t = jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        w = linear_rampup(t, ramp_duration=1.0)
        np.testing.assert_allclose(np.array(w), [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-6)

    def test_tukey_endpoints_and_flat_top(self):
        t = jnp.linspace(0, 1, 101)
        w = tukey_envelope(t, start=0.0, end=1.0, alpha=0.5)
        assert float(w[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(w[-1]) == pytest.approx(0.0, abs=1e-6)
        assert float(w[50]) == pytest.approx(1.0, abs=1e-6)  # flat middle

    def test_tukey_alpha_limits(self):
        t = jnp.linspace(0, 1, 51)
        rect = tukey_envelope(t, 0.0, 1.0, alpha=0.0)
        np.testing.assert_allclose(np.array(rect), 1.0, atol=1e-6)  # rectangular
        hann = tukey_envelope(t, 0.0, 1.0, alpha=1.0)
        # Hann: 0.5(1 - cos(2 pi x))
        np.testing.assert_allclose(np.array(hann), 0.5 * (1 - np.cos(2 * np.pi * np.array(t))), atol=1e-6)

    def test_tukey_zero_outside_range(self):
        t = jnp.array([-0.1, 0.5, 1.1])
        w = tukey_envelope(t, 0.0, 1.0, alpha=0.5)
        assert float(w[0]) == 0.0 and float(w[2]) == 0.0


class TestWindowedDFT:
    def test_matches_numpy_reference(self):
        dt = 1e-16
        n = 200
        sig = np.cos(2 * np.pi * 3e14 * np.arange(n) * dt)
        window = np.hanning(n)
        freqs = np.array([2e14, 3e14, 4e14])
        ref = np.array([np.sum(window * sig * np.exp(1j * 2 * np.pi * f * np.arange(n) * dt)) for f in freqs])
        got = windowed_dft(jnp.asarray(sig), jnp.asarray(window), jnp.asarray(freqs), dt)
        np.testing.assert_allclose(np.array(got), ref, rtol=1e-3, atol=1e-3 * np.max(np.abs(ref)))


class TestWindowProfiles:
    def test_gaussian_window_profile_is_pure_envelope(self):
        t = jnp.linspace(0, 1e-12, 50)
        prof = GaussianWindowProfile(center_time=5e-13, sigma_time=1e-13)
        # carrier-free: ignores period / phase_shift, equals the envelope util
        np.testing.assert_allclose(
            np.array(prof.get_amplitude(t, period=1e-15, phase_shift=2.0)),
            np.array(gaussian_envelope(t, 5e-13, 1e-13)),
            atol=1e-7,
        )

    def test_tukey_window_profile_is_pure_envelope(self):
        t = jnp.linspace(0, 1e-12, 50)
        prof = TukeyWindowProfile(start_time=0.0, end_time=1e-12, alpha=0.5)
        np.testing.assert_allclose(
            np.array(prof.get_amplitude(t, period=1e-15)),
            np.array(tukey_envelope(t, 0.0, 1e-12, 0.5)),
            atol=1e-7,
        )


class TestProfileRefactorRegression:
    """The DRY refactor must keep the source profiles' output unchanged."""

    def test_gaussian_pulse_unchanged(self):
        center = WaveCharacter(wavelength=1e-6)
        width = WaveCharacter(wavelength=2e-6)
        prof = GaussianPulseProfile(center_wave=center, spectral_width=width)
        t = jnp.linspace(0, 5e-14, 60)
        sigma_t = 1.0 / (2 * np.pi * width.get_frequency())
        t0 = 6 * sigma_t
        envelope = np.exp(-((np.array(t) - t0) ** 2) / (2 * sigma_t**2))
        carrier = np.real(np.exp(-1j * (2 * np.pi * center.get_frequency() * np.array(t) + center.phase_shift)))
        np.testing.assert_allclose(
            np.array(prof.get_amplitude(t, period=1e-15)), envelope * carrier, rtol=1e-4, atol=1e-5
        )

    def test_single_frequency_unchanged(self):
        prof = SingleFrequencyProfile(num_startup_periods=4)
        period = 1e-15
        t = jnp.linspace(0, 10 * period, 80)
        time_phase = 2 * np.pi * np.array(t) / period + 0.0 + prof.phase_shift
        raw = np.real(np.exp(-1j * time_phase))
        factor = np.clip(np.array(t) / (4 * period), 0, 1)
        np.testing.assert_allclose(np.array(prof.get_amplitude(t, period=period)), factor * raw, rtol=1e-4, atol=1e-5)
