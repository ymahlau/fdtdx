"""Unit tests for the broadband impedance-correction helpers used by
dispersive TFSF sources.

These exercise :func:`compute_eps_spectrum_from_coefficients` and
:func:`compute_impedance_corrected_temporal_profile` in isolation so
failures in the full simulation test are easier to attribute to the
underlying helper rather than the source wiring.
"""

import numpy as np
import pytest

from fdtdx.dispersion import (
    DrudePole,
    LorentzPole,
    compute_eps_spectrum_from_coefficients,
    compute_impedance_corrected_temporal_profile,
    compute_pole_coefficients,
)

_DT = 5e-18
_SPATIAL_SHAPE = (2, 3, 4)  # arbitrary small block of cells


def _broadcast_coeffs(c1_poles, c2_poles, c3_poles, spatial_shape):
    """Expand a (num_poles,) coefficient vector to shape (num_poles, 1, *spatial).

    Matches the ArrayContainer storage layout used by fdtdx's ADE backend.
    """
    num_poles = c1_poles.shape[0]
    target_shape = (num_poles, 1) + spatial_shape
    c1 = np.broadcast_to(c1_poles.reshape(num_poles, 1, 1, 1, 1), target_shape).copy()
    c2 = np.broadcast_to(c2_poles.reshape(num_poles, 1, 1, 1, 1), target_shape).copy()
    c3 = np.broadcast_to(c3_poles.reshape(num_poles, 1, 1, 1, 1), target_shape).copy()
    return c1, c2, c3


class TestEpsSpectrum:
    def test_non_dispersive_returns_eps_inf(self):
        """Zero ADE coefficients → spectrum is flat at eps_inf."""
        c1 = np.zeros((0, 1) + _SPATIAL_SHAPE)
        c2 = np.zeros((0, 1) + _SPATIAL_SHAPE)
        c3 = np.zeros((0, 1) + _SPATIAL_SHAPE)
        eps_inf_value = 2.25
        inv_eps_inf = np.full((1,) + _SPATIAL_SHAPE, 1.0 / eps_inf_value)

        omegas = np.array([1e14, 1e15, 5e15], dtype=np.float64)
        spectrum = compute_eps_spectrum_from_coefficients(
            c1=c1,
            c2=c2,
            c3=c3,
            inv_eps_inf=inv_eps_inf,
            omegas=omegas,
            dt=_DT,
        )

        assert spectrum.shape == omegas.shape
        assert np.allclose(spectrum.real, eps_inf_value)
        assert np.allclose(spectrum.imag, 0.0)

    def test_matches_lorentz_model(self):
        """Reconstructed ε(ω) must match the analytic Lorentz formula."""
        pole = LorentzPole(resonance_frequency=2.0e15, damping=1.0e13, delta_epsilon=2.25)
        c1_vec, c2_vec, c3_vec = compute_pole_coefficients((pole,), dt=_DT)
        c1, c2, c3 = _broadcast_coeffs(
            np.asarray(c1_vec),
            np.asarray(c2_vec),
            np.asarray(c3_vec),
            _SPATIAL_SHAPE,
        )
        eps_inf = 1.0
        inv_eps_inf = np.full((1,) + _SPATIAL_SHAPE, 1.0 / eps_inf)

        omegas = np.array([0.5e15, 1.0e15, 1.8e15, 3.0e15], dtype=np.float64)
        spectrum = compute_eps_spectrum_from_coefficients(
            c1=c1,
            c2=c2,
            c3=c3,
            inv_eps_inf=inv_eps_inf,
            omegas=omegas,
            dt=_DT,
        )

        # The ADE discretization introduces O(ω·dt) warping; for dt=5e-18 and
        # ω≈1e15 that is ~5e-3, so a 2% tolerance is comfortable.
        for w, got in zip(omegas, spectrum, strict=True):
            omega_0 = pole.omega_0
            gamma = pole.gamma
            chi_analytic = (pole.delta_epsilon * omega_0**2) / (omega_0**2 - w**2 - 1j * gamma * w)
            eps_analytic = eps_inf + chi_analytic
            assert abs(got - eps_analytic) / abs(eps_analytic) < 2e-2, (
                f"ω={w:.2e}: numeric {got} vs analytic {eps_analytic}"
            )

    def test_matches_drude_model(self):
        """Same check for a Drude pole (omega_0 = 0)."""
        pole = DrudePole(plasma_frequency=3.0e15, damping=5.0e13)
        c1_vec, c2_vec, c3_vec = compute_pole_coefficients((pole,), dt=_DT)
        c1, c2, c3 = _broadcast_coeffs(
            np.asarray(c1_vec),
            np.asarray(c2_vec),
            np.asarray(c3_vec),
            _SPATIAL_SHAPE,
        )
        eps_inf = 1.0
        inv_eps_inf = np.full((1,) + _SPATIAL_SHAPE, 1.0 / eps_inf)

        omegas = np.array([1.5e15, 2.5e15, 4.0e15], dtype=np.float64)
        spectrum = compute_eps_spectrum_from_coefficients(
            c1=c1,
            c2=c2,
            c3=c3,
            inv_eps_inf=inv_eps_inf,
            omegas=omegas,
            dt=_DT,
        )
        for w, got in zip(omegas, spectrum, strict=True):
            chi_analytic = -(pole.plasma_frequency**2) / (w**2 + 1j * pole.gamma * w)
            eps_analytic = eps_inf + chi_analytic
            assert abs(got - eps_analytic) / abs(eps_analytic) < 2e-2


class TestImpedanceCorrectedProfile:
    def _build_raw_profile(self, n_time: int, period_steps: float) -> np.ndarray:
        """Gaussian-modulated sine at a chosen period."""
        t = np.arange(n_time)
        center = n_time / 2
        sigma = n_time / 8
        envelope = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        carrier = np.sin(2.0 * np.pi * t / period_steps)
        return envelope * carrier

    def _make_spectrum(self, n_time: int, eps_value: complex) -> np.ndarray:
        """Flat ε(ω) spectrum of a length matching the zero-padded signal."""
        m = 1
        while m < 2 * n_time:
            m *= 2
        return np.full(m // 2 + 1, eps_value, dtype=np.complex128)

    def test_identity_when_spectrum_equals_center(self):
        """Constant ε(ω) = ε_center → filter is the identity (up to FFT round-off)."""
        n = 256
        raw = self._build_raw_profile(n, period_steps=16.0)
        eps_c = complex(4.0, 0.0)
        spectrum = self._make_spectrum(n, eps_c)

        filtered = compute_impedance_corrected_temporal_profile(
            raw_samples=raw,
            dt=_DT,
            eps_spectrum=spectrum,
            eps_center=eps_c,
        )
        assert filtered.shape == raw.shape
        assert np.allclose(filtered, raw, atol=1e-10)

    def test_flat_scaling_with_different_eps(self):
        """Constant ε(ω) ≠ ε_center applies a uniform sqrt(ε/ε_c) scale."""
        n = 256
        raw = self._build_raw_profile(n, period_steps=16.0)
        eps_c = complex(4.0, 0.0)
        eps_other = complex(9.0, 0.0)
        spectrum = self._make_spectrum(n, eps_other)

        filtered = compute_impedance_corrected_temporal_profile(
            raw_samples=raw,
            dt=_DT,
            eps_spectrum=spectrum,
            eps_center=eps_c,
        )

        expected_scale = float(np.sqrt(eps_other.real / eps_c.real))
        # Compare amplitudes in the region where raw is well above numerical noise.
        mask = np.abs(raw) > 0.1 * np.max(np.abs(raw))
        ratio = filtered[mask] / raw[mask]
        assert np.allclose(ratio, expected_scale, atol=5e-3), (
            f"expected scale {expected_scale}, got mean ratio {ratio.mean()}"
        )

    def test_narrowband_single_tone_scaling(self):
        """A narrowband tone at ω_tone sees scale ≈ √(ε(ω_tone)/ε_c) even when
        ε(ω) varies across the FFT grid."""
        n = 1024
        period_steps = 32.0  # tone frequency
        raw = self._build_raw_profile(n, period_steps=period_steps)

        m = 1
        while m < 2 * n:
            m *= 2
        dt = 1.0  # sample-based units — the filter only cares about the ratio
        omegas = 2.0 * np.pi * np.fft.rfftfreq(m, d=dt)

        # Build a linear ε(ω) ramp so the filter clearly differs from identity.
        eps_real = 4.0 + 2.0 * (omegas / omegas[-1])  # from 4 at DC to 6 at Nyquist
        spectrum = eps_real.astype(np.complex128)

        omega_tone = 2.0 * np.pi / period_steps
        # Find the ε at the FFT bin closest to the tone frequency.
        tone_bin = int(np.argmin(np.abs(omegas - omega_tone)))
        eps_at_tone = complex(spectrum[tone_bin])
        eps_c = complex(spectrum[0])  # choose DC as the carrier
        expected_scale = float(np.sqrt(eps_at_tone.real / eps_c.real))

        filtered = compute_impedance_corrected_temporal_profile(
            raw_samples=raw,
            dt=dt,
            eps_spectrum=spectrum,
            eps_center=eps_c,
        )

        # Compare peak-to-peak amplitudes around the pulse centre where the
        # Gaussian envelope is near its maximum and transient edge effects
        # from the FIR filter have died out.
        center = n // 2
        window = slice(center - 32, center + 32)
        raw_amp = np.max(np.abs(raw[window]))
        filt_amp = np.max(np.abs(filtered[window]))
        assert raw_amp > 0
        got_scale = filt_amp / raw_amp
        assert abs(got_scale - expected_scale) / expected_scale < 0.03, (
            f"narrowband tone scale: expected {expected_scale}, got {got_scale}"
        )

    def test_zero_input_returns_zero(self):
        """Filtering a zero signal must give zero back."""
        n = 128
        raw = np.zeros(n)
        eps_c = complex(3.0, 0.0)
        spectrum = self._make_spectrum(n, eps_c * 2.0)

        filtered = compute_impedance_corrected_temporal_profile(
            raw_samples=raw,
            dt=_DT,
            eps_spectrum=spectrum,
            eps_center=eps_c,
        )
        assert np.allclose(filtered, 0.0)

    def test_rejects_too_short_spectrum(self):
        """The spectrum length must correspond to M ≥ N FFT points."""
        n = 128
        raw = self._build_raw_profile(n, period_steps=16.0)
        # Spectrum of length 33 corresponds to M = 64 < N = 128
        bad_spectrum = np.full(33, complex(4.0, 0.0), dtype=np.complex128)
        with pytest.raises(ValueError, match="smaller than the raw profile"):
            compute_impedance_corrected_temporal_profile(
                raw_samples=raw,
                dt=_DT,
                eps_spectrum=bad_spectrum,
                eps_center=complex(4.0, 0.0),
            )
