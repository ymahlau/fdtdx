"""Unit tests for the static parts of the frequency-domain adjoint gradient method."""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.fdtd.adjoint import (
    _ADJOINT_PULSE_FRACTION,
    _DFT_BLOCK,
    _GROUP_SEPARATION,
    _block_phases,
    _blocked_dft,
    _Box,
    _phase_table,
    _plan_groups,
    _source_regime,
    _support_box,
)

_DT = 9.5e-17
_STEPS = 1000


def _synthesized_spectrum(group, r):
    """DTFT at the group frequencies of the adjoint source ``sum_k Re(r_k beta_k(m))`` for scalar ``r_k``."""
    signal = np.real(group.source_signals @ r)  # (T_adj,)
    return group.dtft_phases.T @ signal


@pytest.mark.parametrize(
    "frequencies",
    [
        (3.7e14,),
        (3.0e14, 5.0e14),
        (3.5e14, 3.55e14, 3.6e14, 4.0e14),
    ],
)
def test_adjoint_source_spectrum_is_exact(frequencies):
    """The synthesized source has exactly the requested spectrum at every frequency of its run."""
    omegas = 2 * np.pi * np.asarray(frequencies)
    groups = _plan_groups(omegas, _STEPS, _DT, source_bandwidth=None)
    assert sorted(i for g in groups for i in g.freq_indices) == list(range(len(omegas)))
    rng = np.random.default_rng(0)
    for group in groups:
        r = rng.normal(size=len(group.freq_indices)) + 1j * rng.normal(size=len(group.freq_indices))
        np.testing.assert_allclose(_synthesized_spectrum(group, r), r, rtol=1e-8, atol=1e-8 * np.abs(r).max())


def test_close_frequencies_are_split_into_separate_runs():
    omegas = 2 * np.pi * np.asarray([3.5e14, 3.55e14, 3.6e14])
    groups = _plan_groups(omegas, _STEPS, _DT, source_bandwidth=None)
    sigma_min = _ADJOINT_PULSE_FRACTION / (_STEPS * _DT)
    for group in groups:
        w = omegas[list(group.freq_indices)]
        assert len(w) == 1 or np.min(np.diff(np.sort(w))) >= _GROUP_SEPARATION * sigma_min
    assert len(groups) > 1


def test_adjoint_pulse_fits_in_first_half_of_run():
    omegas = 2 * np.pi * np.asarray([3.0e14, 5.0e14])
    (group,) = _plan_groups(omegas, _STEPS, _DT, source_bandwidth=None)
    signals = np.abs(group.source_signals)
    assert signals[_STEPS // 2 :].max() < 1e-6 * signals.max()


def test_adjoint_pulse_is_limited_to_source_bandwidth():
    omegas = 2 * np.pi * np.asarray([3.7e14])
    steps = 10 * _STEPS  # long enough that the run length does not bound the bandwidth
    wide = _plan_groups(omegas, steps, _DT, source_bandwidth=None)[0].sigma_omega
    narrow = _plan_groups(omegas, steps, _DT, source_bandwidth=0.5 * wide)[0].sigma_omega
    assert narrow == pytest.approx(0.5 * wide)
    # but never narrower than the run allows
    floor = _plan_groups(omegas, _STEPS, _DT, source_bandwidth=1e-3 * wide)[0].sigma_omega
    assert floor == pytest.approx(_ADJOINT_PULSE_FRACTION / (_STEPS * _DT))


def test_phase_table_is_reduced_modulo_two_pi():
    omegas = np.asarray([2.4e15])
    times = np.asarray([0.0, 1e-12])
    table = _phase_table(omegas, times)
    np.testing.assert_allclose(table[:, 0], np.exp(1j * np.mod(omegas[0] * times, 2 * np.pi)), rtol=1e-12)


def test_support_box_clips_and_wraps():
    shape = (10, 10, 10)
    box = _support_box(((3, 5), (0, 2), (8, 10)), 2, shape, (False, False, False))
    assert box == _Box(lo=(1, 0, 6), hi=(7, 4, 10))
    # a stencil leaving a periodic axis re-enters on the other side: the whole axis is needed
    wrapped = _support_box(((3, 5), (0, 2), (8, 10)), 2, shape, (False, True, True))
    assert wrapped == _Box(lo=(1, 0, 0), hi=(7, 10, 10))
    assert wrapped.shape == (6, 10, 10)
    assert wrapped.slices == (slice(1, 7), slice(0, 10), slice(0, 10))


def _sources(*profiles):
    return SimpleNamespace(sources=[SimpleNamespace(name=f"s{i}", temporal_profile=p) for i, p in enumerate(profiles)])


def test_source_regime():
    pulse = fdtdx.GaussianPulseProfile(
        spectral_width=fdtdx.WaveCharacter(frequency=1e13), center_wave=fdtdx.WaveCharacter(frequency=3e14)
    )
    cw = fdtdx.SingleFrequencyProfile()
    assert _source_regime(_sources(cw, cw)) == "cw"
    assert _source_regime(_sources(pulse)) == "pulse"
    assert _source_regime(_sources()) == "pulse"
    with pytest.raises(ValueError, match="mix"):
        _source_regime(_sources(cw, pulse))


def test_gradient_config_adjoint():
    config = fdtdx.GradientConfig(method="adjoint")
    assert config.recorder is None and config.adjoint_mode == "exact"
    with pytest.raises(Exception, match="adjoint_time"):
        fdtdx.GradientConfig(method="adjoint", adjoint_time=-1.0)
    with pytest.raises(Exception, match="Unknown gradient computation method"):
        fdtdx.GradientConfig(method="adjoints")  # type: ignore[arg-type]
    with pytest.raises(Exception, match="adjoint_mode"):
        fdtdx.GradientConfig(method="adjoint", adjoint_mode="exactly")  # type: ignore[arg-type]


def test_gradient_config_exact_mode_options():
    config = fdtdx.GradientConfig(method="adjoint", adjoint_mode="exact", adjoint_band_tolerance=1e-4)
    assert config.adjoint_band_tolerance == 1e-4
    # frequency-mode options are checked at run time, since CW sources use the frequency mode by default too
    fdtdx.GradientConfig(method="adjoint", adjoint_frequencies=(fdtdx.WaveCharacter(wavelength=1e-6),))
    with pytest.raises(Exception, match="only applies to adjoint_mode='exact'"):
        fdtdx.GradientConfig(method="adjoint", adjoint_mode="frequency", adjoint_band_tolerance=1e-4)
    with pytest.raises(Exception, match=r"must lie in \(0, 1\)"):
        fdtdx.GradientConfig(method="adjoint", adjoint_mode="exact", adjoint_band_tolerance=1.5)


def _block_phase(phases, m):
    starts, within = phases
    return starts[m // _DFT_BLOCK] * within[m % _DFT_BLOCK]


def test_exact_mode_parseval_pairing():
    """Forward phases over n and the adjoint's reversed phases (offset 1) pair to the exact time-domain sum."""
    num_steps = 2 * _DFT_BLOCK + 7
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=num_steps), rng.normal(size=num_steps)
    bins = np.arange(num_steps // 2 + 1)
    forward, adjoint = _block_phases(bins, num_steps, 0), _block_phases(bins, num_steps, 1)
    X = sum(x[n] * _block_phase(forward, n) for n in range(num_steps))
    # the adjoint loop visits n = N - 1 - m and records conj(Y_j) = sum_n y_n exp(-2 pi i j n / N), numpy's FFT
    Y_conj = sum(y[num_steps - 1 - m] * _block_phase(adjoint, m) for m in range(num_steps))
    np.testing.assert_allclose(Y_conj, np.fft.fft(y)[: bins.size], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(X, np.conj(np.fft.fft(x)[: bins.size]), rtol=1e-9, atol=1e-9)
    weights = np.where((bins == 0) | (2 * bins == num_steps), 1.0, 2.0) / num_steps
    assert np.sum(weights * np.real(X * Y_conj)) == pytest.approx(np.dot(x, y), rel=1e-12)


def test_blocked_dft_matches_direct_sum():
    """Buffered blocks, including a partial last block, give the per-step DFT and the energy diagnostics."""
    num_steps = 2 * _DFT_BLOCK + 5
    box = _Box(lo=(0, 0, 0), hi=(2, 1, 3))
    rng = np.random.default_rng(1)
    E = rng.normal(size=(num_steps, 3, *box.shape)).astype(np.float32)
    H = rng.normal(size=(num_steps, 3, *box.shape)).astype(np.float32)
    bins = np.asarray([0, 3, 11])

    def body(m, carry):
        return carry + 1, [(jnp.asarray(E)[m], jnp.asarray(H)[m])]

    carry, dft, diag, series = jax.jit(
        lambda: _blocked_dft(
            body, jnp.int32(0), num_steps, (box,), _block_phases(bins, num_steps, 0), jnp.float32, True
        )
    )()
    phases = np.exp(2j * np.pi * np.outer(np.arange(num_steps), bins) / num_steps)  # (N, B)
    for computed, fields in zip(dft[0], (E, H)):
        expected = np.tensordot(phases.T, fields, axes=1)
        np.testing.assert_allclose(np.asarray(computed), expected, rtol=1e-4, atol=1e-4)
    energy = np.sum(E**2, axis=(1, 2, 3, 4)) + np.sum(H**2, axis=(1, 2, 3, 4))
    assert int(carry) == num_steps
    np.testing.assert_allclose(np.asarray(diag), [energy.max(), energy[-1]], rtol=1e-5)
    assert series.shape == (num_steps,)


def test_cw_window_leakage_correction_recovers_phasor():
    """The windowed DFT of a steady tone is ``X + kappa conj(X)``; removing kappa recovers ``X`` exactly."""
    omega = 2 * np.pi * 3.7e14
    times = np.arange(_STEPS) * _DT
    on = np.zeros(_STEPS, dtype=bool)
    on[_STEPS // 2 + 3 :] = True  # deliberately not a whole number of periods
    weights = np.where(on, 2.0 / on.sum(), 0.0)
    X = 0.3 - 1.2j
    signal = np.real(X * np.exp(-1j * omega * times))
    X_hat = (weights * signal) @ _phase_table(np.asarray([omega]), times)[:, 0]
    kappa = 0.5 * (weights @ _phase_table(np.asarray([2 * omega]), times)[:, 0])
    assert abs(X_hat - X) > 1e-4  # the leakage is not negligible here
    assert (X_hat - kappa * np.conj(X_hat)) / (1 - abs(kappa) ** 2) == pytest.approx(X, rel=1e-10)
    assert math.isfinite(abs(kappa))
