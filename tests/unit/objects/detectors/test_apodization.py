"""Tests for PhasorDetector temporal apodization (smooth windows)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.temporal.profile import TukeyWindowProfile
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.detectors.phasor import PhasorDetector


@pytest.fixture
def config():
    return SimulationConfig(time=2e-13, grid=UniformGrid(spacing=1e-7), backend="cpu")


@pytest.fixture
def plane():
    return ((0, 4), (0, 4), (0, 1))


def _run_cw(det, config, key, freq):
    det = det.place_on_grid(((0, 4), (0, 4), (0, 1)), config, key)
    state = det.init_state()
    dt = config.time_step_duration
    for n in range(config.time_steps_total):
        t = n * dt
        E = jnp.zeros((3, 4, 4, 1)).at[0].set(jnp.cos(2 * np.pi * freq * t))
        H = jnp.zeros((3, 4, 4, 1))
        state = det.update(jnp.array(n), E, H, state, jnp.ones((3, 4, 4, 1)), 1.0)
    return det, state


def test_no_apodization_is_rectangular(config, plane):
    """Without apodization the window weights are the plain on-mask (sum == recorded steps)."""
    det = PhasorDetector(name="d", wave_characters=(WaveCharacter(wavelength=1e-6),))
    det = det.place_on_grid(plane, config, jax.random.PRNGKey(0))
    assert det._window_sum == pytest.approx(float(det.num_time_steps_recorded))
    np.testing.assert_allclose(
        np.array(det._window_at_time_step_arr), np.array(det._is_on_at_time_step_arr, dtype=float), atol=0
    )


def test_continuous_cw_amplitude_preserved_without_window(config):
    """Backward-compat: continuous-mode CW amplitude reconstruction is ~1."""
    det = PhasorDetector(name="d", wave_characters=(WaveCharacter(wavelength=1e-6),), reduce_volume=True)
    _, state = _run_cw(det, config, jax.random.PRNGKey(0), WaveCharacter(wavelength=1e-6).get_frequency())
    assert float(jnp.abs(state["phasor"][0, 0, 0])) == pytest.approx(1.0, abs=2e-2)


def test_continuous_cw_amplitude_preserved_with_tukey(config):
    """The 2/sum(w) coherent-gain correction keeps CW amplitude ~1 under apodization."""
    f = WaveCharacter(wavelength=1e-6).get_frequency()
    win = TukeyWindowProfile(
        start_time=0.0, end_time=(config.time_steps_total - 1) * config.time_step_duration, alpha=0.5
    )
    det = PhasorDetector(
        name="d", wave_characters=(WaveCharacter(wavelength=1e-6),), reduce_volume=True, apodization=win
    )
    placed, state = _run_cw(det, config, jax.random.PRNGKey(0), f)
    assert placed._window_sum < placed.num_time_steps_recorded  # window tapers the edges
    assert float(jnp.abs(state["phasor"][0, 0, 0])) == pytest.approx(1.0, abs=3e-2)


def test_window_changes_a_transient_spectrum(config):
    """For a non-CW (decaying) signal, the apodized phasor differs from the rectangular one."""
    f = WaveCharacter(wavelength=1e-6).get_frequency()
    win = TukeyWindowProfile(
        start_time=0.0, end_time=(config.time_steps_total - 1) * config.time_step_duration, alpha=0.8
    )

    def run(det):
        det = det.place_on_grid(((0, 4), (0, 4), (0, 1)), config, jax.random.PRNGKey(0))
        state = det.init_state()
        dt = config.time_step_duration
        n_total = config.time_steps_total
        for n in range(n_total):
            t = n * dt
            decay = np.exp(-3.0 * n / n_total)  # transient that is strong at the (tapered) edges
            E = jnp.zeros((3, 4, 4, 1)).at[0].set(decay * jnp.cos(2 * np.pi * f * t))
            state = det.update(jnp.array(n), E, jnp.zeros((3, 4, 4, 1)), state, jnp.ones((3, 4, 4, 1)), 1.0)
        return complex(state["phasor"][0, 0, 0])

    rect = run(
        PhasorDetector(
            name="d", wave_characters=(WaveCharacter(wavelength=1e-6),), reduce_volume=True, scaling_mode="pulse"
        )
    )
    apod = run(
        PhasorDetector(
            name="d",
            wave_characters=(WaveCharacter(wavelength=1e-6),),
            reduce_volume=True,
            scaling_mode="pulse",
            apodization=win,
        )
    )
    assert abs(rect - apod) > 1e-3 * abs(rect)
