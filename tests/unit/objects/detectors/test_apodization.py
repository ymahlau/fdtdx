"""Tests for PhasorDetector temporal apodization (smooth windows)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.core.window import GaussianWindow, TukeyWindow
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
    assert float(jnp.abs(state["phasor"][0, 0, 0])) == pytest.approx(1.0, abs=2e-3)


def test_continuous_cw_amplitude_preserved_with_tukey(config):
    """The 2/sum(w) coherent-gain correction keeps CW amplitude ~1 under apodization."""
    f = WaveCharacter(wavelength=1e-6).get_frequency()
    win = TukeyWindow(start_time=0.0, end_time=(config.time_steps_total - 1) * config.time_step_duration, alpha=0.5)
    det = PhasorDetector(
        name="d", wave_characters=(WaveCharacter(wavelength=1e-6),), reduce_volume=True, apodization=win
    )
    placed, state = _run_cw(det, config, jax.random.PRNGKey(0), f)
    assert placed._window_sum < placed.num_time_steps_recorded  # window tapers the edges
    # Tapering suppresses the 2w residual, so the apodized reconstruction is ~1000x more
    # accurate than the rectangular one (4.8e-7 vs 5.7e-4). 1e-5 is ~84 float32 ULP.
    assert float(jnp.abs(state["phasor"][0, 0, 0])) == pytest.approx(1.0, abs=1e-5)


def test_window_changes_a_transient_spectrum(config):
    """For a non-CW (decaying) signal, the apodized phasor differs from the rectangular one."""
    f = WaveCharacter(wavelength=1e-6).get_frequency()
    win = TukeyWindow(start_time=0.0, end_time=(config.time_steps_total - 1) * config.time_step_duration, alpha=0.8)

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


# Strides stay above 4 samples/period (17.5 undecimated), so amplitude sits on the numerical
# floor and the same tolerance applies to all of them. Heavier strides alias by design --
# that path is covered in test_phasor_subsample.py.
@pytest.mark.parametrize("stride", [1, 2, 3, 4])
def test_apodization_composes_with_dft_subsampling(config, stride):
    """Apodization and ``dft_subsample`` stack: CW amplitude survives both together.

    The coherent gain is summed over the *thinned* on-mask, so ``sum(w)`` scales with the
    kept-sample count. Getting that wrong (summing over every step) would rescale the phasor
    by the stride.
    """
    wc = WaveCharacter(wavelength=1e-6)
    win = TukeyWindow(start_time=0.0, end_time=(config.time_steps_total - 1) * config.time_step_duration, alpha=0.5)
    det = PhasorDetector(
        name="d",
        wave_characters=(wc,),
        reduce_volume=True,
        apodization=win,
        dft_subsample=stride,
    )
    placed, state = _run_cw(det, config, jax.random.PRNGKey(0), wc.get_frequency())

    # The window is only accumulated on kept steps, so its sum tracks the thinned count.
    assert placed._window_sum <= placed.num_time_steps_recorded
    assert placed._window_sum == pytest.approx(float(jnp.sum(placed._window_at_time_step_arr)), rel=1e-6)
    assert float(jnp.abs(state["phasor"][0, 0, 0])) == pytest.approx(1.0, abs=1e-5)


def test_source_profile_rejected_as_apodization():
    """A source profile is refused as an apodization window."""
    from fdtdx.objects.sources.profile import SingleFrequencyProfile

    with pytest.raises(Exception, match="must be a TemporalWindow"):
        PhasorDetector(
            name="d",
            wave_characters=(WaveCharacter(wavelength=1e-6),),
            apodization=SingleFrequencyProfile(),
        )


def test_window_outside_recorded_interval_is_rejected(config):
    """A window whose support misses the recording window would divide by a zero gain."""
    dur = (config.time_steps_total - 1) * config.time_step_duration
    det = PhasorDetector(
        name="d",
        wave_characters=(WaveCharacter(wavelength=1e-6),),
        apodization=TukeyWindow(start_time=10 * dur, end_time=11 * dur),
    )
    with pytest.raises(Exception, match="must be finite and positive"):
        det.place_on_grid(((0, 4), (0, 4), (0, 1)), config, jax.random.PRNGKey(0))


def test_underflowed_gaussian_window_is_rejected(config):
    """A Gaussian centred far outside the interval underflows to an all-zero window."""
    dur = (config.time_steps_total - 1) * config.time_step_duration
    det = PhasorDetector(
        name="d",
        wave_characters=(WaveCharacter(wavelength=1e-6),),
        apodization=GaussianWindow(center_time=1e6 * dur, sigma_time=1e-18),
    )
    with pytest.raises(Exception, match="must be finite and positive"):
        det.place_on_grid(((0, 4), (0, 4), (0, 1)), config, jax.random.PRNGKey(0))


def test_alpha_zero_tukey_is_identical_to_no_apodization(config):
    """A rectangular Tukey over the full recorded interval must reproduce the un-apodized
    result exactly.

    Tolerance-free: one assertion covers window construction, alignment of the window's
    support with the OnOffSwitch mask, and the coherent-gain path. Verified bit-identical
    (0.0) at both unit and simulation tier, so an off-by-one at either recording edge -
    which every amplitude tolerance above would absorb - fails here immediately.
    """
    f = WaveCharacter(wavelength=1e-6).get_frequency()
    end = (config.time_steps_total - 1) * config.time_step_duration
    win = TukeyWindow(start_time=0.0, end_time=end, alpha=0.0)

    def phasor(apodization):
        det = PhasorDetector(
            name="d",
            wave_characters=(WaveCharacter(wavelength=1e-6),),
            reduce_volume=True,
            apodization=apodization,
        )
        placed, state = _run_cw(det, config, jax.random.PRNGKey(0), f)
        return placed, np.asarray(state["phasor"])

    placed_rect, rect = phasor(None)
    placed_a0, a0 = phasor(win)

    assert placed_a0._window_sum == pytest.approx(placed_rect._window_sum, rel=0, abs=0)
    np.testing.assert_array_equal(a0, rect)
