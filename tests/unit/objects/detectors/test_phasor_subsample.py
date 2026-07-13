"""Unit tests for opt-in phasor DFT auto-subsampling.

Subsampling records only every ``stride``-th active step and rescales the kept samples so the
magnitude matches every-step recording. Default is exact (stride 1, byte-identical to before).
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest
from loguru import logger

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.detectors.phasor import _DFT_OVERSAMPLE, PhasorDetector


@pytest.fixture
def wave():
    return WaveCharacter(frequency=1e14)


def _place(det, simulation_config, small_grid_slice, random_key):
    return det.place_on_grid(small_grid_slice, simulation_config, random_key)


def test_default_is_exact(simulation_config, small_grid_slice, random_key, wave):
    det = PhasorDetector(wave_characters=(wave,))
    det = _place(det, simulation_config, small_grid_slice, random_key)
    assert det._dft_stride == 1
    # No steps dropped: recorded count equals the base (every-step) count.
    assert det.num_time_steps_recorded == simulation_config.time_steps_total


def test_stride_one_matches_default(simulation_config, small_grid_slice, random_key, wave):
    default_det = _place(PhasorDetector(wave_characters=(wave,)), simulation_config, small_grid_slice, random_key)
    one_det = _place(
        PhasorDetector(wave_characters=(wave,), dft_subsample=1), simulation_config, small_grid_slice, random_key
    )
    assert one_det._dft_stride == default_det._dft_stride == 1
    assert np.array_equal(np.asarray(one_det._is_on_at_time_step_arr), np.asarray(default_det._is_on_at_time_step_arr))


def test_explicit_stride_thins_on_list(simulation_config, small_grid_slice, random_key, wave):
    stride = 3
    det = PhasorDetector(wave_characters=(wave,), dft_subsample=stride)
    det = _place(det, simulation_config, small_grid_slice, random_key)
    assert det._dft_stride == stride

    on = np.asarray(det._is_on_at_time_step_arr)
    active = np.nonzero(on)[0]
    # Kept steps are exactly every stride-th of the base active steps (which are all steps here).
    expected = np.arange(simulation_config.time_steps_total)[::stride]
    assert np.array_equal(active, expected)
    assert det.num_time_steps_recorded == len(expected)


def test_explicit_stride_near_nyquist_warns(simulation_config, small_grid_slice, random_key, wave):
    """An explicit stride leaving fewer than 4 samples per period triggers an aliasing warning."""
    dt = float(simulation_config.time_step_duration)
    stride = math.ceil(1.0 / (2.0 * 1e14 * dt))  # at/above Nyquist for the 1e14 Hz wave
    messages = []
    sink_id = logger.add(lambda message: messages.append(message), level="WARNING")
    try:
        det = PhasorDetector(wave_characters=(wave,), dft_subsample=stride)
        _place(det, simulation_config, small_grid_slice, random_key)
    finally:
        logger.remove(sink_id)
    assert any("alias" in message for message in messages)


def test_auto_derives_stride(simulation_config, small_grid_slice, random_key, wave):
    det = PhasorDetector(wave_characters=(wave,), dft_subsample="auto")
    det = _place(det, simulation_config, small_grid_slice, random_key)
    dt = float(simulation_config.time_step_duration)
    f_max = 1e14
    expected = max(1, math.floor(1.0 / (_DFT_OVERSAMPLE * f_max * dt)))
    assert expected > 1  # sanity: this config is well-oversampled
    assert det._dft_stride == expected


def test_pulse_scaling_multiplies_by_stride(simulation_config, small_grid_slice, random_key, wave):
    """Pulse mode weights each kept sample by the stride (Riemann sum), restoring magnitude."""
    field = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
    common = dict(wave_characters=(wave,), scaling_mode="pulse", components=("Ex",))

    exact = _place(PhasorDetector(**common, dft_subsample=1), simulation_config, small_grid_slice, random_key)
    strided = _place(PhasorDetector(**common, dft_subsample=3), simulation_config, small_grid_slice, random_key)

    # At time_step 0 the phasor factor is exp(0) = 1, so the recorded value is field * static_scale.
    def _record(det):
        return det.update(
            time_step=jnp.array(0),
            E=field,
            H=field,
            state=det.init_state(),
            inv_permittivity=jnp.ones((3, 8, 8, 8)),
            inv_permeability=1.0,
        )["phasor"]

    exact_val = _record(exact)
    strided_val = _record(strided)
    assert jnp.allclose(strided_val, 3.0 * exact_val)


def _accumulate_phasor(det, config, omega):
    """Drive the detector with a synthetic CW field cos(omega*t) over its active steps."""
    dt = float(config.time_step_duration)
    on = np.asarray(det._is_on_at_time_step_arr)
    state = det.init_state()
    H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
    for t in np.nonzero(on)[0]:
        amp = float(np.cos(omega * t * dt))
        E = jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * amp
        state = det.update(
            time_step=jnp.array(int(t)),
            E=E,
            H=H,
            state=state,
            inv_permittivity=jnp.ones((3, 8, 8, 8)),
            inv_permeability=1.0,
        )
    return complex(state["phasor"][0, 0, 0])


def test_subsample_recovers_amplitude(random_key):
    """Physics: an auto-subsampled phasor recovers the same amplitude as exact recording.

    A single-tone cos(omega t) sampled at dt has amplitude 1; the continuous-mode phasor magnitude
    should recover ~1 both with every-step recording and with auto-subsampling, since the FDTD dt
    keeps the tone far below Nyquist (the whole point of the oversampling margin).
    """
    # Short run (few hundred steps) that is still well-oversampled, so the loop stays fast.
    config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=1e-7))
    freq = 1e14
    omega = 2 * math.pi * freq
    small_slice = ((0, 8), (0, 8), (0, 8))
    common = dict(
        wave_characters=(WaveCharacter(frequency=freq),),
        scaling_mode="continuous",
        components=("Ex",),
        reduce_volume=True,
    )

    exact = PhasorDetector(**common, dft_subsample=1).place_on_grid(small_slice, config, random_key)
    auto = PhasorDetector(**common, dft_subsample="auto").place_on_grid(small_slice, config, random_key)
    assert auto._dft_stride > 1  # sanity: subsampling is actually active

    mag_exact = abs(_accumulate_phasor(exact, config, omega))
    mag_auto = abs(_accumulate_phasor(auto, config, omega))

    assert mag_exact == pytest.approx(1.0, abs=0.05)
    assert mag_auto == pytest.approx(mag_exact, rel=0.05)


def test_continuous_scaling_uses_kept_count(simulation_config, small_grid_slice, random_key, wave):
    """Continuous mode's 2/N scaling reflects the thinned kept-sample count."""
    strided = _place(
        PhasorDetector(wave_characters=(wave,), scaling_mode="continuous", components=("Ex",), dft_subsample=4),
        simulation_config,
        small_grid_slice,
        random_key,
    )
    field = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
    val = strided.update(
        time_step=jnp.array(0),
        E=field,
        H=field,
        state=strided.init_state(),
        inv_permittivity=jnp.ones((3, 8, 8, 8)),
        inv_permeability=1.0,
    )["phasor"]
    # static_scale = 2 / num_time_steps_recorded at time_step 0 (phasor factor 1).
    expected = 2.0 / strided.num_time_steps_recorded
    assert jnp.allclose(val[0, 0], expected)
