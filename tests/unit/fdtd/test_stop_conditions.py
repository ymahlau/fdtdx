# tests/unit/fdtd/test_stop_conditions.py

import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.fdtd.container import ArrayContainer
from fdtdx.fdtd.stop_conditions import DetectorConvergenceCondition, EnergyThresholdCondition, TimeStepCondition

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Config with ~525 time steps; time_step_duration ≈ 1.906e-12 s
_CONFIG = SimulationConfig(time=100e-11, resolution=1e-3, courant_factor=0.99)


def _make_arrays(time_steps_total=None, detector_states=None):
    """Build a minimal ArrayContainer; detector_states dict is mutable."""
    if detector_states is None:
        detector_states = {}
    return ArrayContainer(
        E=jnp.ones((3, 4, 4, 4)),
        H=jnp.ones((3, 4, 4, 4)),
        psi_E=jnp.zeros((6, 4, 4, 4)),
        psi_H=jnp.zeros((6, 4, 4, 4)),
        alpha=jnp.zeros((3, 4, 4, 4)),
        kappa=jnp.ones((3, 4, 4, 4)),
        sigma=jnp.zeros((3, 4, 4, 4)),
        inv_permittivities=jnp.ones((4, 4, 4)),
        inv_permeabilities=jnp.ones((4, 4, 4)),
        detector_states=detector_states,
        recording_state=None,
    )


def _make_state(time_step, arrays=None):
    if arrays is None:
        arrays = _make_arrays()
    return (jnp.array(time_step), arrays)


# ---------------------------------------------------------------------------
# TimeStepCondition
# ---------------------------------------------------------------------------


class TestTimeStepCondition:
    def test_continues_before_end(self):
        config = _CONFIG
        state = _make_state(0)
        cond = TimeStepCondition().setup(state, config, None)
        result = cond(state, config, None)
        assert bool(result) is True

    def test_stops_at_end(self):
        config = _CONFIG
        state = _make_state(config.time_steps_total)
        cond = TimeStepCondition().setup(state, config, None)
        assert bool(cond(state, config, None)) is False

    def test_stops_past_end(self):
        config = _CONFIG
        state = _make_state(config.time_steps_total + 5)
        cond = TimeStepCondition().setup(state, config, None)
        assert bool(cond(state, config, None)) is False

    def test_output_is_bool_scalar(self):
        config = _CONFIG
        state = _make_state(0)
        cond = TimeStepCondition().setup(state, config, None)
        result = cond(state, config, None)
        assert result.dtype == jnp.bool_
        assert result.shape == ()

    def test_setup_returns_self(self):
        cond = TimeStepCondition()
        returned = cond.setup(_make_state(0), _CONFIG, None)
        assert returned is cond


# ---------------------------------------------------------------------------
# EnergyThresholdCondition
# ---------------------------------------------------------------------------


class TestEnergyThresholdCondition:
    def test_zero_threshold_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            EnergyThresholdCondition(threshold=0.0, min_steps=5).setup(_make_state(0), _CONFIG, None)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            EnergyThresholdCondition(threshold=-1e-5, min_steps=5).setup(_make_state(0), _CONFIG, None)

    def test_negative_min_steps_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            EnergyThresholdCondition(threshold=1e-5, min_steps=-1).setup(_make_state(0), _CONFIG, None)

    def test_default_max_steps_uses_total(self):
        config = _CONFIG
        cond = EnergyThresholdCondition(threshold=1e-5).setup(_make_state(0), config, None)
        assert cond.max_steps == config.time_steps_total

    def test_explicit_max_steps_preserved(self):
        config = _CONFIG
        cond = EnergyThresholdCondition(threshold=1e-5, max_steps=100).setup(_make_state(0), config, None)
        assert cond.max_steps == 100

    def test_default_min_steps_is_10_percent(self):
        config = _CONFIG
        cond = EnergyThresholdCondition(threshold=1e-5).setup(_make_state(0), config, None)
        expected = int(round(config.time_steps_total * 0.1))
        assert cond.min_steps == expected

    def test_explicit_min_steps_preserved(self):
        config = _CONFIG
        cond = EnergyThresholdCondition(threshold=1e-5, min_steps=7).setup(_make_state(0), config, None)
        assert cond.min_steps == 7

    def test_continues_before_min_steps(self):
        """Condition continues before min_steps even when energy is below threshold."""
        config = _CONFIG
        min_steps = 20
        # Use an extremely high threshold so energy would normally stop the simulation,
        # but min_steps forces continuation regardless.
        zero_arrays = ArrayContainer(
            E=jnp.zeros((3, 4, 4, 4)),
            H=jnp.zeros((3, 4, 4, 4)),
            psi_E=jnp.zeros((6, 4, 4, 4)),
            psi_H=jnp.zeros((6, 4, 4, 4)),
            alpha=jnp.zeros((3, 4, 4, 4)),
            kappa=jnp.ones((3, 4, 4, 4)),
            sigma=jnp.zeros((3, 4, 4, 4)),
            inv_permittivities=jnp.ones((4, 4, 4)),
            inv_permeabilities=jnp.ones((4, 4, 4)),
            detector_states={},
            recording_state=None,
        )
        cond = EnergyThresholdCondition(threshold=1e10, min_steps=min_steps).setup(_make_state(0), config, None)
        # Energy is near zero (below threshold=1e10), but time_step < min_steps → must continue
        state = _make_state(min_steps - 5, zero_arrays)
        assert bool(cond(state, config, None)) is True

    def test_continues_when_above_threshold(self):
        """After min_steps, energy above threshold → condition continues."""
        config = _CONFIG
        min_steps = 5
        # E=H=ones, inv_perm=ones → total_energy >> 1e-100
        cond = EnergyThresholdCondition(threshold=1e-100, min_steps=min_steps).setup(_make_state(0), config, None)
        state = _make_state(min_steps + 1)
        assert bool(cond(state, config, None)) is True

    def test_stops_when_below_threshold(self):
        """After min_steps, energy below threshold → condition stops."""
        config = _CONFIG
        min_steps = 5
        threshold = 1.0
        # Craft tiny E and H so energy < threshold
        tiny = jnp.full((3, 4, 4, 4), 1e-10)
        arrays = _make_arrays()
        arrays = arrays.aset("E", tiny).aset("H", tiny)
        cond = EnergyThresholdCondition(threshold=threshold, min_steps=min_steps).setup(
            _make_state(0, arrays), config, None
        )
        state = _make_state(min_steps + 1, arrays)
        assert bool(cond(state, config, None)) is False

    def test_stops_at_max_steps(self):
        """At max_steps the simulation always stops."""
        config = _CONFIG
        max_steps = 50
        cond = EnergyThresholdCondition(threshold=1e-100, min_steps=5, max_steps=max_steps).setup(
            _make_state(0), config, None
        )
        state = _make_state(max_steps)
        assert bool(cond(state, config, None)) is False

    def test_output_is_bool_scalar(self):
        config = _CONFIG
        cond = EnergyThresholdCondition(threshold=1e-5, min_steps=5).setup(_make_state(0), config, None)
        result = cond(_make_state(0), config, None)
        assert result.dtype == jnp.bool_
        assert result.shape == ()


# ---------------------------------------------------------------------------
# DetectorConvergenceCondition
# ---------------------------------------------------------------------------


def _detector_config():
    """Config with ~525 steps; dt ≈ 1.906e-12 s."""
    return SimulationConfig(time=100e-11, resolution=1e-3, courant_factor=0.99)


def _detector_arrays(config, name="det", *, readings=None):
    """Arrays with a valid 2-D detector state for DetectorConvergenceCondition."""
    if readings is None:
        readings = jnp.ones((config.time_steps_total, 1))
    det_states = {name: {"energy": readings}}
    return _make_arrays(detector_states=det_states)


class TestDetectorConvergenceCondition:
    # -- setup / spp --

    def test_spp_computation(self):
        config = _detector_config()
        spp = 10
        period = spp * config.time_step_duration
        arrays = _detector_arrays(config)
        cond = DetectorConvergenceCondition(
            detector_name="det",
            wave_character=WaveCharacter(period=period),
            prev_periods=2,
            threshold=1e-6,
            min_steps=30,
        ).setup(_make_state(0, arrays), config, None)
        assert cond._spp == spp

    def test_default_max_steps_uses_total(self):
        config = _detector_config()
        spp = 10
        period = spp * config.time_step_duration
        arrays = _detector_arrays(config)
        cond = DetectorConvergenceCondition(
            detector_name="det",
            wave_character=WaveCharacter(period=period),
            prev_periods=2,
            threshold=1e-6,
            min_steps=30,
        ).setup(_make_state(0, arrays), config, None)
        assert cond.max_steps == config.time_steps_total

    def test_explicit_max_steps_preserved(self):
        config = _detector_config()
        spp = 10
        period = spp * config.time_step_duration
        arrays = _detector_arrays(config)
        cond = DetectorConvergenceCondition(
            detector_name="det",
            wave_character=WaveCharacter(period=period),
            prev_periods=2,
            threshold=1e-6,
            min_steps=30,
            max_steps=200,
        ).setup(_make_state(0, arrays), config, None)
        assert cond.max_steps == 200

    def test_default_min_steps_is_prev_plus_one_times_spp(self):
        config = _detector_config()
        spp = 10
        prev_periods = 2
        period = spp * config.time_step_duration
        arrays = _detector_arrays(config)
        cond = DetectorConvergenceCondition(
            detector_name="det",
            wave_character=WaveCharacter(period=period),
            prev_periods=prev_periods,
            threshold=1e-6,
        ).setup(_make_state(0, arrays), config, None)
        assert cond.min_steps == (prev_periods + 1) * spp

    # -- _validate errors --

    def test_missing_detector_raises(self):
        config = _detector_config()
        spp = 10
        arrays = _make_arrays(detector_states={})  # no detector at all
        with pytest.raises(KeyError, match="not found"):
            DetectorConvergenceCondition(
                detector_name="missing",
                wave_character=WaveCharacter(period=spp * config.time_step_duration),
                prev_periods=2,
                threshold=1e-6,
                min_steps=30,
            ).setup(_make_state(0, arrays), config, None)

    def test_wrong_detector_type_raises(self):
        config = _detector_config()
        spp = 10
        # Detector state with an unexpected key (not energy/poynting_flux/fields)
        det_states = {"det": {"unknown_key": jnp.zeros((config.time_steps_total, 1))}}
        arrays = _make_arrays(detector_states=det_states)
        with pytest.raises(KeyError, match="EnergyDetector|PoyntingFluxDetector|FieldDetector"):
            DetectorConvergenceCondition(
                detector_name="det",
                wave_character=WaveCharacter(period=spp * config.time_step_duration),
                prev_periods=2,
                threshold=1e-6,
                min_steps=30,
            ).setup(_make_state(0, arrays), config, None)

    def test_1d_readings_raises(self):
        config = _detector_config()
        spp = 10
        det_states = {"det": {"energy": jnp.zeros((config.time_steps_total,))}}  # 1-D, not 2-D
        arrays = _make_arrays(detector_states=det_states)
        with pytest.raises(ValueError, match="reduce_volume"):
            DetectorConvergenceCondition(
                detector_name="det",
                wave_character=WaveCharacter(period=spp * config.time_step_duration),
                prev_periods=2,
                threshold=1e-6,
                min_steps=30,
            ).setup(_make_state(0, arrays), config, None)

    def test_wrong_reading_count_raises(self):
        config = _detector_config()
        spp = 10
        det_states = {"det": {"energy": jnp.zeros((config.time_steps_total - 1, 1))}}
        arrays = _make_arrays(detector_states=det_states)
        with pytest.raises(ValueError, match="number of detector readings must be exactly"):
            DetectorConvergenceCondition(
                detector_name="det",
                wave_character=WaveCharacter(period=spp * config.time_step_duration),
                prev_periods=2,
                threshold=1e-6,
                min_steps=30,
            ).setup(_make_state(0, arrays), config, None)

    def test_prev_periods_lt1_raises(self):
        config = _detector_config()
        spp = 10
        arrays = _detector_arrays(config)
        with pytest.raises(ValueError, match="prev_periods must be >= 1"):
            DetectorConvergenceCondition(
                detector_name="det",
                wave_character=WaveCharacter(period=spp * config.time_step_duration),
                prev_periods=0,
                threshold=1e-6,
                min_steps=30,
            ).setup(_make_state(0, arrays), config, None)

    def test_negative_threshold_raises(self):
        config = _detector_config()
        spp = 10
        arrays = _detector_arrays(config)
        with pytest.raises(ValueError, match="must be non-negative"):
            DetectorConvergenceCondition(
                detector_name="det",
                wave_character=WaveCharacter(period=spp * config.time_step_duration),
                prev_periods=2,
                threshold=-1e-6,
                min_steps=30,
            ).setup(_make_state(0, arrays), config, None)

    def test_min_steps_too_small_raises(self):
        config = _detector_config()
        spp = 10
        prev_periods = 2
        arrays = _detector_arrays(config)
        # (prev_periods + 1) * spp = 30; min_steps must be >= 30 → set to 5 to trigger error
        with pytest.raises(ValueError, match="must be larger"):
            DetectorConvergenceCondition(
                detector_name="det",
                wave_character=WaveCharacter(period=spp * config.time_step_duration),
                prev_periods=prev_periods,
                threshold=1e-6,
                min_steps=5,
            ).setup(_make_state(0, arrays), config, None)

    def test_period_exceeds_total_raises(self):
        config = _detector_config()
        # Use a period so large that (prev_periods+1)*spp > time_steps_total
        large_period = config.time_step_duration * config.time_steps_total
        arrays = _detector_arrays(config)
        with pytest.raises(ValueError, match="Number of samples over which"):
            DetectorConvergenceCondition(
                detector_name="det",
                wave_character=WaveCharacter(period=large_period),
                prev_periods=4,
                threshold=1e-6,
            ).setup(_make_state(0, arrays), config, None)

    # -- __call__ logic --

    def test_continues_before_min_steps(self):
        config = _detector_config()
        spp = 10
        prev_periods = 2
        min_steps = (prev_periods + 1) * spp  # = 30
        arrays = _detector_arrays(config)
        cond = DetectorConvergenceCondition(
            detector_name="det",
            wave_character=WaveCharacter(period=spp * config.time_step_duration),
            prev_periods=prev_periods,
            threshold=1e-6,
            min_steps=min_steps,
        ).setup(_make_state(0, arrays), config, None)

        state = _make_state(min_steps - 5, arrays)
        assert bool(cond(state, config, None)) is True

    def test_stops_at_time_steps_total(self):
        config = _detector_config()
        spp = 10
        prev_periods = 2
        min_steps = (prev_periods + 1) * spp
        arrays = _detector_arrays(config)
        cond = DetectorConvergenceCondition(
            detector_name="det",
            wave_character=WaveCharacter(period=spp * config.time_step_duration),
            prev_periods=prev_periods,
            threshold=1e-6,
            min_steps=min_steps,
        ).setup(_make_state(0, arrays), config, None)

        state = _make_state(config.time_steps_total, arrays)
        assert bool(cond(state, config, None)) is False

    def test_continues_when_not_converged(self):
        """After min_steps, large spectral distance → not converged → continue."""
        config = _detector_config()
        spp = 10
        prev_periods = 2
        min_steps = (prev_periods + 1) * spp  # = 30

        # Reference periods (indices 0..29) = 0.0; last period (indices 30..40) = 100.0
        readings = jnp.zeros((config.time_steps_total, 1))
        readings = readings.at[min_steps : min_steps + spp].set(100.0)
        arrays = _detector_arrays(config, readings=readings)

        cond = DetectorConvergenceCondition(
            detector_name="det",
            wave_character=WaveCharacter(period=spp * config.time_step_duration),
            prev_periods=prev_periods,
            threshold=1e-6,
            min_steps=min_steps,
        ).setup(_make_state(0, arrays), config, None)

        # Step just after the last period is fully populated
        state = _make_state(min_steps + spp, arrays)
        assert bool(cond(state, config, None)) is True

    def test_stops_when_converged(self):
        """After min_steps, all-constant readings → zero spectral distance → stop."""
        config = _detector_config()
        spp = 10
        prev_periods = 2
        min_steps = (prev_periods + 1) * spp  # = 30

        # All ones → FFT of ref == FFT of last → distance = 0 < any positive threshold
        readings = jnp.ones((config.time_steps_total, 1))
        arrays = _detector_arrays(config, readings=readings)

        cond = DetectorConvergenceCondition(
            detector_name="det",
            wave_character=WaveCharacter(period=spp * config.time_step_duration),
            prev_periods=prev_periods,
            threshold=1e-6,
            min_steps=min_steps,
        ).setup(_make_state(0, arrays), config, None)

        state = _make_state(min_steps + spp, arrays)
        assert bool(cond(state, config, None)) is False

    def test_output_is_bool_scalar(self):
        config = _detector_config()
        spp = 10
        arrays = _detector_arrays(config)
        cond = DetectorConvergenceCondition(
            detector_name="det",
            wave_character=WaveCharacter(period=spp * config.time_step_duration),
            prev_periods=2,
            threshold=1e-6,
            min_steps=30,
        ).setup(_make_state(0, arrays), config, None)
        result = cond(_make_state(0, arrays), config, None)
        assert result.dtype == jnp.bool_
        assert result.shape == ()
