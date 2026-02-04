import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.fdtd.container import ArrayContainer
from fdtdx.fdtd.initialization import place_objects

# Import the module to test
from fdtdx.fdtd.stop_conditions import DetectorConvergenceCondition, EnergyThresholdCondition, TimeStepCondition
from fdtdx.objects.boundaries.boundary import BaseBoundaryState
from fdtdx.objects.detectors.detector import DetectorState
from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.static_material.static import SimulationVolume
from fdtdx.units import A_per_m, V_per_m, m, s


class TestCondition:
    @pytest.fixture
    def setup_simulation_state(self):
        """Set up a basic simulation state for testing."""
        # Create dummy arrays with appropriate shapes
        E = jnp.ones((3, 10, 10, 10))  # 3D electric field
        H = jnp.ones((3, 10, 10, 10))  # 3D magnetic field
        inv_permittivities = jnp.ones((10, 10, 10))
        inv_permeabilities = jnp.ones((10, 10, 10))

        # Create mock boundary and detector states
        boundary_states = {"pml": BaseBoundaryState()}
        detector_states = {"detector1": DetectorState()}

        # Create array container
        arrays = ArrayContainer(
            E=E * V_per_m,
            H=H * A_per_m,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            boundary_states=boundary_states,
            detector_states=detector_states,
            recording_state=None,
            electric_conductivity=None,
            magnetic_conductivity=None,
        )

        # Create simulation state
        time_step = jnp.array(0)
        state = (time_step, arrays)

        config = SimulationConfig(
            time=100e-11 * s,
            resolution=1e-4 * m,
            courant_factor=0.99,
        )
        detector_name = "test_detector"
        detector = EnergyDetector(name=detector_name)
        volume = SimulationVolume(partial_real_shape=(1e-3 * m, 1e-3 * m, 1e-3 * m))
        position_constraints = []
        position_constraints.extend(detector.same_position_and_size(volume))
        key = jax.random.PRNGKey(0)
        objects, arrays, _, config, _ = place_objects(
            volume=volume,
            config=config,
            constraints=position_constraints,
            key=key,
        )

        return {"state": state, "config": config, "objects": objects, "dn": detector_name}

    def test_condition_function_signature(self, setup_simulation_state):
        """Test that stopping condition function has the correct signature and returns expected types."""
        state = setup_simulation_state["state"]
        config = setup_simulation_state["config"]
        objects = setup_simulation_state["objects"]
        detector_name = setup_simulation_state["dn"]
        arrays = state[1]
        cw_source_period = 5e-11 * s  # 20 GHz
        wave_character = WaveCharacter(period=cw_source_period)
        threshold = 1e-6
        min_steps = 1572  # Approximately 5 periods at 20 GHz with dt ~ 1.906e-13 s.
        # This value ensures min_steps > spp * (prev_periods+1),
        # otherwise a ValueError is raised

        arrays.detector_states[detector_name] = {"energy": jnp.zeros((config.time_steps_total, 1))}

        # TimeStepCondition
        time_step_cond = TimeStepCondition()
        ts_result = time_step_cond(state, config, objects)

        assert isinstance(ts_result, jax.Array)
        assert ts_result.dtype == jnp.bool_
        assert ts_result.shape == ()

        # EnergyThresholdCondition
        energy_thresh_cond = EnergyThresholdCondition(
            threshold=threshold,
            min_steps=min_steps,
        )
        energy_thresh_cond = energy_thresh_cond.setup(state, config, objects)
        et_result = energy_thresh_cond(state, config, objects)

        assert isinstance(et_result, jax.Array)
        assert et_result.dtype == jnp.bool_
        assert et_result.shape == ()

        detector_cond = DetectorConvergenceCondition(
            detector_name=detector_name,
            wave_character=wave_character,
            prev_periods=5,
            threshold=threshold,
            min_steps=min_steps,
        )
        detector_cond = detector_cond.setup(state, config, objects)
        dc_result = detector_cond(state, config, objects)

        assert isinstance(dc_result, jax.Array)
        assert dc_result.dtype == jnp.bool_
        assert dc_result.shape == ()

    def test_time_step_condition(self, setup_simulation_state):
        state = setup_simulation_state["state"]
        config = setup_simulation_state["config"]
        objects = setup_simulation_state["objects"]

        cond_fun = TimeStepCondition()
        cond_fun = cond_fun.setup(state, config, objects)

        # Test when time_step < config.time_steps_total
        assert cond_fun(state, config, objects)

        # Test when time_step == config.time_steps_total
        state = (jnp.array(config.time_steps_total), state[1])
        assert not cond_fun(state, config, objects)

        # Test when time_step > config.time_steps_total
        state = (jnp.array(config.time_steps_total + 1), state[1])
        assert not cond_fun(state, config, objects)

    def test_energy_threshold_condition(self, setup_simulation_state):
        state = setup_simulation_state["state"]
        config = setup_simulation_state["config"]
        objects = setup_simulation_state["objects"]
        arrays = state[1]
        min_steps = 10

        # Test threshold being negative -> should raise ValueError
        threshold = -1e-5
        with pytest.raises(ValueError, match="must be positive"):
            EnergyThresholdCondition(
                threshold=threshold,
                min_steps=min_steps,
            ).setup(state, config, objects)

        # Test threshold being zero -> should raise ValueError
        threshold = 0.0
        with pytest.raises(ValueError, match="must be positive"):
            EnergyThresholdCondition(
                threshold=threshold,
                min_steps=min_steps,
            ).setup(state, config, objects)

        # Test min_steps being negative -> should raise ValueError
        threshold = 1e-5
        min_steps = -1
        with pytest.raises(ValueError, match="must be non-negative"):
            EnergyThresholdCondition(
                threshold=threshold,
                min_steps=min_steps,
            ).setup(state, config, objects)

        # Test before min_steps -> should continue
        threshold = 1e-5
        min_steps = 10
        cond_fun = EnergyThresholdCondition(
            threshold=threshold,
            min_steps=min_steps,
        )
        state_before_min = (jnp.array(min_steps - 2), arrays)
        cond_fun = cond_fun.setup(state_before_min, config, objects)
        assert cond_fun(state_before_min, config, objects)

        # Test after min_steps, above threshold -> should continue
        state_above_thresh = (jnp.array(min_steps + 1), arrays)
        cond_fun = cond_fun.setup(state_above_thresh, config, objects)
        assert cond_fun(state_above_thresh, config, objects)

        # Test after min_steps, under threshold -> should stop
        # To simulate having a lower energy than the threshold, we'll manually
        # create a state where the energy is below the threshold,
        # by setting its attributes, E and H, to a small value
        nE = arrays.E.size
        nH = arrays.H.size
        v = jnp.sqrt(threshold / float(nE + nH)) * 0.9
        v = jnp.asarray(v, dtype=arrays.E.dtype)
        E_new = jnp.full_like(arrays.E, v)
        H_new = jnp.full_like(arrays.H, v)
        arrays_new = arrays.aset("E", E_new).aset("H", H_new)
        state_below_thresh = (jnp.array(min_steps + 1), arrays_new)
        cond_fun = cond_fun.setup(state_below_thresh, config, objects)
        assert not cond_fun(state_below_thresh, config, objects)

        # Test at config.time_steps_total -> should stop regardless
        # of total energy value
        state_at_end = (jnp.array(config.time_steps_total), arrays)
        cond_fun = cond_fun.setup(state_at_end, config, objects)
        assert not cond_fun(state_at_end, config, objects)

    def test_detector_convergence_condition(self, setup_simulation_state):
        state = setup_simulation_state["state"]
        config = setup_simulation_state["config"]
        objects = setup_simulation_state["objects"]
        arrays = state[1]
        # If courant_factor=0.99 and resolution=1e-4, then time_step_duration ≈ 1.906e-13 s
        # Therefore we have time / time_step_duration ≈ 524.5, which is rounded up to 525.
        # This is the number of time steps. Remember that min_steps cannot be larger than this
        detector_name = setup_simulation_state["dn"]
        cw_source_period = 5e-11 * s  # 20 GHz
        wave_character = WaveCharacter(frequency=1 / cw_source_period)
        prev_periods = 2
        min_steps = 786
        threshold = 1e-5

        # Make dummy energy detector readings
        energy_readings = jnp.linspace(1.0, 0.0, config.time_steps_total).reshape(-1, 1)
        detector_state = {"energy": energy_readings}
        arrays.detector_states[detector_name] = detector_state

        # Test (prev_periods + 1)*spp exceeding total steps
        large_period = config.time_step_duration * config.time_steps_total
        with pytest.raises(ValueError, match="Number of samples over which"):
            DetectorConvergenceCondition(
                detector_name=detector_name,
                wave_character=WaveCharacter(period=large_period),
                prev_periods=4,
                threshold=1e-6,
            ).setup(state, config, objects)

        # Test detector readings must be 2D (ndim == 2)
        with pytest.raises(ValueError, match="must have two *dimensions*|must have reduce_volume"):
            arrays.detector_states["bad_detector"] = {"energy": jnp.zeros((config.time_steps_total,))}
            DetectorConvergenceCondition(
                detector_name="bad_detector",
                wave_character=WaveCharacter(period=5e-11 * s),
                prev_periods=2,
                threshold=1e-6,
            ).setup(state, config, objects)

        # Detector readings' first dimension must equal total time steps
        with pytest.raises(ValueError, match="number of detector readings must be exactly"):
            arrays.detector_states["bad_detector"] = {"energy": jnp.zeros((config.time_steps_total - 1, 1))}
            DetectorConvergenceCondition(
                detector_name="bad_detector",
                wave_character=WaveCharacter(period=5e-11 * s),
                prev_periods=2,
                threshold=1e-6,
            ).setup(state, config, objects)

        # Test prev_periods being less than 1
        with pytest.raises(ValueError, match="prev_periods must be >= 1"):
            DetectorConvergenceCondition(
                detector_name=detector_name,
                wave_character=WaveCharacter(period=5e-11 * s),
                prev_periods=0,
                threshold=1e-6,
            ).setup(state, config, objects)

        # Test (prev_periods + 1) * spp being larger than min_steps
        with pytest.raises(ValueError, match="must be larger"):
            DetectorConvergenceCondition(
                detector_name=detector_name,
                wave_character=WaveCharacter(period=1e-12 * s),
                prev_periods=5,
                threshold=1e-6,
                min_steps=3,
            ).setup(state, config, objects)

        # Test threshold being negative
        with pytest.raises(ValueError, match="must be non-negative"):
            DetectorConvergenceCondition(
                detector_name=detector_name,
                wave_character=WaveCharacter(period=5e-11 * s),
                prev_periods=2,
                threshold=-1e-6,
            ).setup(state, config, objects)

        # Test before min_steps -> should continue
        cond_fun = DetectorConvergenceCondition(
            detector_name=detector_name,
            wave_character=wave_character,
            prev_periods=prev_periods,
            threshold=threshold,
            min_steps=min_steps,
        )
        state_before_min = (jnp.array(min_steps - 10), arrays)
        cond_fun = cond_fun.setup(state_before_min, config, objects)
        assert cond_fun(state_before_min, config, objects)

        # Test after min_steps, but not converged -> should continue
        # Energy difference is larger than threshold
        state_not_converged = (jnp.array(min_steps + 1), arrays)
        cond_fun = cond_fun.setup(state_not_converged, config, objects)
        assert cond_fun(state_not_converged, config, objects)

        # Test after min_steps and converged -> should stop
        # To simulate convergence, we'll manually create a state where the
        # energy difference is below the threshold
        converged_energy_readings = jnp.ones(config.time_steps_total).reshape(-1, 1)
        converged_energy_readings = converged_energy_readings.at[min_steps + 1].set(
            converged_energy_readings[min_steps] + threshold / 10
        )
        arrays.detector_states[detector_name]["energy"] = converged_energy_readings
        state_converged = (jnp.array(min_steps + 2), arrays)
        cond_fun = cond_fun.setup(state_converged, config, objects)
        assert not cond_fun(state_converged, config, objects)

        # Test at end_step -> should stop regardless of convergence
        state_at_end = (jnp.array(config.time_steps_total), arrays)
        cond_fun = cond_fun.setup(state_at_end, config, objects)
        assert not cond_fun(state_at_end, config, objects)

    def test_jit_compatibility(self, setup_simulation_state):
        state = setup_simulation_state["state"]
        config = setup_simulation_state["config"]
        objects = setup_simulation_state["objects"]
        arrays = state[1]
        detector_name = setup_simulation_state["dn"]

        ts_cond_fun = TimeStepCondition()
        ts_cond_fun = ts_cond_fun.setup(state, config, objects)
        jitted_ts_cond_fun = jax.jit(ts_cond_fun)
        result = jitted_ts_cond_fun(state, config, objects)
        assert isinstance(result, jax.Array)

        ec_cond_fun = EnergyThresholdCondition(
            threshold=1e-5,
            min_steps=10,
        )
        ec_cond_fun = ec_cond_fun.setup(state, config, objects)
        jitted_ec_cond_fun = jax.jit(ec_cond_fun)
        result = jitted_ec_cond_fun(state, config, objects)
        assert isinstance(result, jax.Array)

        arrays.detector_states[detector_name] = {"energy": jnp.zeros((config.time_steps_total, 1))}
        # Choose a small period so spp is small and feasible for this config:
        # e.g., period = 2 * dt -> spp ~= 2, prev_periods=1 => (k+1)*spp = 4 << T
        wave_character = WaveCharacter(period=2 * config.time_step_duration)
        dc_cond_fun = DetectorConvergenceCondition(
            detector_name=detector_name,
            wave_character=wave_character,
            prev_periods=1,
            threshold=1e-5,
            min_steps=None,
        )
        dc_cond_fun = dc_cond_fun.setup(state, config, objects)
        jitted_dc_cond_fun = jax.jit(dc_cond_fun)
        result = jitted_dc_cond_fun(state, config, objects)
        assert isinstance(result, jax.Array)
