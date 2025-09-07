from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer

from fdtdx.interfaces.state import RecordingState
from fdtdx.objects.boundaries.boundary import BaseBoundaryState
from fdtdx.objects.detectors.detector import DetectorState

# Import the module to test
from fdtdx.fdtd.stop_conditions import TimeStepCondition, EnergyThresholdCondition, DetectorConvergenceCondition


class TestCondition:
    @pytest.fixture
    def setup_simulation_state(self):
        """Set up a basic simulation state for testing."""
        # Create mock arrays with appropriate shapes
        E = jnp.ones((10, 10, 10, 3))  # 3D electric field
        H = jnp.ones((10, 10, 10, 3))  # 3D magnetic field
        inv_permittivities = jnp.ones((10, 10, 10))
        inv_permeabilities = jnp.ones((10, 10, 10))

        # Create mock boundary and detector states
        boundary_states = {"pml": Mock(spec=BaseBoundaryState)}
        detector_states = {"detector1": Mock(spec=DetectorState)}
        recording_state = Mock(spec=RecordingState)

        # Create array container
        arrays = ArrayContainer(
            E=E,
            H=H,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            boundary_states=boundary_states,
            detector_states=detector_states,
            recording_state=recording_state,
        )

        # Create simulation state
        time_step = jnp.array(0)
        state = (time_step, arrays)

        # Create mock config and objects
        config = Mock(spec=SimulationConfig)
        objects = Mock(spec=ObjectContainer)

        # Create random key
        key = jax.random.PRNGKey(0)

        return {"state": state, "config": config, "objects": objects, "key": key, "arrays": arrays}

    def test_condition_function_signature(self, setup_simulation_state):
        """Test that stopping condition function has the correct signature and returns expected types."""
        sim_data = setup_simulation_state
        state = sim_data["state"]
        objects = sim_data["objects"]
        arrays = sim_data["arrays"]
        end_step = 100

        # 1. Test TimeStepCondition
        time_step_cond = TimeStepCondition(end_step=end_step)
        ts_result = time_step_cond(state, objects)
        
        assert isinstance(ts_result, jax.Array)
        assert ts_result.dtype == jnp.bool_
        assert ts_result.shape == ()

        # 2. Test EnergyThresholdCondition
        energy_threshold_cond = EnergyThresholdCondition(
            threshold=1e-6,
            end_step=end_step,
            min_steps=10,
        )
        et_result = energy_threshold_cond(state, objects)

        assert isinstance(et_result, jax.Array)
        assert et_result.dtype == jnp.bool_
        assert et_result.shape == ()

        # 3. Test DetectorConvergenceCondition
        # Mock dependencies for DetectorConvergenceCondition
        detector_name = "test_detector"
        arrays.detector_states[detector_name] = {"energy": jnp.zeros((end_step, 1))}
        detector_object_mock = Mock()
        detector_object_mock._time_step_to_arr_idx = jnp.arange(end_step)
        objects[detector_name] = detector_object_mock

        detector_cond = DetectorConvergenceCondition(
            threshold=1e-5,
            end_step=end_step,
            min_steps=10,
            detector_name=detector_name,
        )
        dc_result = detector_cond(state, objects)

        assert isinstance(dc_result, jax.Array)
        assert dc_result.dtype == jnp.bool_
        assert dc_result.shape == ()

    def test_time_step_condition(self, setup_simulation_state):
        """Test the TimeStepCondition stopping condition."""
        state = setup_simulation_state["state"]
        objects = setup_simulation_state["objects"]
        end_step = 100

        # Create the condition function
        cond_fun = TimeStepCondition(end_step=end_step)

        # Test when time_step < end_step
        assert cond_fun(state, objects)

        # Test when time_step == end_step
        state = (jnp.array(end_step), state[1])
        assert not cond_fun(state, objects)

        # Test when time_step > end_step
        state = (jnp.array(end_step + 1), state[1])
        assert not cond_fun(state, objects)

    def test_energy_convergence_condition(self, setup_simulation_state):
        """Test the EnergyConvergenceCondition stopping condition."""
        sim_data = setup_simulation_state
        objects = sim_data["objects"]
        arrays = sim_data["arrays"]

        detector_name = "energy_detector"
        end_step = 200
        min_steps = 100
        threshold = 1e-5

        # Mock the energy detector readings and object properties
        energy_readings = jnp.linspace(1.0, 0.0, end_step).reshape(-1, 1)
        
        # Create a mock for the detector state that can be indexed
        detector_state_mock = {"energy": energy_readings}
        arrays.detector_states[detector_name] = detector_state_mock

        # Mock the object with the time step to array index mapping
        detector_object_mock = Mock()
        # Simple 1-to-1 mapping for this test
        detector_object_mock._time_step_to_arr_idx = jnp.arange(end_step)
        objects[detector_name] = detector_object_mock

        cond_fun = EnergyThresholdCondition(
            threshold=threshold,
            end_step=end_step,
            min_steps=min_steps,
        )

        # Test before min_steps -> should continue
        state_before_min = (jnp.array(min_steps - 10), arrays)
        assert cond_fun(state_before_min, objects)

        # Test after min_steps, but not converged -> should continue
        # Energy difference is larger than threshold
        state_not_converged = (jnp.array(min_steps + 1), arrays)
        assert cond_fun(state_not_converged, objects)

        # Test after min_steps and converged -> should stop
        # To simulate convergence, we'll manually create a state where the
        # energy difference is below the threshold.
        converged_energy_readings = jnp.ones(end_step).reshape(-1, 1)
        converged_energy_readings = converged_energy_readings.at[min_steps + 1].set(
            converged_energy_readings[min_steps] + threshold / 10
        )
        arrays.detector_states[detector_name]["energy"] = converged_energy_readings
        state_converged = (jnp.array(min_steps + 1), arrays)
        assert not cond_fun(state_converged, objects)

        # Test at end_step -> should stop regardless of convergence
        state_at_end = (jnp.array(end_step), arrays)
        assert not cond_fun(state_at_end, objects)

    def test_jit_compatibility(self, setup_simulation_state):
        """Test that stopping conditions are JIT-compatible."""
        sim_data = setup_simulation_state
        state = sim_data["state"]
        objects = sim_data["objects"]
        arrays = sim_data["arrays"]

        # TimeStepCondition
        ts_cond_fun = TimeStepCondition(end_step=100)
        jitted_ts_cond_fun = jax.jit(ts_cond_fun)
        result = jitted_ts_cond_fun(state, objects)
        assert isinstance(result, jax.Array)

        # EnergyThresholdCondition
        detector_name = "energy_detector"
        end_step = 100
        # Mock dependencies for EnergyThresholdCondition
        arrays.detector_states[detector_name] = {"energy": jnp.zeros((end_step, 1))}
        detector_object_mock = Mock()
        detector_object_mock._time_step_to_arr_idx = jnp.arange(end_step)
        objects[detector_name] = detector_object_mock

        ec_cond_fun = EnergyThresholdCondition(
            threshold=1e-5,
            end_step=end_step,
            min_steps=10,
        )
        jitted_ec_cond_fun = jax.jit(ec_cond_fun)
        result = jitted_ec_cond_fun(state, objects)
        assert isinstance(result, jax.Array)
