from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer

# Import the module to test
from fdtdx.fdtd.forward import forward, forward_single_args_wrapper
from fdtdx.interfaces.state import RecordingState
from fdtdx.objects.boundaries.boundary import BaseBoundaryState
from fdtdx.objects.detectors.detector import DetectorState

# Import the module to test
from fdtdx.fdtd.stop_conditions import StoppingCondition, TimeStepCondition, EnergyConvergenceCondition


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
        state = setup_simulation_state["state"]
        objects = setup_simulation_state["objects"]

        # Test with different combinations of flags
        result_time_step_condition = TimeStepCondition(state, objects)
        result_energy_convergence_condition = EnergyConvergenceCondition(state, objects)

        # Check return type
        assert isinstance(result_time_step_condition, tuple)
        assert isinstance(result_energy_convergence_condition, tuple)

        assert len(result_time_step_condition) == 2
        assert len(result_energy_convergence_condition) == 2

        assert isinstance(result[0], jax.Array)  # time_step
        assert hasattr(result[1], "E")  # arrays should have E field

        # Check that time step increased by 1
        assert result[0] == state[0] + 1
