from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer

# Import the module to test
from fdtdx.fdtd.forward import forward, forward_single_args_wrapper
from fdtdx.interfaces.state import RecordingState
from fdtdx.objects.detectors.detector import DetectorState


class TestForward:
    @pytest.fixture
    def setup_simulation_state(self):
        """Set up a basic simulation state for testing."""
        # Create mock arrays with appropriate shapes
        E = jnp.ones((10, 10, 10, 3))  # 3D electric field
        H = jnp.ones((10, 10, 10, 3))  # 3D magnetic field
        psi_E = jnp.zeros((6, 10, 10, 10))  # 3D auxiliary electric field
        psi_H = jnp.zeros((6, 10, 10, 10))  # 3D auxiliary magnetic field
        alpha = jnp.zeros((3, 10, 10, 10))  # 3D alpha array
        kappa = jnp.ones((3, 10, 10, 10))  # 3D kappa array
        sigma = jnp.zeros((3, 10, 10, 10))  # 3D sigma array
        inv_permittivities = jnp.ones((10, 10, 10))
        inv_permeabilities = jnp.ones((10, 10, 10))

        # Create mock detector states
        detector_states = {"detector1": Mock(spec=DetectorState)}
        recording_state = Mock(spec=RecordingState)

        # Create array container
        arrays = ArrayContainer(
            E=E,
            H=H,
            psi_E=psi_E,
            psi_H=psi_H,
            alpha=alpha,
            kappa=kappa,
            sigma=sigma,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
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

    def test_forward_function_signature(self, setup_simulation_state):
        """Test that forward function has the correct signature and returns expected types."""
        state = setup_simulation_state["state"]
        config = setup_simulation_state["config"]
        objects = setup_simulation_state["objects"]
        key = setup_simulation_state["key"]

        # Mock all the external functions that forward() calls
        with (
            patch("fdtdx.fdtd.forward.update_E") as mock_update_E,
            patch("fdtdx.fdtd.forward.update_H") as mock_update_H,
        ):
            # Set up the return values for the mocked functions
            mock_update_E.return_value = setup_simulation_state["arrays"]
            mock_update_H.return_value = setup_simulation_state["arrays"]

            # Test with different combinations of flags
            result = forward(state, config, objects, key, False, False, False)

            # Check return type
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], jax.Array)  # time_step
            assert hasattr(result[1], "E")  # arrays should have E field

            # Check that time step increased by 1
            assert result[0] == state[0] + 1

            # Verify the mocked functions were called
            mock_update_E.assert_called_once()
            mock_update_H.assert_called_once()

    def test_forward_with_detector_recording(self, setup_simulation_state):
        """Test forward function with detector recording enabled."""
        state = setup_simulation_state["state"]
        config = setup_simulation_state["config"]
        objects = setup_simulation_state["objects"]
        key = setup_simulation_state["key"]

        # Mock all the external functions
        with (
            patch("fdtdx.fdtd.forward.update_E") as mock_update_E,
            patch("fdtdx.fdtd.forward.update_H") as mock_update_H,
            patch("fdtdx.fdtd.forward.update_detector_states") as mock_update_detectors,
        ):
            mock_update_E.return_value = setup_simulation_state["arrays"]
            mock_update_H.return_value = setup_simulation_state["arrays"]
            mock_update_detectors.return_value = setup_simulation_state["arrays"]

            forward(state, config, objects, key, True, False, False)

            # Verify update_detector_states was called
            mock_update_detectors.assert_called_once()

            # Verify other functions were also called
            mock_update_E.assert_called_once()
            mock_update_H.assert_called_once()

    def test_forward_with_boundary_recording(self, setup_simulation_state):
        """Test forward function with boundary recording enabled."""
        state = setup_simulation_state["state"]
        config = setup_simulation_state["config"]
        objects = setup_simulation_state["objects"]
        key = setup_simulation_state["key"]

        # Mock all the external functions
        with (
            patch("fdtdx.fdtd.forward.update_E") as mock_update_E,
            patch("fdtdx.fdtd.forward.update_H") as mock_update_H,
            patch("fdtdx.fdtd.forward.collect_interfaces") as mock_collect_interfaces,
            patch("fdtdx.fdtd.forward.jax.lax.stop_gradient") as mock_stop_gradient,
        ):
            mock_update_E.return_value = setup_simulation_state["arrays"]
            mock_update_H.return_value = setup_simulation_state["arrays"]
            mock_collect_interfaces.return_value = setup_simulation_state["arrays"]
            mock_stop_gradient.return_value = setup_simulation_state["arrays"]

            forward(state, config, objects, key, False, True, False)

            # Verify collect_interfaces was called
            mock_collect_interfaces.assert_called_once()
            mock_stop_gradient.assert_called_once()

            # Verify other functions were also called
            mock_update_E.assert_called_once()
            mock_update_H.assert_called_once()

    def test_forward_single_args_wrapper(self, setup_simulation_state):
        """Test the single arguments wrapper function."""
        arrays = setup_simulation_state["arrays"]
        config = setup_simulation_state["config"]
        objects = setup_simulation_state["objects"]
        key = setup_simulation_state["key"]

        # Mock the forward function to verify it's called
        with patch("fdtdx.fdtd.forward.forward") as mock_forward:
            mock_state = (jnp.array(1), arrays)
            mock_forward.return_value = mock_state

            result = forward_single_args_wrapper(
                time_step=jnp.array(0),
                E=arrays.E,
                H=arrays.H,
                psi_E=arrays.psi_E,
                psi_H=arrays.psi_H,
                alpha=arrays.alpha,
                kappa=arrays.kappa,
                sigma=arrays.sigma,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                detector_states=arrays.detector_states,
                recording_state=arrays.recording_state,
                config=config,
                objects=objects,
                key=key,
                record_detectors=False,
                record_boundaries=False,
                simulate_boundaries=False,
            )

            # Verify forward was called
            mock_forward.assert_called_once()

            # Check return values
            assert len(result) == 12
            assert result[0] == 1  # time_step
            assert jnp.array_equal(result[1], arrays.E)  # E field
            assert jnp.array_equal(result[2], arrays.H)  # H field

    def test_forward_with_boundary_simulation(self, setup_simulation_state):
        """Test forward function with boundary simulation enabled."""
        state = setup_simulation_state["state"]
        config = setup_simulation_state["config"]
        objects = setup_simulation_state["objects"]
        key = setup_simulation_state["key"]

        # Mock the update_E and update_H functions to verify they're called with simulate_boundaries=True
        with (
            patch("fdtdx.fdtd.forward.update_E") as mock_update_E,
            patch("fdtdx.fdtd.forward.update_H") as mock_update_H,
        ):
            mock_update_E.return_value = setup_simulation_state["arrays"]
            mock_update_H.return_value = setup_simulation_state["arrays"]

            forward(state, config, objects, key, False, False, True)

            # Verify update_E and update_H were called with simulate_boundaries=True
            mock_update_E.assert_called_once()
            mock_update_H.assert_called_once()

            # Check that simulate_boundaries=True was passed
            assert mock_update_E.call_args[1]["simulate_boundaries"]
            assert mock_update_H.call_args[1]["simulate_boundaries"]
