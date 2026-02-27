from typing import Tuple, cast
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.backward import backward, cond_fn, full_backward
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer


class TestBackwardPropagation:
    @pytest.fixture
    def mock_config(self) -> SimulationConfig:
        """Create a mock simulation configuration."""
        config = Mock(spec=SimulationConfig)
        config.grid_shape = (10, 10, 10)
        config.dt = 0.1
        config.dx = 1.0
        return cast(SimulationConfig, config)

    @pytest.fixture
    def mock_arrays(self) -> ArrayContainer:
        """Create mock field arrays."""
        # Create a mock arrays object with the expected fields
        arrays = Mock(spec=ArrayContainer)
        arrays.E = jnp.ones((3, 10, 10, 10))
        arrays.H = jnp.zeros((3, 10, 10, 10))
        arrays.J = jnp.zeros((3, 10, 10, 10))
        arrays.M = jnp.zeros((3, 10, 10, 10))

        # Mock the aset method
        def aset(field_name, value):
            new_arrays = Mock(spec=ArrayContainer)
            new_arrays.E = value if field_name == "E" else arrays.E
            new_arrays.H = value if field_name == "H" else arrays.H
            new_arrays.J = value if field_name == "J" else arrays.J
            new_arrays.M = value if field_name == "M" else arrays.M
            return cast(ArrayContainer, new_arrays)

        arrays.aset = aset
        return cast(ArrayContainer, arrays)

    @pytest.fixture
    def mock_objects(self) -> ObjectContainer:
        """Create mock object container."""
        objects = Mock(spec=ObjectContainer)
        objects.pml_objects = []
        objects.boundary_objects = []
        objects.source_objects = []
        objects.detector_objects = []
        return cast(ObjectContainer, objects)

    @pytest.fixture
    def mock_state(self, mock_arrays: ArrayContainer) -> Tuple[jax.Array, ArrayContainer]:
        """Create a mock simulation state."""
        return (jnp.array(5), mock_arrays)

    @pytest.fixture
    def key(self) -> jax.Array:
        """Create a JAX PRNG key."""
        return jax.random.PRNGKey(42)

    def test_cond_fn_true(self):
        """Test condition function returns True when time step > start."""
        state = cast(Tuple[jax.Array, ArrayContainer], (jnp.array(10), Mock()))
        assert cond_fn(state, start_time_step=5) is True

    def test_cond_fn_false(self):
        """Test condition function returns False when time step <= start."""
        state = cast(Tuple[jax.Array, ArrayContainer], (jnp.array(5), Mock()))
        assert cond_fn(state, start_time_step=5) is False

        state = cast(Tuple[jax.Array, ArrayContainer], (jnp.array(3), Mock()))
        assert cond_fn(state, start_time_step=5) is False

    @patch("fdtdx.fdtd.backward.eqxi.while_loop")
    def test_full_backward_calls_while_loop(self, mock_while_loop, mock_state, mock_objects, mock_config, key):
        """Test full_backward calls while_loop with correct arguments."""
        # Mock the while_loop to return the final state
        mock_while_loop.return_value = (jnp.array(0), mock_state[1])

        full_backward(
            state=mock_state,
            objects=mock_objects,
            config=mock_config,
            key=key,
            record_detectors=False,
            reset_fields=False,
            start_time_step=0,
        )

        # Verify while_loop was called with correct arguments
        mock_while_loop.assert_called_once()
        call_kwargs = mock_while_loop.call_args.kwargs
        assert call_kwargs["kind"] == "lax"
        assert call_kwargs["cond_fun"] is not None
        assert call_kwargs["body_fun"] is not None
        assert call_kwargs["init_val"] == mock_state

    @patch("fdtdx.fdtd.backward.add_interfaces")
    @patch("fdtdx.fdtd.backward.update_H_reverse")
    @patch("fdtdx.fdtd.backward.update_E_reverse")
    @patch("fdtdx.fdtd.backward.update_detector_states")
    def test_backward_step(
        self,
        mock_update_detectors,
        mock_update_E,
        mock_update_H,
        mock_add_interfaces,
        mock_state,
        mock_objects,
        mock_config,
        key,
        mock_arrays,
    ):
        """Test a single backward step execution."""
        # Mock the function calls
        mock_add_interfaces.return_value = mock_arrays
        mock_update_H.return_value = mock_arrays
        mock_update_E.return_value = mock_arrays
        mock_update_detectors.return_value = mock_arrays

        result = backward(
            state=mock_state,
            config=mock_config,
            objects=mock_objects,
            key=key,
            record_detectors=True,
            reset_fields=False,
        )

        # Verify function calls
        mock_add_interfaces.assert_called_once()
        mock_update_H.assert_called_once()
        mock_update_E.assert_called_once()
        mock_update_detectors.assert_called_once()

        # Verify time step decremented
        assert result[0] == 4  # Original was 5

    @patch("fdtdx.fdtd.backward.add_interfaces")
    @patch("fdtdx.fdtd.backward.update_H_reverse")
    @patch("fdtdx.fdtd.backward.update_E_reverse")
    def test_backward_with_reset_fields(
        self, mock_update_E, mock_update_H, mock_add_interfaces, mock_state, mock_objects, mock_config, key, mock_arrays
    ):
        """Test backward step with field resetting."""
        mock_add_interfaces.return_value = mock_arrays
        mock_update_H.return_value = mock_arrays
        mock_update_E.return_value = mock_arrays

        # Test with reset_fields=True
        result = backward(
            state=mock_state,
            config=mock_config,
            objects=mock_objects,
            key=key,
            record_detectors=False,
            reset_fields=True,
        )

        # Should still call all update functions
        mock_add_interfaces.assert_called_once()
        mock_update_H.assert_called_once()
        mock_update_E.assert_called_once()
        assert result[0] == 4

    @patch("fdtdx.fdtd.backward.add_interfaces")
    @patch("fdtdx.fdtd.backward.update_H_reverse")
    @patch("fdtdx.fdtd.backward.update_E_reverse")
    def test_backward_with_pml_reset(
        self, mock_update_E, mock_update_H, mock_add_interfaces, mock_state, mock_objects, mock_config, key, mock_arrays
    ):
        """Test field resetting with PML objects."""
        # Create a mock PML object
        pml = Mock()
        pml.grid_slice = (slice(0, 2), slice(0, 2), slice(0, 2))

        mock_objects.pml_objects = [pml]
        mock_add_interfaces.return_value = mock_arrays
        mock_update_H.return_value = mock_arrays
        mock_update_E.return_value = mock_arrays

        result = backward(
            state=mock_state,
            config=mock_config,
            objects=mock_objects,
            key=key,
            record_detectors=False,
            reset_fields=True,
        )

        # Should process PML objects during reset
        assert result[0] == 4

    @patch("fdtdx.fdtd.backward.add_interfaces")
    @patch("fdtdx.fdtd.backward.update_H_reverse")
    @patch("fdtdx.fdtd.backward.update_E_reverse")
    def test_backward_with_periodic_boundary(
        self, mock_update_E, mock_update_H, mock_add_interfaces, mock_state, mock_objects, mock_config, key, mock_arrays
    ):
        """Test field resetting with periodic boundaries."""
        # Create a mock periodic boundary
        boundary = Mock()
        boundary.grid_slice = (slice(0, 1), slice(0, 10), slice(0, 10))
        boundary.direction = "+"
        boundary.axis = 0
        boundary._grid_slice_tuple = [(0, 1), (0, 10), (0, 10)]

        mock_objects.boundary_objects = [boundary]
        mock_add_interfaces.return_value = mock_arrays
        mock_update_H.return_value = mock_arrays
        mock_update_E.return_value = mock_arrays

        result = backward(
            state=mock_state,
            config=mock_config,
            objects=mock_objects,
            key=key,
            record_detectors=False,
            reset_fields=True,
        )

        # Should process periodic boundaries during reset
        assert result[0] == 4

    @patch("fdtdx.fdtd.backward.add_interfaces")
    @patch("fdtdx.fdtd.backward.update_H_reverse")
    @patch("fdtdx.fdtd.backward.update_E_reverse")
    def test_backward_custom_fields_to_reset(
        self, mock_update_E, mock_update_H, mock_add_interfaces, mock_state, mock_objects, mock_config, key, mock_arrays
    ):
        """Test backward step with custom fields to reset."""
        mock_add_interfaces.return_value = mock_arrays
        mock_update_H.return_value = mock_arrays
        mock_update_E.return_value = mock_arrays

        result = backward(
            state=mock_state,
            config=mock_config,
            objects=mock_objects,
            key=key,
            record_detectors=False,
            reset_fields=True,
            fields_to_reset=("E",),  # Only reset E field
        )

        # Should work with custom field selection
        assert result[0] == 4

    @patch("fdtdx.fdtd.backward.add_interfaces")
    @patch("fdtdx.fdtd.backward.update_H_reverse")
    @patch("fdtdx.fdtd.backward.update_E_reverse")
    def test_backward_without_detector_recording(
        self, mock_update_E, mock_update_H, mock_add_interfaces, mock_state, mock_objects, mock_config, key, mock_arrays
    ):
        """Test backward step without detector recording."""
        mock_add_interfaces.return_value = mock_arrays
        mock_update_H.return_value = mock_arrays
        mock_update_E.return_value = mock_arrays

        result = backward(
            state=mock_state,
            config=mock_config,
            objects=mock_objects,
            key=key,
            record_detectors=False,  # No detector recording
            reset_fields=False,
        )

        # Should not call update_detector_states
        mock_add_interfaces.assert_called_once()
        mock_update_H.assert_called_once()
        mock_update_E.assert_called_once()
        assert result[0] == 4

    def test_backward_time_step_decrement(self):
        """Test that backward step properly decrements time step."""
        # Create a simple test with minimal mocks
        mock_arrays = Mock(spec=ArrayContainer)
        mock_arrays.E = jnp.ones((3, 5, 5, 5))
        mock_arrays.H = jnp.zeros((3, 5, 5, 5))
        mock_arrays.aset = lambda field, value: mock_arrays

        mock_objects = Mock(spec=ObjectContainer)
        mock_objects_casted = cast(ObjectContainer, mock_objects)
        mock_objects.pml_objects = []
        mock_objects.boundary_objects = []

        mock_config = cast(SimulationConfig, Mock(spec=SimulationConfig))

        state = (jnp.array(10), mock_arrays)

        # Mock the internal functions to just return the arrays
        with (
            patch("fdtdx.fdtd.backward.add_interfaces", return_value=mock_arrays),
            patch("fdtdx.fdtd.backward.update_H_reverse", return_value=mock_arrays),
            patch("fdtdx.fdtd.backward.update_E_reverse", return_value=mock_arrays),
        ):
            result = backward(
                state=state,
                config=mock_config,
                objects=mock_objects_casted,
                key=jax.random.PRNGKey(42),
                record_detectors=False,
                reset_fields=False,
            )

            # Verify time step was decremented
            assert result[0] == 9
