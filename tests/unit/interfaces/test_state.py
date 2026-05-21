"""Tests for interfaces/state.py - recording state utilities."""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp

from fdtdx.interfaces.state import (
    RecordingState,
    init_recording_state,
    init_sharded_dict,
)


class TestRecordingState:
    """Tests for RecordingState class."""

    def test_creation_with_data(self):
        """Test creating RecordingState with data."""
        data = {"field": jnp.zeros((10, 5, 5))}
        state_dict = {"counter": jnp.array(0)}

        recording_state = RecordingState(data=data, state=state_dict)

        assert "field" in recording_state.data
        assert "counter" in recording_state.state

    def test_data_access(self):
        """Test accessing data in RecordingState."""
        data = {"E": jnp.ones((5, 5, 5)), "H": jnp.zeros((5, 5, 5))}
        state_dict = {}

        recording_state = RecordingState(data=data, state=state_dict)

        assert jnp.allclose(recording_state.data["E"], 1.0)
        assert jnp.allclose(recording_state.data["H"], 0.0)

    def test_empty_state(self):
        """Test with empty state dict."""
        data = {"field": jnp.zeros((3, 3, 3))}
        state_dict = {}

        recording_state = RecordingState(data=data, state=state_dict)

        assert len(recording_state.state) == 0


class TestInitShardedDict:
    """Tests for init_sharded_dict function.

    Note: These tests use the default backend to avoid CPU/GPU sharding conflicts.
    """

    def test_empty_dict(self):
        """Test with empty shape_dtypes."""
        result = init_sharded_dict({}, backend=None)
        assert len(result) == 0

    def test_single_field_zero_initialized(self):
        """Arrays returned by init_sharded_dict are zero-initialized."""
        result = init_sharded_dict(
            {"field": jax.ShapeDtypeStruct(shape=(4, 4), dtype=jnp.float32)},
            backend=None,
        )
        assert jnp.allclose(result["field"], 0.0)

    def test_dtype_preserved(self):
        """The dtype of the created array matches the spec dtype."""
        result = init_sharded_dict(
            {"field": jax.ShapeDtypeStruct(shape=(4, 4), dtype=jnp.float16)},
            backend=None,
        )
        assert result["field"].dtype == jnp.float16

    def test_shape_at_least_as_large_as_spec(self):
        """Created arrays have at least the requested shape (first dim may be padded)."""
        spec = jax.ShapeDtypeStruct(shape=(4, 5, 6), dtype=jnp.float32)
        result = init_sharded_dict({"field": spec}, backend=None)
        assert result["field"].shape[0] >= spec.shape[0]
        assert result["field"].shape[1:] == spec.shape[1:]

    def test_multiple_fields_all_created(self):
        """All keys from shape_dtypes appear in the result with correct dtypes."""
        specs = {
            "E": jax.ShapeDtypeStruct(shape=(4, 4, 4), dtype=jnp.float32),
            "H": jax.ShapeDtypeStruct(shape=(4, 4, 4), dtype=jnp.bfloat16),
        }
        result = init_sharded_dict(specs, backend=None)
        assert set(result.keys()) == {"E", "H"}
        assert result["E"].dtype == jnp.float32
        assert result["H"].dtype == jnp.bfloat16

    def test_shape_padded_when_not_divisible_by_num_devices(self):
        """First dim is padded to the next multiple of num_devices when not already divisible."""
        fake_devices = [MagicMock(), MagicMock()]  # simulate 2-device environment
        spec = jax.ShapeDtypeStruct(shape=(3, 4), dtype=jnp.float32)

        with patch.object(jax, "devices", return_value=fake_devices):
            with patch("fdtdx.interfaces.state.create_named_sharded_matrix") as mock_mat:
                mock_mat.return_value = jnp.zeros((4, 4), dtype=jnp.float32)
                init_sharded_dict({"field": spec}, backend=None)

        # With 2 devices, shape (3, 4) → first dim padded to 4 (next multiple of 2)
        called_shape = mock_mat.call_args.kwargs["shape"]
        assert called_shape[0] == 4
        assert called_shape[1:] == (4,)


class TestInitRecordingState:
    """Tests for init_recording_state function.

    Note: Tests use backend=None to let JAX choose the default backend,
    avoiding CPU/GPU sharding conflicts.
    """

    def test_basic_initialization(self):
        """Test basic RecordingState initialization."""
        data_shapes = {"field": jax.ShapeDtypeStruct(shape=(10, 5, 5), dtype=jnp.float32)}
        # State shapes need at least one dimension > 1 for sharding
        state_shapes = {"counter": jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.int32)}

        result = init_recording_state(data_shapes, state_shapes, backend=None)

        assert isinstance(result, RecordingState)
        assert "field" in result.data
        assert "counter" in result.state

    def test_empty_state_shapes(self):
        """Test with empty state shapes."""
        data_shapes = {"field": jax.ShapeDtypeStruct(shape=(8, 4, 4), dtype=jnp.float32)}

        result = init_recording_state(data_shapes, {}, backend=None)

        assert "field" in result.data
        assert len(result.state) == 0

    def test_multiple_data_fields(self):
        """Test with multiple data fields."""
        data_shapes = {
            "E_x": jax.ShapeDtypeStruct(shape=(10, 5, 5), dtype=jnp.float32),
            "E_y": jax.ShapeDtypeStruct(shape=(10, 5, 5), dtype=jnp.float32),
            "E_z": jax.ShapeDtypeStruct(shape=(10, 5, 5), dtype=jnp.float32),
        }

        result = init_recording_state(data_shapes, {}, backend=None)

        assert len(result.data) == 3
        assert all(k in result.data for k in ["E_x", "E_y", "E_z"])

    def test_data_zero_initialized(self):
        """Data arrays in RecordingState are initialized to zero."""
        data_shapes = {"field": jax.ShapeDtypeStruct(shape=(4, 5), dtype=jnp.float32)}
        result = init_recording_state(data_shapes, {}, backend=None)
        assert jnp.allclose(result.data["field"], 0.0)

    def test_data_dtype_preserved(self):
        """Dtype of data arrays matches the requested dtype spec."""
        data_shapes = {"field": jax.ShapeDtypeStruct(shape=(4, 4), dtype=jnp.float16)}
        result = init_recording_state(data_shapes, {}, backend=None)
        assert result.data["field"].dtype == jnp.float16

    def test_state_shapes_populated(self):
        """State arrays are created when state_shapes is non-empty."""
        data_shapes = {"field": jax.ShapeDtypeStruct(shape=(4, 4), dtype=jnp.float32)}
        state_shapes = {"counter": jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.int32)}
        result = init_recording_state(data_shapes, state_shapes, backend=None)
        assert "counter" in result.state
        assert jnp.allclose(result.state["counter"], 0)
