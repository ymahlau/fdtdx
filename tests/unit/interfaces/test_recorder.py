"""Tests for interfaces/recorder.py - data recording utilities."""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.interfaces.modules import DtypeConversion
from fdtdx.interfaces.recorder import Recorder
from fdtdx.interfaces.state import RecordingState
from fdtdx.interfaces.time_filter import LinearReconstructEveryK


class TestRecorder:
    """Tests for Recorder class.

    Note: Tests use backend=None to let JAX choose the default backend,
    avoiding CPU/GPU sharding conflicts.
    """

    def test_init_state_basic(self):
        """Test basic recorder initialization."""
        recorder = Recorder(modules=[LinearReconstructEveryK(k=5)])
        input_shapes = {"E": jax.ShapeDtypeStruct(shape=(10, 10, 10), dtype=jnp.float32)}

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=100,
            backend=None,
        )

        assert updated_recorder is not None
        assert state is not None
        assert "E" in state.data

    def test_init_state_multiple_fields(self):
        """Test initialization with multiple fields."""
        recorder = Recorder(modules=[LinearReconstructEveryK(k=10)])
        input_shapes = {
            "E_x": jax.ShapeDtypeStruct(shape=(5, 5, 5), dtype=jnp.float32),
            "E_y": jax.ShapeDtypeStruct(shape=(5, 5, 5), dtype=jnp.float32),
            "E_z": jax.ShapeDtypeStruct(shape=(5, 5, 5), dtype=jnp.float32),
        }

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=50,
            backend=None,
        )

        assert "E_x" in state.data
        assert "E_y" in state.data
        assert "E_z" in state.data

    def test_init_state_preserves_float32_dtype(self):
        """Test that float32 dtype is preserved during initialization."""
        recorder = Recorder(modules=[LinearReconstructEveryK(k=5)])
        input_shapes = {
            "float32_field": jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32),
        }

        _, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=50,
            backend=None,
        )

        assert state.data["float32_field"].dtype == jnp.float32

    def test_compress_at_saved_step(self):
        """Test compression at a saved time step."""
        recorder = Recorder(modules=[LinearReconstructEveryK(k=10)])
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32)}

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=100,
            backend=None,
        )

        values = {"field": jnp.ones((5, 5)) * 42.0}
        key = jax.random.PRNGKey(0)

        # Compress at time step 0 (should be saved)
        new_state = updated_recorder.compress(values, state, jnp.array(0), key)

        # Data should be updated at index 0
        assert jnp.allclose(new_state.data["field"][0], 42.0)

    def test_compress_at_unsaved_step(self):
        """Test compression at unsaved time step."""
        recorder = Recorder(modules=[LinearReconstructEveryK(k=10)])
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32)}

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=100,
            backend=None,
        )

        values = {"field": jnp.ones((5, 5)) * 99.0}
        key = jax.random.PRNGKey(0)

        # Compress at time step 5 (not saved for k=10)
        new_state = updated_recorder.compress(values, state, jnp.array(5), key)

        # State should not change (time step not saved)
        # All values should still be 0
        assert not jnp.any(new_state.data["field"] == 99.0)

    def test_compress_updates_state_progressively(self):
        """Test that compress updates state at multiple time steps."""
        recorder = Recorder(modules=[LinearReconstructEveryK(k=10)])
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(3, 3), dtype=jnp.float32)}

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=100,
            backend=None,
        )

        key = jax.random.PRNGKey(0)

        # Compress at step 0
        values_0 = {"field": jnp.ones((3, 3)) * 1.0}
        state = updated_recorder.compress(values_0, state, jnp.array(0), key)

        # Compress at step 10
        values_10 = {"field": jnp.ones((3, 3)) * 2.0}
        key, subkey = jax.random.split(key)
        state = updated_recorder.compress(values_10, state, jnp.array(10), subkey)

        # Check both values are stored
        assert jnp.allclose(state.data["field"][0], 1.0)
        assert jnp.allclose(state.data["field"][1], 2.0)

    def test_decompress_at_saved_step(self):
        """Test decompression at a saved time step."""
        recorder = Recorder(modules=[LinearReconstructEveryK(k=10)])
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(3, 3), dtype=jnp.float32)}

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=100,
            backend=None,
        )

        # Store a value at step 0
        key = jax.random.PRNGKey(0)
        values = {"field": jnp.ones((3, 3)) * 42.0}
        state = updated_recorder.compress(values, state, jnp.array(0), key)

        # Store a value at step 10
        values_10 = {"field": jnp.ones((3, 3)) * 84.0}
        key, subkey = jax.random.split(key)
        state = updated_recorder.compress(values_10, state, jnp.array(10), subkey)

        # Decompress at step 0
        key, subkey = jax.random.split(key)
        result, _ = updated_recorder.decompress(state, jnp.array(0), subkey)

        assert jnp.allclose(result["field"], 42.0)

    def test_decompress_with_interpolation(self):
        """Test decompression with linear interpolation."""
        recorder = Recorder(modules=[LinearReconstructEveryK(k=10)])
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(1,), dtype=jnp.float32)}

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=100,
            backend=None,
        )

        key = jax.random.PRNGKey(0)

        # Store value 0 at step 0
        state = updated_recorder.compress({"field": jnp.array([0.0])}, state, jnp.array(0), key)

        # Store value 10 at step 10
        key, subkey = jax.random.split(key)
        state = updated_recorder.compress({"field": jnp.array([10.0])}, state, jnp.array(10), subkey)

        # Decompress at step 5 (midpoint)
        key, subkey = jax.random.split(key)
        result, _ = updated_recorder.decompress(state, jnp.array(5), subkey)

        # Should be interpolated to 5.0
        assert jnp.isclose(result["field"][0], 5.0, atol=0.1)

    def test_empty_modules_list(self):
        """Test recorder with empty modules list."""
        recorder = Recorder(modules=[])
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32)}

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=10,
            backend=None,
        )

        assert "field" in state.data

    def test_decompress_returns_recording_state(self):
        """decompress returns a (values, RecordingState) tuple."""
        recorder = Recorder(modules=[LinearReconstructEveryK(k=10)])
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(3, 3), dtype=jnp.float32)}

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=100,
            backend=None,
        )

        key = jax.random.PRNGKey(0)
        state = updated_recorder.compress({"field": jnp.ones((3, 3))}, state, jnp.array(0), key)
        key, subkey = jax.random.split(key)
        state = updated_recorder.compress({"field": jnp.ones((3, 3)) * 2.0}, state, jnp.array(10), subkey)

        key, subkey = jax.random.split(key)
        result, returned_state = updated_recorder.decompress(state, jnp.array(0), subkey)

        assert isinstance(returned_state, RecordingState)

    def test_compress_wrong_shape_raises(self):
        """compress raises when input values have wrong shape."""
        recorder = Recorder(modules=[LinearReconstructEveryK(k=10)])
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32)}

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=100,
            backend=None,
        )

        wrong_values = {"field": jnp.ones((3, 3), dtype=jnp.float32)}
        key = jax.random.PRNGKey(0)

        with pytest.raises(Exception):
            updated_recorder.compress(wrong_values, state, jnp.array(0), key)

    def test_recorder_with_dtype_conversion(self):
        """Recorder pipeline with DtypeConversion stores float16 and restores float32."""
        recorder = Recorder(modules=[DtypeConversion(dtype=jnp.float16), LinearReconstructEveryK(k=10)])
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(4, 4), dtype=jnp.float32)}

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=50,
            backend=None,
        )

        # Latent array should be stored as float16
        assert state.data["field"].dtype == jnp.float16

        key = jax.random.PRNGKey(0)
        original = jnp.ones((4, 4)) * 3.0
        state = updated_recorder.compress({"field": original}, state, jnp.array(0), key)
        key, subkey = jax.random.split(key)
        state = updated_recorder.compress({"field": jnp.ones((4, 4)) * 6.0}, state, jnp.array(10), subkey)

        key, subkey = jax.random.split(key)
        result, _ = updated_recorder.decompress(state, jnp.array(0), subkey)

        # Decompressed values should be back to float32 and match original within float16 precision
        assert result["field"].dtype == jnp.float32
        assert jnp.allclose(result["field"], original, atol=1e-2)

    def test_empty_modules_compress_and_decompress(self):
        """Recorder with no modules stores and retrieves values at any time index."""
        recorder = Recorder(modules=[])
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(3,), dtype=jnp.float32)}

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=10,
            backend=None,
        )

        key = jax.random.PRNGKey(0)
        original = jnp.array([1.0, 2.0, 3.0])
        state = updated_recorder.compress({"field": original}, state, jnp.array(3), key)

        key, subkey = jax.random.split(key)
        result, _ = updated_recorder.decompress(state, jnp.array(3), subkey)

        assert jnp.allclose(result["field"], original)


class TestRecorderEdgeCases:
    """Edge case tests for Recorder."""

    def test_single_time_step(self):
        """Test with single time step."""
        recorder = Recorder(modules=[LinearReconstructEveryK(k=1)])
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(3,), dtype=jnp.float32)}

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=1,
            backend=None,
        )

        assert state is not None

    def test_k_larger_than_time_steps(self):
        """Test with k larger than max time steps."""
        recorder = Recorder(modules=[LinearReconstructEveryK(k=100)])
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(3,), dtype=jnp.float32)}

        updated_recorder, state = recorder.init_state(
            input_shape_dtypes=input_shapes,
            max_time_steps=50,
            backend=None,
        )

        # Should still work, just save first and last
        assert state is not None
