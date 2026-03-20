"""Tests for interfaces/time_filter.py - time step filtering utilities."""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.interfaces.state import RecordingState
from fdtdx.interfaces.time_filter import LinearReconstructEveryK, TimeStepFilter


class TestLinearReconstructEveryK:
    """Tests for LinearReconstructEveryK class."""

    def test_init_shapes_basic(self):
        """Test basic shape initialization."""
        filter_obj = LinearReconstructEveryK(k=5)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(10, 10), dtype=jnp.float32)}

        updated_filter, array_size, out_shapes, state_shapes = filter_obj.init_shapes(input_shapes, time_steps_max=100)

        assert array_size > 0
        assert array_size < 100  # Should be reduced
        assert "field" in out_shapes
        assert len(state_shapes) == 0  # LinearReconstructEveryK has no state

    def test_init_shapes_preserves_dtypes(self):
        """Test that init_shapes preserves dtypes."""
        filter_obj = LinearReconstructEveryK(k=10)
        input_shapes = {
            "E": jax.ShapeDtypeStruct(shape=(5, 5, 5), dtype=jnp.float32),
            "H": jax.ShapeDtypeStruct(shape=(5, 5, 5), dtype=jnp.bfloat16),
        }

        _, _, out_shapes, _ = filter_obj.init_shapes(input_shapes, time_steps_max=50)

        assert out_shapes["E"].dtype == jnp.float32
        assert out_shapes["H"].dtype == jnp.bfloat16

    def test_init_shapes_includes_final_step(self):
        """Test that final time step is always included."""
        filter_obj = LinearReconstructEveryK(k=7)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float32)}

        updated_filter, array_size, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=50)

        # The last time step (49) should be included
        assert 49 in updated_filter._save_time_steps.tolist()

    def test_time_to_array_index_saved_step(self):
        """Test time_to_array_index for saved steps."""
        filter_obj = LinearReconstructEveryK(k=10)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float32)}
        updated_filter, _, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=100)

        # Time step 0 should map to array index 0
        idx = updated_filter.time_to_array_index(0)
        assert idx == 0

        # Time step 10 should map to array index 1
        idx = updated_filter.time_to_array_index(10)
        assert idx == 1

    def test_time_to_array_index_unsaved_step(self):
        """Test time_to_array_index for unsaved steps returns -1."""
        filter_obj = LinearReconstructEveryK(k=10)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float32)}
        updated_filter, _, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=100)

        # Time step 5 is not saved (between 0 and 10)
        idx = updated_filter.time_to_array_index(5)
        assert idx == -1

    def test_compress_returns_values_unchanged(self):
        """Test that compress returns values unchanged."""
        filter_obj = LinearReconstructEveryK(k=10)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(5, 5), dtype=jnp.float32)}
        updated_filter, _, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=100)

        values = {"field": jnp.ones((5, 5))}
        state = RecordingState(data={}, state={})
        key = jax.random.PRNGKey(0)

        out_values, out_state = updated_filter.compress(values, state, jnp.array(0), key)

        assert jnp.allclose(out_values["field"], values["field"])

    def test_indices_to_decompress(self):
        """Test indices_to_decompress returns two indices."""
        filter_obj = LinearReconstructEveryK(k=10)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float32)}
        updated_filter, _, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=100)

        indices = updated_filter.indices_to_decompress(jnp.array(5))

        assert indices.shape == (2,)
        # Should return consecutive indices
        assert indices[1] - indices[0] == 1

    def test_decompress_at_saved_step(self):
        """Test decompress at a saved time step."""
        filter_obj = LinearReconstructEveryK(k=10)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float32)}
        updated_filter, _, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=100)

        # Create mock values
        values = [
            {"field": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])},
            {"field": jnp.array([2.0, 3.0, 4.0, 5.0, 6.0])},
        ]
        state = RecordingState(data={}, state={})
        arr_indices = jnp.array([0, 1])
        key = jax.random.PRNGKey(0)

        result = updated_filter.decompress(values, state, arr_indices, jnp.array(0), key)

        # At saved step, should return first value unchanged
        assert jnp.allclose(result["field"], values[0]["field"])

    def test_decompress_interpolates(self):
        """Test decompress performs linear interpolation."""
        filter_obj = LinearReconstructEveryK(k=10)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(1,), dtype=jnp.float32)}
        updated_filter, _, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=100)

        # Create values at steps 0 and 10
        values = [
            {"field": jnp.array([0.0])},  # at t=0
            {"field": jnp.array([10.0])},  # at t=10
        ]
        state = RecordingState(data={}, state={})
        arr_indices = jnp.array([0, 1])
        key = jax.random.PRNGKey(0)

        # Decompress at t=5 (midpoint)
        result = updated_filter.decompress(values, state, arr_indices, jnp.array(5), key)

        # Should be linearly interpolated to 5.0
        assert jnp.isclose(result["field"][0], 5.0)

    def test_start_recording_after(self):
        """Test start_recording_after parameter."""
        filter_obj = LinearReconstructEveryK(k=10, start_recording_after=20)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float32)}
        updated_filter, _, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=100)

        # First saved step should be 20
        assert updated_filter._save_time_steps[0] == 20

    def test_k_equals_1(self):
        """Test with k=1 (save every step)."""
        filter_obj = LinearReconstructEveryK(k=1)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float32)}
        updated_filter, array_size, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=50)

        # Should save all 50 steps
        assert array_size == 50

    def test_init_shapes_array_size_equals_saved_steps_count(self):
        """Array size returned by init_shapes equals the number of saved time steps."""
        filter_obj = LinearReconstructEveryK(k=7)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(3,), dtype=jnp.float32)}
        updated_filter, array_size, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=50)
        assert array_size == len(updated_filter._save_time_steps)

    def test_time_to_array_index_last_step(self):
        """Last time step is always saved and maps to a valid (non-negative) array index."""
        filter_obj = LinearReconstructEveryK(k=10)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(5,), dtype=jnp.float32)}
        updated_filter, _, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=100)

        # Step 99 (time_steps_max - 1) is always appended to the save list
        idx = updated_filter.time_to_array_index(99)
        assert idx >= 0

    def test_compress_state_unchanged(self):
        """compress returns the RecordingState object unchanged."""
        filter_obj = LinearReconstructEveryK(k=10)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.float32)}
        updated_filter, _, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=100)

        values = {"field": jnp.ones((4,))}
        state = RecordingState(data={}, state={})
        key = jax.random.PRNGKey(0)

        _, returned_state = updated_filter.compress(values, state, jnp.array(0), key)
        assert returned_state is state

    def test_init_shapes_state_shapes_empty(self):
        """LinearReconstructEveryK has no persistent state — state_shapes is empty."""
        filter_obj = LinearReconstructEveryK(k=5)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.float32)}
        _, _, _, state_shapes = filter_obj.init_shapes(input_shapes, time_steps_max=50)
        assert state_shapes == {}

    def test_time_to_array_index_multiple_saved_steps_all_valid(self):
        """All saved time steps return a valid non-negative array index."""
        filter_obj = LinearReconstructEveryK(k=10)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.float32)}
        updated_filter, _, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=50)

        for step in updated_filter._save_time_steps.tolist():
            idx = updated_filter.time_to_array_index(step)
            assert idx >= 0, f"Saved step {step} returned invalid index {idx}"

    def test_decompress_at_last_step(self):
        """Decompress at the last (always-saved) time step returns values[0]."""
        filter_obj = LinearReconstructEveryK(k=10)
        input_shapes = {"field": jax.ShapeDtypeStruct(shape=(3,), dtype=jnp.float32)}
        updated_filter, _, _, _ = filter_obj.init_shapes(input_shapes, time_steps_max=50)

        last_step = int(updated_filter._save_time_steps[-1])
        arr_idx = int(updated_filter.time_to_array_index(last_step))
        expected = jnp.array([7.0, 8.0, 9.0])
        values = [
            {"field": expected},
            {"field": jnp.zeros((3,))},  # second slot, not used for saved step
        ]
        state = RecordingState(data={}, state={})
        arr_indices = jnp.array([arr_idx, arr_idx + 1])
        key = jax.random.PRNGKey(0)

        result = updated_filter.decompress(values, state, arr_indices, jnp.array(last_step), key)
        assert jnp.allclose(result["field"], expected)


# ─── TestTimeStepFilterAbstract ───────────────────────────────────────────────


class TestTimeStepFilterAbstract:
    """TimeStepFilter is an ABC — subclasses must implement all abstract methods."""

    def test_cannot_instantiate_directly(self):
        """Direct instantiation of TimeStepFilter raises TypeError."""
        with pytest.raises(TypeError):
            TimeStepFilter()  # type: ignore

    def test_subclass_must_implement_init_shapes(self):
        """A subclass omitting init_shapes is not instantiable."""

        class Incomplete(TimeStepFilter):
            def time_to_array_index(self, time_idx):
                return time_idx

            def compress(self, values, state, time_idx, key):
                return values, state

            def indices_to_decompress(self, time_idx):
                return jnp.array([time_idx, time_idx + 1])

            def decompress(self, values, state, arr_indices, time_idx, key):
                return values[0]

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_must_implement_compress(self):
        """A subclass omitting compress is not instantiable."""

        class Incomplete(TimeStepFilter):
            def init_shapes(self, input_shape_dtypes, time_steps_max):
                return self, time_steps_max, input_shape_dtypes, {}

            def time_to_array_index(self, time_idx):
                return time_idx

            def indices_to_decompress(self, time_idx):
                return jnp.array([time_idx, time_idx + 1])

            def decompress(self, values, state, arr_indices, time_idx, key):
                return values[0]

        with pytest.raises(TypeError):
            Incomplete()
