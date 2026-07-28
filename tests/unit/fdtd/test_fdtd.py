"""Unit tests for fdtdx.fdtd.fdtd module."""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.fdtd.container import ArrayContainer, FieldState, ObjectContainer
from fdtdx.fdtd.fdtd import _reversible_slice_boundaries, checkpointed_fdtd, custom_fdtd_forward, reversible_fdtd
from fdtdx.fdtd.stop_conditions import TimeStepCondition
from fdtdx.interfaces.recorder import Recorder
from fdtdx.objects.object import SimulationObject


class DummySimulationObject(SimulationObject):
    """Minimal mock implementation of a SimulationObject."""

    def __init__(self, name="dummy"):
        self.name = name

    def update(self, *args, **kwargs):
        return args, kwargs


@pytest.fixture
def field_shape():
    return (3, 2, 2, 2)


@pytest.fixture
def dummy_arrays(field_shape):
    return ArrayContainer(
        fields=FieldState(
            E=jnp.zeros(field_shape),
            H=jnp.zeros(field_shape),
            psi_E={},
            psi_H={},
        ),
        inv_permittivities=jnp.ones(field_shape),
        inv_permeabilities=jnp.ones(field_shape),
        detector_states={},
        recording_state=None,
        electric_conductivity=None,
        magnetic_conductivity=None,
    )


@pytest.fixture
def dummy_objects():
    return ObjectContainer(object_list=[DummySimulationObject()], volume_idx=0)


@pytest.fixture
def empty_objects():
    return ObjectContainer(object_list=[], volume_idx=0)


@pytest.fixture
def config_few_steps():
    """Config with ~5 time steps for fast tests."""
    return SimulationConfig(
        time=400e-15,
        grid=UniformGrid(spacing=40e-6),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )


@pytest.fixture
def config_checkpointed():
    return SimulationConfig(
        time=400e-15,
        grid=UniformGrid(spacing=40e-6),
        backend="gpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=GradientConfig(
            method="checkpointed",
            num_checkpoints=10,
        ),
    )


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


class TestReversibleSliceBoundaries:
    """Tests for the slice-boundary helper of the sliced reversible backward pass."""

    @pytest.mark.parametrize(
        "time_steps_total,num_slices",
        [(10, 1), (10, 2), (10, 3), (10, 7), (10, 10), (37, 5), (420, 4), (5, 5), (1, 1)],
    )
    def test_boundaries_are_valid_partition(self, time_steps_total, num_slices):
        b = _reversible_slice_boundaries(time_steps_total, num_slices)
        # k + 1 boundaries, from 0 to T inclusive
        assert len(b) == num_slices + 1
        assert b[0] == 0
        assert b[-1] == time_steps_total
        # strictly increasing (distinct) and every slice has length >= 1
        assert b == sorted(b)
        assert len(set(b)) == len(b)
        assert all(b[i + 1] - b[i] >= 1 for i in range(num_slices))

    def test_single_slice_is_full_range(self):
        """num_slices == 1 (num_checkpoints_reversible == 0) yields no interior boundaries."""
        b = _reversible_slice_boundaries(123, 1)
        assert b == [0, 123]
        assert b[1:-1] == []  # no interior checkpoints

    def test_every_step_checkpointed(self):
        """num_slices == T yields a boundary at every integer time step."""
        b = _reversible_slice_boundaries(6, 6)
        assert b == [0, 1, 2, 3, 4, 5, 6]


def _reversible_grad_config(config, num_ckpt):
    """Attach a reversible GradientConfig (empty recorder) with the given interior-checkpoint count.

    Returns the updated config and the matching zero-initialized recording_state. The dummy scene
    has no PML, so the recorder needs no interface buffers (``input_shape_dtypes={}``).
    """
    recorder = Recorder(modules=[])
    recorder, recording_state = recorder.init_state(
        input_shape_dtypes={},
        max_time_steps=config.time_steps_total,
        backend="cpu",
    )
    config = config.aset(
        "gradient_config",
        GradientConfig(method="reversible", recorder=recorder, num_checkpoints_reversible=num_ckpt),
    )
    return config, recording_state


class TestSlicedReversibleFdtd:
    """Smoke tests exercising the segmented forward + checkpoint-reset backward under jax.grad."""

    def test_grad_runs_with_interior_checkpoints(self, dummy_arrays, dummy_objects, config_few_steps, key):
        config, recording_state = _reversible_grad_config(config_few_steps, num_ckpt=2)
        arrays = dummy_arrays.aset("recording_state", recording_state)

        def loss(inv_eps):
            a = arrays.aset("inv_permittivities", inv_eps)
            _, out = reversible_fdtd(a, dummy_objects, config, key, show_progress=False)
            return jnp.sum(out.fields.E**2) + jnp.sum(out.fields.H**2)

        val, grad = jax.value_and_grad(loss)(arrays.inv_permittivities)
        assert jnp.isfinite(val)
        assert jnp.all(jnp.isfinite(grad))
        assert grad.shape == arrays.inv_permittivities.shape

    def test_too_many_checkpoints_raises(self, dummy_arrays, dummy_objects, config_few_steps, key):
        # num_checkpoints_reversible == time_steps_total means num_slices == T + 1 > T.
        num_ckpt = config_few_steps.time_steps_total
        config, recording_state = _reversible_grad_config(config_few_steps, num_ckpt=num_ckpt)
        arrays = dummy_arrays.aset("recording_state", recording_state)
        with pytest.raises(Exception, match="num_checkpoints_reversible must be <="):
            reversible_fdtd(arrays, dummy_objects, config, key, show_progress=False)


class TestReversibleFdtd:
    def test_returns_simulation_state(self, dummy_arrays, dummy_objects, config_few_steps, key):
        t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, config_few_steps, key)
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_output_shapes_match_input(self, dummy_arrays, dummy_objects, config_few_steps, key, field_shape):
        _, arrs = reversible_fdtd(dummy_arrays, dummy_objects, config_few_steps, key)
        assert arrs.fields.E.shape == field_shape
        assert arrs.fields.H.shape == field_shape
        assert arrs.inv_permittivities.shape == field_shape
        assert arrs.inv_permeabilities.shape == field_shape

    def test_advances_time_steps(self, dummy_arrays, dummy_objects, config_few_steps, key):
        t, _ = reversible_fdtd(dummy_arrays, dummy_objects, config_few_steps, key)
        assert int(t) == config_few_steps.time_steps_total

    def test_zero_time_produces_no_evolution(self, dummy_arrays, dummy_objects, key):
        config = SimulationConfig(
            time=0.0,
            grid=UniformGrid(spacing=40e-6),
            backend="cpu",
            dtype=jnp.float32,
            courant_factor=0.99,
        )
        t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, config, key)
        assert int(t) == 0
        assert isinstance(arrs, ArrayContainer)

    def test_preserves_conductivity_fields(self, dummy_arrays, dummy_objects, config_few_steps, key):
        _, arrs = reversible_fdtd(dummy_arrays, dummy_objects, config_few_steps, key)
        assert arrs.electric_conductivity is None
        assert arrs.magnetic_conductivity is None

    def test_with_nonzero_initial_fields(self, dummy_objects, config_few_steps, key, field_shape):
        arrays = ArrayContainer(
            fields=FieldState(
                E=jnp.ones(field_shape) * 0.5,
                H=jnp.ones(field_shape) * 0.3,
                psi_E={},
                psi_H={},
            ),
            inv_permittivities=jnp.ones(field_shape),
            inv_permeabilities=jnp.ones(field_shape),
            detector_states={},
            recording_state=None,
            electric_conductivity=None,
            magnetic_conductivity=None,
        )
        t, arrs = reversible_fdtd(arrays, dummy_objects, config_few_steps, key)
        assert int(t) == config_few_steps.time_steps_total
        assert arrs.fields.E.shape == field_shape

    def test_empty_objects(self, dummy_arrays, empty_objects, config_few_steps, key):
        t, arrs = reversible_fdtd(dummy_arrays, empty_objects, config_few_steps, key)
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)


class TestCheckpointedFdtd:
    def test_returns_simulation_state(self, dummy_arrays, dummy_objects, config_checkpointed, key):
        t, arrs = checkpointed_fdtd(dummy_arrays, dummy_objects, config_checkpointed, key)
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_advances_time_steps(self, dummy_arrays, dummy_objects, config_checkpointed, key):
        t, _ = checkpointed_fdtd(dummy_arrays, dummy_objects, config_checkpointed, key)
        assert int(t) == config_checkpointed.time_steps_total

    def test_output_shapes_match_input(self, dummy_arrays, dummy_objects, config_checkpointed, key, field_shape):
        _, arrs = checkpointed_fdtd(dummy_arrays, dummy_objects, config_checkpointed, key)
        assert arrs.fields.E.shape == field_shape
        assert arrs.fields.H.shape == field_shape

    def test_empty_objects(self, dummy_arrays, empty_objects, config_checkpointed, key):
        t, arrs = checkpointed_fdtd(dummy_arrays, empty_objects, config_checkpointed, key)
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_with_custom_stopping_condition(self, dummy_arrays, dummy_objects, config_checkpointed, key):
        """Covers the stopping_condition.setup() path (line 369)."""
        condition = TimeStepCondition()
        t, arrs = checkpointed_fdtd(dummy_arrays, dummy_objects, config_checkpointed, key, stopping_condition=condition)
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)
        assert int(t) == config_checkpointed.time_steps_total

    def test_preserves_output_structure(self, dummy_arrays, dummy_objects, config_checkpointed, key):
        _, arrs = checkpointed_fdtd(dummy_arrays, dummy_objects, config_checkpointed, key)
        assert arrs.detector_states == {}
        assert arrs.recording_state is None
        assert arrs.electric_conductivity is None
        assert arrs.magnetic_conductivity is None


class TestCustomFdtdForward:
    def test_returns_simulation_state(self, dummy_arrays, dummy_objects, config_few_steps, key):
        t, arrs = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            config_few_steps,
            key,
            reset_container=True,
            record_detectors=False,
            start_time=0,
            end_time=1,
        )
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_advances_to_end_time(self, dummy_arrays, dummy_objects, config_few_steps, key):
        end = min(3, config_few_steps.time_steps_total)
        t, _ = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            config_few_steps,
            key,
            reset_container=True,
            record_detectors=False,
            start_time=0,
            end_time=end,
        )
        assert int(t) == end

    def test_same_start_end_no_evolution(self, dummy_arrays, dummy_objects, config_few_steps, key):
        t, arrs = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            config_few_steps,
            key,
            reset_container=False,
            record_detectors=False,
            start_time=5,
            end_time=5,
        )
        assert int(t) == 5
        assert isinstance(arrs, ArrayContainer)

    def test_output_shapes_match_input(self, dummy_arrays, dummy_objects, config_few_steps, key, field_shape):
        _, arrs = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            config_few_steps,
            key,
            reset_container=True,
            record_detectors=False,
            start_time=0,
            end_time=1,
        )
        assert arrs.fields.E.shape == field_shape
        assert arrs.fields.H.shape == field_shape

    def test_reset_container_false(self, dummy_arrays, dummy_objects, config_few_steps, key):
        t, arrs = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            config_few_steps,
            key,
            reset_container=False,
            record_detectors=False,
            start_time=0,
            end_time=1,
        )
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_with_record_detectors(self, dummy_arrays, dummy_objects, config_few_steps, key):
        t, arrs = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            config_few_steps,
            key,
            reset_container=True,
            record_detectors=True,
            start_time=0,
            end_time=1,
        )
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_partial_time_range(self, dummy_arrays, dummy_objects, config_few_steps, key):
        t, _ = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            config_few_steps,
            key,
            reset_container=True,
            record_detectors=False,
            start_time=2,
            end_time=4,
        )
        assert int(t) == 4

    def test_empty_objects(self, dummy_arrays, empty_objects, config_few_steps, key):
        t, arrs = custom_fdtd_forward(
            dummy_arrays,
            empty_objects,
            config_few_steps,
            key,
            reset_container=True,
            record_detectors=False,
            start_time=0,
            end_time=1,
        )
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)


class TestShowProgressFlag:
    """Verify that show_progress=True/False does not alter simulation outputs."""

    def test_reversible_fdtd_with_progress(self, dummy_arrays, dummy_objects, config_few_steps):
        key = jax.random.PRNGKey(0)
        t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, config_few_steps, key, show_progress=True)
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)
        assert arrs.fields.E.shape == dummy_arrays.fields.E.shape

    def test_checkpointed_fdtd_with_progress(self, dummy_arrays, dummy_objects, config_checkpointed):
        key = jax.random.PRNGKey(0)
        t, arrs = checkpointed_fdtd(dummy_arrays, dummy_objects, config_checkpointed, key, show_progress=True)
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_custom_fdtd_forward_with_progress(self, dummy_arrays, dummy_objects, config_few_steps):
        key = jax.random.PRNGKey(0)
        t, arrs = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            config_few_steps,
            key,
            reset_container=True,
            record_detectors=False,
            start_time=0,
            end_time=5,
            show_progress=True,
        )
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_reversible_fdtd_without_progress(self, dummy_arrays, dummy_objects, config_few_steps):
        key = jax.random.PRNGKey(0)
        t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, config_few_steps, key, show_progress=False)
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_checkpointed_fdtd_without_progress(self, dummy_arrays, dummy_objects, config_checkpointed):
        key = jax.random.PRNGKey(0)
        t, arrs = checkpointed_fdtd(
            dummy_arrays,
            dummy_objects,
            config_checkpointed,
            key,
            show_progress=False,
        )
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_custom_fdtd_no_steps_no_bar(self, dummy_arrays, dummy_objects, config_few_steps):
        """When start_time == end_time no loop iterations run and no bar is created."""
        key = jax.random.PRNGKey(0)
        t, arrs = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            config_few_steps,
            key,
            reset_container=False,
            record_detectors=False,
            start_time=3,
            end_time=3,
            show_progress=True,
        )
        assert int(t) == 3
        assert isinstance(arrs, ArrayContainer)

    def test_custom_fdtd_jax_array_times_disables_progress(self, dummy_arrays, dummy_objects, config_few_steps):
        """Passing jax.Array start_time/end_time must not cause concretization errors."""
        key = jax.random.PRNGKey(0)
        start = jnp.asarray(0, dtype=jnp.int32)
        end = jnp.asarray(2, dtype=jnp.int32)
        t, arrs = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            config_few_steps,
            key,
            reset_container=True,
            record_detectors=False,
            start_time=start,
            end_time=end,
            show_progress=True,
        )
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)
