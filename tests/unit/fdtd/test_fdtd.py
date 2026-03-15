"""Unit tests for fdtdx.fdtd.fdtd module."""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.fdtd import checkpointed_fdtd, custom_fdtd_forward, reversible_fdtd
from fdtdx.fdtd.stop_conditions import TimeStepCondition
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
    auxiliary_field_shape = (6, 2, 2, 2)
    return ArrayContainer(
        E=jnp.zeros(field_shape),
        H=jnp.zeros(field_shape),
        psi_E=jnp.zeros(auxiliary_field_shape),
        psi_H=jnp.zeros(auxiliary_field_shape),
        alpha=jnp.zeros(field_shape),
        kappa=jnp.ones(field_shape),
        sigma=jnp.zeros(field_shape),
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
        resolution=40e-6,
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )


@pytest.fixture
def config_checkpointed():
    return SimulationConfig(
        time=400e-15,
        resolution=40e-6,
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


class TestReversibleFdtd:
    def test_returns_simulation_state(self, dummy_arrays, dummy_objects, config_few_steps, key):
        t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, config_few_steps, key)
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_output_shapes_match_input(self, dummy_arrays, dummy_objects, config_few_steps, key, field_shape):
        _, arrs = reversible_fdtd(dummy_arrays, dummy_objects, config_few_steps, key)
        assert arrs.E.shape == field_shape
        assert arrs.H.shape == field_shape
        assert arrs.inv_permittivities.shape == field_shape
        assert arrs.inv_permeabilities.shape == field_shape

    def test_advances_time_steps(self, dummy_arrays, dummy_objects, config_few_steps, key):
        t, _ = reversible_fdtd(dummy_arrays, dummy_objects, config_few_steps, key)
        assert int(t) == config_few_steps.time_steps_total

    def test_zero_time_produces_no_evolution(self, dummy_arrays, dummy_objects, key):
        config = SimulationConfig(
            time=0.0,
            resolution=40e-6,
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
        auxiliary_field_shape = (6, 2, 2, 2)
        arrays = ArrayContainer(
            E=jnp.ones(field_shape) * 0.5,
            H=jnp.ones(field_shape) * 0.3,
            psi_E=jnp.zeros(auxiliary_field_shape),
            psi_H=jnp.zeros(auxiliary_field_shape),
            alpha=jnp.zeros(field_shape),
            kappa=jnp.ones(field_shape),
            sigma=jnp.zeros(field_shape),
            inv_permittivities=jnp.ones(field_shape),
            inv_permeabilities=jnp.ones(field_shape),
            detector_states={},
            recording_state=None,
            electric_conductivity=None,
            magnetic_conductivity=None,
        )
        t, arrs = reversible_fdtd(arrays, dummy_objects, config_few_steps, key)
        assert int(t) == config_few_steps.time_steps_total
        assert arrs.E.shape == field_shape

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
        assert arrs.E.shape == field_shape
        assert arrs.H.shape == field_shape

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
        assert arrs.E.shape == field_shape
        assert arrs.H.shape == field_shape

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
        assert arrs.E.shape == dummy_arrays.E.shape

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
