import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.fdtd import checkpointed_fdtd, custom_fdtd_forward, reversible_fdtd
from fdtdx.objects.object import SimulationObject
from fdtdx.units import A_per_m, V_per_m, inv_eps, inv_mu, m, s


class DummySimulationObject(SimulationObject):
    """Minimal mock implementation of a SimulationObject."""

    def __init__(self, name="dummy"):
        self.name = name

    def update(self, *args, **kwargs):
        return args, kwargs


@pytest.fixture
def dummy_arrays():
    field_shape = (3, 2, 2, 2)  # (components, nx, ny, nz)
    mat_shape = (2, 2, 2)  # scalar per voxel

    return ArrayContainer(
        E=jnp.zeros(field_shape) * V_per_m,
        H=jnp.zeros(field_shape) * A_per_m,
        inv_permittivities=jnp.ones(mat_shape) * inv_eps,
        inv_permeabilities=jnp.ones(mat_shape) * inv_mu,
        boundary_states={},
        detector_states={},
        recording_state=None,
        electric_conductivity=None,
        magnetic_conductivity=None,
    )


@pytest.fixture
def dummy_objects():
    # Include one mock SimulationObject in the object list
    return ObjectContainer(object_list=[DummySimulationObject()], volume_idx=0)


@pytest.fixture
def empty_objects():
    return ObjectContainer(object_list=[], volume_idx=0)


@pytest.fixture
def dummy_config():
    return SimulationConfig(
        time=0.1 * s,
        resolution=1.0 * m,
        backend="gpu",
        dtype=jnp.float32,
        courant_factor=0.99 * s,
        gradient_config=None,
    )


def test_reversible_fdtd_runs(dummy_arrays, dummy_objects, dummy_config):
    key = jax.random.PRNGKey(0)
    t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, dummy_config, key)

    assert isinstance(t, jax.Array)
    assert isinstance(arrs, ArrayContainer)
    assert arrs.E.shape == dummy_arrays.E.shape


def test_checkpointed_fdtd_runs(dummy_arrays, dummy_objects, dummy_config):
    key = jax.random.PRNGKey(0)
    t, arrs = checkpointed_fdtd(dummy_arrays, dummy_objects, dummy_config, key)

    assert isinstance(t, jax.Array)
    assert isinstance(arrs, ArrayContainer)


def test_custom_fdtd_forward_runs(dummy_arrays, dummy_objects, dummy_config):
    key = jax.random.PRNGKey(0)
    t, arrs = custom_fdtd_forward(
        dummy_arrays,
        dummy_objects,
        dummy_config,
        key,
        reset_container=True,
        record_detectors=False,
        start_time=0,
        end_time=1,
    )

    assert isinstance(t, jax.Array)
    assert isinstance(arrs, ArrayContainer)


# --- Edge case tests ---


def test_zero_time(dummy_arrays, dummy_objects):
    config = SimulationConfig(
        time=0.0 * s,
        resolution=1.0 * m,
        backend="gpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )
    key = jax.random.PRNGKey(0)
    t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, config, key)
    assert int(t) == 0
    assert isinstance(arrs, ArrayContainer)


def test_empty_objects(dummy_arrays, empty_objects, dummy_config):
    key = jax.random.PRNGKey(0)
    t, arrs = checkpointed_fdtd(dummy_arrays, empty_objects, dummy_config, key)
    assert isinstance(t, jax.Array)
    assert isinstance(arrs, ArrayContainer)


def test_custom_fdtd_forward_same_start_end(dummy_arrays, dummy_objects, dummy_config):
    key = jax.random.PRNGKey(0)
    t, arrs = custom_fdtd_forward(
        dummy_arrays,
        dummy_objects,
        dummy_config,
        key,
        reset_container=False,
        record_detectors=False,
        start_time=1,
        end_time=1,
    )
    # No evolution should occur
    assert int(t) == 1
    assert isinstance(arrs, ArrayContainer)
