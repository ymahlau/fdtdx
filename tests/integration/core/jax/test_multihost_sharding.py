"""Multi-process regression coverage for named-sharded array construction.

This test skips in ordinary single-process CI. On a TPU VM slice, use
``checks/check_multihost_sharding.py`` to initialize JAX distributed and invoke
pytest in the same Python process on all hosts simultaneously.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.core.jax.sharding import create_named_sharded_matrix


@pytest.mark.skipif(jax.process_count() == 1, reason="requires a multi-process JAX cluster")
def test_create_named_sharded_matrix_uses_addressable_devices():
    """Construct the global array from only this process's addressable devices."""
    assert jax.process_count() > 1
    assert jax.default_backend() == "tpu"

    global_device_count = jax.device_count()
    local_device_count = jax.local_device_count()
    assert global_device_count > local_device_count

    shape = (3, 2 * global_device_count, 8, 8)
    array = create_named_sharded_matrix(
        shape=shape,
        value=2.0,
        sharding_axis=1,
        dtype=jnp.float32,
        backend="tpu",
    )
    addressable_map = array.sharding.addressable_devices_indices_map(shape)

    assert array.shape == shape
    assert len(array.devices()) == global_device_count
    assert len(array.addressable_shards) == local_device_count
    assert len(array.addressable_shards) == len(addressable_map)

    for shard in array.addressable_shards:
        assert shard.data.shape == (3, 2, 8, 8)
        assert bool(jnp.all(shard.data == 2.0).block_until_ready())
