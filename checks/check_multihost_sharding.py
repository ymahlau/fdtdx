"""Run the multi-process sharding regression test on a TPU VM slice.

Invoke this script simultaneously from the repository root on every TPU
worker. It initializes JAX distributed and validates the addressable-device
contract in the same interpreter.
"""

import json

import jax
import jax.numpy as jnp


def main() -> None:
    jax.distributed.initialize()
    if jax.default_backend() != "tpu":
        raise RuntimeError(f"Expected TPU backend, got {jax.default_backend()!r}")

    import fdtdx
    from fdtdx.core.jax.sharding import create_named_sharded_matrix

    process_count = jax.process_count()
    global_device_count = jax.device_count()
    local_device_count = jax.local_device_count()

    assert process_count > 1
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

    print(
        json.dumps(
            {
                "status": "PASS",
                "process_index": jax.process_index(),
                "process_count": process_count,
                "global_device_count": global_device_count,
                "local_device_count": local_device_count,
                "addressable_shards": len(array.addressable_shards),
                "fdtdx_file": fdtdx.__file__,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
