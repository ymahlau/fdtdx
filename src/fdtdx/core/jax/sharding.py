import math
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils

from fdtdx.constants import SHARD_STR
from fdtdx.typing import BackendOption


def get_named_sharding_from_shape(
    shape: tuple[int, ...],
    sharding_axis: int,
    backend: BackendOption | None = None,
) -> jax.sharding.NamedSharding:
    """Creates a NamedSharding object for distributing an array across devices.

    Args:
        shape (tuple[int, ...]): Shape of the array to be sharded
        sharding_axis (int): Which axis to shard the array along
        backend (BackendOption | None): Device backend used to construct the mesh

    Returns:
        jax.sharding.NamedSharding: NamedSharding object specifying how to distribute the array
            across available devices.
    """
    compute_devices = jax.devices(backend=backend)
    num_dims = len(shape)
    device_shape = (len(compute_devices),)
    axis_names = tuple(SHARD_STR if i == sharding_axis else None for i in range(num_dims))
    devices = mesh_utils.create_device_mesh(
        device_shape,
        devices=compute_devices,
    )
    mesh = jax.sharding.Mesh(devices=devices, axis_names=(SHARD_STR,))
    spec = jax.sharding.PartitionSpec(*axis_names)
    sharding = jax.sharding.NamedSharding(mesh=mesh, spec=spec)
    return sharding


counter = 0


def get_dtype_bytes(dtype: jnp.dtype) -> int:
    return jnp.dtype(dtype).itemsize


def _local_shape_from_global_index(
    shape: tuple[int, ...],
    global_index: tuple[Any, ...] | None,
) -> tuple[int, ...]:
    """Derive a process-local shard shape from its global array index."""
    if global_index is None or len(global_index) != len(shape):
        raise ValueError(f"Invalid shard index {global_index!r} for global shape {shape}")

    local_shape = []
    for axis, (axis_size, axis_index) in enumerate(zip(shape, global_index, strict=True)):
        if not isinstance(axis_index, slice):
            raise ValueError(
                f"Expected a slice for shard axis {axis}, got {axis_index!r} in global index {global_index!r}"
            )
        start, stop, step = axis_index.indices(axis_size)
        local_shape.append(len(range(start, stop, step)))

    return tuple(local_shape)


def pretty_print_sharding(sharding: jax.sharding.Sharding) -> str:
    if isinstance(sharding, jax.sharding.NamedSharding):
        return f"NamedSharding({sharding.mesh.devices}, {sharding.spec})"
    #
    # elif isinstance(sharding, jax.sharding.PositionalSharding):
    #    return f"PositionalSharding({sharding._devices}, {sharding._ids.shape})"
    elif isinstance(sharding, jax.sharding.SingleDeviceSharding):
        return f"SingleDeviceSharding({sharding._device})"
    else:
        return str(sharding)


def _sharding_preserving_indexed_update(
    array: jax.Array,
    index: Any,
    values: Any,
    *,
    operation: str,
) -> jax.Array:
    """Apply an indexed update without losing the destination sharding."""
    if operation == "set":

        def update(destination, update_values):
            return destination.at[index].set(update_values)

    elif operation == "add":

        def update(destination, update_values):
            return destination.at[index].add(update_values)

    else:
        raise ValueError(f"Unsupported indexed update operation: {operation}")

    # A fully addressable array may still span multiple devices. In that case,
    # an unconstrained indexed update can silently return a replicated result.
    # Only single-device arrays are safe to update without an output-sharding
    # contract.
    if len(array.devices()) == 1:
        return update(array, values)

    sharded_update = jax.jit(
        update,
        out_shardings=array.sharding,
        donate_argnums=(0,),
    )
    return cast(Any, sharded_update)(array, values)


def sharding_preserving_set(array: jax.Array, index: Any, values: Any) -> jax.Array:
    """Set indexed values while preserving the destination sharding.

    Multi-device input buffers may be donated and must not be reused.
    """
    return _sharding_preserving_indexed_update(array, index, values, operation="set")


def sharding_preserving_add(array: jax.Array, index: Any, values: Any) -> jax.Array:
    """Add indexed values while preserving the destination sharding.

    Multi-device input buffers may be donated and must not be reused.
    """
    return _sharding_preserving_indexed_update(array, index, values, operation="add")


def create_named_sharded_matrix(
    shape: tuple[int, ...],
    value: float,
    sharding_axis: int,
    dtype: jnp.dtype,
    backend: BackendOption,
) -> jax.Array:
    """Creates a sharded matrix distributed across available devices.

    Creates a matrix of the given shape filled with the specified value,
    sharded across available devices along the specified axis.

    Args:
        shape (tuple[int, ...]): Shape of the matrix to create
        value (float): Value to fill the matrix with
        sharding_axis (int): Which axis to shard along
        dtype (jnp.dtype): Data type of the matrix elements
        backend (BackendOption): Which device backend to use ("gpu", "tpu", or "cpu")

    Returns:
        jax.Array: Sharded matrix distributed across devices

    Raises:
        ValueError: If shape[sharding_axis] is not divisible by number of devices
    """
    global counter
    if shape[sharding_axis] == 1:
        sharding_axis = next(i for i, dim in enumerate(shape) if dim != 1)
    named_sharding = get_named_sharding_from_shape(
        shape=shape,
        sharding_axis=sharding_axis,
        backend=backend,
    )
    compute_devices = jax.devices(backend=backend)
    num_devices = len(compute_devices)
    if shape[sharding_axis] % num_devices != 0:
        raise ValueError(
            "Grid shape in sharding axis must be divisible by num_devices"
            f", got {shape[sharding_axis]=} and {num_devices=}"
        )
    sharding_axis_size = shape[sharding_axis] // num_devices

    per_device_shape = tuple(shape[i] if i != sharding_axis else sharding_axis_size for i in range(len(shape)))

    @partial(jax.jit, donate_argnames="arr")
    def value_fn(arr, val):
        return arr * val

    addressable_map = named_sharding.addressable_devices_indices_map(shape)
    matrices = []
    for device, global_index in addressable_map.items():
        local_shape = _local_shape_from_global_index(shape, global_index)
        if local_shape != per_device_shape:
            raise ValueError(
                "Mapped local shard shape does not match the expected evenly divided shape, "
                f"got {local_shape=} and {per_device_shape=} for {device=} and {global_index=}"
            )
        device_matrix = jnp.ones(
            local_shape,
            dtype=dtype,
            device=device,
        )
        device_matrix = cast(Any, value_fn)(device_matrix, value)
        matrices.append(device_matrix)
    num_bytes = get_dtype_bytes(dtype)
    counter += math.prod(shape) * num_bytes
    return jax.make_array_from_single_device_arrays(shape, named_sharding, matrices)
