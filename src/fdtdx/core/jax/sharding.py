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
) -> jax.sharding.NamedSharding:
    """Creates a NamedSharding object for distributing an array across devices.

    Args:
        shape (tuple[int, ...]): Shape of the array to be sharded
        sharding_axis (int): Which axis to shard the array along

    Returns:
        jax.sharding.NamedSharding: NamedSharding object specifying how to distribute the array
            across available devices.
    """
    compute_devices = jax.devices()
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


def pin_to_single_device(arr: jax.Array) -> jax.Array:
    """Pin ``arr`` to a single-device replicated layout via ``with_sharding_constraint``.

    On a multi-device run the field/material arrays are x-sharded (NamedSharding
    ``P(None,'shard',None,None)``). When a thin transverse plane is sliced out of such an
    array and fed to a host ``pure_callback`` (the Tidy3D mode solver, which runs at
    ``{maximal device=0}``), XLA must materialise a single-device operand for the callback.
    Without an explicit constraint XLA may conservatively all-gather the *parent* array to
    satisfy the callback's single-device requirement. Pinning the already-sliced PLANE to a
    single-device replicated sharding tells XLA to gather only the thin band (a few-hundred-KB
    plane), never the full ~100M-cell domain.

    No-op on a single-device run (``len(jax.devices()) == 1``): returns ``arr`` unchanged so the
    single-GPU path stays byte-identical (no sharding constraint is inserted at all).
    """
    if len(jax.devices()) == 1:
        return arr
    # REPLICATED sharding over the FULL device mesh -- NOT SingleDeviceSharding, whose {device 0}
    # device set is incompatible with the [0,1,...] set of the surrounding x-sharded computation
    # (jit raises "Received incompatible devices for jitted computation"). A replicated
    # PartitionSpec() keeps the thin sliced plane on the full mesh but UN-sharded, so XLA gathers
    # only the small plane (a few-hundred KB) to materialise the callback operand, never the full
    # x-sharded ~100M-cell domain, while staying device-set-compatible with the parent computation.
    compute_devices = jax.devices()
    devices = mesh_utils.create_device_mesh((len(compute_devices),), devices=compute_devices)
    mesh = jax.sharding.Mesh(devices=devices, axis_names=(SHARD_STR,))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return jax.lax.with_sharding_constraint(arr, replicated)


def get_dtype_bytes(dtype: jnp.dtype) -> int:
    return jnp.dtype(dtype).itemsize


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

    matrices = []
    for device in compute_devices[::-1]:
        device_matrix = jnp.ones(
            per_device_shape,
            dtype=dtype,
            device=device,
        )
        device_matrix = cast(Any, value_fn)(device_matrix, value)
        matrices.append(device_matrix)
    num_bytes = get_dtype_bytes(dtype)
    counter += math.prod(shape) * num_bytes
    return jax.make_array_from_single_device_arrays(shape, named_sharding, matrices)
