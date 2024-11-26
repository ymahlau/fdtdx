from functools import partial
import math
from typing import Literal
import jax
import jax.numpy as jnp
from loguru import logger

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

from fdtdx.core.physics.constants import SHARD_STR


def get_named_sharding_from_shape(
    shape: tuple[int, ...],
    sharding_axis: int,
) -> jax.sharding.NamedSharding:
    compute_devices = jax.devices()
    num_dims = len(shape)
    device_shape = (len(compute_devices),)
    axis_names = tuple(
        SHARD_STR if i == sharding_axis else None
        for i in range(num_dims)
    )
    devices = mesh_utils.create_device_mesh(
        device_shape,
        devices=compute_devices,
    )
    mesh = jax.sharding.Mesh(
        devices=devices,
        axis_names=(SHARD_STR,)
    )
    spec = jax.sharding.PartitionSpec(*axis_names)
    sharding = jax.sharding.NamedSharding(
        mesh=mesh,
        spec=spec
    )
    return sharding

counter = 0

def get_dtype_bytes(dtype):
    return jnp.dtype(dtype).itemsize



def pretty_print_sharding(sharding):
    if isinstance(sharding, jax.sharding.NamedSharding):
        return f"NamedSharding({sharding.mesh.devices}, {sharding.spec})"
    elif isinstance(sharding, jax.sharding.PositionalSharding):
        return f"PositionalSharding({sharding._devices}, {sharding._ids.shape})"
    elif isinstance(sharding, jax.sharding.SingleDeviceSharding):
        return f"SingleDeviceSharding({sharding._device})"
    else:
        return str(sharding)


def create_named_sharded_matrix(
    shape: tuple[int, ...],
    value: float,
    sharding_axis: int,
    dtype: jnp.dtype,
    backend: Literal["gpu", "tpu", "cpu"],
):
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
    
    per_device_shape = tuple(
        shape[i] if i != sharding_axis else sharding_axis_size
        for i in range(len(shape))
    )
    
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
        device_matrix = value_fn(device_matrix, value)
        matrices.append(device_matrix)
    num_bytes = get_dtype_bytes(dtype)
    counter += math.prod(shape) * num_bytes
    return jax.make_array_from_single_device_arrays(shape, named_sharding, matrices)
