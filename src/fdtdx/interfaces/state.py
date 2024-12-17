from typing import Literal

import jax
import pytreeclass as tc

from fdtdx.core.jax.sharding import create_named_sharded_matrix


@tc.autoinit
class RecordingState(tc.TreeClass):
    """Container for simulation recording state data.

    Holds field data and state information for FDTD simulations.

    Attributes:
        data: Dictionary mapping field names to their array values.
        state: Dictionary mapping state variable names to their array values.
    """

    data: dict[str, jax.Array]
    state: dict[str, jax.Array]


def init_recording_state(
    data_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    state_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    backend: Literal["gpu", "tpu", "cpu"],
) -> RecordingState:
    """Initialize a new recording state with sharded arrays.

    Creates a RecordingState instance with data and state arrays sharded across
    available devices based on the provided shapes/dtypes and backend.

    Args:
        data_shape_dtypes: Dictionary mapping field names to their shape/dtype specs.
        state_shape_dtypes: Dictionary mapping state names to their shape/dtype specs.
        backend: Hardware backend to use ("gpu", "tpu", or "cpu").

    Returns:
        A new RecordingState instance with initialized sharded arrays.
    """
    data = init_sharded_dict(data_shape_dtypes, backend=backend)
    state = init_sharded_dict(state_shape_dtypes, backend=backend)
    return RecordingState(
        data=data,
        state=state,
    )


def init_sharded_dict(
    shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    backend: Literal["gpu", "tpu", "cpu"],
) -> dict[str, jax.Array]:
    """Initialize a dictionary of sharded arrays.

    Creates arrays sharded across available devices based on the provided
    shapes/dtypes and backend.

    Args:
        shape_dtypes: Dictionary mapping names to shape/dtype specifications.
        backend: Hardware backend to use ("gpu", "tpu", or "cpu").

    Returns:
        Dictionary mapping names to initialized sharded arrays.
    """
    data = {}
    for k, v in shape_dtypes.items():
        num_devices = len(jax.devices(backend="gpu"))
        shape = v.shape
        if v.shape[0] % num_devices != 0:
            new_shape = list(v.shape)
            new_shape[0] = new_shape[0] + num_devices - (new_shape[0] % num_devices)
            shape = tuple(new_shape)
        arr = create_named_sharded_matrix(
            shape=shape,
            value=0,
            sharding_axis=0,
            dtype=v.dtype,
            backend=backend,
        )
        data[k] = arr
    return data
