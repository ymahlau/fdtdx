import jax

from fdtdx.core.jax.pytrees import TreeClass, autoinit
from fdtdx.core.jax.sharding import create_named_sharded_matrix
from fdtdx.typing import BackendOption


@autoinit
class RecordingState(TreeClass):
    """Container for simulation recording state data.

    Holds field data and state information for FDTD simulations.

    Attributes:
        data (dict[str, jax.Array]): Dictionary mapping field names to their array values.
        state (dict[str, jax.Array]): Dictionary mapping state variable names to their array values.
    """

    data: dict[str, jax.Array]
    state: dict[str, jax.Array]


def init_recording_state(
    data_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    state_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    backend: BackendOption,
) -> RecordingState:
    """Initialize a new recording state with sharded arrays.

    Creates a RecordingState instance with data and state arrays sharded across
    available devices based on the provided shapes/dtypes and backend.

    Args:
        data_shape_dtypes (dict[str, jax.ShapeDtypeStruct]): Dictionary mapping field names to their shape/dtype specs.
        state_shape_dtypes (dict[str, jax.ShapeDtypeStruct]): Dictionary mapping state names to their shape/dtype specs.
        backend (BackendOption): Hardware backend to use ("gpu", "tpu", or "cpu").

    Returns:
        RecordingState: A new RecordingState instance with initialized sharded arrays.
    """
    data = init_sharded_dict(data_shape_dtypes, backend=backend)
    state = init_sharded_dict(state_shape_dtypes, backend=backend)
    return RecordingState(
        data=data,
        state=state,
    )


def init_sharded_dict(
    shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    backend: BackendOption,
) -> dict[str, jax.Array]:
    """Initialize a dictionary of sharded arrays.

    Creates arrays sharded across available devices based on the provided
    shapes/dtypes and backend.

    Args:
        shape_dtypes (dict[str, jax.ShapeDtypeStruct]): Dictionary mapping names to shape/dtype specifications.
        backend (BackendOption): Hardware backend to use ("gpu", "tpu", or "cpu").

    Returns:
        dict[str, jax.Array]: Dictionary mapping names to initialized sharded arrays.
    """
    data = {}
    for k, v in shape_dtypes.items():
        num_devices = len(jax.devices(backend=backend))
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
