from typing import Literal
import jax
import pytreeclass as tc

from fdtdx.core.jax.sharding import create_named_sharded_matrix

@tc.autoinit
class RecordingState(tc.TreeClass):
    data: dict[str, jax.Array]
    state: dict[str, jax.Array]
    


def init_recording_state(
    data_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    state_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    backend: Literal["gpu", "tpu", "cpu"],
) -> RecordingState:
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
