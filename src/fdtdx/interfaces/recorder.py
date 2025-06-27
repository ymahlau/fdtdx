from typing import Self, Sequence

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_private_field
from fdtdx.core.jax.utils import check_shape_dtype
from fdtdx.interfaces.modules import CompressionModule
from fdtdx.interfaces.state import RecordingState, init_recording_state
from fdtdx.interfaces.time_filter import TimeStepFilter
from fdtdx.typing import BackendOption


@autoinit
class Recorder(TreeClass):
    """Records and compresses simulation data over time using a sequence of processing modules.

    The Recorder manages a pipeline of modules that process simulation data at each timestep.
    It supports both compression modules that reduce data size and time filters that control
    when data is recorded. The recorder handles initialization, compression and decompression
    of simulation data through its module pipeline.

    Attributes:
        modules (Sequence[CompressionModule | TimeStepFilter]): Sequence of processing modules to apply to the
            simulation data. Can be either CompressionModule for data reduction or TimeStepFilter
            for controlling recording frequency.
    """

    modules: Sequence[CompressionModule | TimeStepFilter]
    _input_shape_dtypes: dict[str, jax.ShapeDtypeStruct] = frozen_private_field(default=None)  # type:ignore
    _output_shape_dtypes: dict[str, jax.ShapeDtypeStruct] = frozen_private_field(default=None)  # type:ignore
    _max_time_steps: int = frozen_private_field()
    _latent_array_size: int = frozen_private_field()

    def init_state(
        self: Self,
        input_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
        max_time_steps: int,
        backend: BackendOption,
    ) -> tuple[Self, RecordingState]:
        self = self.aset("_max_time_steps", max_time_steps, create_new_ok=True)
        self = self.aset("_input_shape_dtypes", input_shape_dtypes, create_new_ok=True)

        latent_arr_size, out_shapes = max_time_steps, input_shape_dtypes
        state_sizes: dict[str, jax.ShapeDtypeStruct] = {}
        new_modules = []
        for m in self.modules:
            if isinstance(m, CompressionModule):
                m, out_shapes, state_shapes = m.init_shapes(out_shapes)
            else:
                m, latent_arr_size, out_shapes, state_shapes = m.init_shapes(out_shapes, latent_arr_size)
            state_sizes.update(state_shapes)
            new_modules.append(m)

        self = self.aset("modules", new_modules)
        self = self.aset("_output_shape_dtypes", out_shapes)
        self = self.aset("_latent_array_size", latent_arr_size)

        expanded_out_shapes = {
            k: jax.ShapeDtypeStruct(
                shape=(self._latent_array_size, *v.shape),
                dtype=v.dtype,
            )
            for k, v in self._output_shape_dtypes.items()
        }
        state = init_recording_state(
            data_shape_dtypes=expanded_out_shapes,
            state_shape_dtypes=state_sizes,
            backend=backend,
        )
        return self, state

    def compress(
        self,
        values: dict[str, jax.Array],
        state: RecordingState,
        time_step: jax.Array,
        key: jax.Array,
    ) -> RecordingState:
        check_shape_dtype(values, self._input_shape_dtypes)
        latent_idx = time_step

        def helper_fn(m, values, state, latent_idx, key):
            if isinstance(m, CompressionModule):
                values, state = m.compress(values, state, key=key)
            elif isinstance(m, TimeStepFilter):
                values, state = m.compress(values, state, latent_idx, key=key)
                latent_idx = m.time_to_array_index(latent_idx)
            else:
                raise Exception(f"Invalid module: {m}")
            check_shape_dtype(values, m._output_shape_dtypes)
            return values, state, latent_idx

        def dummy_fn(m, values, state, latent_idx, key):
            del key
            # Only create zero arrays for keys that exist in the input values
            # This ensures structure matching with helper_fn for periodic boundaries
            values = {k: jnp.zeros(v.shape, v.dtype) for k, v in m._output_shape_dtypes.items() if k in values}
            check_shape_dtype(values, m._output_shape_dtypes)
            return values, state, latent_idx

        for m in self.modules:
            key, subkey = jax.random.split(key)
            values, state, latent_idx = jax.lax.cond(
                latent_idx == -1, dummy_fn, helper_fn, m, values, state, latent_idx, subkey
            )
            check_shape_dtype(values, m._output_shape_dtypes)

        def update_state_fn(state, values, latent_idx):
            # Only update state data for keys that exist in values
            for k in state.data.keys():
                if k in values:
                    state.data[k] = state.data[k].at[latent_idx].set(values[k])
            return state

        def update_dummy_fn(state, values, latent_idx):
            del values, latent_idx
            return state

        state = jax.lax.cond(latent_idx == -1, update_dummy_fn, update_state_fn, state, values, latent_idx)

        return state

    def decompress(
        self,
        state: RecordingState,
        time_step: jax.Array,
        key: jax.Array,
    ) -> tuple[
        dict[str, jax.Array],
        RecordingState,
    ]:
        # gather indices necessary to reconstruct
        time_filters = [m for m in self.modules if isinstance(m, TimeStepFilter)]
        indices: list[jax.Array] = [jnp.asarray([time_step])]
        time_indices: list[jax.Array] = []

        for tf in time_filters:
            cur_time_indices = indices[-1].flatten()
            cur_indices = jnp.asarray([tf.indices_to_decompress(idx) for idx in cur_time_indices])
            time_indices.append(cur_time_indices)
            indices.append(cur_indices)

        def reconstruction_iteration(
            m: CompressionModule | TimeStepFilter,
            state: RecordingState,
            key: jax.Array,
            latent: list[dict[str, jax.Array]],
            cur_tf_idx: int,
        ) -> tuple[
            int,
            list[dict[str, jax.Array]],
            RecordingState,
        ]:
            if isinstance(m, CompressionModule):
                latent = [m.decompress(v, state, key=key) for v in latent]
            else:
                num_time_idx = indices[cur_tf_idx].shape[0]
                num_arr_idx = indices[cur_tf_idx].shape[1]
                next_latent = []
                for cur_idx in range(0, num_time_idx):
                    # for idx in range(start_idx, start_idx + num_idx):
                    start_idx = cur_idx * num_arr_idx
                    cur_v = [latent[i] for i in range(start_idx, start_idx + num_arr_idx)]
                    arr_indices = indices[cur_tf_idx][cur_idx]
                    time_idx = time_indices[cur_tf_idx - 1][cur_idx]
                    next_v = m.decompress(
                        values=cur_v,
                        state=state,
                        arr_indices=arr_indices,
                        time_idx=time_idx,
                        key=key,
                    )
                    next_latent.append(next_v)
                latent = next_latent
                cur_tf_idx = cur_tf_idx - 1
            for v in latent:
                check_shape_dtype(v, m._input_shape_dtypes)
            return cur_tf_idx, latent, state

        def bottom_up_reconstruction(state: RecordingState, key):
            cur_tf_idx = len(time_filters)
            latent: list[dict[str, jax.Array]] = [
                {k: jnp.take(v, indices=idx.reshape(1), axis=0).squeeze(axis=0) for k, v in state.data.items()}
                for idx in indices[cur_tf_idx].flatten()
            ]
            for m in self.modules[::-1]:
                key, subkey = jax.random.split(key)
                cur_tf_idx, latent, state = reconstruction_iteration(
                    m=m,
                    state=state,
                    key=subkey,
                    latent=latent,
                    cur_tf_idx=cur_tf_idx,
                )
            return latent, state

        values, state = bottom_up_reconstruction(state, key)

        if len(values) != 1:
            raise Exception("This should never happen")
        return values[0], state
