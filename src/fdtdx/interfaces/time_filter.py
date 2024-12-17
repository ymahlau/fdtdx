import math
from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp
import pytreeclass as tc

from fdtdx.core.jax.pytrees import ExtendedTreeClass
from fdtdx.core.misc import index_1d_array
from fdtdx.interfaces.state import RecordingState

tc.autoinit


class TimeStepFilter(ExtendedTreeClass, ABC):
    """Abstract base class for filtering and processing time steps in FDTD simulations.

    This class provides an interface for filters that process simulation data at specific
    time steps. Implementations can perform operations like downsampling, collation, or
    other temporal processing of field data.

    Attributes:
        _time_steps_max: Maximum number of time steps in the simulation.
        _array_size: Size of the array used to store filtered data.
        _input_shape_dtypes: Dictionary mapping field names to their input shapes/dtypes.
        _output_shape_dtypes: Dictionary mapping field names to their output shapes/dtypes.
    """

    _time_steps_max: int = tc.field(default=-1, init=False)  # type: ignore
    _array_size: int = tc.field(default=-1, init=False)  # type: ignore
    _input_shape_dtypes: dict[str, jax.ShapeDtypeStruct] = tc.field(
        default=None,
        init=False,
        on_getattr=[tc.unfreeze],
        on_setattr=[tc.freeze],
    )  # type: ignore
    _output_shape_dtypes: dict[str, jax.ShapeDtypeStruct] = tc.field(
        default=None,
        init=False,
        on_getattr=[tc.unfreeze],
        on_setattr=[tc.freeze],
    )  # type: ignore

    @abstractmethod
    def init_shapes(
        self,
        input_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
        time_steps_max: int,  # maximum number of time steps
    ) -> tuple[
        Self,
        int,  # array size (number of latent time steps)
        dict[str, jax.ShapeDtypeStruct],  # data
        dict[str, jax.ShapeDtypeStruct],  # state shapes
    ]:
        """Initialize shapes and sizes for the time step filter.

        Args:
            input_shape_dtypes: Dictionary mapping field names to their shape/dtype information.
            time_steps_max: Maximum number of time steps in the simulation.

        Returns:
            A tuple containing:
            - Updated filter instance
            - Size of array for storing filtered data
            - Dictionary of data shapes/dtypes
            - Dictionary of state shapes/dtypes
        """
        del input_shape_dtypes, time_steps_max
        raise NotImplementedError()

    @abstractmethod
    def time_to_array_index(
        self,
        time_idx: int,  # scalar
    ) -> int:  # array index if not filtered, else -1
        """Convert a time step index to its corresponding array index.

        Args:
            time_idx: Time step index to convert.

        Returns:
            The corresponding array index if the time step is not filtered,
            or -1 if the time step is filtered out.
        """
        del time_idx
        raise NotImplementedError()

    @abstractmethod
    def compress(
        self,
        values: dict[str, jax.Array],
        state: RecordingState,
        time_idx: int,  # scalar
        key: jax.Array,
    ) -> tuple[
        dict[str, jax.Array],
        RecordingState,  # updated recording state
    ]:
        """Compress field values at a given time step.

        Args:
            values: Dictionary mapping field names to their values.
            state: Current recording state.
            time_idx: Current time step index.
            key: Random key for stochastic operations.

        Returns:
            Tuple containing:
            - Dictionary of compressed field values
            - Updated recording state
        """
        del values, state, time_idx, key
        raise NotImplementedError()

    @abstractmethod
    def indices_to_decompress(
        self,
        time_idx: jax.Array,  # scalar
    ) -> jax.Array:  # 1d-list of array indices necessary to reconstruct
        """Get array indices needed to reconstruct data for a given time step.

        Args:
            time_idx: Time step index to reconstruct.

        Returns:
            Array of indices needed to reconstruct the data for this time step.
        """
        del time_idx
        raise NotImplementedError()

    @abstractmethod
    def decompress(
        self,
        values: list[dict[str, jax.Array]],  # array values requested above
        state: RecordingState,
        arr_indices: jax.Array,
        time_idx: jax.Array,  # scalar
        key: jax.Array,
    ) -> dict[str, jax.Array]:  # reconstructed value
        """Decompress field values to reconstruct data for a time step.

        Args:
            values: List of dictionaries containing array values needed for reconstruction.
            state: Current recording state.
            arr_indices: Array indices needed for reconstruction.
            time_idx: Time step index to reconstruct.
            key: Random key for stochastic operations.

        Returns:
            Dictionary of reconstructed field values.
        """
        del values, state, arr_indices, time_idx, key
        raise NotImplementedError()


@tc.autoinit
class LinearReconstructEveryK(TimeStepFilter):
    """Time step filter that performs linear reconstruction between sampled steps.

    This filter saves field values every k time steps and uses linear interpolation
    to reconstruct values at intermediate time steps.

    Attributes:
        k: Number of time steps between saved values.
        start_recording_after: Time step to start recording from.
        _save_time_steps: Array of time steps that are saved.
        _time_to_arr_idx: Mapping from time steps to array indices.
    """

    k: int = tc.field(init=True, kind="POS_OR_KW")  # type:ignore
    start_recording_after: int = 0
    _save_time_steps: jax.Array = tc.field(default=None, init=False)  # type:ignore
    _time_to_arr_idx: jax.Array = tc.field(default=None, init=False)  # type:ignore

    def init_shapes(
        self,
        input_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
        time_steps_max: int,  # maximum number of time steps
    ) -> tuple[
        Self,
        int,
        dict[str, jax.ShapeDtypeStruct],  # data
        dict[str, jax.ShapeDtypeStruct],  # state shapes
    ]:
        self = self.aset("_time_steps_max", time_steps_max)
        self = self.aset("_input_shape_dtypes", input_shape_dtypes)
        self = self.aset("_output_shape_dtypes", input_shape_dtypes)

        # init list of all time steps to save
        all_time_steps = jnp.arange(self.start_recording_after, self._time_steps_max, self.k).tolist()
        if all_time_steps[-1] != self._time_steps_max - 1:
            all_time_steps.append(self._time_steps_max - 1)

        self = self.aset("_save_time_steps", jnp.asarray(all_time_steps, dtype=jnp.int32))
        self = self.aset("_array_size", len(all_time_steps))

        # mapping between time steps and array indices
        index_tmp = jnp.arange(0, self._array_size, dtype=jnp.int32)
        time_indices = jnp.zeros(shape=(self._time_steps_max,), dtype=jnp.int32)
        time_indices = time_indices.at[self._save_time_steps].set(index_tmp)
        for _ in range(self.k - 1):
            rolled = jnp.roll(time_indices, 1)
            time_indices = jnp.where(
                time_indices == 0,
                rolled,
                time_indices,
            )
            time_indices = time_indices.at[: self.k].set(0)
        self = self.aset("_time_to_arr_idx", time_indices)
        return self, self._array_size, input_shape_dtypes, {}

    def time_to_array_index(
        self,
        time_idx: int,  # scalar
    ) -> int:  # scalar, array index if not filtered, else -1
        result = jax.lax.cond(
            jnp.any(time_idx == self._save_time_steps),
            lambda: self._time_to_arr_idx[time_idx],
            lambda: jnp.asarray(-1, dtype=jnp.int32),
        )
        return result

    def indices_to_decompress(
        self,
        time_idx: jax.Array,  # scalar
    ) -> jax.Array:  # 1d-list of array indices necessary to reconstruct
        arr_idx = self._time_to_arr_idx[time_idx]
        result = jnp.asarray([arr_idx, arr_idx + 1], dtype=jnp.int32)
        return result

    def compress(
        self,
        values: dict[str, jax.Array],
        state: RecordingState,
        time_idx: jax.Array,  # scalar
        key: jax.Array,
    ) -> tuple[
        dict[str, jax.Array],
        RecordingState,  # updated recording state
    ]:
        del time_idx, key
        return values, state

    def decompress(
        self,
        values: list[dict[str, jax.Array]],  # array values requested above
        state: RecordingState,
        arr_indices: jax.Array,
        time_idx: jax.Array,  # scalar
        key: jax.Array,
    ) -> dict[str, jax.Array]:  # reconstructed value
        del key, state

        def value_was_saved():
            return values[0]

        def linear_reconstruct():
            arr_idx = arr_indices[0]

            prev_save_time = index_1d_array(self._time_to_arr_idx, arr_idx)
            next_save_time = index_1d_array(self._time_to_arr_idx, arr_idx + 1)
            interp_factor = (time_idx - prev_save_time) / (next_save_time - prev_save_time)

            prev_vals, next_vals = values[0], values[1]
            res = {}
            for k, prev in prev_vals.items():
                next = next_vals[k]
                interp = prev + interp_factor.astype(next.dtype) * (next - prev)
                res[k] = interp
            return res

        result = jax.lax.cond(
            jnp.any(time_idx == self._save_time_steps),
            value_was_saved,
            linear_reconstruct,
        )
        return result


@tc.autoinit
class CollateTimeSteps(TimeStepFilter):
    """Time step filter that collates multiple time steps into single records.

    This filter groups consecutive time steps together into single records,
    useful for batch processing or analyzing temporal patterns.

    Attributes:
        num_steps: Number of time steps to collate together.
        _start_time_steps: Array of starting time steps for each collation.
        _end_time_steps: Array of ending time steps for each collation.
        _time_to_arr_idx: Mapping from time steps to array indices.
        _state_name_map: Dictionary mapping field names to state names.
    """

    num_steps: int = tc.field(init=True, kind="KW_ONLY")  # type: ignore
    _start_time_steps: jax.Array = tc.field(default=None, init=False)  # type:ignore
    _end_time_steps: jax.Array = tc.field(default=None, init=False)  # type:ignore
    _time_to_arr_idx: jax.Array = tc.field(default=None, init=False)  # type:ignore
    # _saved_time_idx: jax.Array = tc.field(default=None, init=False) # type:ignore
    _state_name_map: dict[str, str] = tc.field(
        default=None, on_setattr=[tc.freeze], on_getattr=[tc.unfreeze], init=False
    )  # type:ignore

    def init_shapes(
        self,
        input_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
        time_steps_max: int,  # maximum number of time steps
    ) -> tuple[
        Self,
        int,
        dict[str, jax.ShapeDtypeStruct],  # data
        dict[str, jax.ShapeDtypeStruct],  # state shapes
    ]:
        self = self.aset("_time_steps_max", time_steps_max)
        self = self.aset("_input_shape_dtypes", input_shape_dtypes)

        # init list of all time steps to save
        self = self.aset("_start_time_steps", jnp.arange(0, self._time_steps_max, self.num_steps))
        self = self.aset("array_size", len(self._start_time_steps))

        end_time_steps = (self._start_time_steps + self.num_steps - 1).tolist()
        if end_time_steps[-1] != self._time_steps_max - 1:
            del end_time_steps[-1]
            end_time_steps.append(self._time_steps_max - 1)

        self = self.aset("_end_time_steps", jnp.asarray(end_time_steps, dtype=jnp.int32))

        # mapping between time steps and array indices
        index_tmp = jnp.arange(0, self._array_size, dtype=jnp.int32)
        time_indices = jnp.zeros(shape=(self._time_steps_max,), dtype=jnp.int32)
        time_indices = time_indices.at[self._start_time_steps].set(index_tmp)
        for _ in range(self.num_steps - 1):
            rolled = jnp.roll(time_indices, 1)
            time_indices = jnp.where(
                time_indices == 0,
                rolled,
                time_indices,
            )
            time_indices = time_indices.at[: self.num_steps].set(0)
        self = self.aset("_time_to_arr_idx", time_indices)

        result_shapes, state_shapes = {}, {}
        state_map = {}
        for k, v in input_shape_dtypes.items():
            result_shapes[k] = jax.ShapeDtypeStruct(
                shape=(self.num_steps, *v.shape),
                dtype=v.dtype,
            )
            state_map[k] = f"{k}_coll_state_{self.num_steps}"
            state_shapes[state_map[k]] = jax.ShapeDtypeStruct(
                shape=(self.num_steps, *v.shape),
                dtype=v.dtype,
            )
        state_shapes["cached_arr_idx"] = jax.ShapeDtypeStruct(
            shape=(1,),
            dtype=jnp.int32,
        )
        self = self.aset("_state_name_map", state_map)
        self = self.aset("_output_shape_dtypes", result_shapes)
        return self, self._array_size, result_shapes, state_shapes

    def time_to_array_index(
        self,
        time_idx: int,  # scalar
    ) -> int:  # scalar, array index if not filtered, else -1
        result = jax.lax.cond(
            jnp.any(time_idx == self._end_time_steps),
            lambda: self._time_to_arr_idx[time_idx],
            lambda: jnp.asarray(-1, dtype=jnp.int32),
        )
        return result

    def indices_to_decompress(
        self,
        time_idx: jax.Array,  # scalar
    ) -> jax.Array:  # 1d-list of array indices necessary to reconstruct
        arr_idx = self._time_to_arr_idx[time_idx]
        result = jnp.asarray([arr_idx])
        return result

    def compress(
        self,
        values: dict[str, jax.Array],
        state: RecordingState,
        time_idx: jax.Array,  # scalar
        key: jax.Array,
    ) -> tuple[
        dict[str, jax.Array],
        RecordingState,  # updated recording state
    ]:
        del key
        state_idx = time_idx % self.num_steps

        def save_to_state(values: dict[str, jax.Array], state: RecordingState):
            result = {}
            for k, v in values.items():
                k_state = self._state_name_map[k]
                state.state[k_state] = state.state[k_state].at[state_idx].set(v)
                result[k] = jnp.zeros(
                    shape=self._output_shape_dtypes[k].shape,
                    dtype=self._output_shape_dtypes[k].dtype,
                )
            return result, state

        def save_to_value_and_reset(values: dict[str, jax.Array], state: RecordingState):
            result = {}
            for k, v in values.items():
                k_state = self._state_name_map[k]
                arr = state.state[k_state]
                arr = arr.at[state_idx].set(v)
                result[k] = arr
                state.state[k_state] = state.state[k_state].at[:].set(0)
            return result, state

        values, state = jax.lax.cond(
            jnp.any(time_idx == self._end_time_steps),
            save_to_value_and_reset,
            save_to_state,
            values,
            state,
        )
        return values, state

    def decompress(
        self,
        values: list[dict[str, jax.Array]],  # array values requested above
        state: RecordingState,
        arr_indices: jax.Array,
        time_idx: jax.Array,  # scalar
        key: jax.Array,
    ) -> dict[str, jax.Array]:  # reconstructed value
        del key
        arr_idx = time_idx % self.num_steps
        result: dict[str, jax.Array] = {}
        for k, v in values[0].items():
            result[k] = v[arr_idx]
            state.state[self._state_name_map[k]] = v
        state.state["cached_arr_idx"] = arr_indices
        return result


@tc.autoinit
class RepeatingTimeStepFilter(TimeStepFilter):
    """Time step filter that handles repeating temporal patterns.

    This filter is designed for simulations with periodic behavior,
    saving memory by exploiting temporal repetition in the field values.

    Attributes:
        time_per_step: Time duration of each step.
        period: Duration of one complete period.
        num_periods: Number of periods to process.
        _save_time_steps: Array of time steps that are saved.
        _relative_pos: Array of relative positions within periods.
        _time_to_arr_idx: Mapping from time steps to array indices.
    """

    time_per_step: float = tc.field(init=True, kind="KW_ONLY")  # type:ignore
    period: float = tc.field(init=True, kind="KW_ONLY")  # type:ignore
    num_periods: int = 1
    _save_time_steps: jax.Array = tc.field(default=None, init=False)  # type:ignore
    _relative_pos: jax.Array = tc.field(default=None, init=False)  # type:ignore
    _time_to_arr_idx: jax.Array = tc.field(default=None, init=False)  # type:ignore

    def init_shapes(
        self,
        input_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
        time_steps_max: int,  # maximum number of time steps
    ) -> tuple[
        Self,
        int,
        dict[str, jax.ShapeDtypeStruct],  # data
        dict[str, jax.ShapeDtypeStruct],  # state shapes
    ]:
        self = self.aset("_time_steps_max", time_steps_max)
        self = self.aset("_input_shape_dtypes", input_shape_dtypes)
        self = self.aset("_output_shape_dtypes", input_shape_dtypes)

        # init list of all time steps to save
        steps_per_period = math.ceil(self.period / self.time_per_step)
        all_time_steps = jnp.arange(
            self._time_steps_max - self.num_periods * steps_per_period,
            self._time_steps_max,
        ).tolist()

        self = self.aset("_save_time_steps", jnp.asarray(all_time_steps, dtype=jnp.int32))
        self = self.aset("_array_size", len(all_time_steps))

        # mapping between time steps and array indices
        time_indices = jnp.zeros(shape=(self._time_steps_max, 2), dtype=jnp.int32)
        relative_pos = jnp.zeros(shape=(self._time_steps_max,), dtype=jnp.float32)
        # exactly saved time steps
        for i in range(self._time_steps_max - 1, self._time_steps_max - self.num_periods * steps_per_period - 1, -1):
            negative_offset = self._time_steps_max - 1 - i
            idx = self._array_size - 1 - (negative_offset % (self.num_periods * steps_per_period))
            time_indices = time_indices.at[i, 0].set(idx)
            relative_pos = relative_pos.at[i].set(0)
        # everything after last period needs to be interpolated
        last_period_step_interval = (self.period / self.time_per_step) % 1
        for i in range(self._time_steps_max - self.num_periods * steps_per_period - 1, -1, -1):
            negative_offset = self._time_steps_max - i - 1
            time_since_end = negative_offset * self.time_per_step
            pos_in_period_float = (time_since_end / (self.num_periods * self.period)) % 1
            time_since_period_start = pos_in_period_float * (self.num_periods * self.period)
            step_float = time_since_period_start / self.time_per_step
            idx0 = math.floor(step_float)
            if idx0 == self._array_size:
                idx0 = 0
            idx1 = idx0 + 1
            if idx1 == self._array_size:
                idx1 = 0
            cur_relative_pos = step_float % 1
            if idx1 == 0:
                cur_relative_pos = (step_float % 1) / last_period_step_interval

            idx0 = self._array_size - 1 - idx0
            idx1 = self._array_size - 1 - idx1
            time_indices = time_indices.at[i, 0].set(idx0)
            time_indices = time_indices.at[i, 1].set(idx1)

            relative_pos = relative_pos.at[i].set(cur_relative_pos)

        self = self.aset("_time_to_arr_idx", time_indices)
        self = self.aset("_relative_pos", relative_pos)

        return self, self._array_size, input_shape_dtypes, {}

    def time_to_array_index(
        self,
        time_idx: int,  # scalar
    ) -> int:  # scalar, array index if not filtered, else -1
        result = jax.lax.cond(
            jnp.any(time_idx == self._save_time_steps),
            lambda: self._time_to_arr_idx[time_idx, 0],
            lambda: jnp.asarray(-1, dtype=jnp.int32),
        )
        return result

    def indices_to_decompress(
        self,
        time_idx: jax.Array,  # scalar
    ) -> jax.Array:  # 1d-list of array indices necessary to reconstruct
        arr_indices = self._time_to_arr_idx[time_idx]
        result = jnp.asarray([arr_indices[0], arr_indices[1]])
        return result

    def compress(
        self,
        values: dict[str, jax.Array],
        state: RecordingState,
        time_idx: jax.Array,  # scalar
        key: jax.Array,
    ) -> tuple[
        dict[str, jax.Array],
        RecordingState,  # updated recording state
    ]:
        del time_idx, key
        return values, state

    def decompress(
        self,
        values: list[dict[str, jax.Array]],  # array values requested above
        state: RecordingState,
        arr_indices: jax.Array,
        time_idx: jax.Array,  # scalar
        key: jax.Array,
    ) -> dict[str, jax.Array]:  # reconstructed value
        del state, arr_indices, key
        rel_pos = self._relative_pos[time_idx]
        result = {}
        for k in values[0].keys():
            arr0, arr1 = values[0][k], values[1][k]
            interp = (1 - rel_pos) * arr0 + rel_pos * arr1
            result[k] = interp
        return result
