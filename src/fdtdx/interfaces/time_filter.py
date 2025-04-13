from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_field, frozen_private_field
from fdtdx.core.misc import index_1d_array
from fdtdx.interfaces.state import RecordingState


@extended_autoinit
class TimeStepFilter(ExtendedTreeClass, ABC):
    """Abstract base class for filtering and processing time steps in FDTD simulations.

    This class provides an interface for filters that process simulation data at specific
    time steps. Implementations can perform operations like downsampling, collation, or
    other temporal processing of field data.
    """

    _time_steps_max: int = frozen_private_field(default=-1)  # type: ignore
    _array_size: int = frozen_private_field(default=-1)  # type: ignore
    _input_shape_dtypes: dict[str, jax.ShapeDtypeStruct] = frozen_private_field(default=None)  # type: ignore
    _output_shape_dtypes: dict[str, jax.ShapeDtypeStruct] = frozen_private_field(default=None)  # type: ignore

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

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
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

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
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

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
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

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
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

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        del values, state, arr_indices, time_idx, key
        raise NotImplementedError()


@extended_autoinit
class LinearReconstructEveryK(TimeStepFilter):
    """Time step filter that performs linear reconstruction between sampled steps.

    This filter saves field values every k time steps and uses linear interpolation
    to reconstruct values at intermediate time steps.

    Attributes:
        k: Number of time steps between saved values.
        start_recording_after: Time step to start recording from.
    """

    k: int = frozen_field()
    start_recording_after: int = 0
    _save_time_steps: jax.Array = frozen_private_field(default=None)  # type: ignore
    _time_to_arr_idx: jax.Array = frozen_private_field(default=None)  # type: ignore

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
