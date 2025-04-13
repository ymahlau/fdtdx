from abc import ABC, abstractmethod
from typing import Self, Sequence

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_field, frozen_private_field
from fdtdx.interfaces.state import RecordingState


@extended_autoinit
class CompressionModule(ExtendedTreeClass, ABC):
    """Abstract base class for compression modules that process simulation data.

    This class provides an interface for modules that compress and decompress field data
    during FDTD simulations. Implementations can perform operations like quantization,
    dimensionality reduction, or other compression techniques.
    """

    _input_shape_dtypes: dict[str, jax.ShapeDtypeStruct] = frozen_private_field(default=None)
    _output_shape_dtypes: dict[str, jax.ShapeDtypeStruct] = frozen_private_field(default=None)

    @abstractmethod
    def init_shapes(
        self,
        input_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    ) -> tuple[
        Self,
        dict[str, jax.ShapeDtypeStruct],  # data
        dict[str, jax.ShapeDtypeStruct],  # state shapes/dtypes
    ]:
        """Initialize shapes and sizes for the compression module.

        Args:
            input_shape_dtypes: Dictionary mapping field names to their input shapes/dtypes.

        Returns:
            Tuple containing:
                - Self: Updated instance of the compression module
                - Dictionary mapping field names to their output shapes/dtypes
                - Dictionary mapping field names to their state shapes/dtypes

        Raises:
            NotImplementedError: This is an abstract method that should be implemented
                by subclasses.
        """
        del input_shape_dtypes
        raise NotImplementedError()

    @abstractmethod
    def compress(
        self,
        values: dict[str, jax.Array],
        state: RecordingState,
        key: jax.Array,
    ) -> tuple[
        dict[str, jax.Array],  # compressed data
        RecordingState,  # updated recording state
    ]:
        """Compress field values at the current time step.

        Args:
            values: Dictionary mapping field names to their values.
            state: Current recording state.
            key: Random key for stochastic operations.

        Returns:
            Tuple containing:
                - Dictionary of compressed field values
                - Updated recording state

        Raises:
            NotImplementedError: This is an abstract method that should be implemented
                by subclasses.
        """
        del values, state, key
        raise NotImplementedError()

    @abstractmethod
    def decompress(
        self,
        values: dict[str, jax.Array],
        state: RecordingState,
        key: jax.Array,
    ) -> dict[str, jax.Array]:
        """Decompress field values back to their original form.

        Args:
            values: Dictionary mapping field names to their compressed values.
            state: Current recording state.
            key: Random key for stochastic operations.

        Returns:
            Dictionary mapping field names to their decompressed values.

        Raises:
            NotImplementedError: This is an abstract method that should be implemented
                by subclasses.
        """
        del (
            values,
            state,
            key,
        )
        raise NotImplementedError()


@extended_autoinit
class DtypeConversion(CompressionModule):
    """Compression module that converts data types of field values.

    This module changes the data type of field values while preserving their shape,
    useful for reducing memory usage or meeting precision requirements.

    Attributes:
        dtype: Target data type for conversion.
        exclude_filter: List of field names to exclude from conversion.
    """

    dtype: jnp.dtype = frozen_field(kind="KW_ONLY")  # type: ignore
    exclude_filter: Sequence[str] = frozen_field(default=tuple([]), kind="KW_ONLY")

    def init_shapes(
        self,
        input_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    ) -> tuple[
        Self,
        dict[str, jax.ShapeDtypeStruct],  # data
        dict[str, jax.ShapeDtypeStruct],  # state shapes/dtypes
    ]:
        self = self.aset("_input_shape_dtypes", input_shape_dtypes)
        exclude = [] if self.exclude_filter is None else self.exclude_filter
        out_shape_dtypes = {
            k: (jax.ShapeDtypeStruct(v.shape, self.dtype) if not any(e in k for e in exclude) else v)
            for k, v in input_shape_dtypes.items()
        }
        self = self.aset("_output_shape_dtypes", out_shape_dtypes)
        return self, self._output_shape_dtypes, {}

    def compress(
        self,
        values: dict[str, jax.Array],
        state: RecordingState,
        key: jax.Array,
    ) -> tuple[
        dict[str, jax.Array],
        RecordingState,
    ]:
        del key
        out_vals = {
            k: (v.astype(self.dtype) if not any(e in k for e in self.exclude_filter) else v) for k, v in values.items()
        }
        return out_vals, state

    def decompress(
        self,
        values: dict[str, jax.Array],
        state: RecordingState,
        key: jax.Array,
    ) -> dict[str, jax.Array]:
        del key, state
        out_vals = {k: v.astype(self._input_shape_dtypes[k].dtype) for k, v in values.items()}
        return out_vals
