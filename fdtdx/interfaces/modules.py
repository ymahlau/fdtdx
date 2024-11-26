from abc import ABC, abstractmethod
from typing import Self

import pytreeclass as tc
import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import ExtendedTreeClass
from fdtdx.interfaces.state import RecordingState

@tc.autoinit
class CompressionModule(ExtendedTreeClass, ABC):
    _input_shape_dtypes: dict[str, jax.ShapeDtypeStruct] = tc.field(
        default=None,
        on_getattr=[tc.unfreeze], 
        on_setattr=[tc.freeze],
        init=False,
    )  # type: ignore
    _output_shape_dtypes: dict[str, jax.ShapeDtypeStruct] = tc.field(
        default=None,
        on_getattr=[tc.unfreeze], 
        on_setattr=[tc.freeze],
        init=False,
    )  # type: ignore
    
    
    @abstractmethod
    def init_shapes(
        self,
        input_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    ) -> tuple[
        Self,
        dict[str, jax.ShapeDtypeStruct],  # data
        dict[str, jax.ShapeDtypeStruct]  # state
    ]:
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
        RecordingState,
    ]:
        del values, state, key
        raise NotImplementedError()
    
    @abstractmethod
    def decompress(
        self, 
        values: dict[str, jax.Array],
        state: RecordingState,
        key: jax.Array,
    ) -> dict[str, jax.Array]:
        del values, state, key,
        raise NotImplementedError()


@tc.autoinit
class DtypeConversion(CompressionModule):
    dtype: jnp.dtype = tc.field(
        init=True,
        kind="KW_ONLY",
        on_getattr=[tc.unfreeze], 
        on_setattr=[tc.freeze]
    )  # type: ignore
    exclude_filter: list[str] = tc.field(default=None, init=False)  # type: ignore

    def init_shapes(
        self,
        input_shape_dtypes: dict[str, jax.ShapeDtypeStruct],
    ) -> tuple[
        Self,
        dict[str, jax.ShapeDtypeStruct],  # data
        dict[str, jax.ShapeDtypeStruct]  # state
    ]:
        self = self.aset("_input_shape_dtypes", input_shape_dtypes)
        exclude = [] if self.exclude_filter is None else self.exclude_filter
        out_shape_dtypes = {
            k: (
                jax.ShapeDtypeStruct(
                    v.shape, 
                    self.dtype
                )
                if not any(e in k for e in exclude)
                else v
            )
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
        exclude = [] if self.exclude_filter is None else self.exclude_filter
        out_vals = {
            k: (
                v.astype(self.dtype) 
                if not any(e in k for e in exclude)
                else v
            )
            for k, v in values.items() 
        }
        return out_vals, state
    
    def decompress(
        self, 
        values: dict[str, jax.Array],
        state: RecordingState,
        key: jax.Array,
    ) -> dict[str, jax.Array]:
        del key, state
        out_vals = {
            k: v.astype(self._input_shape_dtypes[k].dtype)
            for k, v in values.items()
        }
        return out_vals