from functools import partial
import jax
import jax.numpy as jnp
from jax._src import pjit
from typing import Any, Callable

from fdtdx.core.jax.pytrees import TreeClass, autoinit
from fdtdx.core.jax.utils import hash_abstract_pytree, is_currently_jitting
from fdtdx.units import unitful
from fdtdx.units.unitful import Unitful



class CustomJitWrapped:
    def __init__(
        self,
        fun: Callable,
        jit_kwargs: dict[str, Any],
    ):
        self.fun = fun
        self.jit_kwargs = jit_kwargs
        self.jitted_fun = jax._orig_jit(  # type: ignore
            fun,
            **jit_kwargs
        )
        self.cache = {}
    
    
    def _call_from_cache(self, input_hash, *args, **kwargs):
        to_materialise_mask = self.cache[input_hash]
        # do actual jit compilation on function
        optimized_result = self.jitted_fun(*args, **kwargs)
        # materialise the results according to mask from unitful to arrays
        materialised_result = jax.tree.map(
            lambda x, y: x.array_materialise() if y else x,
            optimized_result,
            to_materialise_mask,
            is_leaf=lambda x: isinstance(x, Unitful),
        )
        return materialised_result
    
    
    def _compute_materialisation_mask(
        self,
        *args,
        **kwargs,
    ):
        # first get shape dtypes without any array to unitful optimizations
        # The functools.partial prevents jax from caching the function call
        unitful.STATIC_OPTIM_STOP_FLAG = True
        desired_shape_dtype = jax.eval_shape(
            jax._orig_jit(  # type: ignore
                partial(self.fun), 
                **self.jit_kwargs
            ),
            *args,
            **kwargs,
        )
        is_array_mask = jax.tree.map(
            lambda x: isinstance(x, jax.ShapeDtypeStruct),
            desired_shape_dtype,
            is_leaf=lambda x: isinstance(x, Unitful),
        )
        
        # then get the result of the actual jit step
        unitful.STATIC_OPTIM_STOP_FLAG = False
        optimized_shape_dtype = jax.eval_shape(
            jax._orig_jit(   # type: ignore
                partial(self.fun), 
                **self.jit_kwargs
            ),
            *args,
            **kwargs,
        )
        is_unitful_mask = jax.tree.map(
            lambda x: isinstance(x, Unitful),
            optimized_shape_dtype,
            is_leaf=lambda x: isinstance(x, Unitful),
        )
        # we need to convert all unitfuls back to arrays which are only arrays because of scale optimization
        to_materialise_mask = jax.tree.map(
            lambda x, y: x and y,
            is_unitful_mask,
            is_array_mask,
        )
        return to_materialise_mask
        
    
    def __call__(self, *args, **kwargs):
        # input hashing to avoid recompilation
        input_hash = hash_abstract_pytree((args, kwargs))
        if input_hash not in self.cache:
            # compute the materialisation mask
            to_materialise_mask = self._compute_materialisation_mask(*args, **kwargs)
            self.cache[input_hash] = to_materialise_mask
            
        return self._call_from_cache(input_hash, *args, **kwargs)
          


def jit(fun: Callable, **kwargs) -> CustomJitWrapped:
    custom_wrapper = CustomJitWrapped(fun=jax.tree_util.Partial(fun), jit_kwargs=kwargs)
    return custom_wrapper
