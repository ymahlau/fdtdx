import numpy as np
from fdtdx.core.jax.utils import is_traced
from fdtdx.units.typing import PhysicalArrayLike
from fdtdx.units.unitful import Unitful, get_static_operand
from plum import dispatch, overload
import jax
import jax.numpy as jnp


## norm #####################################
@overload
def norm(
    x: Unitful,
    *args,
    **kwargs,
) -> Unitful:
    new_val = jnp.linalg._orig_norm(x.val, *args, **kwargs)  # type: ignore
    # static computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        if x_arr is not None:
            new_static_arr = np.linalg.norm(x_arr, *args, **kwargs)
    return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)

@overload
def norm(
    x: PhysicalArrayLike,
    *args,
    **kwargs,
) -> jax.Array:
    return jnp.linalg._orig_norm(x, *args, **kwargs)  # type: ignore

@dispatch
def norm(  # type: ignore
    x,
    *args,
    **kwargs,
):
    del x, args, kwargs
    raise NotImplementedError()


