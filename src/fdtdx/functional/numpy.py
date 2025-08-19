import math
import jax
import jax.numpy as jnp
from plum import dispatch, overload

from fdtdx.core.fraction import Fraction
from fdtdx.typing import SI
from fdtdx.units.unitful import Unit, Unitful

## Square Root ###########################
@overload
def sqrt(
    x: Unitful, 
) -> Unitful:
    new_dim: dict[SI, int | Fraction] = {}
    for k, v in x.unit.dim.items():
        if isinstance(v, int):
            if v % 2 == 0:
                new_dim[k] = v // 2
            else:
                new_dim[k] = Fraction(v, 2)
        elif isinstance(v, Fraction):
            new_dim[k] = v / 2
        else:
            raise Exception(f"Invalid dimension exponent: {v}")
    if x.unit.scale % 2 == 0:
        new_val = jnp._orig_sqrt(x.val)  # type: ignore
        new_scale = x.unit.scale // 2
    else:
        new_val = jnp._orig_sqrt(x.val) * math.sqrt(10)  # type: ignore
        new_scale = math.floor(x.unit.scale / 2)
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=new_dim))

@overload
def sqrt(x: int | float) -> float: return math.sqrt(x)

@overload
def sqrt(x: jax.Array) -> jax.Array: return jnp._orig_sqrt(x)  # type: ignore

@dispatch
def sqrt(x):  # type: ignore
    del x
    raise NotImplementedError()


