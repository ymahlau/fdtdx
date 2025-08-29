import math
import jax
import jax.numpy as jnp
from plum import dispatch, overload

from fdtdx.core.fraction import Fraction
from fdtdx.units.typing import SI
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


## Roll #####################################
@overload
def roll(
    x: Unitful,
    *args,
    **kwargs, 
) -> Unitful:
    new_val = jnp._orig_roll(x.val, *args, **kwargs)  # type: ignore
    return Unitful(val=new_val, unit=x.unit)

@overload
def roll(
    x: int | float | complex | jax.Array,
    *args,
    **kwargs, 
) -> jax.Array: 
    return jnp._orig_roll(x, *args, **kwargs)  # type: ignore

@dispatch
def roll(  # type: ignore
    x,
    *args,
    **kwargs,
):
    del x, args, kwargs
    raise NotImplementedError()

## Square #####################################
@overload
def square(
    x: Unitful,
) -> Unitful:
    return x * x

@overload
def square(
    x: int | float
) -> float: 
    return x * x

@overload
def square(
    x: complex
) -> complex: 
    return x * x

@overload
def square(
    x: jax.Array
) -> jax.Array: 
    return x * x

@dispatch
def square(  # type: ignore
    x,
):
    del x
    raise NotImplementedError()

## Cross #####################################
@overload
def cross(
    a: Unitful,
    b: Unitful,
    *args,
    **kwargs,
) -> Unitful:
    new_val = jnp._orig_cross(a.val, b.val, *args, **kwargs)  # type: ignore
    new_scale = a.unit.scale + b.unit.scale
    # compute new unit: same as multiplication
    unit_dict = a.unit.dim.copy()
    for k, v in b.unit.dim.items():
        if k in unit_dict:
            unit_dict[k] += v
            if unit_dict[k] == 0:
                del unit_dict[k]
        else:
            unit_dict[k] = v
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=unit_dict))

@overload
def cross(
    a: jax.Array,
    b: jax.Array,
    *args,
    **kwargs,
) -> jax.Array: 
    return jnp._orig_cross(a, b, *args, **kwargs)  # type: ignore

@dispatch
def cross(  # type: ignore
    a,
    b,
    *args,
    **kwargs,
):
    del a, b, args, kwargs
    raise NotImplementedError()

## Cross #####################################
@overload
def conj(
    x: Unitful,
) -> Unitful:
    new_val = jnp._orig_conj(x.val)  # type: ignore
    return Unitful(val=new_val, unit=x.unit)

@overload
def conj(
    x: int | float | complex | jax.Array,
) -> jax.Array: 
    return jnp._orig_conj(x)  # type: ignore

@dispatch
def conj(  # type: ignore
    x,
):
    del x
    raise NotImplementedError()


