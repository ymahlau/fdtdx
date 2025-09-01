import math
from typing import Sequence
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
import numpy as np
from plum import dispatch, overload

from fdtdx.core.fraction import Fraction
from fdtdx.units.typing import SI
from fdtdx.units.unitful import Unit, Unitful
from fdtdx.units.utils import dim_after_multiplication, handle_n_scales

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
    unit_dict = dim_after_multiplication(a.unit.dim, b.unit.dim)
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

## Conjugate #####################################
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


## dot #####################################
@overload
def dot(
    a: Unitful,
    b: Unitful,
    *args,
    **kwargs,
) -> Unitful:
    new_val = jnp._orig_dot(a.val, b.val, *args, **kwargs)  # type: ignore
    unit_dict = dim_after_multiplication(a.unit.dim, b.unit.dim)
    new_scale = a.unit.scale + b.unit.scale
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=unit_dict))

@overload
def dot(
    a: Unitful,
    b: jax.Array,
    *args,
    **kwargs,
):
    new_val = jnp._orig_dot(a.val, b, *args, **kwargs)  # type: ignore
    return Unitful(val=new_val, unit=a.unit)

@overload
def dot(
    a: jax.Array,
    b: Unitful,
    *args,
    **kwargs,
):
    new_val = jnp._orig_dot(a, b.val, *args, **kwargs)  # type: ignore
    return Unitful(val=new_val, unit=b.unit)

@overload
def dot(
    a: jax.Array,
    b: jax.Array,
    *args,
    **kwargs,
) -> jax.Array: 
    return jnp._orig_dot(a, b, *args, **kwargs)  # type: ignore

@dispatch
def dot(  # type: ignore
    a,
    b,
    *args,
    **kwargs,
):
    del a, b, args, kwargs
    raise NotImplementedError()


## Transpose #####################################
@overload
def transpose(
    x: Unitful,
    *args,
    **kwargs,
) -> Unitful:
    new_val = jnp._orig_transpose(x.val, *args, **kwargs)  # type: ignore
    return Unitful(val=new_val, unit=x.unit)

@overload
def transpose(
    x: jax.Array,
    *args,
    **kwargs,
) -> jax.Array: 
    return jnp._orig_transpose(x, *args, **kwargs)  # type: ignore

@dispatch
def transpose(  # type: ignore
    x,
    *args,
    **kwargs,
):
    del x, args, kwargs
    raise NotImplementedError()


## pad #####################################
@overload
def pad(
    x: Unitful,
    *args,
    **kwargs,
) -> Unitful:
    new_val = jnp._orig_pad(x.val, *args, **kwargs)  # type: ignore
    return Unitful(val=new_val, unit=x.unit)

@overload
def pad(
    x: jax.Array,
    *args,
    **kwargs,
) -> jax.Array: 
    return jnp._orig_pad(x, *args, **kwargs)  # type: ignore

@dispatch
def pad(  # type: ignore
    x,
    *args,
    **kwargs,
):
    del x, args, kwargs
    raise NotImplementedError()


## stack #####################################
@overload
def stack(
    arrays: Unitful | Sequence[Unitful],
    *args,
    **kwargs,
) -> Unitful:
    if isinstance(arrays, Sequence):
        for a in arrays:
            if a.unit.dim != arrays[0].unit.dim:
                raise Exception(
                    f"jnp.stack requires all Unitful to have the same dimension, but got: {arrays}"
                )
        # bring all values to same scale
        new_scale, factors = handle_n_scales([a.unit.scale for a in arrays])
        scaled = [a.val * f for a, f in zip(arrays, factors)]
        # simply call original function
        new_val = jnp._orig_stack(scaled, *args, **kwargs)  # type: ignore
        new_unit = Unit(scale=new_scale, dim=arrays[0].unit.dim)
    else:
        new_val = jnp._orig_stack(arrays, *args, **kwargs)  # type: ignore
        new_unit = arrays.unit
    return Unitful(val=new_val, unit=new_unit)

@overload
def stack(
    x: jax.Array | np.ndarray | Sequence[ArrayLike],
    *args,
    **kwargs,
) -> jax.Array: 
    return jnp._orig_stack(x, *args, **kwargs)  # type: ignore

@dispatch
def stack(  # type: ignore
    x,
    *args,
    **kwargs,
):
    del x, args, kwargs
    raise NotImplementedError()
