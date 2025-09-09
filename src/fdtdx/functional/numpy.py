import cmath
from functools import partial
import math
from typing import Sequence
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
import numpy as np
from plum import dispatch, overload

from fdtdx.core.fraction import Fraction
from fdtdx.core.jax.utils import is_currently_jitting, is_traced
from fdtdx.units.typing import PHYSICAL_DTYPES, SI, PhysicalArrayLike
from fdtdx.units.unitful import EMPTY_UNIT, MAX_OPTIMIZED_ARR_SIZE, Unit, Unitful, get_static_operand, output_unitful_for_array, unary_fn
from fdtdx.units.utils import dim_after_multiplication, handle_n_scales, is_struct_optimizable

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
    # static arr computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        if x_arr is not None:
            new_static_arr = np.sqrt(x_arr)
            if x.unit.scale % 2 != 0:
                new_static_arr = new_static_arr * math.sqrt(10)
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=new_dim), static_arr=new_static_arr)

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
    # static arr computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        if x_arr is not None:
            new_static_arr = np.roll(x_arr, *args, **kwargs)
    return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)

@overload
def roll(
    x: jax.Array,
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
    # static computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(a)
        y_arr = get_static_operand(b)
        if x_arr is not None and y_arr is not None:
            new_static_arr = np.cross(x_arr, y_arr, *args, **kwargs)
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=unit_dict), static_arr=new_static_arr)

@overload
def cross(
    a: jax.Array,
    b: jax.Array,
    *args,
    **kwargs,
) -> jax.Array: 
    return jnp._orig_cross(a, b, *args, **kwargs)  # type: ignore

@overload
def cross(  # type: ignore
    a,
    b,
    *args,
    **kwargs,
):
    raise NotImplementedError(f"Currently not implemented for {a}, {b}, {args}, {kwargs}")

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
    # static arr computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        if x_arr is not None:
            new_static_arr = np.conj(x_arr)
    return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)

@overload
def conj(
    x: int,
) -> int:
    return x

@overload
def conj(
    x: float,
) -> float: 
    return x

@overload
def conj(
    x: complex,
) -> complex: 
    return x.conjugate()

@overload
def conj(
    x: jax.Array,
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
    # static computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(a)
        y_arr = get_static_operand(b)
        if x_arr is not None and y_arr is not None:
            new_static_arr = np.dot(x_arr, y_arr, *args, **kwargs)
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=unit_dict), static_arr=new_static_arr)

@overload
def dot(
    a: Unitful,
    b: jax.Array | np.ndarray,
    *args,
    **kwargs,
) -> Unitful:
    new_val = jnp._orig_dot(a.val, b, *args, **kwargs)  # type: ignore
    # static computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(a)
        y_arr = get_static_operand(b)
        if x_arr is not None and y_arr is not None:
            new_static_arr = np.dot(x_arr, y_arr, *args, **kwargs)
    return Unitful(val=new_val, unit=a.unit, static_arr=new_static_arr)

@overload
def dot(
    a: jax.Array | np.ndarray,
    b: Unitful,
    *args,
    **kwargs,
) -> Unitful:
    new_val = jnp._orig_dot(a, b.val, *args, **kwargs)  # type: ignore
    # static computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(a)
        y_arr = get_static_operand(b)
        if x_arr is not None and y_arr is not None:
            new_static_arr = np.dot(x_arr, y_arr, *args, **kwargs)
    return Unitful(val=new_val, unit=b.unit, static_arr=new_static_arr)

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
    # static computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        if x_arr is not None:
            new_static_arr = np.transpose(x_arr, *args, **kwargs)
    return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)

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
    # static computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        if x_arr is not None:
            new_static_arr = np.pad(x_arr, *args, **kwargs)
    return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)

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
        # static computation
        new_static_arr = None
        if is_traced(new_val):
            arrs = [get_static_operand(v) for v in arrays]
            if all([v is not None for v in arrs]):
                scaled_arrs = [a * f for a, f in zip(arrs, factors)]  # type: ignore
                new_static_arr = np.stack(scaled_arrs, *args, **kwargs) # type: ignore
    else:
        new_val = jnp._orig_stack(arrays, *args, **kwargs)  # type: ignore
        new_unit = arrays.unit
        # static computation
        new_static_arr = None
        if is_traced(new_val):
            x_arr = get_static_operand(arrays)
            if x_arr is not None:
                new_static_arr = np.stack(x_arr, *args, **kwargs)  # type: ignore
    return Unitful(val=new_val, unit=new_unit, static_arr=new_static_arr)

@overload
def stack(
    x: jax.Array | np.ndarray | Sequence[ArrayLike],
    *args,
    **kwargs,
) -> jax.Array:
    all_physical = (
        True 
        if isinstance(x, jax.Array | np.ndarray) 
        else all([isinstance(x_i, PhysicalArrayLike) for x_i in x])
    )
    # axis/dtype args/kwargs needs to be static for eval_shape
    partial_orig_fn = partial(jnp._orig_stack, x, *args, **kwargs)  # type: ignore
    result_shape_dtype = jax.eval_shape(partial_orig_fn)
    if not output_unitful_for_array(result_shape_dtype) or not all_physical:
        return jnp._orig_stack(x, *args, **kwargs)  # type: ignore        
    
    if isinstance(x, Sequence):
        unit_input = [Unitful(val=x_i, unit=EMPTY_UNIT) for x_i in x] # type: ignore
    else:
        unit_input = Unitful(val=x, unit=EMPTY_UNIT)
    unit_result = stack(unit_input, *args, **kwargs)
    return unit_result  # type: ignore
    

@dispatch
def stack(  # type: ignore
    x,
    *args,
    **kwargs,
):
    del x, args, kwargs
    raise NotImplementedError()


## isfinite #####################################
@overload
def isfinite(
    x: Unitful,
    *args,
    **kwargs,
) -> jax.Array:
    new_val = jnp._orig_isfinite(x.val, *args, **kwargs)  # type: ignore
    return new_val

@overload
def isfinite(
    x: jax.Array,
    *args,
    **kwargs,
) -> jax.Array: 
    return jnp._orig_isfinite(x, *args, **kwargs)  # type: ignore

@dispatch
def isfinite(  # type: ignore
    x,
    *args,
    **kwargs,
):
    del x, args, kwargs
    raise NotImplementedError()


## real #####################################
@overload
def real(val: Unitful, *args, **kwargs,) -> Unitful:
    new_val = jnp._orig_real(val.val, *args, **kwargs)  # type: ignore
    # static computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(val)
        if x_arr is not None:
            new_static_arr = np.real(x_arr, *args, **kwargs)
    return Unitful(val=new_val, unit=val.unit, static_arr=new_static_arr)

@overload
def real(val: int, *args, **kwargs) -> int:
    return val

@overload
def real(val: float, *args, **kwargs) -> float:
    return val

@overload
def real(
    val: complex, *args, **kwargs) -> complex:
    return val.real

@overload
def real(val: jax.Array, *args, **kwargs) -> jax.Array: 
    return jnp._orig_real(val, *args, **kwargs)  # type: ignore

@dispatch
def real(  # type: ignore
    val,
    *args,
    **kwargs,
):
    del val, args, kwargs
    raise NotImplementedError()


## imag #####################################
@overload
def imag(val: Unitful, *args, **kwargs) -> Unitful:
    new_val = jnp._orig_imag(val.val, *args, **kwargs)  # type: ignore
    # static computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(val)
        if x_arr is not None:
            new_static_arr = np.imag(x_arr, *args, **kwargs)
    return Unitful(val=new_val, unit=val.unit, static_arr=new_static_arr)

@overload
def imag(val: int, *args, **kwargs) -> int:
    return val

@overload
def imag(val: float, *args, **kwargs) -> float:
    return val

@overload
def imag(
    val: complex, *args, **kwargs) -> complex:
    return val.imag

@overload
def imag(val: jax.Array, *args, **kwargs) -> jax.Array: 
    return jnp._orig_imag(val, *args, **kwargs)  # type: ignore

@dispatch
def imag(  # type: ignore
    val,
    *args,
    **kwargs,
):
    del val, args, kwargs
    raise NotImplementedError()


## sin #####################################
@overload
def sin(x: Unitful) -> Unitful:
    new_val = jnp._orig_sin(x.val)  # type: ignore
    # static computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        if x_arr is not None:
            new_static_arr = np.sin(x_arr)
    return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)

@overload
def sin(x: int | float) -> float:
    return math.sin(x)

@overload
def sin(x: np.ndarray | np.number) -> np.ndarray:
    return np.sin(x)

@overload
def sin(x: jax.Array) -> jax.Array:
    return jnp._orig_sin(x)  # type: ignore

@dispatch
def sin(  # type: ignore
    x,
):
    del x
    raise NotImplementedError()


## cos #####################################
@overload
def cos(x: Unitful) -> Unitful:
    new_val = jnp._orig_cos(x.val)  # type: ignore
    # static computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        if x_arr is not None:
            new_static_arr = np.cos(x_arr)
    return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)

@overload
def cos(x: jax.Array) -> jax.Array:
    return jnp._orig_cos(x)  # type: ignore

@overload
def cos(x: int | float) -> float:
    return math.cos(x)

@overload
def cos(x: np.ndarray | np.number) -> np.ndarray:
    return np.cos(x)

@dispatch
def cos(  # type: ignore
    x,
):
    del x
    raise NotImplementedError()


## tan #####################################
@overload
def tan(x: Unitful) -> Unitful:
    new_val = jnp._orig_tan(x.val)  # type: ignore
    # static computation
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        if x_arr is not None:
            new_static_arr = np.tan(x_arr)
    return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)

@overload
def tan(x: jax.Array) -> jax.Array:
    return jnp._orig_tan(x)  # type: ignore

@overload
def tan(x: int | float) -> float:
    return math.tan(x)

@overload
def tan(x: np.ndarray | np.number) -> np.ndarray:
    return np.tan(x)

@dispatch
def tan(  # type: ignore
    x,
    *args,
    **kwargs,
):
    del x, args, kwargs
    raise NotImplementedError()



## asarray #####################################
def asarray(
    a,
    *args,
    **kwargs,
) -> jax.Array:
    result: jax.Array = jnp._orig_asarray(a, *args, **kwargs)  # type: ignore
    if not output_unitful_for_array(result):
        # cannot use this as Unitful, wrong dtype
        return result
    
    # try to get a static version of the array and save to trace metadata
    static_arr = None
    result_size = math.prod(result.shape)
    if is_struct_optimizable(a) and result_size <= MAX_OPTIMIZED_ARR_SIZE:
        static_arr = np.asarray(a, copy=True)
    
    # return Unitful without unit. We lie to typechecker here
    return Unitful(val=result, unit=Unit(scale=0, dim={}), static_arr=static_arr)  # type: ignore
    

def array(
    a,
    *args,
    **kwargs,
) -> jax.Array:
    return asarray(a, *args, **kwargs)

