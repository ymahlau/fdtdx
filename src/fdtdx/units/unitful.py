from enum import Enum
from numbers import Number
from typing import TYPE_CHECKING, cast

import jax
import jax.random
from jaxtyping import ArrayLike
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field

from plum import dispatch, overload
from pytreeclass import tree_repr
from fastcore.foundation import patch_to

from fdtdx.units.utils import handle_different_scales, patch_fn_to_module


class SI(Enum):
    s = "second"
    m = "meter"
    kg = "kilogram"
    A = "ampere"
    K = "kelvin"
    mol = "mole"
    cd = "candela"

class Unit:
    def __init__(
        self,
        scale: int,
        dim: dict[SI, int],
    ):
        self.scale = scale
        self.dim = dim

    def __str__(self):
        res_str = f"10^{self.scale} " if self.scale != 0 else ""
        for k, v in self.dim.items():
            res_str += f"{k.name}^{v} " if v != 1 else f"{k.name} "
        return res_str[:-1]

    def __repr__(self):
        return str(self)


@autoinit
class Unitful(TreeClass):
    val: ArrayLike
    unit: Unit = frozen_field()
    
    def materialise(self) -> ArrayLike:
        if self.unit.dim:
            raise ValueError(f"Cannot materialise unitful array with a non-zero unit: {self.unit}")
        factor = 10 ** self.unit.scale
        return self.val * factor

    def __str__(self) -> str:
        return f"Unitful [{self.unit}]: {tree_repr(self.val)}"

    def __repr__(self) -> str:
        return str(self)
    
    def __mul__(self, other: ArrayLike | "Unitful") -> "Unitful":
        return multiply(self, other)
    
    def __rmul__(self, other: ArrayLike | "Unitful") -> "Unitful":
        return multiply(self, other)
    
    def __add__(self, other: "Unitful") -> "Unitful":
        return add(self, other)
    
    def __radd__(self, other: "Unitful") -> "Unitful":
        return add(self, other)
    
    def __sub__(self, other: "Unitful") -> "Unitful":
        return subtract(self, other)
    
    def __rsub__(self, other: "Unitful") -> "Unitful":
        return subtract(other, self)
    
    def __lt__(self, other: "Unitful") -> ArrayLike:
        return lt(self, other)
    
    def __le__(self, other: "Unitful") -> ArrayLike:
        return le(self, other)
    
    def __eq__(self, other: "Unitful") -> ArrayLike:
        return eq(self, other)
    
    def __ne__(self, other: "Unitful") -> ArrayLike:  # type: ignore
        return ne(self, other)
    
    def __ge__(self, other: "Unitful") -> ArrayLike:
        return ge(self, other)
    
    def __gt__(self, other: "Unitful") -> ArrayLike:
        return gt(self, other)
    


def align_scales(
    u1: Unitful,
    u2: Unitful,
) -> tuple[Unitful, Unitful]:
    if u1.unit.dim != u2.unit.dim:
        raise Exception(f"Cannot align arrays with different units")
    new_scale, factor1, factor2 = handle_different_scales(
        u1.unit.scale,
        u2.unit.scale,
    )
    if new_scale != u1.unit.scale:
        u1 = Unitful(
            val=u1.val * factor1,
            unit=Unit(new_scale, u1.unit.dim)
        )
    if new_scale != u2.unit.scale:
        u2 = Unitful(
            val=u2.val * factor2,
            unit=Unit(new_scale, u2.unit.dim)
        )
    return u1, u2


## Multiplication ###########################
@overload
def multiply(  # type: ignore
    x: Unitful, 
    y: Unitful
) -> Unitful:
    unit_dict = x.unit.dim.copy()
    for k, v in y.unit.dim.items():
        if k in unit_dict:
            unit_dict[k] += v
            if unit_dict[k] == 0:
                del unit_dict[k]
        else:
            unit_dict[k] = v
    new_scale = x.unit.scale + y.unit.scale
    new_unit = Unit(new_scale, unit_dict)
    return Unitful(val=x.val * y.val, unit=new_unit)

@overload
def multiply(  # type: ignore
    x: ArrayLike, 
    y: Unitful
) -> Unitful:
    return Unitful(val=y.val * x, unit=y.unit)

@overload
def multiply(  # type: ignore
    x: Unitful, 
    y: ArrayLike
) -> Unitful:
    return Unitful(val=x.val * y, unit=x.unit)

@overload
def multiply(  # type: ignore
    x: ArrayLike, 
    y: ArrayLike
) -> ArrayLike:
    return x * y


@dispatch
def multiply(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## Addition ###########################
def _same_dim_binary_fn(x: Unitful, y: Unitful, op_str: str) -> Unitful:
    if x.unit.dim != y.unit.dim:
        raise ValueError(f"Cannot {op_str} two arrays with units {x.unit} and {y.unit}.")
    x, y = align_scales(x, y)
    orig_fn = getattr(jax.numpy, op_str)
    new_val = orig_fn(x.val, y.val)
    return Unitful(val=new_val, unit=y.unit)

@overload
def add(  # type: ignore
    x: Unitful, 
    y: Unitful
) -> Unitful:
    return _same_dim_binary_fn(x, y, "add")

@overload
def add(  # type: ignore
    x: ArrayLike, 
    y: ArrayLike
) -> ArrayLike:
    return jnp._orig_add(x, y)  # type: ignore

@dispatch
def add(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## Subtractions ###########################
@overload
def subtract(  # type: ignore
    x: Unitful, 
    y: Unitful
) -> Unitful:
    return _same_dim_binary_fn(x, y, "subtract")

@overload
def subtract(  # type: ignore
    x: ArrayLike, 
    y: ArrayLike
) -> ArrayLike:
    return jnp._orig_subtract(x, y)  # type: ignore

@dispatch
def subtract(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## Remainder ###########################
@overload
def remainder(  # type: ignore
    x: Unitful, 
    y: Unitful
) -> Unitful:
    return _same_dim_binary_fn(x, y, "remainder")

@overload
def remainder(  # type: ignore
    x: ArrayLike, 
    y: ArrayLike
) -> ArrayLike:
    return jnp._orig_remainder(x, y)  # type: ignore

@dispatch
def remainder(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## less than ##########################
def _same_dim_binary_fn_array_return(x: Unitful, y: Unitful, op_str: str) -> ArrayLike:
    if x.unit.dim != y.unit.dim:
        raise ValueError(f"Cannot {op_str} two arrays with units {x.unit} and {y.unit}.")
    x, y = align_scales(x, y)
    orig_fn = getattr(jax.lax, op_str)
    new_val = orig_fn(x.val, y.val)
    return new_val

@overload
def lt(  # type: ignore
    x: Unitful, 
    y: Unitful
) -> ArrayLike:
    return _same_dim_binary_fn_array_return(x, y, "lt")

@overload
def lt(  # type: ignore
    x: ArrayLike, 
    y: ArrayLike
) -> ArrayLike:
    return jax.lax._orig_lt(x, y)  # type: ignore

@dispatch
def lt(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## less equal ##########################
@overload
def le(  # type: ignore
    x: Unitful, 
    y: Unitful
) -> ArrayLike:
    return _same_dim_binary_fn_array_return(x, y, "le")

@overload
def le(  # type: ignore
    x: ArrayLike, 
    y: ArrayLike
) -> ArrayLike:
    return jax.lax._orig_le(x, y)  # type: ignore

@dispatch
def le(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## equal ##########################
@overload
def eq(  # type: ignore
    x: Unitful, 
    y: Unitful
) -> ArrayLike:
    return _same_dim_binary_fn_array_return(x, y, "eq")

@overload
def eq(  # type: ignore
    x: ArrayLike, 
    y: ArrayLike
) -> ArrayLike:
    return jax.lax._orig_eq(x, y)  # type: ignore

@dispatch
def eq(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## not equal ##########################
@overload
def ne(  # type: ignore
    x: Unitful, 
    y: Unitful
) -> ArrayLike:
    return _same_dim_binary_fn_array_return(x, y, "ne")

@overload
def ne(  # type: ignore
    x: ArrayLike, 
    y: ArrayLike
) -> ArrayLike:
    return jax.lax._orig_ne(x, y)  # type: ignore

@dispatch
def ne(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## greater equal ##########################
@overload
def ge(  # type: ignore
    x: Unitful, 
    y: Unitful
) -> ArrayLike:
    return _same_dim_binary_fn_array_return(x, y, "ge")

@overload
def ge(  # type: ignore
    x: ArrayLike, 
    y: ArrayLike
) -> ArrayLike:
    return jax.lax._orig_ge(x, y)  # type: ignore

@dispatch
def ge(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## greater than ##########################
@overload
def gt(  # type: ignore
    x: Unitful, 
    y: Unitful
) -> ArrayLike:
    return _same_dim_binary_fn_array_return(x, y, "gt")

@overload
def gt(  # type: ignore
    x: ArrayLike, 
    y: ArrayLike
) -> ArrayLike:
    return jax.lax._orig_gt(x, y)  # type: ignore

@dispatch
def gt(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## add to original jax.numpy ###################
_full_patch_list_numpy = [
    (multiply, None),
    (add, None),
    (subtract, None),
    (remainder, None),
    (lt, "less"),
    (le, "less_equal"),
    (eq, "equal"),
    (ne, "not_equal"),
    (ge, "greater_equal"),
    (gt, "greater"),
]
for fn, orig in _full_patch_list_numpy:
    patch_fn_to_module(
        f=fn, 
        mod=jax.numpy,
        fn_name=orig,
    )

## add to jax.lax ###################
_full_patch_list_lax = [
    (lt, None),
    (le, None),
    (eq, None),
    (ne, None),
    (ge, None),
    (gt, None),
]
for fn, orig in _full_patch_list_lax:
    patch_fn_to_module(
        f=fn, 
        mod=jax.lax,
        fn_name=orig,
    )
