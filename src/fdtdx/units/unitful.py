from dataclasses import dataclass
import jax
import jax.numpy as jnp

from fdtdx.core.fraction import Fraction
from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field

from plum import dispatch, overload
from pytreeclass import tree_repr

from fdtdx.typing import SI, PhysicalArrayLike
from fdtdx.units.utils import best_scale, handle_different_scales, patch_fn_to_module

@autoinit
class Unit(TreeClass):
    scale: int = frozen_field()
    dim: dict[SI, int | Fraction] = frozen_field()

    def __str__(self) -> str:
        res_str = f"10^{self.scale} " if self.scale != 0 else ""
        for k, v in self.dim.items():
            res_str += f"{k.name}^{v} " if v != 1 else f"{k.name} "
        return res_str[:-1]

    def __repr__(self) -> str:
        return str(self)


@autoinit
class Unitful(TreeClass):
    val: PhysicalArrayLike
    unit: Unit = frozen_field()
    
    def __post_init__(self):
        optimized_val, power = best_scale(self.val)
        new_scale = self.unit.scale - power
        self.val = optimized_val
        self.unit = Unit(scale=new_scale, dim=self.unit.dim)
    
    def materialise(self) -> PhysicalArrayLike:
        if self.unit.dim:
            raise ValueError(f"Cannot materialise unitful array with a non-zero unit: {self.unit}")
        factor = 10 ** self.unit.scale
        return self.val * factor
    
    def value(self) -> PhysicalArrayLike:
        scale = self.unit.scale
        if isinstance(scale, Fraction):
            scale = scale.value()
        return self.val * 10 ** scale

    def __str__(self) -> str:
        return f"Unitful [{self.unit}]: {tree_repr(self.val)}"

    def __repr__(self) -> str:
        return str(self)
    
    def __mul__(self, other: PhysicalArrayLike | "Unitful") -> "Unitful":
        return multiply(self, other)
    
    def __rmul__(self, other: PhysicalArrayLike | "Unitful") -> "Unitful":
        return multiply(self, other)
    
    def __add__(self, other: "Unitful") -> "Unitful":
        return add(self, other)
    
    def __radd__(self, other: "Unitful") -> "Unitful":
        return add(self, other)
    
    def __sub__(self, other: "Unitful") -> "Unitful":
        return subtract(self, other)
    
    def __rsub__(self, other: "Unitful") -> "Unitful":
        return subtract(other, self)
    
    def __lt__(self, other: "Unitful") -> PhysicalArrayLike:
        return lt(self, other)
    
    def __le__(self, other: "Unitful") -> PhysicalArrayLike:
        return le(self, other)
    
    def __eq__(self, other: "Unitful") -> PhysicalArrayLike:
        return eq(self, other)
    
    def __ne__(self, other: "Unitful") -> PhysicalArrayLike:  # type: ignore
        return ne(self, other)
    
    def __ge__(self, other: "Unitful") -> PhysicalArrayLike:
        return ge(self, other)
    
    def __gt__(self, other: "Unitful") -> PhysicalArrayLike:
        return gt(self, other)
    
    def __len__(self):
        if isinstance(self.val, int | float | complex):
            return 1
        return len(self.val)
    
    def __getitem__(self, key) -> "Unitful":
        """Enable numpy-style indexing"""
        if isinstance(self.val, int | float | complex):
            raise Exception(f"Cannot slice Unitful with python scalar value ({self.val})")
        new_val = self.val[key]
        return Unitful(val=new_val, unit=self.unit)


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
        u1 = u1.aset("val", u1.val * factor1)
        u1 = u1.aset("unit->scale", new_scale)
    if new_scale != u2.unit.scale:
        u2 = u2.aset("val", u2.val * factor2)
        u2 = u2.aset("unit->scale", new_scale)
    return u1, u2


## Multiplication ###########################
@overload
def multiply(
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
    new_val = x.val * y.val
    new_scale = x.unit.scale + y.unit.scale
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=unit_dict))

@overload
def multiply(
    x: PhysicalArrayLike, 
    y: Unitful
) -> Unitful:
    new_val = y.val * x
    return Unitful(val=new_val, unit=y.unit)

@overload
def multiply(
    x: Unitful, 
    y: PhysicalArrayLike
) -> Unitful:
    return multiply(y, x)

@overload
def multiply(x: int, y: int) -> int: return x * y

@overload
def multiply(x: float, y: float) -> float: return x * y

@overload
def multiply(x: complex, y: complex) -> complex: return x * y

@overload
def multiply(x: jax.Array, y: jax.Array) -> jax.Array: return x * y

@dispatch
def multiply(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## Addition ###########################
def _same_dim_binary_fn(x: Unitful, y: Unitful, op_str: str) -> Unitful:
    if x.unit.dim != y.unit.dim:
        raise ValueError(f"Cannot {op_str} two arrays with units {x.unit} and {y.unit}.")
    x_align, y_align = align_scales(x, y)
    orig_fn = getattr(jax.numpy, op_str)
    new_val = orig_fn(x_align.val, y_align.val)
    return Unitful(val=new_val, unit=x_align.unit)

@overload
def add(
    x: Unitful, 
    y: Unitful
) -> Unitful:
    return _same_dim_binary_fn(x, y, "add")

@overload
def add(x: int, y: int) -> int: return x + y

@overload
def add(x: float, y: float) -> float: return x + y

@overload
def add(x: complex, y: complex) -> complex: return x + y

@overload
def add(x: jax.Array, y: jax.Array) -> jax.Array: return x + y

@dispatch
def add(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## Subtractions ###########################
@overload
def subtract(
    x: Unitful, 
    y: Unitful
) -> Unitful:
    return _same_dim_binary_fn(x, y, "subtract")

@overload
def subtract(x: int, y: int) -> int: return x - y

@overload
def subtract(x: float, y: float) -> float: return x - y

@overload
def subtract(x: complex, y: complex) -> complex: return x - y

@overload
def subtract(x: jax.Array, y: jax.Array) -> jax.Array: return x - y

@dispatch
def subtract(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## Remainder ###########################
@overload
def remainder(
    x: Unitful, 
    y: Unitful
) -> Unitful:
    return _same_dim_binary_fn(x, y, "remainder")

@overload
def remainder(x: int, y: int) -> int: return x % y

@overload
def remainder(x: float, y: float) -> float: return x % y

@overload
def remainder(x: jax.Array, y: jax.Array) -> jax.Array: return x % y

@dispatch
def remainder(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## less than ##########################
def _same_dim_binary_fn_array_return(x: Unitful, y: Unitful, op_str: str) -> PhysicalArrayLike:
    if x.unit.dim != y.unit.dim:
        raise ValueError(f"Cannot {op_str} two arrays with units {x.unit} and {y.unit}.")
    x, y = align_scales(x, y)
    orig_fn = getattr(jax.lax, op_str)
    new_val = orig_fn(x.val, y.val)
    return new_val

@overload
def lt(
    x: Unitful, 
    y: Unitful
) -> PhysicalArrayLike:
    return _same_dim_binary_fn_array_return(x, y, "lt")

@overload
def lt(x: int | float, y: int | float) -> bool: return x < y

@overload
def lt(x: jax.Array, y: jax.Array) -> jax.Array: return x < y

@dispatch
def lt(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## less equal ##########################
@overload
def le(
    x: Unitful, 
    y: Unitful
) -> PhysicalArrayLike:
    return _same_dim_binary_fn_array_return(x, y, "le")

@overload
def le(x: int | float, y: int | float) -> bool: return x <= y

@overload
def le(x: jax.Array, y: jax.Array) -> jax.Array: return x <= y

@dispatch
def le(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## equal ##########################
@overload
def eq(
    x: Unitful, 
    y: Unitful
) -> PhysicalArrayLike:
    return _same_dim_binary_fn_array_return(x, y, "eq")

@overload
def eq(x: int | float | complex, y: int | float | complex) -> bool: return x == y

@overload
def eq(x: jax.Array, y: jax.Array) -> jax.Array: return x == y

@dispatch
def eq(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## not equal ##########################
@overload
def ne(
    x: Unitful, 
    y: Unitful
) -> PhysicalArrayLike:
    return _same_dim_binary_fn_array_return(x, y, "ne")

@overload
def ne(x: int | float | complex, y: int | float | complex) -> bool: return x != y

@overload
def ne(x: jax.Array, y: jax.Array) -> jax.Array: return x != y

@dispatch
def ne(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## greater equal ##########################
@overload
def ge(
    x: Unitful, 
    y: Unitful
) -> PhysicalArrayLike:
    return _same_dim_binary_fn_array_return(x, y, "ge")

@overload
def ge(x: int | float, y: int | float) -> bool: return x >= y

@overload
def ge(x: jax.Array, y: jax.Array) -> jax.Array: return x >= y

@dispatch
def ge(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()

## greater than ##########################
@overload
def gt(
    x: Unitful, 
    y: Unitful
) -> PhysicalArrayLike:
    return _same_dim_binary_fn_array_return(x, y, "gt")

@overload
def gt(x: int | float, y: int | float) -> bool: return x > y

@overload
def gt(x: jax.Array, y: jax.Array) -> jax.Array: return x > y

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