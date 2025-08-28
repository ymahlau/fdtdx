from dataclasses import dataclass
from typing import Any, Self
import jax
import jax.numpy as jnp

from fdtdx.core.fraction import Fraction
from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field

from plum import dispatch, overload
from pytreeclass import tree_repr

from fdtdx.units.typing import SI, PhysicalArrayLike
from fdtdx.units.utils import best_scale, handle_different_scales

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


@dataclass(frozen=True)
class UnitfulIndexer:
    unitful: "Unitful"
    where: Any | None = None

    def __getitem__(self, where: Any) -> "UnitfulIndexer":
        if self.where is not None:
            raise Exception("Already called [] on Unitful.at! Double Indexing [][] is currently not supported")
        return UnitfulIndexer(self.unitful, where)
    
    def get(self) -> "Unitful":
        """Get the leaf values at the specified location."""
        if self.where is None:
            return self.unitful
        if isinstance(self.unitful.val, int | float | complex):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        new_val = self.unitful.val[self.where]
        return Unitful(val=new_val, unit=self.unitful.unit)
    
    def set(self, value: "Unitful") -> "Unitful":
        """Set the leaf values at the specified location."""
        if self.where is None:
            raise Exception(f"Cannot update value if no where clause is given")
        if isinstance(self.unitful.val, int | float | complex):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        if self.unitful.unit.dim != value.unit.dim:
            raise Exception(
                f"Cannot update array value with different unit: {self.unitful.unit.dim} != {value.unit.dim}"
            )
        align_self, align_other = align_scales(self.unitful, value)
        align_self_arr = align_self.val
        assert not isinstance(align_self_arr, int | complex | float)
        new_val: jax.Array = align_self_arr.at[self.where].set(align_other.val)
        return Unitful(val=new_val, unit=align_self.unit)
    
    def add(self, value: "Unitful") -> "Unitful":
        """Set the leaf values at the specified location."""
        if self.where is None:
            raise Exception(f"Cannot update value if no where clause is given")
        if isinstance(self.unitful.val, int | float | complex):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        if self.unitful.unit.dim != value.unit.dim:
            raise Exception(
                f"Cannot update array value with different unit: {self.unitful.unit.dim} != {value.unit.dim}"
            )
        align_self, align_other = align_scales(self.unitful, value)
        align_self_arr = align_self.val
        assert not isinstance(align_self_arr, int | complex | float)
        new_val: jax.Array = align_self_arr.at[self.where].add(align_other.val)
        return Unitful(val=new_val, unit=align_self.unit)
    
    def subtract(self, value: "Unitful") -> "Unitful":
        """Set the leaf values at the specified location."""
        if self.where is None:
            raise Exception(f"Cannot update value if no where clause is given")
        if isinstance(self.unitful.val, int | float | complex):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        if self.unitful.unit.dim != value.unit.dim:
            raise Exception(
                f"Cannot update array value with different unit: {self.unitful.unit.dim} != {value.unit.dim}"
            )
        align_self, align_other = align_scales(self.unitful, value)
        align_self_arr = align_self.val
        assert not isinstance(align_self_arr, int | complex | float)
        new_val: jax.Array = align_self_arr.at[self.where].subtract(align_other.val)
        return Unitful(val=new_val, unit=align_self.unit)
    
    def multiply(self, value: jax.Array) -> "Unitful":
        if isinstance(value, Unitful):
            raise Exception(
                f"Multiplying part of an array with another Unitful would lead to different units within the Array!"
            )
        if self.where is None:
            raise Exception(f"Cannot update value if no where clause is given")
        if isinstance(self.unitful.val, int | float | complex):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        new_val = self.unitful.val.at[self.where].multiply(value)
        return Unitful(val=new_val, unit=self.unitful.unit)
    
    def divide(self, value: jax.Array) -> "Unitful":
        if isinstance(value, Unitful):
            raise Exception(
                f"Multiplying part of an array with another Unitful would lead to different units within the Array!"
            )
        if self.where is None:
            raise Exception(f"Cannot update value if no where clause is given")
        if isinstance(self.unitful.val, int | float | complex):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        new_val = self.unitful.val.at[self.where].divide(value)
        return Unitful(val=new_val, unit=self.unitful.unit)
    
    def power(self, value: Any) -> "Unitful":
        del value
        raise Exception(
            f"Raising part of an array to a power is an undefined operation for a Unitful"
        )


@autoinit
class Unitful(TreeClass):
    val: PhysicalArrayLike
    unit: Unit = field()
    optimize_scale: bool = frozen_field(default=True)
    
    def __post_init__(self):
        if self.optimize_scale:
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
    
    @property
    def at(self) -> UnitfulIndexer:  # type: ignore
        """Gets the indexer for this tree.

        Returns:
            UnitfulIndexer: Indexer that preserves type information
        """
        return UnitfulIndexer(self)
    
    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.val, int | float | complex):
            return ()
        return self.val.shape
    
    @property
    def dtype(self):
        if isinstance(self.val, int | float | complex):
            raise Exception(f"Python scalar does not have dtype attribute")
        return self.val.dtype
    
    @property
    def ndim(self):
        if isinstance(self.val, int | float | complex):
            return 0
        return self.val.ndim
    
    @property
    def size(self):
        if isinstance(self.val, int | float | complex):
            return 1
        return self.val.size
    
    def aset(
        self,
        attr_name: str,
        val: Any,
        create_new_ok: bool = False,
    ) -> Self:
        del attr_name, val, create_new_ok
        raise Exception(
            "the aset-method is unsafe for Unitful and therefore not implemented. Please use .at[].set() intead."
        )

    def __str__(self) -> str:
        return f"Unitful [{self.unit}]: {tree_repr(self.val)}"

    def __repr__(self) -> str:
        return str(self)
    
    def __mul__(self, other: PhysicalArrayLike | "Unitful") -> "Unitful":
        return multiply(self, other)
    
    def __rmul__(self, other: PhysicalArrayLike | "Unitful") -> "Unitful":
        return multiply(other, self)
    
    def __truediv__(self, other: PhysicalArrayLike | "Unitful") -> "Unitful":
        return divide(self, other)
    
    def __rtruediv__(self, other: PhysicalArrayLike | "Unitful") -> "Unitful":
        return divide(other, self)
    
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
    
    def __iter__(self):
        """Use a generator for simplicity"""
        if isinstance(self.val, int | float | complex):
            raise Exception(f"Cannot iterate over Unitful with python scalar value ({self.val})")
        for v in self.val:
            yield(Unitful(val=v, unit=self.unit))
    
    def __reversed__(self):
        return iter(self[::-1])
    
    def __neg__(self):
        return Unitful(val=-self.val, unit=self.unit)

    def __pos__(self):
        """Unary plus: +x"""
        return Unitful(val=+self.val, unit=self.unit)

    def __abs__(self):
        return abs(self)
    
    def __matmul__(self, other: "Unitful") -> "Unitful":
        return matmul(self, other)
    
    def __pow__(self, other: int) -> "Unitful":
        return pow(self, other)
    
    def min(self, **kwargs) -> "Unitful":
        return min(self, **kwargs)
    
    def max(self, **kwargs) -> "Unitful":
        return max(self, **kwargs)
    
    def mean(self, **kwargs) -> "Unitful":
        return mean(self, **kwargs)
    
    def sum(self, **kwargs) -> "Unitful":
        return sum(self, **kwargs)
        


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
            unit=Unit(scale=new_scale, dim=u1.unit.dim),
            optimize_scale=False,
        )
    if new_scale != u2.unit.scale:
        u2 = Unitful(
            val=u2.val * factor2,
            unit=Unit(scale=new_scale, dim=u2.unit.dim),
            optimize_scale=False,
        )
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


## Division ###########################
@overload
def divide(
    x1: Unitful,
    x2: Unitful
) -> Unitful:
    unit_dict = x1.unit.dim.copy()
    for k, v in x2.unit.dim.items():
        if k in unit_dict:
            unit_dict[k] -= v
            if unit_dict[k] == 0:
                del unit_dict[k]
        else:
            unit_dict[k] = -v
    new_val = x1.val / x2.val
    new_scale = x1.unit.scale - x2.unit.scale
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=unit_dict))

@overload
def divide(
    x1: PhysicalArrayLike, 
    x2: Unitful
) -> Unitful:
    new_val = x1 / x2.val
    new_dim = {k: -v for k, v in x2.unit.dim.items()}
    new_scale = -x2.unit.scale 
    return Unitful(val=new_val, unit=Unit(dim=new_dim, scale=new_scale))

@overload
def divide(
    x1: Unitful, 
    x2: PhysicalArrayLike
) -> Unitful:
    new_val = x1.val / x2
    return Unitful(val=new_val, unit=x1.unit)

@overload
def divide(x1: int, x2: int) -> float: return x1 / x2

@overload
def divide(x1: float, x2: float) -> float: return x1 / x2

@overload
def divide(x1: complex, x2: complex) -> complex: return x1 / x2

@overload
def divide(x1: jax.Array, x2: jax.Array) -> jax.Array: return x1 / x2

@dispatch
def divide(x1, x2):  # type: ignore
    del x1, x2
    raise NotImplementedError()


## Addition ###########################
def _same_dim_binary_fn(x: Unitful, y: Unitful, op_str: str) -> Unitful:
    if x.unit.dim != y.unit.dim:
        raise ValueError(f"Cannot {op_str} two arrays with units {x.unit} and {y.unit}.")
    x_align, y_align = align_scales(x, y)
    orig_fn = getattr(jax.numpy, f"_orig_{op_str}")
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

## Matrix Multiplication ###################################
@overload
def matmul(
    x: Unitful, 
    y: Unitful,
    **kwargs
) -> Unitful:
    if x.unit.dim != y.unit.dim:
        raise ValueError(f"Cannot matmul two arrays with units {x.unit} and {y.unit}.")
    x_align, y_align = align_scales(x, y)
    new_val = jnp._orig_matmul(x.val, y.val)  # type: ignore
    new_dim = {k: x_align.unit.dim[k] + y_align.unit.dim[k] for k in x_align.unit.dim.keys()}
    return Unitful(val=new_val, unit=Unit(scale=x_align.unit.scale, dim=new_dim))

@overload
def matmul(x: jax.Array, y: jax.Array, **kwargs) -> jax.Array:
    return jnp._orig_matmul(x, y, **kwargs)  # type: ignore

@dispatch
def matmul(x, y):  # type: ignore
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
    orig_fn = getattr(jax.lax, f"_orig_{op_str}")
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


## power #######################################
@overload
def pow(x: Unitful, y: int) -> Unitful:
    new_dim = {}
    for k, v in x.unit.dim.items():
        new_dim[k] = v * y
    new_val = x.val ** y
    new_scale = x.unit.scale * y
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=new_dim))

@overload
def pow(x: jax.Array, y: jax.Array) -> jax.Array: return x ** y

@dispatch
def pow(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## min #######################################
def unary_fn(x: Unitful, op_str: str, **kwargs) -> Unitful:
    orig_fn = getattr(jax.numpy, f"_orig_{op_str}")
    new_val = orig_fn(x.val, **kwargs)
    if not isinstance(new_val, PhysicalArrayLike):
        raise Exception(f"This is an internal error: {op_str} produced {type(new_val)}")
    return Unitful(val=new_val, unit=x.unit)

@overload
def min(x: Unitful, **kwargs) -> Unitful: return unary_fn(x, "min", **kwargs)

@overload
def min(x: jax.Array, **kwargs) -> jax.Array: return jnp._orig_min(x, **kwargs)  # type: ignore

@dispatch
def min(x, **kwargs):  # type: ignore
    del x, kwargs
    raise NotImplementedError()

## max #######################################
@overload
def max(x: Unitful, **kwargs) -> Unitful: return unary_fn(x, "max", **kwargs)

@overload
def max(x: jax.Array, **kwargs) -> jax.Array: return jnp._orig_max(x, **kwargs)  # type: ignore

@dispatch
def max(x, **kwargs):  # type: ignore
    del x, kwargs
    raise NotImplementedError()

## mean #######################################
@overload
def mean(x: Unitful, **kwargs) -> Unitful: return unary_fn(x, "mean", **kwargs)

@overload
def mean(x: jax.Array, **kwargs) -> jax.Array: return jnp._orig_mean(x, **kwargs)  # type: ignore

@dispatch
def mean(x, **kwargs):  # type: ignore
    del x, kwargs
    raise NotImplementedError()

## mean #######################################
@overload
def sum(x: Unitful, **kwargs) -> Unitful: return unary_fn(x, "sum", **kwargs)

@overload
def sum(x: jax.Array, **kwargs) -> jax.Array: return jnp._orig_sum(x, **kwargs)  # type: ignore

@dispatch
def sum(x, **kwargs):  # type: ignore
    del x, kwargs
    raise NotImplementedError()


## abs #######################################
@overload
def abs(x: Unitful) -> Unitful: return unary_fn(x, "abs")

@overload
def abs(x: jax.Array) -> jax.Array: return jnp._orig_abs(x)  # type: ignore

@overload
def abs(x: int) -> int: return abs(x)

@overload
def abs(x: float) -> float: return abs(x)

@overload
def abs(x: complex) -> complex: return abs(x)

@dispatch
def abs(x):  # type: ignore
    del x
    raise NotImplementedError()

