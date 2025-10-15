# ruff: noqa: F811

from dataclasses import dataclass
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike
from plum import add_conversion_method, dispatch, overload
from pytreeclass import tree_repr

from fdtdx.core.fraction import Fraction
from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field
from fdtdx.core.jax.utils import is_currently_compiling, is_traced
from fdtdx.units.typing import (
    PHYSICAL_DTYPES,
    SI,
    NonPhysicalArrayLike,
    PhysicalArrayLike,
    RealPhysicalArrayLike,
    StaticArrayLike,
    StaticPhysicalArrayLike,
)
from fdtdx.units.utils import best_scale, dim_after_multiplication, handle_different_scales

""" This determines the maximum size where an array can be saved statically for optimization during jit tracing"""
MAX_STATIC_OPTIMIZED_SIZE = int(1e5)

""" Global optimization stop flag. If True, then no arrays are statically saved. Additionally, no jax functions 
like jnp.ones will return a Unitful instead of Array for scale optimization.
"""
STATIC_OPTIM_STOP_FLAG = False


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


EMPTY_UNIT = Unit(scale=0, dim={})


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
        if not isinstance(self.unitful.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        new_val = self.unitful.val[self.where]
        return Unitful(val=new_val, unit=self.unitful.unit)

    def set(self, value: "Unitful") -> "Unitful":
        """Set the leaf values at the specified location."""
        if self.where is None:
            raise Exception("Cannot update value if no where clause is given")
        if not isinstance(self.unitful.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        if self.unitful.unit.dim != value.unit.dim:
            raise Exception(
                f"Cannot update array value with different unit: {self.unitful.unit.dim} != {value.unit.dim}"
            )
        align_self, align_other = align_scales(self.unitful, value)
        align_self_arr = align_self.val
        assert isinstance(align_self_arr, jax.Array | np.ndarray)
        if isinstance(align_self_arr, np.ndarray):
            new_val = np.copy(align_self_arr)
            new_val[self.where] = align_other.val
        else:
            new_val = align_self_arr.at[self.where].set(align_other.val)
        return Unitful(val=new_val, unit=align_self.unit)

    def add(self, value: "Unitful") -> "Unitful":
        """Set the leaf values at the specified location."""
        if self.where is None:
            raise Exception("Cannot update value if no where clause is given")
        if not isinstance(self.unitful.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        if self.unitful.unit.dim != value.unit.dim:
            raise Exception(
                f"Cannot update array value with different unit: {self.unitful.unit.dim} != {value.unit.dim}"
            )
        align_self, align_other = align_scales(self.unitful, value)
        align_self_arr = align_self.val
        assert isinstance(align_self_arr, jax.Array | np.ndarray)
        if isinstance(align_self_arr, np.ndarray):
            new_val = np.copy(align_self_arr)
            new_val[self.where] += align_other.val
        else:
            new_val = align_self_arr.at[self.where].add(align_other.val)
        return Unitful(val=new_val, unit=align_self.unit)

    def subtract(self, value: "Unitful") -> "Unitful":
        """Set the leaf values at the specified location."""
        if self.where is None:
            raise Exception("Cannot update value if no where clause is given")
        if not isinstance(self.unitful.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        if self.unitful.unit.dim != value.unit.dim:
            raise Exception(
                f"Cannot update array value with different unit: {self.unitful.unit.dim} != {value.unit.dim}"
            )
        align_self, align_other = align_scales(self.unitful, value)
        align_self_arr = align_self.val
        assert isinstance(align_self_arr, jax.Array | np.ndarray)
        if isinstance(align_self_arr, np.ndarray):
            new_val = np.copy(align_self_arr)
            new_val[self.where] += align_other.val
        else:
            new_val = align_self_arr.at[self.where].subtract(align_other.val)
        return Unitful(val=new_val, unit=align_self.unit)

    def multiply(self, value: jax.Array) -> "Unitful":
        if isinstance(value, Unitful):
            raise Exception(
                "Multiplying part of an array with another Unitful would lead to different units within the Array!"
            )
        if self.where is None:
            raise Exception("Cannot update value if no where clause is given")
        if not isinstance(self.unitful.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        v = self.unitful.val
        if isinstance(v, np.ndarray):
            new_val = np.copy(self.unitful.val)
            new_val[self.where] *= value
        else:
            new_val = v.at[self.where].multiply(value)
        return Unitful(val=new_val, unit=self.unitful.unit)

    def divide(self, value: jax.Array) -> "Unitful":
        if isinstance(value, Unitful):
            raise Exception(
                "Multiplying part of an array with another Unitful would lead to different units within the Array!"
            )
        if self.where is None:
            raise Exception("Cannot update value if no where clause is given")
        if not isinstance(self.unitful.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot index scalar value: {self.unitful}")
        v = self.unitful.val
        if isinstance(v, np.ndarray):
            new_val = np.copy(self.unitful.val)
            new_val[self.where] /= value
        else:
            new_val = v.at[self.where].divide(value)
        return Unitful(val=new_val, unit=self.unitful.unit)

    def power(self, value: Any) -> "Unitful":
        del value
        raise Exception("Raising part of an array to a power is an undefined operation for a Unitful")


@autoinit
class Unitful(TreeClass):
    val: ArrayLike
    unit: Unit = field()
    optimize_scale: bool = frozen_field(default=True)
    static_arr: StaticArrayLike | None = frozen_field(default=None)

    def _validate(self):
        bad_dtype = isinstance(self.val, jax.Array | np.ndarray | np.number) and self.dtype not in PHYSICAL_DTYPES
        if isinstance(self.val, NonPhysicalArrayLike) or bad_dtype:
            if self.unit.scale != 0:
                if isinstance(self.val, int):
                    raise Exception(f"Cannot have non-zero scale for integer {self}. Consider using float as input.")
                else:
                    raise Exception(f"Cannot have non-zero scale for non-physical {self}")
            if self.unit.dim != {}:
                if isinstance(self.val, int):
                    raise Exception(
                        f"Cannot have non-empty dimension for integer {self}. Consider using float as input."
                    )
                else:
                    raise Exception(f"Cannot have non-empty dimension for non-physical {self}")

    def __post_init__(self):
        self._validate()
        if not self.optimize_scale or not can_optimize_scale(self):
            return
        orig_val = self.val
        if is_traced(self.val):
            # if value is traced then optimize with static array
            assert self.static_arr is not None and isinstance(self.static_arr, StaticPhysicalArrayLike)
            optimized_val, power = best_scale(self.static_arr, self.unit.scale)
            assert isinstance(optimized_val, StaticPhysicalArrayLike)
            self.static_arr = optimized_val
            self.val = self.val * (10**power)
        else:
            # non-traced case: optimize scale
            assert isinstance(self.val, PhysicalArrayLike)
            optimized_val, power = best_scale(self.val, self.unit.scale)
            self.val = optimized_val
        new_scale = self.unit.scale - power
        self.unit = Unit(scale=new_scale, dim=self.unit.dim)
        # special case: The value might have been static jax array within jit context and scale optimization converted
        # it to tracer. In this case, create a static array from the original array
        if not is_traced(orig_val) and is_traced(self.val):
            self.static_arr = get_static_operand(orig_val) * (10**power)

    def materialise(self) -> ArrayLike:
        if self.unit.dim:
            raise ValueError(f"Cannot materialise unitful array with a non-zero unit: {self.unit}")
        return self.value()

    def float_materialise(self) -> float:
        v = self.materialise()
        assert isinstance(v, float), f"safe float_materialise called on Unitful with non-float value: {self}"
        return v

    def array_materialise(self) -> jax.Array:
        v = self.materialise()
        assert isinstance(v, jax.Array), f"safe array_materialise called on Unitful with non-array value: {self}"
        return v

    def value(self) -> ArrayLike:
        scale = self.unit.scale
        if isinstance(scale, Fraction):
            scale = scale.value()
        if scale == 0:
            return self.val
        return self.val * (10**scale)

    def array_value(self) -> jax.Array:
        v = self.value()
        assert isinstance(v, jax.Array), f"safe array_value called on Unitful with non-array value: {self}"
        return v

    def float_value(self) -> float:
        v = self.value()
        assert isinstance(v, float), f"safe float_value called on Unitful with non-float value: {self}"
        return v

    def static_value(self) -> StaticArrayLike | None:
        if self.static_arr is None:
            return None
        scale = self.unit.scale
        if isinstance(scale, Fraction):
            scale = scale.value()
        return self.static_arr * (10**scale)

    @property
    def at(self) -> UnitfulIndexer:  # type: ignore
        """Gets the indexer for this tree.

        Returns:
            UnitfulIndexer: Indexer that preserves type information
        """
        return UnitfulIndexer(self)

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.val, int | float | complex | bool):
            return ()
        return self.val.shape

    @property
    def dtype(self):
        if isinstance(self.val, int | float | complex | bool):
            raise Exception("Python scalar does not have dtype attribute")
        return self.val.dtype

    @property
    def ndim(self):
        if isinstance(self.val, int | float | complex | bool):
            return 0
        return self.val.ndim

    @property
    def size(self):
        if isinstance(self.val, int | float | complex | bool):
            return 1
        return self.val.size

    @property
    def T(self):
        if not isinstance(self.val, jax.Array | np.ndarray | np.number | np.bool):
            raise Exception(f"Cannot call .T on {self}")
        new_val = self.val.T
        new_static_arr = None
        if is_traced(new_val):
            arr = get_static_operand(self)
            if arr is not None:
                assert isinstance(arr, jax.Array | np.ndarray | np.number | np.bool)
                new_static_arr = arr.T
        return Unitful(val=new_val, unit=self.unit, static_arr=new_static_arr)

    def aset(
        self,
        attr_name: str,
        val: Any,
        create_new_ok: bool = False,
    ) -> Self:
        del attr_name, val, create_new_ok
        raise Exception(
            "the aset-method is unsafe for Unitful internals and therefore not implemented. "
            "Please use .at[].set() intead. Note that using aset on structures containing unitful is safe and "
            " implemented, just the internals of Unitful should not changed this way."
        )

    def astype(self, *args, **kwargs) -> "Unitful":
        return astype(self, *args, **kwargs)

    def squeeze(
        self,
        axis: int | None = None,
    ) -> "Unitful":
        return squeeze(self, axis)

    def reshape(
        self,
        *args,
        **kwargs,
    ) -> "Unitful":
        return reshape(self, args, **kwargs)

    def __str__(self) -> str:
        try:
            return f"Unitful [{self.unit}]: {tree_repr(self.val)}"
        except Exception:
            return f"Unitful [{self.unit}]: {self.shape}"

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

    def __add__(self, other: "Unitful | PhysicalArrayLike") -> "Unitful":
        return add(self, other)

    def __radd__(self, other: "Unitful | PhysicalArrayLike") -> "Unitful":
        return add(self, other)

    def __sub__(self, other: "Unitful | PhysicalArrayLike") -> "Unitful":
        return subtract(self, other)

    def __rsub__(self, other: "Unitful | PhysicalArrayLike") -> "Unitful":
        return subtract(other, self)

    def __lt__(self, other: "Unitful | RealPhysicalArrayLike") -> np.bool | np.ndarray | jax.Array | bool:
        return lt(self, other)

    def __le__(self, other: "Unitful | RealPhysicalArrayLike") -> np.bool | np.ndarray | jax.Array | bool:
        return le(self, other)

    def __eq__(self, other: "Unitful | RealPhysicalArrayLike") -> np.bool | np.ndarray | jax.Array | bool:
        return eq(self, other)

    def __ne__(self, other: "Unitful | RealPhysicalArrayLike") -> np.bool | np.ndarray | jax.Array | bool:  # type: ignore
        return ne(self, other)

    def __ge__(self, other: "Unitful | RealPhysicalArrayLike") -> np.bool | np.ndarray | jax.Array | bool:
        return ge(self, other)

    def __gt__(self, other: "Unitful | RealPhysicalArrayLike") -> np.bool | np.ndarray | jax.Array | bool:
        return gt(self, other)

    def __len__(self):
        if not isinstance(self.val, jax.Array | np.ndarray):
            return 1
        return len(self.val)

    def __getitem__(self, key: Any) -> "Unitful":
        """Enable numpy-style indexing"""
        if not isinstance(self.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot slice Unitful with python scalar value ({self.val})")
        if isinstance(key, Unitful):
            key = key.materialise()
        if isinstance(key, tuple):
            new_list = []
            for k in key:
                if isinstance(k, Unitful):
                    new_list.append(k.materialise())
                else:
                    new_list.append(k)
            key = tuple(new_list)
        new_val = self.val[key]
        return Unitful(val=new_val, unit=self.unit)

    def __iter__(self):
        """Use a generator for simplicity"""
        if not isinstance(self.val, jax.Array | np.ndarray):
            raise Exception(f"Cannot iterate over Unitful with python scalar value ({self.val})")
        for v in self.val:
            yield (Unitful(val=v, unit=self.unit))

    def __reversed__(self):
        return iter(self[::-1])

    def __neg__(self):
        if isinstance(self.val, NonPhysicalArrayLike):
            raise Exception(f"Cannot perform negation on non-physcal value {self}")
        return Unitful(val=-self.val, unit=self.unit)

    def __pos__(self):
        """Unary plus: +x"""
        if isinstance(self.val, NonPhysicalArrayLike):
            raise Exception(f"Cannot perform unary plus on non-physcal value {self}")
        return Unitful(val=+self.val, unit=self.unit)

    def __abs__(self):
        return abs_impl(self)

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

    def prod(self, **kwargs) -> "Unitful":
        return prod(self, **kwargs)

    def argmax(self, **kwargs) -> "Unitful":
        return argmax(self, **kwargs)

    def argmin(self, **kwargs) -> "Unitful":
        return argmin(self, **kwargs)


def align_scales(
    u1: Unitful,
    u2: Unitful,
) -> tuple[Unitful, Unitful]:
    if u1.unit.dim != u2.unit.dim:
        raise Exception("Cannot align arrays with different units")
    # non physical ArrayLikes need to keep scale 0
    force_zero_scale = False
    if not u1.optimize_scale or not can_optimize_scale(u1):
        assert u1.unit.scale == 0
        force_zero_scale = True
    if not u2.optimize_scale or not can_optimize_scale(u2):
        assert u2.unit.scale == 0
        force_zero_scale = True
    # calculate new scale
    if force_zero_scale:
        new_scale, factor1, factor2 = 0, 10**u1.unit.scale, 10**u2.unit.scale
    else:
        new_scale, factor1, factor2 = handle_different_scales(
            u1.unit.scale,
            u2.unit.scale,
        )
    # update unitfuls
    if new_scale != u1.unit.scale:
        u1 = Unitful(
            val=u1.val * factor1,
            unit=Unit(scale=new_scale, dim=u1.unit.dim),
            optimize_scale=False,
            static_arr=None if u1.static_arr is None else u1.static_arr * factor1,
        )
    if new_scale != u2.unit.scale:
        u2 = Unitful(
            val=u2.val * factor2,
            unit=Unit(scale=new_scale, dim=u2.unit.dim),
            optimize_scale=False,
            static_arr=None if u2.static_arr is None else u2.static_arr * factor2,
        )
    return u1, u2


def get_static_operand(
    x: Unitful | ArrayLike,
) -> StaticArrayLike | None:
    if STATIC_OPTIM_STOP_FLAG:
        return None
    # Physical arraylike without a unit
    if isinstance(x, ArrayLike):
        if is_traced(x):
            return None
        if isinstance(x, jax.Array):
            if x.size >= MAX_STATIC_OPTIMIZED_SIZE:
                return None
            return np.asarray(x, copy=True)
        assert isinstance(x, StaticArrayLike), "Internal error, please report"
        return x
    # Unitful
    x_arr = None
    if x.static_arr is not None:
        x_arr = x.static_arr
    elif not is_traced(x.val):
        x_arr = x.val
        if isinstance(x_arr, jax.Array):
            if x_arr.size >= MAX_STATIC_OPTIMIZED_SIZE:
                return None
            x_arr = np.asarray(x_arr, copy=True)
    return x_arr


def can_perform_scatic_ops(x: StaticArrayLike | None):
    if x is None:
        return False
    if isinstance(x, NonPhysicalArrayLike):
        return False
    return True


def output_unitful_for_array(static_arr: ArrayLike | jax.ShapeDtypeStruct | None) -> bool:
    if static_arr is None:
        return False
    if is_traced(static_arr):
        return False
    if isinstance(static_arr, jax.Array | np.ndarray | jax.ShapeDtypeStruct):
        if static_arr.size > MAX_STATIC_OPTIMIZED_SIZE:
            return False
    return is_currently_compiling() and not STATIC_OPTIM_STOP_FLAG


def can_optimize_scale(obj: Unitful | ArrayLike) -> bool:
    v = obj.val if isinstance(obj, Unitful) else obj
    if isinstance(v, NonPhysicalArrayLike):
        return False
    if isinstance(v, jax.Array | np.ndarray):
        if v.dtype not in PHYSICAL_DTYPES:
            return False
    if is_traced(v) and not isinstance(obj, Unitful):
        return False
    if is_traced(v):
        assert isinstance(obj, Unitful)
        if obj.static_arr is None:
            return False
        if not can_optimize_scale(obj.static_arr):
            return False
    return True


# This conversion method is necessary, because within jit-context we lie to the dispatcher.
# Specifically, functions that are supposed to return a jax array will return a unitful to be able to perform
# scale optimization.
def unitful_to_array_conversion(obj: Unitful):
    assert obj.unit.dim == {}
    if is_currently_compiling():
        return obj  # type: ignore
    return obj.array_materialise()


add_conversion_method(type_from=Unitful, type_to=jax.Array, f=unitful_to_array_conversion)


def unitful_to_array_conversion_with_bool(obj: Unitful) -> np.bool | np.ndarray | jax.Array | bool:
    assert obj.unit.dim == {}
    if is_currently_compiling():
        result: np.bool | np.ndarray | jax.Array | bool = obj  # type: ignore
    else:
        result: np.bool | np.ndarray | jax.Array | bool = obj.materialise()  # type: ignore
    return result


add_conversion_method(
    type_from=Unitful,
    type_to=np.bool | np.ndarray | jax.Array | bool,  # type: ignore
    f=unitful_to_array_conversion_with_bool,
)


## Multiplication ###########################
@overload
def multiply(
    x: Unitful,
    y: Unitful,
) -> Unitful:
    unit_dict = dim_after_multiplication(x.unit.dim, y.unit.dim)
    new_val = x.val * y.val
    new_scale = x.unit.scale + y.unit.scale
    # if static arrays exist, perform mul with static arrs
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        y_arr = get_static_operand(y)
        if x_arr is not None and y_arr is not None:
            new_static_arr = x_arr * y_arr
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=unit_dict), static_arr=new_static_arr)


@overload
def multiply(x: ArrayLike, y: Unitful) -> Unitful:
    scale_offset = 0
    if can_optimize_scale(x):
        # optimize scale of x
        unit_x = Unitful(val=x, unit=Unit(scale=0, dim={}))
        x = unit_x.val
        scale_offset = unit_x.unit.scale
    new_val = y.val * x
    new_scale = y.unit.scale + scale_offset
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        y_arr = get_static_operand(y)
        if x_arr is not None and y_arr is not None:
            new_static_arr = x_arr * y_arr
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=y.unit.dim), static_arr=new_static_arr)


@overload
def multiply(
    x: Unitful,
    y: ArrayLike,
) -> Unitful:
    return multiply(y, x)


@overload
def multiply(x: int, y: int) -> int:
    return x * y


@overload
def multiply(x: float, y: float) -> float:
    return x * y


@overload
def multiply(x: complex, y: complex) -> complex:
    return x * y


@overload
def multiply(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * y


@overload
def multiply(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    return x * y


@dispatch
def multiply(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## Division ###########################
@overload
def divide(x1: Unitful, x2: Unitful) -> Unitful:
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
    # if static arrays exist, perform div with static arrs
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x1)
        y_arr = get_static_operand(x2)
        if x_arr is not None and y_arr is not None:
            new_static_arr = x_arr / y_arr
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=unit_dict), static_arr=new_static_arr)


@overload
def divide(x1: ArrayLike, x2: Unitful) -> Unitful:
    new_dim = {k: -v for k, v in x2.unit.dim.items()}
    scale_offset = 0
    if not is_traced(x1):
        unit_x1 = Unitful(val=x1, unit=Unit(scale=0, dim={}))
        x1 = unit_x1.val
        scale_offset = unit_x1.unit.scale
    new_val = x1 / x2.val
    new_scale = scale_offset - x2.unit.scale
    # if static arrays exist, perform div with static arrs
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x1)
        y_arr = get_static_operand(x2)
        if x_arr is not None and y_arr is not None:
            new_static_arr = x_arr / y_arr
    return Unitful(val=new_val, unit=Unit(dim=new_dim, scale=new_scale), static_arr=new_static_arr)


@overload
def divide(x1: Unitful, x2: ArrayLike) -> Unitful:
    scale_offset = 0
    if not is_traced(x2):
        unit_x2 = Unitful(val=x2, unit=Unit(scale=0, dim={}))
        x2 = unit_x2.val
        scale_offset = -unit_x2.unit.scale
    new_val = x1.val / x2
    new_scale = x1.unit.scale - scale_offset
    # if static arrays exist, perform div with static arrs
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x1)
        y_arr = get_static_operand(x2)
        if x_arr is not None and y_arr is not None:
            new_static_arr = x_arr / y_arr
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=x1.unit.dim), static_arr=new_static_arr)


@overload
def divide(x1: int, x2: int) -> float:
    return x1 / x2


@overload
def divide(x1: float, x2: float) -> float:
    return x1 / x2


@overload
def divide(x1: complex, x2: complex) -> complex:
    return x1 / x2


@overload
def divide(x1: jax.Array, x2: jax.Array) -> jax.Array:
    return x1 / x2


@overload
def divide(x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
    return x1 / x2


@dispatch
def divide(x1, x2):  # type: ignore
    del x1, x2
    raise NotImplementedError()


## Addition ###########################
@overload
def add(x: Unitful, y: Unitful) -> Unitful:
    if x.unit.dim != y.unit.dim:
        raise ValueError(f"Cannot add two arrays with units {x.unit} and {y.unit}.")
    x_align, y_align = align_scales(x, y)
    new_val = x_align.val + y_align.val
    # if static arrays exist, perform add with static arrs
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x_align)
        y_arr = get_static_operand(y_align)
        if x_arr is not None and y_arr is not None:
            new_static_arr = x_arr + y_arr
    return Unitful(val=new_val, unit=x_align.unit, static_arr=new_static_arr)


@overload
def add(x: Unitful, y: ArrayLike) -> Unitful:
    if x.unit.dim:
        raise ValueError(f"Cannot add non-unitful to array with unit {x.unit}.")
    y_unitful = Unitful(val=y, unit=Unit(scale=0, dim={}))
    return add(x, y_unitful)


@overload
def add(x: ArrayLike, y: Unitful) -> Unitful:
    return add(y, x)


@overload
def add(x: int, y: int) -> int:
    return x + y


@overload
def add(x: float, y: float) -> float:
    return x + y


@overload
def add(x: complex, y: complex) -> complex:
    return x + y


@overload
def add(x: jax.Array, y: jax.Array) -> jax.Array:
    return x + y


@overload
def add(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    return x + y


@dispatch
def add(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## Matrix Multiplication ###################################
@overload
def matmul(x: Unitful, y: Unitful, **kwargs) -> Unitful:
    # TODO: numpy values should stay numpy if possible
    x_align, y_align = align_scales(x, y)
    new_val = jnp._orig_matmul(x_align.val, y_align.val)  # type: ignore
    unit_dict = dim_after_multiplication(x.unit.dim, y.unit.dim)
    new_scale = 2 * x_align.unit.scale
    # if static arrays exist, perform subtract with static arrs
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x_align)
        y_arr = get_static_operand(y_align)
        if can_perform_scatic_ops(x_arr) and can_perform_scatic_ops(y_arr):
            new_static_arr = x_arr @ y_arr  # type: ignore
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=unit_dict), static_arr=new_static_arr)


@overload
def matmul(x: jax.Array, y: jax.Array, **kwargs) -> jax.Array:
    return jnp._orig_matmul(x, y, **kwargs)  # type: ignore


@dispatch
def matmul(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## Subtractions ###########################
@overload
def subtract(x: Unitful, y: Unitful) -> Unitful:
    if x.unit.dim != y.unit.dim:
        raise ValueError(f"Cannot subtract two arrays with units {x.unit} and {y.unit}.")
    x_align, y_align = align_scales(x, y)
    if isinstance(x_align.val, np.bool) or isinstance(y_align.val, np.bool):
        raise Exception(f"Subtract not supported for bool: {x}, {y}")
    new_val = x_align.val - y_align.val
    # if static arrays exist, perform subtract with static arrs
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x_align)
        y_arr = get_static_operand(y_align)
        if can_perform_scatic_ops(x_arr) and can_perform_scatic_ops(y_arr):
            new_static_arr = x_arr - y_arr  # type: ignore
    return Unitful(val=new_val, unit=x_align.unit, static_arr=new_static_arr)


@overload
def subtract(x: Unitful, y: ArrayLike) -> Unitful:
    if x.unit.dim:
        raise ValueError(f"Cannot add non-unitful to array with unit {x.unit}.")
    y_unitful = Unitful(val=y, unit=EMPTY_UNIT)
    return subtract(x, y_unitful)


@overload
def subtract(x: ArrayLike, y: Unitful) -> Unitful:
    if y.unit.dim:
        raise ValueError(f"Cannot add non-unitful to array with unit {y.unit}.")
    x_unitful = Unitful(val=x, unit=EMPTY_UNIT)
    return subtract(x_unitful, y)


@overload
def subtract(x: int, y: int) -> int:
    return x - y


@overload
def subtract(x: float, y: float) -> float:
    return x - y


@overload
def subtract(x: complex, y: complex) -> complex:
    return x - y


@overload
def subtract(x: jax.Array, y: jax.Array) -> jax.Array:
    return x - y


@overload
def subtract(x: PhysicalArrayLike, y: PhysicalArrayLike) -> PhysicalArrayLike:
    return x - y


@dispatch
def subtract(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## less than ##########################
@overload
def lt(
    x: Unitful,
    y: Unitful,
) -> np.bool | np.ndarray | jax.Array | bool:
    x_align, y_align = align_scales(x, y)
    if isinstance(x_align.val, complex) or isinstance(y_align.val, complex):
        raise Exception(f"Cannot compare complex values: {x}, {y}")
    new_val = x_align.val < y_align.val
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x_align)
        y_arr = get_static_operand(y_align)
        if x_arr is not None and y_arr is not None:
            assert not isinstance(x_arr, complex) and not isinstance(y_arr, complex)
            new_static_arr = x_arr < y_arr
    if output_unitful_for_array(new_static_arr):
        return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)  # type: ignore
    return new_val


@overload
def lt(
    x: Unitful,
    y: ArrayLike,
) -> np.bool | np.ndarray | jax.Array | bool:
    assert x.unit.dim == {}, f"Cannot compare unitful with dim {x.unit.dim} to unitless quantity"
    result = lt(x, Unitful(val=y, unit=EMPTY_UNIT))
    return result


@overload
def lt(
    x: ArrayLike,
    y: Unitful,
) -> np.bool | np.ndarray | jax.Array | bool:
    return ge(y, x)


@overload
def lt(x: int | float, y: int | float) -> bool:
    return x < y


@overload
def lt(x: jax.Array, y: jax.Array) -> jax.Array:
    return x < y


@overload
def lt(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x < y


@overload
def lt(x: np.number, y: np.number) -> np.bool:
    return x < y


@dispatch
def lt(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## less equal ##########################
@overload
def le(
    x: Unitful,
    y: Unitful,
) -> np.bool | np.ndarray | jax.Array | bool:
    x_align, y_align = align_scales(x, y)
    if isinstance(x_align.val, complex) or isinstance(y_align.val, complex):
        raise Exception(f"Cannot compare complex values: {x}, {y}")
    new_val = x_align.val <= y_align.val
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x_align)
        y_arr = get_static_operand(y_align)
        if x_arr is not None and y_arr is not None:
            assert not isinstance(x_arr, complex) and not isinstance(y_arr, complex)
            new_static_arr = x_arr <= y_arr
    if output_unitful_for_array(new_static_arr):
        return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)  # type: ignore
    return new_val


@overload
def le(
    x: Unitful,
    y: ArrayLike,
) -> np.bool | np.ndarray | jax.Array | bool:
    assert x.unit.dim == {}, f"Cannot compare unitful with dim {x.unit.dim} to unitless quantity"
    result = le(x, Unitful(val=y, unit=EMPTY_UNIT))
    return result


@overload
def le(
    x: ArrayLike,
    y: Unitful,
) -> np.bool | np.ndarray | jax.Array | bool:
    return gt(y, x)


@overload
def le(x: int | float, y: int | float) -> bool:
    return x <= y


@overload
def le(x: jax.Array, y: jax.Array) -> jax.Array:
    return x <= y


@overload
def le(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x <= y


@overload
def le(x: np.number, y: np.number) -> np.bool:
    return x <= y


@dispatch
def le(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## equal ##########################
@overload
def eq(
    x: Unitful,
    y: Unitful,
) -> np.bool | np.ndarray | jax.Array | bool:
    x_align, y_align = align_scales(x, y)
    new_val = x_align.val == y_align.val
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x_align)
        y_arr = get_static_operand(y_align)
        if x_arr is not None and y_arr is not None:
            new_static_arr = x_arr == y_arr
    if output_unitful_for_array(new_static_arr):
        return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)  # type: ignore
    return new_val


@overload
def eq(
    x: Unitful,
    y: ArrayLike,
) -> np.bool | np.ndarray | jax.Array | bool:
    assert x.unit.dim == {}, f"Cannot compare unitful with dim {x.unit.dim} to unitless quantity"
    result = eq(x, Unitful(val=y, unit=EMPTY_UNIT))
    return result


@overload
def eq(
    x: ArrayLike,
    y: Unitful,
) -> np.bool | np.ndarray | jax.Array | bool:
    return eq(y, x)


@overload
def eq(x: int | float, y: int | float) -> bool:
    return x == y


@overload
def eq(x: jax.Array, y: jax.Array) -> jax.Array:
    return x == y


@overload
def eq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x == y


@overload
def eq(x: np.number, y: np.number) -> np.bool:
    return x == y


@dispatch
def eq(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## not equal ##########################
@overload
def ne(
    x: Unitful,
    y: Unitful,
) -> np.bool | np.ndarray | jax.Array | bool:
    x_align, y_align = align_scales(x, y)
    new_val = x_align.val != y_align.val
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x_align)
        y_arr = get_static_operand(y_align)
        if x_arr is not None and y_arr is not None:
            new_static_arr = x_arr != y_arr
    if output_unitful_for_array(new_static_arr):
        return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)  # type: ignore
    return new_val


@overload
def ne(
    x: Unitful,
    y: ArrayLike,
) -> np.bool | np.ndarray | jax.Array | bool:
    assert x.unit.dim == {}, f"Cannot compare unitful with dim {x.unit.dim} to unitless quantity"
    result = ne(x, Unitful(val=y, unit=EMPTY_UNIT))
    return result


@overload
def ne(
    x: ArrayLike,
    y: Unitful,
) -> np.bool | np.ndarray | jax.Array | bool:
    return ne(y, x)


@overload
def ne(x: int | float, y: int | float) -> bool:
    return x != y


@overload
def ne(x: jax.Array, y: jax.Array) -> jax.Array:
    return x != y


@overload
def ne(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x != y


@overload
def ne(x: np.number, y: np.number) -> np.bool:
    return x != y


@dispatch
def ne(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## greater equal ##########################
@overload
def ge(
    x: Unitful,
    y: Unitful,
) -> np.bool | np.ndarray | jax.Array | bool:
    x_align, y_align = align_scales(x, y)
    if isinstance(x_align.val, complex) or isinstance(y_align.val, complex):
        raise Exception(f"Cannot compare complex values: {x}, {y}")
    new_val = x_align.val >= y_align.val
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x_align)
        y_arr = get_static_operand(y_align)
        if x_arr is not None and y_arr is not None:
            assert not isinstance(x_arr, complex) and not isinstance(y_arr, complex)
            new_static_arr = x_arr >= y_arr
    if output_unitful_for_array(new_static_arr):
        return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)  # type: ignore
    return new_val


@overload
def ge(
    x: Unitful,
    y: ArrayLike,
) -> np.bool | np.ndarray | jax.Array | bool:
    assert x.unit.dim == {}, f"Cannot compare unitful with dim {x.unit.dim} to unitless quantity"
    result = ge(x, Unitful(val=y, unit=EMPTY_UNIT))
    return result


@overload
def ge(
    x: ArrayLike,
    y: Unitful,
) -> np.bool | np.ndarray | jax.Array | bool:
    return lt(y, x)


@overload
def ge(x: int | float, y: int | float) -> bool:
    return x >= y


@overload
def ge(x: jax.Array, y: jax.Array) -> jax.Array:
    return x >= y


@overload
def ge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x >= y


@overload
def ge(x: np.number, y: np.number) -> np.bool:
    return x >= y


@dispatch
def ge(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## greater than ##########################
@overload
def gt(
    x: Unitful,
    y: Unitful,
) -> np.bool | np.ndarray | jax.Array | bool:
    x_align, y_align = align_scales(x, y)
    if isinstance(x_align.val, complex) or isinstance(y_align.val, complex):
        raise Exception(f"Cannot compare complex values: {x}, {y}")
    new_val = x_align.val > y_align.val
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x_align)
        y_arr = get_static_operand(y_align)
        if x_arr is not None and y_arr is not None:
            assert not isinstance(x_arr, complex) and not isinstance(y_arr, complex)
            new_static_arr = x_arr > y_arr
    if output_unitful_for_array(new_static_arr):
        return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)  # type: ignore
    return new_val


@overload
def gt(
    x: Unitful,
    y: ArrayLike,
) -> np.bool | np.ndarray | jax.Array | bool:
    assert x.unit.dim == {}, f"Cannot compare unitful with dim {x.unit.dim} to unitless quantity"
    result = gt(x, Unitful(val=y, unit=EMPTY_UNIT))
    return result


@overload
def gt(
    x: ArrayLike,
    y: Unitful,
) -> np.bool | np.ndarray | jax.Array | bool:
    return le(y, x)


@overload
def gt(x: int | float, y: int | float) -> bool:
    return x > y


@overload
def gt(x: jax.Array, y: jax.Array) -> jax.Array:
    return x > y


@overload
def gt(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x > y


@overload
def gt(x: np.number, y: np.number) -> np.bool:
    return x > y


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
    new_val = x.val**y
    new_scale = x.unit.scale * y
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        if x_arr is not None:
            new_static_arr = x_arr**y
    return Unitful(val=new_val, unit=Unit(scale=new_scale, dim=new_dim), static_arr=new_static_arr)


@overload
def pow(x: jax.Array, y: jax.Array) -> jax.Array:
    return x**y


# TODO: more fine-grained overloads
@overload
def pow(x: StaticPhysicalArrayLike, y: StaticPhysicalArrayLike) -> StaticPhysicalArrayLike:
    return x**y


@dispatch
def pow(x, y):  # type: ignore
    del x, y
    raise NotImplementedError()


## min #######################################
def unary_fn(x: Unitful, op_str: str, *args, **kwargs) -> Unitful:
    # handling of numpy arrays
    if isinstance(x.val, np.ndarray | np.number):
        np_fn = getattr(np, f"{op_str}")
        new_val = np_fn(x.val, *args, **kwargs)
        if not isinstance(new_val, np.ndarray) and not isinstance(new_val, np.number):
            raise Exception(f"This is an internal error: {op_str} produced {type(new_val)}")
        if new_val.dtype not in PHYSICAL_DTYPES:
            return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=None)
        return Unitful(val=new_val, unit=x.unit, static_arr=None)

    # handling of other jax array or other types
    orig_fn = getattr(jax.numpy, f"_orig_{op_str}")
    new_val = orig_fn(x.val, *args, **kwargs)
    if not isinstance(new_val, jax.Array):
        raise Exception(f"This is an internal error: {op_str} produced {type(new_val)}")
    new_static_arr = None
    if is_traced(new_val):
        x_arr = get_static_operand(x)
        if x_arr is not None:
            np_fn = getattr(np, f"{op_str}")
            new_static_arr = np_fn(x_arr, *args, **kwargs)
    if new_val.dtype not in PHYSICAL_DTYPES:
        return Unitful(val=new_val, unit=EMPTY_UNIT, static_arr=new_static_arr)
    return Unitful(val=new_val, unit=x.unit, static_arr=new_static_arr)


@overload
def min(x: Unitful, *args, **kwargs) -> Unitful:
    return unary_fn(x, "min", *args, **kwargs)


@overload
def min(x: PhysicalArrayLike, *args, **kwargs) -> jax.Array:
    result_shape_dtype = jax.eval_shape(jnp._orig_min, x, *args, **kwargs)  # type: ignore
    if output_unitful_for_array(result_shape_dtype):
        unit_result = unary_fn(Unitful(val=x, unit=EMPTY_UNIT), "min", *args, **kwargs)
        return unit_result  # type: ignore
    return jnp._orig_min(x, *args, **kwargs)  # type: ignore


@dispatch
def min(x, *args, **kwargs):  # type: ignore
    del x, args, kwargs
    raise NotImplementedError()


## max #######################################
@overload
def max(x: Unitful, *args, **kwargs) -> Unitful:
    return unary_fn(x, "max", *args, **kwargs)


@overload
def max(x: PhysicalArrayLike, *args, **kwargs) -> jax.Array:
    result_shape_dtype = jax.eval_shape(jnp._orig_max, x, *args, **kwargs)  # type: ignore
    if output_unitful_for_array(result_shape_dtype):
        unit_result = unary_fn(Unitful(val=x, unit=EMPTY_UNIT), "max", *args, **kwargs)
        return unit_result  # type: ignore
    return jnp._orig_max(x, *args, **kwargs)  # type: ignore


@dispatch
def max(x, *args, **kwargs):  # type: ignore
    del x, args, kwargs
    raise NotImplementedError()


## mean #######################################
@overload
def mean(x: Unitful, *args, **kwargs) -> Unitful:
    return unary_fn(x, "mean", *args, **kwargs)


@overload
def mean(x: PhysicalArrayLike, *args, **kwargs) -> jax.Array:
    result_shape_dtype = jax.eval_shape(jnp._orig_mean, x, *args, **kwargs)  # type: ignore
    if output_unitful_for_array(result_shape_dtype):
        unit_result = unary_fn(Unitful(val=x, unit=EMPTY_UNIT), "mean", *args, **kwargs)
        return unit_result  # type: ignore
    return jnp._orig_mean(x, *args, **kwargs)  # type: ignore


@dispatch
def mean(x, *args, **kwargs):  # type: ignore
    del x, args, kwargs
    raise NotImplementedError()


## sum #######################################
@overload
def sum(x: Unitful, *args, **kwargs) -> Unitful:
    return unary_fn(x, "sum", *args, **kwargs)


@overload
def sum(x: PhysicalArrayLike, *args, **kwargs) -> jax.Array:
    result_shape_dtype = jax.eval_shape(jnp._orig_sum, x, *args, **kwargs)  # type: ignore
    if output_unitful_for_array(result_shape_dtype):
        unit_result = unary_fn(Unitful(val=x, unit=EMPTY_UNIT), "sum", *args, **kwargs)
        return unit_result  # type: ignore
    return jnp._orig_sum(x, *args, **kwargs)  # type: ignore


@dispatch
def sum(x, *args, **kwargs):  # type: ignore
    del x, args, kwargs
    raise NotImplementedError()


## prod #######################################
@overload
def prod(x: Unitful, *args, **kwargs) -> Unitful:
    if x.unit.dim != {}:
        raise NotImplementedError()
    # TODO: make numerically more stable by avoiding .value()
    new_static_arr = x.static_value()
    new_val = x.value()
    unary_input = Unitful(val=new_val, unit=EMPTY_UNIT, optimize_scale=False, static_arr=new_static_arr)
    return unary_fn(unary_input, "prod", *args, **kwargs)


@overload
def prod(x: PhysicalArrayLike, *args, **kwargs) -> jax.Array:
    result_shape_dtype = jax.eval_shape(jnp._orig_prod, x, *args, **kwargs)  # type: ignore
    if output_unitful_for_array(result_shape_dtype):
        unit_result = unary_fn(Unitful(val=x, unit=EMPTY_UNIT), "sum", *args, **kwargs)
        return unit_result  # type: ignore
    return jnp._orig_prod(x, *args, **kwargs)  # type: ignore


@dispatch
def prod(x, *args, **kwargs):  # type: ignore
    del x, args, kwargs
    raise NotImplementedError()


## abs #######################################
@overload
def abs_impl(x: Unitful) -> Unitful:
    return unary_fn(x, "abs")


@overload
def abs_impl(x: jax.Array) -> jax.Array:
    return jnp._orig_abs(x)  # type: ignore


@overload
def abs_impl(x: int) -> int:
    return abs(x)


@overload
def abs_impl(x: float) -> float:
    return abs(x)


@overload
def abs_impl(x: complex) -> complex:
    return abs(x)


@dispatch
def abs_impl(x):  # type: ignore
    del x
    raise NotImplementedError()


## astype #######################################
@overload
def astype(x: Unitful, *args, **kwargs) -> Unitful:
    return unary_fn(x, "astype", *args, **kwargs)


@overload
def astype(x: PhysicalArrayLike, *args, **kwargs) -> jax.Array:
    partial_fn = jax.tree_util.Partial(jnp._orig_astype, x, *args, **kwargs)  # type: ignore
    result_shape_dtype = jax.eval_shape(partial_fn)
    if output_unitful_for_array(result_shape_dtype):
        unit_result = unary_fn(Unitful(val=x, unit=EMPTY_UNIT), "astype", *args, **kwargs)
        return unit_result  # type: ignore
    return jnp._orig_astype(x, *args, **kwargs)  # type: ignore


@dispatch
def astype(x, *args, **kwargs):  # type: ignore
    del x, args, kwargs
    raise NotImplementedError()


## squeeze #######################################
@overload
def squeeze(x: Unitful, *args, **kwargs) -> Unitful:
    return unary_fn(x, "squeeze", *args, **kwargs)


@overload
def squeeze(x: jax.Array, *args, **kwargs) -> jax.Array:
    return jnp._orig_squeeze(x, *args, **kwargs)  # type: ignore


@dispatch
def squeeze(x, *args, **kwargs):  # type: ignore
    del x, args, kwargs
    raise NotImplementedError()


## reshape #######################################
@overload
def reshape(x: Unitful, *args, **kwargs) -> Unitful:
    return unary_fn(x, "reshape", *args, **kwargs)


@overload
def reshape(x: jax.Array, *args, **kwargs) -> jax.Array:
    return jnp._orig_reshape(x, *args, **kwargs)  # type: ignore


@dispatch
def reshape(x, *args, **kwargs):  # type: ignore
    del x, args, kwargs
    raise NotImplementedError()


## argmax #######################################
@overload
def argmax(x: Unitful, *args, **kwargs) -> Unitful:
    return unary_fn(x, "argmax", *args, **kwargs)


@overload
def argmax(x: jax.Array, *args, **kwargs) -> jax.Array:
    return jnp._orig_argmax(x, *args, **kwargs)  # type: ignore


@overload
def argmax(x: np.ndarray, *args, **kwargs) -> np.ndarray:
    return np.argmax(x, *args, **kwargs)


@dispatch
def argmax(x, *args, **kwargs):  # type: ignore
    del x, args, kwargs
    raise NotImplementedError()


## argmin #######################################
@overload
def argmin(x: Unitful, *args, **kwargs) -> Unitful:
    return unary_fn(x, "argmin", *args, **kwargs)


@overload
def argmin(x: jax.Array, *args, **kwargs) -> jax.Array:
    return jnp._orig_argmin(x, *args, **kwargs)  # type: ignore


@overload
def argmin(x: np.ndarray, *args, **kwargs) -> np.ndarray:
    return np.argmin(x, *args, **kwargs)


@dispatch
def argmin(x, *args, **kwargs):  # type: ignore
    del x, args, kwargs
    raise NotImplementedError()


## sign #######################################
from typing import overload as _typing_overload  # ensure name not shadowed

@overload
def sign(x: Unitful) -> Unitful:
    # Drop units and return integer {-1,0,1}
    # Handle NumPy-backed vs JAX-backed values separately, just like other ops.
    if isinstance(x.val, np.ndarray | np.number):
        # NumPy path
        y = np.sign(x.val)
        # Ensure integer dtype (ndarray or scalar)
        if isinstance(y, np.ndarray):
            y_int: np.ndarray | np.number = y.astype(np.int_)
        else:
            # numpy scalar -> numpy integer scalar
            y_int = np.int_(y)
        return Unitful(val=y_int, unit=EMPTY_UNIT, static_arr=None)
    else:
        # JAX path (includes tracers)
        y = jax.lax.sign(x.val)  # avoids recursion and works under jit
        y_int = jax.lax.convert_element_type(y, jnp.int_)  # type: ignore
        if not isinstance(y_int, jax.Array):
            raise Exception(f"This is an internal error: sign produced {type(y_int)}")
        new_static_arr = None
        if is_traced(y_int):
            x_arr = get_static_operand(x)
            if x_arr is not None:
                y_s = np.sign(x_arr)
                if isinstance(y_s, np.ndarray):
                    y_s = y_s.astype(np.int_)
                else:
                    y_s = np.int_(y_s)
                new_static_arr = y_s
        return Unitful(val=y_int, unit=EMPTY_UNIT, static_arr=new_static_arr)  # type: ignore


@overload
def sign(x: jax.Array) -> jax.Array:
    # JAX original op via lax, then cast to integer dtype
    y = jax.lax.sign(x)
    return jax.lax.convert_element_type(y, jnp.int_)  # type: ignore


@overload
def sign(x: np.ndarray) -> np.ndarray:
    # NumPy sign then cast to numpy integer
    y = np.sign(x)
    return y.astype(np.int_)


@overload
def sign(x: int) -> int:
    # python integer → python integer in {-1, 0, 1}
    return -1 if x < 0 else (1 if x > 0 else 0)


@dispatch
def sign(x):  # type: ignore
    del x
    raise NotImplementedError()


# --- make jnp.sign(...) use these overloads, preserving original ---
if not hasattr(jnp, "_orig_sign"):
    jnp._orig_sign = jnp.sign  # type: ignore
jnp.sign = sign  # type: ignore
