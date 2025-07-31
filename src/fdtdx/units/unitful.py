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

from fdtdx.units.utils import patch_fn_to_module


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

    def __str__(self):
        return f"Unitful [{self.unit}]: {tree_repr(self.val)}"

    def __repr__(self):
        return str(self)
    
    def __mul__(self, other: ArrayLike | "Unitful") -> "Unitful":
        return multiply(self, other)
    
    def __rmul__(self, other: ArrayLike | "Unitful") -> "Unitful":
        return multiply(self, other)


s = Unitful(val=1, unit=Unit(0, {SI.s: 1}))
ms = Unitful(val=1, unit=Unit(-3, {SI.s: 1}))
µs = Unitful(val=1, unit=Unit(-6, {SI.s: 1}))
ns = Unitful(val=1, unit=Unit(-9, {SI.s: 1}))
ps = Unitful(val=1, unit=Unit(-12, {SI.s: 1}))
fs = Unitful(val=1, unit=Unit(-15, {SI.s: 1}))

m_per_s = Unitful(val=1, unit=Unit(0, {SI.s: -1, SI.m: 1}))

Hz = Unitful(val=1, unit=Unit(0, {SI.s: -1}))
kHz = Unitful(val=1, unit=Unit(3, {SI.s: -1}))
MHz = Unitful(val=1, unit=Unit(6, {SI.s: -1}))
GHz = Unitful(val=1, unit=Unit(9, {SI.s: -1}))
THz = Unitful(val=1, unit=Unit(12, {SI.s: -1}))
PHz = Unitful(val=1, unit=Unit(15, {SI.s: -1}))


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

patch_fn_to_module(multiply, jax.numpy)
