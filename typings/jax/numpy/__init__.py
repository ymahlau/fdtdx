from fdtdx.units.unitful import Unitful
from jax.numpy import *
from typing import overload

@overload
def multiply(a: Unitful, b: Unitful) -> Unitful: ...

