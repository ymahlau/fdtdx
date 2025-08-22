from enum import Enum
from typing import Union

import jax

PhysicalArrayLike = Union[
    int,
    float,
    complex,
    jax.Array
]

class SI(Enum):
    s = "second"
    m = "meter"
    kg = "kilogram"
    A = "ampere"
    K = "kelvin"
    mol = "mole"
    cd = "candela"