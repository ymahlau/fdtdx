from enum import Enum
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

PhysicalArrayLike = Union[
    int,
    float,
    complex,
    jax.Array,
    np.number,
    np.ndarray,
]

StaticPhysicalArrayLike = Union[
    int,
    float,
    complex,
    np.number,
    np.ndarray,
]

class SI(Enum):
    s = "second"
    m = "meter"
    kg = "kilogram"
    A = "ampere"
    K = "kelvin"
    mol = "mole"
    cd = "candela"

PHYSICAL_DTYPES = [
    jnp.float32,
    jnp.float64,
    jnp.complex64,
    jnp.complex128,
    jnp.float16,
    jnp.bfloat16,
    jnp.float8_e4m3b11fnuz,
    jnp.float8_e4m3fn,
    jnp.float8_e4m3fnuz,
    jnp.float8_e5m2,
    jnp.float8_e5m2fnuz,
]    
