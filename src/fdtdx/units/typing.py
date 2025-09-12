from enum import Enum
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

# Enum of SI units. Intentionally omits mol as this is a "count" rather than an actual unit.
class SI(Enum):
    s = "second"
    m = "meter"
    kg = "kilogram"
    A = "ampere"
    K = "kelvin"
    cd = "candela"

# Types whose scale can be optimized
PhysicalArrayLike = Union[
    float,
    complex,
    jax.Array,
    np.number,
    np.ndarray,
]

# Types whose scale can be optimized and are real
RealPhysicalArrayLike = Union[
    float,
    jax.Array,
    np.number,
    np.ndarray,
]

# Types who remain static during jit-context
StaticArrayLike = Union[
    int,
    bool,
    np.bool,
    float,
    complex,
    np.number,
    np.ndarray,
]

# Types who remain static during jit-context and whose scale can be optimized
StaticPhysicalArrayLike = Union[
    float,
    complex,
    np.number,
    np.ndarray,
]

# Types whose scale cannot be optimized
NonPhysicalArrayLike = Union[
    int,
    bool,
    np.bool,
]

# array data types which allow for scale optimization
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
    np.float16,
    np.float32,
    np.float64,
    np.float128,
    np.complex64,
    np.complex128,
    np.complex256,
]    
