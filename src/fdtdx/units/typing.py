from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any, Tuple, Union

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


def _jax_dtype_works(dtype: Any) -> bool:
    """
    Conservative check: dtype must be constructible and usable in a simple op
    on the current default backend.
    """
    try:
        x = jnp.array([1, 2, 3], dtype=dtype)
        y = x + x
        _ = jnp.sum(y)
        return True
    except Exception:
        return False


def _numpy_dtype_is_real_extended(dtype: Any) -> bool:
    """
    Return True if dtype exists and is wider than float64/complex128 in this NumPy build.
    This filters out platforms where float128/complex256 are missing or aliased.
    """
    if dtype is None:
        return False
    try:
        dt = np.dtype(dtype)
    except Exception:
        return False

    # float128 should be wider than float64
    # complex256 should be wider than complex128 (implicitly depends on float128)
    if dt.kind == "f":
        return dt.itemsize > np.dtype(np.float64).itemsize
    if dt.kind == "c":
        return dt.itemsize > np.dtype(np.complex128).itemsize
    return False


@lru_cache(maxsize=1)
def _physical_dtypes() -> Tuple[Any, ...]:
    """
    Return dtypes whose scale can be optimized.
    Cross-platform policy:
    - Always include stable dtypes (float16/32/64, bfloat16, complex64/128).
    - Conditionally include float8 dtypes if supported by the active JAX backend.
    - Conditionally include NumPy extended precision (float128/complex256) only if the build truly supports wider-than-float64 types (typically Linux x86_64).
    """
    dtypes: list[Any] = []

    dtypes.extend(
        [
            jnp.float16,
            jnp.bfloat16,
            jnp.float32,
            jnp.float64,
            jnp.complex64,
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ]
    )

    float8_names = (
        "float8_e4m3b11fnuz",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    )
    for name in float8_names:
        dt = getattr(jnp, name, None)
        if dt is not None and _jax_dtype_works(dt):
            dtypes.append(dt)

    if _numpy_dtype_is_real_extended(getattr(np, "float128", None)):
        dtypes.append(np.float128)
    if _numpy_dtype_is_real_extended(getattr(np, "complex256", None)):
        dtypes.append(np.complex256)

    out: list[Any] = []
    seen: set[Any] = set()
    for dt in dtypes:
        if dt not in seen:
            seen.add(dt)
            out.append(dt)

    return tuple(out)


# array data types which allow for scale optimization
PHYSICAL_DTYPES = _physical_dtypes()
