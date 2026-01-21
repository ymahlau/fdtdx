import jax.numpy as jnp
import numpy as np

from fdtdx.units.typing import PHYSICAL_DTYPES


def test_physical_dtypes_contains_basic_set():
    """Stable, cross-platform dtypes must always be present."""
    expected = {
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
    }

    for dt in expected:
        assert dt in PHYSICAL_DTYPES


def test_physical_dtypes_no_duplicates():
    """PHYSICAL_DTYPES should not contain duplicate entries"""
    assert len(PHYSICAL_DTYPES) == len(set(PHYSICAL_DTYPES))


def test_float8_dtypes_are_usable_if_present():
    """Float8 dtypes are included only if they are actually executable on the current backend"""
    FLOAT8_NAMES = (
        "float8_e4m3b11fnuz",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    )

    for name in FLOAT8_NAMES:
        dt = getattr(jnp, name, None)
        if dt is None:
            continue

        if dt in PHYSICAL_DTYPES:
            x = jnp.array([1, 2, 3], dtype=dt)
            y = x + x
            _ = jnp.sum(y)


def test_numpy_extended_precision_semantics():
    """Include float128 / complex256 only when they provide real extended precision (not aliases)."""
    if hasattr(np, "float128"):
        is_real = np.dtype(np.float128).itemsize > np.dtype(np.float64).itemsize
        assert (np.float128 in PHYSICAL_DTYPES) == is_real

    if hasattr(np, "complex256"):
        is_real = np.dtype(np.complex256).itemsize > np.dtype(np.complex128).itemsize
        assert (np.complex256 in PHYSICAL_DTYPES) == is_real
