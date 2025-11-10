import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.units.unitful import Unit, Unitful, SI, sign



def test_sign_overload_unitful_mixed():
    """Unitful → Unitful with unit dropped and integer values {-1,0,1}"""
    speed_unit = Unit(scale=3, dim={SI.m: 1, SI.s: -1})  # km/s
    speeds = Unitful(val=jnp.array([-2.5, 0.0, 3.7, -1.2, 4.8]), unit=speed_unit)

    result = jnp.sign(speeds)  # type: ignore

    assert isinstance(result, Unitful)
    assert result.unit.dim == {}
    assert isinstance(result.val, jax.Array)
    assert np.issubdtype(result.val.dtype, np.integer)
    assert jnp.all(result.value() == jnp.array([-1, 0, 1, -1, 1]))


def test_sign_overload_jax_array():
    """JAX array → JAX integer array"""
    x = jnp.array([-3.0, -0.0, 0.0, 2.2])
    y = sign(x)
    assert isinstance(y, jax.Array)
    assert np.issubdtype(y.dtype, np.integer)
    assert jnp.all(y == jnp.array([-1, 0, 0, 1]))


def test_sign_overload_numpy_array():
    """NumPy array → NumPy integer array"""
    x = np.array([-4.5, 0.0, 7.1], dtype=np.float64)
    y = sign(x)
    assert isinstance(y, np.ndarray)
    assert np.issubdtype(y.dtype, np.integer)
    assert np.array_equal(y, np.array([-1, 0, 1], dtype=y.dtype))


def test_sign_python_int():
    """Python int → Python int in {-1,0,1}"""
    assert isinstance(sign(-7), int)
    assert sign(-7) == -1
    assert sign(0) == 0
    assert sign(9) == 1


def test_sign_unitful_scalar_and_numpy_backend():
    """Unitful with NumPy scalar should yield Unitful, unitless, NumPy integer inside"""
    unit = Unit(scale=-2, dim={SI.kg: 1})  # centi-scale, but should be dropped
    x = Unitful(val=np.array(-0.3), unit=unit)
    y = sign(x)  # or jnp.sign(x)  # type: ignore
    assert isinstance(y, Unitful)
    assert y.unit.dim == {}
    assert isinstance(y.val, np.integer)
    assert y ==  -1


def test_sign_jitted_with_unitful_and_static():
    """JIT path keeps integer output dtype and drops unit"""
    unit = Unit(scale=0, dim={SI.A: 1})
    x = Unitful(
        val=jnp.asarray([-1.0, 0.0, 5.0]),
        unit=unit,
        static_arr=np.array([-1.0, 0.0, 4.9]),
    )

    def fn(u: Unitful) -> Unitful:
        r = jnp.sign(u)  # type: ignore
        return r

    y = jax.jit(fn)(x)
    assert isinstance(y, Unitful)
    assert y.unit.dim == {}
    assert isinstance(y.val, jax.Array)
    assert np.issubdtype(y.val.dtype, np.integer)
    assert jnp.all(y.value() == jnp.array([-1, 0, 1]))
