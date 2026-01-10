import cmath
import math

import jax
import jax.numpy as jnp
import numpy as np
import plum
import pytest

import fdtdx.units.unitful as unitful
from fdtdx.core.fraction import Fraction
from fdtdx.core.jax.pytrees import TreeClass, autoinit
from fdtdx.core.jax.utils import is_currently_compiling
from fdtdx.units import Hz, ms, s
from fdtdx.units.unitful import (
    SI,
    Unit,
    Unitful,
    add,
    cond,
    eq,
    ge,
    gt,
    le,
    lt,
    matmul,
    multiply,
    ne,
    pow,
    reshape,
    squeeze,
    subtract,
)


def test_multiply_unitful_unitful_same_dimensions():
    """Test multiplication of two Unitful objects with same dimensions"""
    time1 = s * 2
    time2 = 3 * s

    result = multiply(time1, time2)

    assert jnp.allclose(result.value(), 6.0)
    assert result.unit.dim == {SI.s: 2}


def test_multiply_unitful_unitful_different_dimensions():
    """Test multiplication of two Unitful objects with different dimensions"""
    time_val = 5 * s
    freq_val = 10 * Hz

    result = multiply(time_val, freq_val)

    assert math.isclose(result.value(), 50.0)  # type: ignore
    assert result.unit.dim == {}  # s^1 * s^-1 = dimensionless


def test_multiply_unitful_unitful_different_scales():
    """Test multiplication of Unitful objects with different scales"""
    time1 = 2 * s  # scale 0
    time2 = 3 * ms  # scale -3

    result = multiply(time1, time2)

    assert math.isclose(result.value(), 6.0e-3)  # type: ignore
    assert result.unit.dim == {SI.s: 2}


def test_multiply_unitful_unitful_python_python_type_and_dim():
    """Test multiplication of Unitful objects with value Python scalar and Python scalar"""
    a = 2 * s
    b = 3.5 * s
    res = multiply(a, b)

    assert math.isclose(res.value(), 7.0)  # type: ignore
    assert res.unit.dim == {SI.s: 2}
    assert isinstance(res.val, float)


def test_multiply_unitful_unitful_numpy_numpy_type_and_dim():
    """Test multiplication of Unitful objects with value numpy Array and numpy Array"""
    a = s * np.array([1.0, 2.0, 3.0])
    b = s * np.array([4.0, 5.0, 6.0])
    res = multiply(a, b)

    assert isinstance(res.val, np.ndarray)
    assert jnp.allclose(res.value(), jnp.array([4.0, 10.0, 18.0]))
    assert res.unit.dim == {SI.s: 2}


def test_multiply_unitful_unitful_jax_jax_type_and_dim():
    """Test multiplication of Unitful objects with value Jax Array and Jax Array"""
    a = s * jnp.array([1.0, 2.0, 3.0])
    b = s * jnp.array([4.0, 5.0, 6.0])
    res = multiply(a, b)

    assert isinstance(res.val, jax.Array)
    assert jnp.allclose(res.value(), jnp.array([4.0, 10.0, 18.0]))
    assert res.unit.dim == {SI.s: 2}


def test_multiply_with_arrays():
    """Test multiplication with array values"""
    time_array = s * jnp.array([1.0, 2.0, 3.0])
    freq_array: Unitful = jnp.array([2.0, 3.0, 4.0]) * Hz  # type: ignore
    assert isinstance(freq_array, Unitful)
    assert isinstance(time_array, Unitful)
    result = multiply(time_array, freq_array)

    expected_val = jnp.array([2.0, 6.0, 12.0])
    assert jnp.allclose(result.value(), expected_val)
    assert result.unit.dim == {}  # dimensionless


def test_multiply_overload_jax_jax():
    """Test multiplication of two jax Arrays"""
    arr1 = jnp.array([1.0, 2.0, 3.0])
    arr2 = jnp.array([4.0, 5.0, 6.0])

    result = multiply(arr1, arr2)

    expected = jnp.array([4.0, 10.0, 18.0])
    assert jnp.allclose(result, expected)
    assert isinstance(result, jax.Array)


def test_multiply_overload_numpy_numpy():
    """Test multiplication of two numpy arrays"""
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([4.0, 5.0, 6.0])

    result = multiply(arr1, arr2)

    expected = np.array([4.0, 10.0, 18.0])
    assert np.allclose(result, expected)
    assert isinstance(result, np.ndarray)


def test_multiply_python_scalars_promotion():
    """Test multiplication with Python Scalar int, float and complex"""
    assert multiply(2, 3.5) == 7.0
    assert multiply(2, 1 + 2j) == 2 + 4j
    assert multiply(2.0, 1 - 1j) == 2 - 2j
    assert multiply(2, 3) == 6
    assert multiply(2.0, 3.0) == 6.0
    assert multiply(1 + 2j, 3 - 1j) == (5 + 5j)


# mixed type:
# unitful-unitful with val jax-numpy -> should return jax type
def test_multiply_unitful_unitful_mixed_jax_numpy_backend_priority():
    """Test multiplication of Unitful objects with value Jax Array and numpy Array"""
    a = s * jnp.array([1.0, 2.0])
    b = s * np.array([3.0, 4.0])
    res1 = multiply(a, b)
    res2 = multiply(b, a)
    assert isinstance(res1.val, jax.Array)
    assert isinstance(res2.val, jax.Array)
    assert jnp.allclose(res1.value(), jnp.array([3.0, 8.0]))
    assert jnp.allclose(res2.value(), jnp.array([3.0, 8.0]))
    assert res1.unit.dim == {SI.s: 2}
    assert res2.unit.dim == {SI.s: 2}


# unitful-unitful with val jax-python -> should return jax type
def test_multiply_unitful_unitful_mixed_jax_python_backend_priority():
    """Test multiplication of Unitful objects with value Jax Array and Python scalar"""
    a = s * jnp.array([2.0, 3.0])
    b = 5 * s  # Python value
    res1 = multiply(a, b)
    res2 = multiply(b, a)
    assert isinstance(res1.val, jax.Array)
    assert isinstance(res2.val, jax.Array)
    assert jnp.allclose(res1.value(), jnp.array([10.0, 15.0]))
    assert jnp.allclose(res2.value(), jnp.array([10.0, 15.0]))
    assert res1.unit.dim == {SI.s: 2}
    assert res2.unit.dim == {SI.s: 2}


# unitful-unitful with val numpy-python -> should return numpy type
def test_multiply_unitful_unitful_mixed_numpy_python_backend_priority():
    """Test multiplication of Unitful objects with value numpy array and Python scalar"""
    a = s * np.array([2.0, 3.0])
    b = s * 4
    res = multiply(a, b)
    assert isinstance(res.val, np.ndarray)
    assert jnp.allclose(res.value(), jnp.array([8.0, 12.0]))
    assert res.unit.dim == {SI.s: 2}


def test_multiply_arraylike_unitful():
    """Test multiplication of ArrayLike with Unitful"""
    scalar = 5.0
    time_val = 3 * s

    result: Unitful = jnp.multiply(scalar, time_val)  # type: ignore

    assert jnp.allclose(result.value(), 15.0)
    assert result.unit.dim == {SI.s: 1}


# jax array - numpy array
def test_multiply_jaxarray_numpy_array_both_orders():
    """Test multiplication with regular Jax Array and Numpy Array"""
    x = jnp.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    r1 = multiply(x, y)
    r2 = multiply(y, x)

    assert isinstance(r1, jax.Array)
    assert isinstance(r2, jax.Array)
    expected = jnp.array([4.0, 10.0, 18.0])
    assert jnp.allclose(r1, expected)
    assert jnp.allclose(r2, expected)


# jax array - python scalar
def test_multiply_jaxarray_python_scalar_both_orders():
    """Test multiplication with regular Jax Array and Python Scalar"""
    x = jnp.array([1.0, 2.0, 3.0])
    s1 = 2.5

    r1 = multiply(x, s1)
    r2 = multiply(s1, x)

    assert isinstance(r1, jax.Array)
    assert isinstance(r2, jax.Array)
    expected = jnp.array([2.5, 5.0, 7.5])
    assert jnp.allclose(r1, expected)
    assert jnp.allclose(r2, expected)


# numpy array - python scalar
def test_multiply_numpy_array_python_scalar_both_orders():
    """Test multiplication with regular Numpy Array and Python Scalar"""
    a = np.array([2.0, 3.0])
    s1 = 4.0

    r1 = multiply(a, s1)
    r2 = multiply(s1, a)

    assert isinstance(r1, np.ndarray)
    assert isinstance(r2, np.ndarray)
    expected = np.array([8.0, 12.0])
    assert jnp.allclose(r1, expected)
    assert jnp.allclose(r2, expected)


# numpy number - python scalar
def test_multiply_numpy_scalar_python_scalar_both_orders():
    """Test multiplication with regular Numpy Number and Python Scalar"""
    ns = np.float64(2.5)
    py = 4

    r1 = multiply(ns, py)
    r2 = multiply(py, ns)

    assert isinstance(r1, np.generic)
    assert isinstance(r2, np.generic)
    assert r1 == np.float64(10.0)
    assert r2 == np.float64(10.0)


# numpy number - python scalar
def test_multiply_numpy_scalar_with_jax_both_orders():
    """Test multiplication with regular Numpy Number and Jax Array"""
    ns = np.int64(3)
    x = jnp.array([5.0, 6.0])

    r1 = multiply(ns, x)
    r2 = multiply(x, ns)

    assert isinstance(r1, jax.Array)
    assert isinstance(r2, jax.Array)
    assert jnp.allclose(r1, jnp.array([15.0, 18.0]))
    assert jnp.allclose(r2, jnp.array([15.0, 18.0]))


def test_multiply_not_implemented():
    """Test that multiply raises NotFoundLookupError for unsupported types"""
    with pytest.raises(plum.NotFoundLookupError):
        multiply("string", "string")  # type: ignore


def test_multiply_jitted():
    """Test jitting of multplication"""
    arr1 = s * jnp.array([1.0, 2.0, 3.0])
    arr2 = ms * jnp.array([1.0, 2.0, 3.0])
    arr3 = ms * jnp.array([1.0, 2.0, 3.0])

    def fn(a: Unitful, b: Unitful) -> Unitful:
        return jnp.multiply(a, b)  # type: ignore

    jitted_fn = jax.jit(fn)
    res1 = jitted_fn(arr1, arr2)
    res2 = jitted_fn(arr2, arr3)

    assert jnp.allclose(res1.value(), res2.value() * 1e3)


def test_add_same_units():
    """Test addition of two Unitful objects with the same dimensions."""
    # Create two lengths in meters
    unit_m = Unit(scale=0, dim={SI.m: 1})
    u1 = Unitful(val=jnp.array(5.0), unit=unit_m)
    u2 = Unitful(val=jnp.array(3.0), unit=unit_m)

    result = jnp.add(u1, u2)  # type: ignore

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 8.0)
    assert result.unit.dim == {SI.m: 1}


def test_add_different_units_raises_error():
    """Test that addition of Unitful objects with different dimensions raises ValueError."""
    # Create length and time units
    unit_m = Unit(scale=0, dim={SI.m: 1})
    unit_s = Unit(scale=0, dim={SI.s: 1})
    u1 = Unitful(val=jnp.array(5.0), unit=unit_m)
    u2 = Unitful(val=jnp.array(3.0), unit=unit_s)

    with pytest.raises(ValueError):
        add(u1, u2)


def test_subtract_same_units():
    """Test subtraction of two Unitful objects with the same dimensions."""
    # Create two masses in kilograms
    unit_kg = Unit(scale=0, dim={SI.kg: 1})
    u1 = Unitful(val=jnp.array(10.0), unit=unit_kg)
    u2 = Unitful(val=jnp.array(4.0), unit=unit_kg)

    result = jnp.subtract(u1, u2)  # type: ignore

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 6.0)
    assert result.unit.dim == {SI.kg: 1}


def test_subtract_different_units_raises_error():
    """Test that subtraction of Unitful objects with different dimensions raises ValueError."""
    # Create temperature and current units
    unit_K = Unit(scale=0, dim={SI.K: 1})
    unit_A = Unit(scale=0, dim={SI.A: 1})
    u1 = Unitful(val=jnp.array(300.0), unit=unit_K)
    u2 = Unitful(val=jnp.array(2.0), unit=unit_A)

    with pytest.raises(ValueError):
        subtract(u1, u2)


def test_le_magic_method():
    """Test less than or equal magic method of Unitful objects"""
    time1 = 2 * s
    time2 = 3 * s
    result = time1 <= time2

    assert isinstance(result, Unitful)
    assert isinstance(result.val, bool)
    assert jnp.allclose(result.val, True)


def test_le_method_with_numpy_arrays():
    """Test less than or equal with Unitful objects containing NumPy arrays"""
    pressure_unit = Unit(scale=3, dim={SI.kg: 1, SI.m: -1, SI.s: -2})
    pressures1 = Unitful(val=np.array([101.0, 95.5, 110.2, 88.7, 105.3]), unit=pressure_unit)
    pressures2 = Unitful(val=np.array([100.0, 95.5, 118.2, 12.7, 45.3]), unit=pressure_unit)

    result = pressures1 <= pressures2

    assert isinstance(result, Unitful)
    assert isinstance(result.val, np.ndarray)
    assert result.val.dtype == bool
    assert np.allclose(result.val, np.array([False, True, True, False, False]))


def test_le_method_with_jax_arrays():
    """Test less than or equal with Unitful objects containing Jax arrays"""
    length_unit = Unit(scale=-2, dim={SI.m: 1})
    lengths1 = Unitful(val=jnp.array([12.0, 25.5, 18.3, 30.1, 22.7]), unit=length_unit)
    lengths2 = Unitful(val=jnp.array([12.0, 14.5, 100.3, 302.1, 2.7]), unit=length_unit)
    result = lengths1 <= lengths2

    assert isinstance(result, Unitful)
    assert isinstance(result.val, jax.Array)
    assert result.val.dtype == bool
    assert jnp.allclose(result.val, np.array([True, False, True, True, False]))


def test_le_method_overload_numpy_array():
    """Test less than or equal with regular NumPy arrays"""
    array1 = np.array([3.4, 7.1, 2.8, 9.3, 6.2])
    array2 = np.array([7.9, 3.1, 2.8, 23.4, 23.4])

    result = array1 <= array2

    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert np.allclose(result, np.array([True, False, True, True, True]))


def test_le_method_overload_jax_array():
    """Test less than or equal with regular Jax arrays"""
    array1 = jnp.array([3.4, 7.1, 2.8, 9.3, 6.2])
    array2 = jnp.array([7.9, 3.1, 2.8, 23.4, 23.4])

    result = array1 <= array2

    assert isinstance(result, jax.Array)
    assert result.dtype == bool
    assert np.allclose(result, jnp.array([True, False, True, True, True]))


def test_le_same_units_success():
    """Test less than or equal with same units"""
    unit_m = Unit(scale=0, dim={SI.m: 1})
    u1 = Unitful(val=jnp.array(5.0), unit=unit_m)
    u2 = Unitful(val=jnp.array(5.0), unit=unit_m)
    result = jnp.less_equal(u1, u2)  # type: ignore

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, True)


def test_le_untiful_jitted():
    """Test less than or equal within a JIT-compiled function when input as unitful"""

    length_unit = Unit(scale=-2, dim={SI.m: 1})
    lengths1 = Unitful(
        val=jnp.array([12.0, 25.5, 18.3, 30.1, 22.7]),
        unit=length_unit,
        static_arr=np.array([12.0, 25.5, 18.3, 30.1, 22.7]),
    )
    lengths2 = Unitful(
        val=jnp.array([12.0, 14.5, 100.3, 302.1, 2.7]),
        unit=length_unit,
        static_arr=np.array([12.0, 14.5, 100.3, 302.1, 2.7]),
    )

    def fn(x: Unitful, y: Unitful) -> Unitful:
        return x <= y

    jitted_fn = jax.jit(fn)
    result = jitted_fn(lengths1, lengths2)

    assert isinstance(result, Unitful)
    assert isinstance(result.val, jax.Array)
    assert isinstance(result.static_arr, np.ndarray)
    assert jnp.allclose(result.val, jnp.array([True, False, True, True, False]))


def test_le_jitted_static():
    """Test less than or equal within a JIT-compiled function when input as unitful with static arrays"""
    length_unit = Unit(scale=-2, dim={SI.m: 1})
    lengths1 = Unitful(
        val=jnp.array([12.0, 5.5, 18.3, 30.1]),
        unit=length_unit,
        static_arr=np.array([12.0, 5.5, 18.3, 30.1]),
    )
    lengths2 = Unitful(
        val=jnp.array([12.0, 14.5, 1.3, 3012.1]),
        unit=length_unit,
        static_arr=np.array([12.0, 14.5, 100.3, 3012.1]),
    )

    def fn(x: Unitful, y: Unitful) -> Unitful:
        if is_currently_compiling() and not unitful.STATIC_OPTIM_STOP_FLAG:
            assert x.static_arr is not None
            assert y.static_arr is not None
        result = x <= y
        if is_currently_compiling() and not unitful.STATIC_OPTIM_STOP_FLAG:
            assert result.static_arr is not None
        return result

    jitted_fn = jax.jit(fn)
    result = jitted_fn(lengths1, lengths2)

    assert isinstance(result, Unitful)
    assert isinstance(result.val, jax.Array)
    assert isinstance(result.static_arr, np.ndarray)
    assert jnp.allclose(result.val, jnp.array([True, True, False, True]))


def test_le_different_units_raises_error():
    """Test that le with different units raises ValueError"""
    unit_m = Unit(scale=0, dim={SI.m: 1})
    unit_s = Unit(scale=0, dim={SI.s: 1})
    u1 = Unitful(val=jnp.array(5.0), unit=unit_m)
    u2 = Unitful(val=jnp.array(3.0), unit=unit_s)
    with pytest.raises(Exception):
        le(u1, u2)


def test_lt_magic_method():
    """Test less than magic method of Unitful objects"""
    time1 = 2 * s
    time2 = 3 * s
    result = jax.jit(lambda: time1 < time2)()
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, True)


def test_lt_same_units_success():
    """Test less than with same units"""
    unit_kg = Unit(scale=0, dim={SI.kg: 1})
    u1 = Unitful(val=jnp.array(4.0), unit=unit_kg)
    u2 = Unitful(val=jnp.array(10.0), unit=unit_kg)
    result = jnp.less(u1, u2)  # type: ignore

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, True)


def test_lt_different_units_raises_error():
    """Test that lt with different units raises ValueError"""
    unit_K = Unit(scale=0, dim={SI.K: 1})
    unit_A = Unit(scale=0, dim={SI.A: 1})
    u1 = Unitful(val=jnp.array(300.0), unit=unit_K)
    u2 = Unitful(val=jnp.array(2.0), unit=unit_A)
    with pytest.raises(Exception):
        lt(u1, u2)


def test_eq_magic_method():
    """Test equality magic method of Unitful objects"""
    time1 = 5 * s
    time2 = 5 * s
    result = time1 == time2
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, True)


def test_eq_method_with_jax_arrays():
    """Test less than or equal with Unitful objects containing Jax arrays"""
    length_unit = Unit(scale=-2, dim={SI.m: 1})
    lengths1 = Unitful(val=jnp.array([12.0, 25.5, 18.3, 30.1, 22.7]), unit=length_unit)
    lengths2 = Unitful(val=jnp.array([12.0, 14.5, 100.3, 302.1, 22.7]), unit=length_unit)
    result = lengths1 == lengths2

    assert isinstance(result, Unitful)
    assert isinstance(result.val, jax.Array)
    assert result.val.dtype == bool
    assert jnp.allclose(result.val, np.array([True, False, False, False, True]))


def test_eq_same_units_success():
    """Test equality with same units"""
    unit_kg = Unit(scale=0, dim={SI.kg: 1})
    u1 = Unitful(val=jnp.array(7.0), unit=unit_kg)
    u2 = Unitful(val=jnp.array(7.0), unit=unit_kg)
    result = jnp.equal(u1, u2)  # type: ignore
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, True)


def test_eq_different_units_raises_error():
    """Test that eq with different units raises ValueError"""
    unit_kg = Unit(scale=0, dim={SI.kg: 1})
    unit_cd = Unit(scale=0, dim={SI.cd: 1})
    u1 = Unitful(val=jnp.array(7.0), unit=unit_kg)
    u2 = Unitful(val=jnp.array(7.0), unit=unit_cd)
    with pytest.raises(Exception):
        eq(u1, u2)


def test_neq_magic_method():
    """Test not equal magic method of Unitful objects"""
    time1 = 5 * s
    time2 = 3 * s
    result = time1 != time2
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, True)


def test_neq_same_units_success():
    """Test not equal with same units"""
    unit_cd = Unit(scale=0, dim={SI.cd: 1})
    u1 = Unitful(val=jnp.array(5.0), unit=unit_cd)
    u2 = Unitful(val=jnp.array(3.0), unit=unit_cd)
    result = jnp.not_equal(u1, u2)  # type: ignore
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, True)


def test_neq_different_units_raises_error():
    """Test that neq with different units raises ValueError"""
    unit_m = Unit(scale=0, dim={SI.m: 1})
    unit_kg = Unit(scale=0, dim={SI.kg: 1})
    u1 = Unitful(val=jnp.array(5.0), unit=unit_m)
    u2 = Unitful(val=jnp.array(5.0), unit=unit_kg)
    with pytest.raises(Exception):
        ne(u1, u2)


def test_ge_magic_method():
    """Test greater than or equal magic method of Unitful objects"""
    time1 = 3 * s
    time2 = 2 * s
    result = time1 >= time2
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, True)


def test_ge_method_with_jax_arrays():
    """Test less than or equal with Unitful objects containing Jax arrays"""
    length_unit = Unit(scale=-2, dim={SI.m: 1})
    lengths1 = Unitful(val=jnp.array([12.0, 25.5, 18.3, 30.1, 22.7]), unit=length_unit)
    lengths2 = Unitful(val=jnp.array([12.0, 14.5, 100.3, 302.1, 22.7]), unit=length_unit)
    result = lengths1 >= lengths2

    assert isinstance(result, Unitful)
    assert isinstance(result.val, jax.Array)
    assert result.val.dtype == bool
    assert jnp.allclose(result.val, np.array([True, True, False, False, True]))


def test_ge_same_units_success():
    """Test greater than or equal with same units"""
    unit_A = Unit(scale=0, dim={SI.A: 1})
    u1 = Unitful(val=jnp.array(8.0), unit=unit_A)
    u2 = Unitful(val=jnp.array(8.0), unit=unit_A)
    result = jnp.greater_equal(u1, u2)  # type: ignore
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, True)


def test_ge_different_units_raises_error():
    """Test that ge with different units raises ValueError"""
    unit_s = Unit(scale=0, dim={SI.s: 1})
    unit_m = Unit(scale=0, dim={SI.m: 1})
    u1 = Unitful(val=jnp.array(10.0), unit=unit_s)
    u2 = Unitful(val=jnp.array(3.0), unit=unit_m)
    with pytest.raises(Exception):
        ge(u1, u2)


def test_gt_magic_method():
    """Test greater than magic method of Unitful objects"""
    time1 = 5 * s
    time2 = 3 * s
    result = time1 > time2
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, True)


def test_gt_same_units_success():
    """Test greater than with same units"""
    unit_K = Unit(scale=0, dim={SI.K: 1})
    u1 = Unitful(val=jnp.array(400.0), unit=unit_K)
    u2 = Unitful(val=jnp.array(300.0), unit=unit_K)
    result = jnp.greater(u1, u2)  # type: ignore
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, True)


def test_gt_different_units_raises_error():
    """Test that gt with different units raises ValueError"""
    unit_kg = Unit(scale=0, dim={SI.kg: 1})
    unit_A = Unit(scale=0, dim={SI.A: 1})
    u1 = Unitful(val=jnp.array(15.0), unit=unit_kg)
    u2 = Unitful(val=jnp.array(5.0), unit=unit_A)
    with pytest.raises(Exception):
        gt(u1, u2)


def test_multiply_unitful_with_fractional_dimensions():
    """Test multiplication creating and using fractional dimensions"""
    # Create a unit with m^(1/2) dimension (like square root of area)
    unit_sqrt_m = Unit(scale=0, dim={SI.m: Fraction(1, 2)})
    u1 = Unitful(val=jnp.array(4.0), unit=unit_sqrt_m)
    u2 = Unitful(val=jnp.array(3.0), unit=unit_sqrt_m)

    # Multiplying m^(1/2) * m^(1/2) should give m^1
    result = multiply(u1, u2)

    assert jnp.allclose(result.value(), 12.0)
    assert result.unit.dim == {SI.m: 1}  # 1/2 + 1/2 = 1


def test_multiply_fractional_with_integer_dimensions():
    """Test multiplication of fractional and integer dimensions"""
    # Create units: m^(1/3) and m^(2/3)
    unit_cube_root_m = Unit(scale=0, dim={SI.m: Fraction(1, 3)})
    unit_two_thirds_m = Unit(scale=0, dim={SI.m: Fraction(2, 3)})

    u1 = Unitful(val=jnp.array(8.0), unit=unit_cube_root_m)
    u2 = Unitful(val=jnp.array(2.0), unit=unit_two_thirds_m)

    # m^(1/3) * m^(2/3) = m^1
    result = multiply(u1, u2)

    assert jnp.allclose(result.value(), 16.0)
    assert result.unit.dim == {SI.m: 1}  # 1/3 + 2/3 = 1


def test_multiply_fractional_dimensions_cancel_out():
    """Test that fractional dimensions can cancel out to become dimensionless"""
    # Create units: s^(3/4) and s^(-3/4)
    unit_pos_frac = Unit(scale=0, dim={SI.s: Fraction(3, 4)})
    unit_neg_frac = Unit(scale=0, dim={SI.s: Fraction(-3, 4)})

    u1 = Unitful(val=jnp.array(5.0), unit=unit_pos_frac)
    u2 = Unitful(val=jnp.array(7.0), unit=unit_neg_frac)

    # s^(3/4) * s^(-3/4) = s^0 = dimensionless
    result = multiply(u1, u2)

    assert jnp.allclose(result.value(), 35.0)
    assert result.unit.dim == {}  # 3/4 + (-3/4) = 0, so dimension is removed


def test_add_same_fractional_dimensions():
    """Test addition of unitful objects with the same fractional dimensions"""
    # Create two units with kg^(2/5) dimension
    unit_frac_kg = Unit(scale=0, dim={SI.kg: Fraction(2, 5)})
    u1 = Unitful(val=jnp.array(10.0), unit=unit_frac_kg)
    u2 = Unitful(val=jnp.array(15.0), unit=unit_frac_kg)

    result = add(u1, u2)

    assert jnp.allclose(result.value(), 25.0)
    assert result.unit.dim == {SI.kg: Fraction(2, 5)}


def test_equality_fractional_dimensions():
    """Test equality comparison with fractional dimensions"""
    # Create units with A^(7/11) dimension
    unit_frac_A = Unit(scale=0, dim={SI.A: Fraction(7, 11)})
    u1 = Unitful(val=jnp.array(42.0), unit=unit_frac_A)
    u2 = Unitful(val=jnp.array(42.0), unit=unit_frac_A)
    u3 = Unitful(val=jnp.array(43.0), unit=unit_frac_A)

    # Test equality
    result_equal = eq(u1, u2)
    result_not_equal = eq(u1, u3)

    assert isinstance(result_equal, Unitful)
    assert isinstance(result_not_equal, Unitful)
    assert jnp.allclose(result_equal.val, True)
    assert jnp.allclose(result_not_equal.val, False)

    # Also test that the units are properly preserved
    assert u1.unit.dim == {SI.A: Fraction(7, 11)}
    assert u2.unit.dim == {SI.A: Fraction(7, 11)}


# Division Test Cases
def test_divide_unitful_unitful_same_dimensions():
    """Test division of two Unitful objects with same dimensions"""
    distance1 = Unit(scale=0, dim={SI.m: 1})
    distance2 = Unit(scale=0, dim={SI.m: 1})
    u1 = Unitful(val=jnp.array(10.0), unit=distance1)
    u2 = Unitful(val=jnp.array(2.0), unit=distance2)

    result = u1 / u2

    assert jnp.allclose(result.value(), 5.0)
    assert result.unit.dim == {}  # m^1 / m^1 = dimensionless


def test_divide_unitful_unitful_different_dimensions():
    """Test division of two Unitful objects with different dimensions"""
    distance = Unit(scale=0, dim={SI.m: 1})
    time = Unit(scale=0, dim={SI.s: 1})
    u1 = Unitful(val=jnp.array(20.0), unit=distance)  # 20 meters
    u2 = Unitful(val=jnp.array(4.0), unit=time)  # 4 seconds

    result: Unitful = jnp.divide(u1, u2)  # type: ignore

    assert jnp.allclose(result.value(), 5.0)  # 20/4 = 5
    assert result.unit.dim == {SI.m: 1, SI.s: -1}  # m/s


def test_divide_unitful_unitful_with_fractional_dimensions():
    """Test division creating fractional dimensions"""
    # Create m^2 (area) and m^(1/2)
    area_unit = Unit(scale=0, dim={SI.m: 2})
    sqrt_unit = Unit(scale=0, dim={SI.m: Fraction(1, 2)})
    u1 = Unitful(val=jnp.array(16.0), unit=area_unit)
    u2 = Unitful(val=jnp.array(4.0), unit=sqrt_unit)

    # m^2 / m^(1/2) = m^(3/2)
    result: Unitful = jnp.divide(u1, u2)  # type: ignore

    assert jnp.allclose(result.value(), 4.0)
    assert result.unit.dim == {SI.m: Fraction(3, 2)}


def test_divide_magic_method():
    """Test division using magic method (__truediv__)"""
    velocity_unit = Unit(scale=0, dim={SI.m: 1, SI.s: -1})
    time_unit = Unit(scale=0, dim={SI.s: 1})
    velocity = Unitful(val=jnp.array(30.0), unit=velocity_unit)  # 30 m/s
    time = Unitful(val=jnp.array(6.0), unit=time_unit)  # 6 s

    # This would require implementing __truediv__ in the Unitful class
    # result = velocity / time  # Should give acceleration (m/s^2)
    result: Unitful = jnp.divide(velocity, time)  # type: ignore

    assert jnp.allclose(result.value(), 5.0)
    assert result.unit.dim == {SI.m: 1, SI.s: -2}  # m/s^2 (acceleration)


def test_len_scalar_unitful():
    """Test __len__ method with scalar Unitful object"""
    time_unit = Unit(scale=0, dim={SI.s: 1})
    scalar_time = Unitful(val=5.0, unit=time_unit)

    result = len(scalar_time)

    assert result == 1


def test_len_array_unitful():
    """Test __len__ method with array Unitful object"""
    time_unit = Unit(scale=0, dim={SI.s: 1})
    array_time = Unitful(val=jnp.array([1.0, 2.0, 3.0, 4.0]), unit=time_unit)

    result = len(array_time)

    assert result == 4


def test_getitem_single_index():
    """Test __getitem__ method with single index"""
    distance_unit = Unit(scale=0, dim={SI.m: 1})
    distances = Unitful(val=jnp.array([10.0, 20.0, 30.0]), unit=distance_unit)

    result = distances[1]

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 20.0)
    assert result.unit.dim == {SI.m: 1}


def test_getitem_slice():
    """Test __getitem__ method with slice"""
    mass_unit = Unit(scale=0, dim={SI.kg: 1})
    masses = Unitful(val=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), unit=mass_unit)

    result = masses[1:4]

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), jnp.array([2.0, 3.0, 4.0]))
    assert result.unit.dim == {SI.kg: 1}


def test_iter_unitful_array():
    """Test __iter__ method with Unitful array"""
    current_unit = Unit(scale=0, dim={SI.A: 1})
    currents = Unitful(val=jnp.array([1.0, 2.0, 3.0]), unit=current_unit)

    result_list = list(currents)

    assert len(result_list) == 3
    for i, item in enumerate(result_list):
        assert isinstance(item, Unitful)
        assert jnp.allclose(item.value(), float(i + 1))
        assert item.unit.dim == {SI.A: 1}


def test_iter_unitful_scalar_raises_exception():
    """Test that __iter__ raises exception for scalar Unitful"""
    temp_unit = Unit(scale=0, dim={SI.K: 1})
    scalar_temp = Unitful(val=300.0, unit=temp_unit)

    with pytest.raises(Exception, match="Cannot iterate over Unitful with python scalar value"):
        list(scalar_temp)


def test_reversed_unitful_array():
    """Test __reversed__ method with Unitful array"""
    energy_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -2})  # Joules
    energies = Unitful(val=jnp.array([10.0, 20.0, 30.0]), unit=energy_unit)

    result_list = list(reversed(energies))

    assert len(result_list) == 3
    expected_values = [30.0, 20.0, 10.0]
    for i, item in enumerate(result_list):
        assert isinstance(item, Unitful)
        assert jnp.allclose(item.value(), expected_values[i])
        assert item.unit.dim == {SI.kg: 1, SI.m: 2, SI.s: -2}


def test_reversed_preserves_units():
    """Test that __reversed__ preserves complex units correctly"""
    # Create a complex unit: m^(3/2) * kg^(-1/3)
    complex_unit = Unit(scale=2, dim={SI.m: Fraction(3, 2), SI.kg: Fraction(-1, 3)})
    values = Unitful(val=jnp.array([1.0, 4.0, 9.0, 16.0]), unit=complex_unit)

    reversed_values = list(reversed(values))

    assert len(reversed_values) == 4
    expected_order = [16.0, 9.0, 4.0, 1.0]
    for i, item in enumerate(reversed_values):
        assert isinstance(item, Unitful)
        assert jnp.allclose(item.value(), expected_order[i] * 100)  # scale=2 means *100
        assert item.unit.dim == {SI.m: Fraction(3, 2), SI.kg: Fraction(-1, 3)}


def test_neg_scalar_unitful():
    """Test __neg__ method with scalar Unitful object"""
    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})  # Newtons
    force = Unitful(val=jnp.array(15.0), unit=force_unit)

    result = -force

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), -15.0)
    assert result.unit.dim == {SI.kg: 1, SI.m: 1, SI.s: -2}
    assert result.unit.scale == force.unit.scale


def test_neg_array_unitful():
    """Test __neg__ method with array Unitful object"""
    voltage_unit = Unit(scale=-3, dim={SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -1})  # millivolts
    voltages = Unitful(val=jnp.array([1.0, -2.0, 3.0, -4.0]), unit=voltage_unit)

    result = -voltages

    assert isinstance(result, Unitful)
    expected_values = jnp.array([-1.0, 2.0, -3.0, 4.0]) * (10**-3)
    assert jnp.allclose(result.value(), expected_values)  # Check the raw values
    assert result.unit.dim == voltages.unit.dim


def test_abs_scalar_unitful_negative():
    """Test __abs__ method with negative scalar Unitful object"""
    charge_unit = Unit(scale=-6, dim={SI.A: 1, SI.s: 1})  # microCoulombs
    charge = Unitful(val=jnp.array(-15.0), unit=charge_unit)

    result = abs(charge)

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 15.0e-6)  # 15 microCoulombs
    assert result.unit.dim == {SI.A: 1, SI.s: 1}
    assert result.unit.scale == charge.unit.scale


def test_abs_scalar_unitful_numpy_negative():
    """Test __abs__ method with negative numpy Unitful object"""
    charge_unit = Unit(scale=-6, dim={SI.A: 1, SI.s: 1})  # microCoulombs
    charge = Unitful(val=np.array([-15.0, -34.9]), unit=charge_unit)

    result = abs(charge)

    assert isinstance(result, Unitful)
    assert isinstance(result.val, np.ndarray)
    assert np.allclose(result.value(), np.array([15.0e-6, 34.9e-6]))  # 15 microCoulombs
    assert result.unit.dim == {SI.A: 1, SI.s: 1}
    assert result.unit.scale == charge.unit.scale


def test_abs_array_unitful_mixed_signs():
    """Test __abs__ method with array Unitful object containing mixed positive/negative values"""
    # Create power unit: kg * m^2 * s^(-3) (Watts)
    power_unit = Unit(scale=3, dim={SI.kg: 1, SI.m: 2, SI.s: -3})  # kiloWatts
    powers = Unitful(val=jnp.array([-2.5, 0.0, 3.7, -1.2, 4.8]), unit=power_unit)

    result = abs(powers)

    assert isinstance(result, Unitful)
    expected_values = jnp.array([2.5, 0.0, 3.7, 1.2, 4.8]) * (10**3)
    assert jnp.allclose(result.value(), expected_values)
    assert result.unit.dim == {SI.kg: 1, SI.m: 2, SI.s: -3}


def test_matmul_magic_method_same_units_with_jax_array():
    """Test matrix multiplication using @ operator with same units"""
    # Create two 2x2 matrices with force units (kg*m*s^-2)
    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})
    matrix1 = Unitful(val=jnp.array([[2.0, 3.0], [4.0, 1.0]]), unit=force_unit)
    matrix2 = Unitful(val=jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=force_unit)

    result = matrix1 @ matrix2

    # Expected calculation: [[2*1+3*3, 2*2+3*4], [4*1+1*3, 4*2+1*4]] = [[11, 16], [7, 12]]
    expected_vals = jnp.array([[11.0, 16.0], [7.0, 12.0]])
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), expected_vals)
    assert isinstance(result.val, jax.Array)
    # Force^2 units: (kg*m*s^-2)^2 = kg^2*m^2*s^-4
    assert result.unit.dim == {SI.kg: 2, SI.m: 2, SI.s: -4}


def test_matmul_magic_method_same_units_with_numpy_array():
    """Test matrix multiplication using @ operator with same units"""
    # Create two 2x2 matrices with force units (kg*m*s^-2)
    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})
    matrix1 = Unitful(val=np.array([[2.0, 3.0], [4.0, 1.0]]), unit=force_unit)
    matrix2 = Unitful(val=np.array([[1.0, 2.0], [3.0, 4.0]]), unit=force_unit)

    result = matrix1 @ matrix2

    # Expected calculation: [[2*1+3*3, 2*2+3*4], [4*1+1*3, 4*2+1*4]] = [[11, 16], [7, 12]]
    expected_vals = np.array([[11.0, 16.0], [7.0, 12.0]])
    assert isinstance(result, Unitful)
    assert np.allclose(result.value(), expected_vals)
    assert isinstance(result.val, np.ndarray)
    # Force^2 units: (kg*m*s^-2)^2 = kg^2*m^2*s^-4
    assert result.unit.dim == {SI.kg: 2, SI.m: 2, SI.s: -4}


def test_matmul_magic_method_same_units_with_mixed_array():
    """Test matrix multiplication using @ operator with same units"""
    # Create two 2x2 matrices with force units (kg*m*s^-2)
    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})
    matrix1 = Unitful(val=np.array([[2.0, 3.0], [4.0, 1.0]]), unit=force_unit)
    matrix2 = Unitful(val=jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=force_unit)

    result = matrix1 @ matrix2

    # Expected calculation: [[2*1+3*3, 2*2+3*4], [4*1+1*3, 4*2+1*4]] = [[11, 16], [7, 12]]
    expected_vals = jnp.array([[11.0, 16.0], [7.0, 12.0]])
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), expected_vals)
    assert isinstance(result.val, jax.Array)
    # Force^2 units: (kg*m*s^-2)^2 = kg^2*m^2*s^-4
    assert result.unit.dim == {SI.kg: 2, SI.m: 2, SI.s: -4}


def test_matmul_overload_different_scales():
    """Test matmul function with Unitful objects having different scales but same dimensions"""
    # Create matrices with time units at different scales
    time_unit_s = Unit(scale=0, dim={SI.s: 1})  # seconds
    time_unit_ms = Unit(scale=-3, dim={SI.s: 1})  # milliseconds

    matrix1 = Unitful(val=jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=time_unit_s)
    matrix2 = Unitful(val=jnp.array([[500.0, 1000.0], [1500.0, 2000.0]]), unit=time_unit_ms)

    result = jnp.matmul(matrix1, matrix2)  # type: ignore

    # The scales should be aligned before multiplication
    # matrix2 values become [0.5, 1.0], [1.5, 2.0] after scale alignment
    # Expected: [[1*0.5+2*1.5, 1*1+2*2], [3*0.5+4*1.5, 3*1+4*2]] = [[3.5, 5], [7.5, 11]]
    expected_vals = jnp.array([[3.5, 5.0], [7.5, 11.0]])
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), expected_vals)
    # Time^2 units: s^2
    assert result.unit.dim == {SI.s: 2}


def test_matmul_overload_jax_arrays():
    """Test matmul function with regular JAX arrays (non-Unitful)"""
    # Test that the overloaded matmul still works with regular JAX arrays
    array1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    array2 = jnp.array([[5.0, 6.0], [7.0, 8.0]])

    result = matmul(array1, array2)

    # Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    expected = jnp.array([[19.0, 22.0], [43.0, 50.0]])
    assert jnp.allclose(result, expected)
    # Should return a regular JAX array, not a Unitful object
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)


def test_matmul_magic_method_same_units_with_numpy_value():
    """Test matmul function with Unitful numpy input"""
    # Create two 2x2 matrices with force units (kg*m*s^-2)
    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})
    matrix1 = Unitful(val=np.array([[2.0, 3.0], [4.0, 1.0]]), unit=force_unit)
    matrix2 = Unitful(val=np.array([[1.0, 2.0], [3.0, 4.0]]), unit=force_unit)

    result = matrix1 @ matrix2

    # Expected calculation: [[2*1+3*3, 2*2+3*4], [4*1+1*3, 4*2+1*4]] = [[11, 16], [7, 12]]
    expected_vals = np.array([[11.0, 16.0], [7.0, 12.0]])
    assert isinstance(result, Unitful)
    assert isinstance(result.val, np.ndarray)
    assert np.allclose(result.value(), expected_vals)
    # Force^2 units: (kg*m*s^-2)^2 = kg^2*m^2*s^-4
    assert result.unit.dim == {SI.kg: 2, SI.m: 2, SI.s: -4}


def test_matmul_overload_numpy_arrays():
    """Test matmul function with regular numpy arrays (non-Unitful)"""
    array1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    array2 = np.array([[5.0, 6.0], [7.0, 8.0]])

    result = matmul(array1, array2)

    # Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(result, expected)
    assert isinstance(result, np.ndarray)
    assert not isinstance(result, Unitful)


def test_pow_magic_method():
    """Test power operation using magic method (**) with integer exponent"""
    # Create a length unit (meters)
    length_unit = Unit(scale=0, dim={SI.m: 1})
    length = Unitful(val=jnp.array(3.0), unit=length_unit)

    # Test cubing: m^3 (volume)
    result = length**3

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 27.0)  # 3^3 = 27
    assert result.unit.dim == {SI.m: 3}


def test_pow_overload_unitful_positive_exponent():
    """Test pow function with Unitful object and positive integer exponent"""
    # Create a velocity unit: m/s
    velocity_unit = Unit(scale=0, dim={SI.m: 1, SI.s: -1})
    velocity = Unitful(val=jnp.array(5.0), unit=velocity_unit)

    # Square it to get m^2/s^2 (like kinetic energy per unit mass)
    result = pow(velocity, 2)

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 25.0)  # 5^2 = 25
    assert result.unit.dim == {SI.m: 2, SI.s: -2}


def test_pow_overload_unitful_with_fractional_dimensions():
    """Test pow function with Unitful object containing fractional dimensions"""
    # Create a unit with fractional dimension: m^(2/3)
    fractional_unit = Unit(scale=0, dim={SI.m: Fraction(2, 3)})
    value = Unitful(val=jnp.array(8.0), unit=fractional_unit)

    # Raise to power 3: (m^(2/3))^3 = m^2
    result = pow(value, 3)

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 512.0)  # 8^3 = 512
    assert result.unit.dim == {SI.m: 2}  # (2/3) * 3 = 2


def test_pow_overload_jax_arrays():
    """Test pow function with regular JAX arrays (non-Unitful)"""
    # Test that the overloaded pow still works with regular JAX arrays
    base = jnp.array([2.0, 3.0, 4.0])
    exponent = jnp.array([2.0, 3.0, 2.0])

    result = jax.lax.pow(base, exponent)

    # Expected: [2^2, 3^3, 4^2] = [4, 27, 16]
    expected = jnp.array([4.0, 27.0, 16.0])
    assert jnp.allclose(result, expected)
    # Should return a regular JAX array, not a Unitful object
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)


def test_pow_unitful_int_exponent_jax_value_dimful():
    """Unitful + int exponent, JAX value, dimensionful unit"""
    length_unit = Unit(scale=0, dim={SI.m: 1})
    length = Unitful(val=jnp.array(3.0), unit=length_unit)

    result = pow(length, 2)

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 9.0)
    # dim^2, scale * 2
    assert result.unit.dim == {SI.m: 2}
    assert result.unit.scale == 1  # scale is optimized by unitful


def test_pow_unitful_int_exponent_numpy_value_dimful():
    """Unitful + int exponent, numpy scalar value, dimensionful unit"""
    length_unit = Unit(scale=1, dim={SI.m: 1})
    length = Unitful(val=np.array([2.0]), unit=length_unit)

    result = pow(length, 3)

    assert isinstance(result, Unitful)
    assert result.value() == 8000.0
    assert result.unit.dim == {SI.m: 3}
    assert result.unit.scale == 4  # scale is optimized by unitful


def test_pow_unitful_fractional_dimension_int_exponent():
    """Fractional dimensions should be multiplied correctly by integer exponent"""
    frac_unit = Unit(scale=0, dim={SI.m: Fraction(1, 2)})
    x = Unitful(val=jnp.array(9.0), unit=frac_unit)

    result = pow(x, 4)

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 9.0**4)
    # (1/2) * 4 = 2
    assert result.unit.dim == {SI.m: 2}


# ---------- tests for non-Unitful overloads ----------
def test_pow_scalar_overloads_basic():
    """Test scalar overloads for int/float/complex combinations"""
    assert pow(2, 3) == 8
    assert pow(2, 0.5) == math.pow(2, 0.5)
    assert pow(2.0, 3) == 8.0
    assert pow(2.0, 2.0) == 4.0
    assert pow(1 + 2j, 2) == (-3 + 4j)

    z = 1 + 1j
    expected = cmath.sqrt(z)
    result = pow(z, 0.5)
    assert np.allclose(result.real, expected.real)
    assert np.allclose(result.imag, expected.imag)


def test_pow_unitful_int_exponent_jit():
    """JIT-compiled pow with Unitful + int exponent should work and preserve units"""

    length_unit = Unit(scale=0, dim={SI.m: 1})
    length = Unitful(val=jnp.array(3.0), unit=length_unit)

    @jax.jit
    def cube(u: Unitful) -> Unitful:
        return pow(u, 3)

    result = cube(length)

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, 27.0)
    assert result.unit.dim == {SI.m: 3}


def test_at_get_single_index():
    """Test getting a single value using .at[].get()"""
    distance_unit = Unit(scale=0, dim={SI.m: 1})
    distances = Unitful(val=jnp.array([10.0, 20.0, 30.0, 40.0]), unit=distance_unit)

    result = distances.at[2].get()

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 30.0)
    assert result.unit.dim == {SI.m: 1}


def test_at_get_slice():
    """Test getting a slice using .at[].get()"""
    time_unit = Unit(scale=-3, dim={SI.s: 1})  # milliseconds
    times = Unitful(val=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), unit=time_unit)

    result = times.at[1:4].get()

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), jnp.array([2.0e-3, 3.0e-3, 4.0e-3]))
    assert result.unit.dim == {SI.s: 1}


def test_at_set_single_index_same_units():
    """Test setting a single value with same units using .at[].set()"""
    mass_unit = Unit(scale=0, dim={SI.kg: 1})
    masses = Unitful(val=jnp.array([1.0, 2.0, 3.0, 4.0]), unit=mass_unit)
    new_mass = Unitful(val=jnp.array(99.0), unit=mass_unit)

    result = masses.at[2].set(new_mass)

    assert isinstance(result, Unitful)
    expected = jnp.array([1.0, 2.0, 99.0, 4.0])
    assert jnp.allclose(result.value(), expected)
    assert result.unit.dim == {SI.kg: 1}


def test_at_set_different_scales_same_dimensions():
    """Test setting values with different scales but same dimensions"""
    time_unit_s = Unit(scale=0, dim={SI.s: 1})  # seconds
    time_unit_ms = Unit(scale=-3, dim={SI.s: 1})  # milliseconds

    times = Unitful(val=jnp.array([1.0, 2.0, 3.0]), unit=time_unit_s)
    new_time = Unitful(val=jnp.array(5000.0), unit=time_unit_ms)  # 5000 ms = 5 s

    result = times.at[1].set(new_time)

    assert isinstance(result, Unitful)
    expected = jnp.array([1.0, 5.0, 3.0])  # 5000 ms converted to 5 s
    assert jnp.allclose(result.value(), expected)
    assert result.unit.dim == {SI.s: 1}


def test_at_add_single_index():
    """Test adding to a single value using .at[].add()"""
    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})  # Newtons
    forces = Unitful(val=jnp.array([10.0, 20.0, 30.0]), unit=force_unit)
    additional_force = Unitful(val=jnp.array(5.0), unit=force_unit)

    result = forces.at[1].add(additional_force)

    assert isinstance(result, Unitful)
    expected = jnp.array([10.0, 25.0, 30.0])  # 20 + 5 = 25
    assert jnp.allclose(result.value(), expected)
    assert result.unit.dim == {SI.kg: 1, SI.m: 1, SI.s: -2}


def test_at_add_with_scale_alignment():
    """Test adding values that require scale alignment"""
    energy_unit_j = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -2})  # Joules
    energy_unit_kj = Unit(scale=3, dim={SI.kg: 1, SI.m: 2, SI.s: -2})  # kiloJoules

    energies = Unitful(val=jnp.array([100.0, 200.0, 300.0]), unit=energy_unit_j)
    additional_energy = Unitful(val=jnp.array(2.0), unit=energy_unit_kj)  # 2 kJ = 2000 J

    result = energies.at[0].add(additional_energy)

    assert isinstance(result, Unitful)
    expected = jnp.array([2100.0, 200.0, 300.0])  # 100 + 2000 = 2100
    assert jnp.allclose(result.value(), expected)
    assert result.unit.dim == {SI.kg: 1, SI.m: 2, SI.s: -2}


def test_at_subtract_single_index():
    """Test subtracting from a single value using .at[].subtract()"""
    voltage_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -1})  # Volts
    voltages = Unitful(val=jnp.array([12.0, 24.0, 36.0]), unit=voltage_unit)
    voltage_drop = Unitful(val=jnp.array(3.0), unit=voltage_unit)

    result = voltages.at[2].subtract(voltage_drop)

    assert isinstance(result, Unitful)
    expected = jnp.array([12.0, 24.0, 33.0])  # 36 - 3 = 33
    assert jnp.allclose(result.value(), expected)
    assert result.unit.dim == {SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -1}


def test_at_subtract_slice_with_fractional_dimensions():
    """Test subtracting from a slice with fractional dimensions"""
    # Create a unit with fractional dimension: kg^(3/4)
    fractional_unit = Unit(scale=0, dim={SI.kg: Fraction(3, 4)})
    values = Unitful(val=jnp.array([10.0, 20.0, 30.0, 40.0]), unit=fractional_unit)
    subtract_value = Unitful(val=jnp.array([1.0, 2.0]), unit=fractional_unit)

    result = values.at[1:3].subtract(subtract_value)

    assert isinstance(result, Unitful)
    expected = jnp.array([10.0, 19.0, 28.0, 40.0])  # [20-1, 30-2] = [19, 28]
    assert jnp.allclose(result.value(), expected)
    assert result.unit.dim == {SI.kg: Fraction(3, 4)}


def test_at_multiply_single_index():
    """Test multiplying a single value using .at[].multiply()"""
    current_unit = Unit(scale=0, dim={SI.A: 1})
    currents = Unitful(val=jnp.array([1.0, 2.0, 3.0, 4.0]), unit=current_unit)
    multiplier = jnp.array(5.0)  # Scalar JAX array

    result = currents.at[2].multiply(multiplier)

    assert isinstance(result, Unitful)
    expected = jnp.array([1.0, 2.0, 15.0, 4.0])  # 3 * 5 = 15
    assert jnp.allclose(result.value(), expected)
    assert result.unit.dim == {SI.A: 1}


def test_at_multiply_array_slice():
    """Test multiplying an array slice using .at[].multiply()"""
    pressure_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: -1, SI.s: -2})  # Pascals
    pressures = Unitful(val=jnp.array([100.0, 200.0, 300.0, 400.0]), unit=pressure_unit)
    multipliers = jnp.array([2.0, 3.0])  # Array of multipliers

    result = pressures.at[1:3].multiply(multipliers)

    assert isinstance(result, Unitful)
    expected = jnp.array([100.0, 400.0, 900.0, 400.0])  # [200*2, 300*3] = [400, 900]
    assert jnp.allclose(result.value(), expected)
    assert result.unit.dim == {SI.kg: 1, SI.m: -1, SI.s: -2}


def test_at_divide_single_index():
    """Test dividing a single value using .at[].divide()"""
    frequency_unit = Unit(scale=0, dim={SI.s: -1})  # Hertz
    frequencies = Unitful(val=jnp.array([100.0, 200.0, 300.0]), unit=frequency_unit)
    divisor = jnp.array(4.0)

    result = frequencies.at[1].divide(divisor)

    assert isinstance(result, Unitful)
    expected = jnp.array([100.0, 50.0, 300.0])  # 200 / 4 = 50
    assert jnp.allclose(result.value(), expected)
    assert result.unit.dim == {SI.s: -1}


def test_at_divide_multiple_indices():
    """Test dividing multiple values using .at[].divide()"""
    temperature_unit = Unit(scale=0, dim={SI.K: 1})
    temperatures = Unitful(val=jnp.array([300.0, 400.0, 500.0, 600.0]), unit=temperature_unit)
    divisors = jnp.array([2.0, 5.0])

    result = temperatures.at[0:2].divide(divisors)

    assert isinstance(result, Unitful)
    expected = jnp.array([150.0, 80.0, 500.0, 600.0])  # [300/2, 400/5] = [150, 80]
    assert jnp.allclose(result.value(), expected)
    assert result.unit.dim == {SI.K: 1}


# Test error cases
def test_at_set_different_units_raises_error():
    """Test that setting with different units raises an exception"""
    length_unit = Unit(scale=0, dim={SI.m: 1})
    time_unit = Unit(scale=0, dim={SI.s: 1})

    lengths = Unitful(val=jnp.array([1.0, 2.0, 3.0]), unit=length_unit)
    time_value = Unitful(val=jnp.array(5.0), unit=time_unit)

    with pytest.raises(Exception, match="Cannot update array value with different unit"):
        lengths.at[1].set(time_value)


def test_at_add_different_units_raises_error():
    """Test that adding with different units raises an exception"""
    mass_unit = Unit(scale=0, dim={SI.kg: 1})
    length_unit = Unit(scale=0, dim={SI.m: 1})

    masses = Unitful(val=jnp.array([10.0, 20.0]), unit=mass_unit)
    length_value = Unitful(val=jnp.array(3.0), unit=length_unit)

    with pytest.raises(Exception, match="Cannot update array value with different unit"):
        masses.at[0].add(length_value)


def test_at_multiply_with_unitful_raises_error():
    """Test that multiplying with a Unitful object raises an exception"""
    power_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -3})  # Watts
    powers = Unitful(val=jnp.array([100.0, 200.0]), unit=power_unit)
    unitful_multiplier = Unitful(val=jnp.array(2.0), unit=power_unit)

    with pytest.raises(
        Exception, match="Multiplying part of an array with another Unitful would lead to different units"
    ):
        powers.at[0].multiply(unitful_multiplier)  # type: ignore


def test_at_divide_with_unitful_raises_error():
    """Test that dividing with a Unitful object raises an exception"""
    energy_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -2})  # Joules
    energies = Unitful(val=jnp.array([1000.0, 2000.0]), unit=energy_unit)
    unitful_divisor = Unitful(val=jnp.array(10.0), unit=energy_unit)

    with pytest.raises(
        Exception, match="Multiplying part of an array with another Unitful would lead to different units"
    ):
        energies.at[0].divide(unitful_divisor)  # type: ignore


def test_at_power_raises_error():
    """Test that the power method raises an exception"""
    area_unit = Unit(scale=0, dim={SI.m: 2})
    areas = Unitful(val=jnp.array([4.0, 9.0, 16.0]), unit=area_unit)

    with pytest.raises(Exception, match="Raising part of an array to a power is an undefined operation"):
        areas.at[1].power(2)


def test_at_scalar_value_raises_error():
    """Test that using .at operations on scalar values raises exceptions"""
    scalar_unit = Unit(scale=0, dim={SI.kg: 1})
    scalar_value = Unitful(val=42.0, unit=scalar_unit)

    # Test various operations on scalar
    with pytest.raises(Exception, match="Cannot index scalar value"):
        scalar_value.at[0].get()

    with pytest.raises(Exception, match="Cannot index scalar value"):
        scalar_value.at[0].set(Unitful(val=1.0, unit=scalar_unit))

    with pytest.raises(Exception, match="Cannot index scalar value"):
        scalar_value.at[0].add(Unitful(val=1.0, unit=scalar_unit))

    with pytest.raises(Exception, match="Cannot index scalar value"):
        scalar_value.at[0].multiply(jnp.array(2.0))


def test_at_no_where_clause_raises_error():
    """Test that operations without a where clause raise exceptions"""
    unit = Unit(scale=0, dim={SI.cd: 1})  # Candela
    values = Unitful(val=jnp.array([1.0, 2.0, 3.0]), unit=unit)

    # Test operations without indexing first
    with pytest.raises(Exception, match="Cannot update value if no where clause is given"):
        values.at.set(Unitful(val=jnp.array(5.0), unit=unit))

    with pytest.raises(Exception, match="Cannot update value if no where clause is given"):
        values.at.add(Unitful(val=jnp.array(1.0), unit=unit))

    with pytest.raises(Exception, match="Cannot update value if no where clause is given"):
        values.at.multiply(jnp.array(2.0))


def test_at_double_indexing_raises_error():
    """Test that double indexing [][] raises an exception"""
    unit = Unit(scale=0, dim={SI.m: 1})
    values = Unitful(val=jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=unit)

    with pytest.raises(Exception, match="Double Indexing .* is currently not supported"):
        values.at[0][1].get()


def test_at_get_preserves_complex_units():
    """Test that .at[].get() preserves complex units with fractional dimensions"""
    # Create a complex unit: m^(5/3) * kg^(-2/7) * s^(1/2)
    complex_unit = Unit(scale=-2, dim={SI.m: Fraction(5, 3), SI.kg: Fraction(-2, 7), SI.s: Fraction(1, 2)})
    values = Unitful(val=jnp.array([1.0, 4.0, 9.0]), unit=complex_unit)

    result = values.at[1].get()

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 4.0e-2)  # scale=-2 means *0.01
    assert result.unit.dim == {SI.m: Fraction(5, 3), SI.kg: Fraction(-2, 7), SI.s: Fraction(1, 2)}


def test_min_magic_method():
    """Test min magic method on Unitful objects"""
    temperature_unit = Unit(scale=0, dim={SI.K: 1})
    temperatures = Unitful(val=jnp.array([300.0, 250.0, 400.0, 275.0]), unit=temperature_unit)

    result = temperatures.min()

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 250.0)
    assert result.unit.dim == {SI.K: 1}


def test_min_overload_unitful():
    """Test min function with Unitful objects"""
    # Create pressure values with scale
    pressure_unit = Unit(scale=3, dim={SI.kg: 1, SI.m: -1, SI.s: -2})  # kilopascals
    pressures = Unitful(val=jnp.array([101.0, 95.5, 110.2, 88.7, 105.3]), unit=pressure_unit)

    result = jnp.min(pressures)  # type: ignore

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 88.7e3)  # 88.7 kPa = 88700 Pa
    assert result.unit.dim == {SI.kg: 1, SI.m: -1, SI.s: -2}


def test_min_overload_jax_arrays():
    """Test min function with regular JAX arrays"""
    array = jnp.array([15.0, 3.2, 42.1, 8.9, 25.6])

    result = jnp.min(array)

    assert jnp.allclose(result, 3.2)
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)


def test_min_with_axis_parameter():
    """Test min method with axis parameter on 2D Unitful array"""
    energy_unit = Unit(scale=-3, dim={SI.kg: 1, SI.m: 2, SI.s: -2})  # millijoules
    energies = Unitful(val=jnp.array([[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]]), unit=energy_unit)

    # Min along axis 0 (columns)
    result = energies.min(axis=0)

    assert isinstance(result, Unitful)
    expected_vals = jnp.array([5.0, 15.0, 25.0])
    assert jnp.allclose(result.value(), expected_vals * 1e-3)  # Convert to joules
    assert result.unit.dim == {SI.kg: 1, SI.m: 2, SI.s: -2}


def test_min_with_fractional_dimensions():
    """Test min with Unitful objects containing fractional dimensions"""
    # Create a unit with fractional dimension: m^(3/2)
    fractional_unit = Unit(scale=0, dim={SI.m: Fraction(3, 2)})
    values = Unitful(val=jnp.array([8.0, 27.0, 64.0, 125.0]), unit=fractional_unit)

    result = values.min()

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 8.0)
    assert result.unit.dim == {SI.m: Fraction(3, 2)}


def test_max_magic_method():
    """Test max magic method on Unitful objects"""
    current_unit = Unit(scale=-3, dim={SI.A: 1})  # milliamps
    currents = Unitful(val=jnp.array([150.0, 200.0, 180.0, 250.0, 120.0]), unit=current_unit)

    result = currents.max()

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 250.0e-3)  # 250 mA = 0.25 A
    assert result.unit.dim == {SI.A: 1}


def test_max_overload_unitful():
    """Test max function with Unitful objects"""
    # Create force values
    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})  # Newtons
    forces = Unitful(val=jnp.array([12.5, 8.3, 19.7, 15.2]), unit=force_unit)

    result = jnp.max(forces)  # type: ignore

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 19.7)
    assert result.unit.dim == {SI.kg: 1, SI.m: 1, SI.s: -2}


def test_max_overload_jax_arrays():
    """Test max function with regular JAX arrays"""
    array = jnp.array([7.1, 23.4, 11.8, 45.2, 19.6])

    result = jnp.max(array)

    assert jnp.allclose(result, 45.2)
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)


def test_max_with_keepdims_parameter():
    """Test max method with keepdims parameter on 2D Unitful array"""
    voltage_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -1})  # Volts
    voltages = Unitful(val=jnp.array([[12.0, 24.0], [36.0, 48.0], [6.0, 18.0]]), unit=voltage_unit)

    # Max along axis 1 with keepdims=True
    result = voltages.max(axis=1, keepdims=True)

    assert isinstance(result, Unitful)
    expected_vals = jnp.array([[24.0], [48.0], [18.0]])
    assert jnp.allclose(result.value(), expected_vals)
    assert result.unit.dim == {SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -1}
    assert result.val.shape == (3, 1)  # type: ignore


def test_max_with_complex_units():
    """Test max with complex composite units"""
    # Create power density unit: kg * s^(-3) (Watts per square meter)
    power_density_unit = Unit(scale=2, dim={SI.kg: 1, SI.s: -3})
    values = Unitful(val=jnp.array([0.5, 1.2, 0.8, 2.1, 1.5]), unit=power_density_unit)

    result = values.max()

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 2.1e2)  # scale=2 means *100
    assert result.unit.dim == {SI.kg: 1, SI.s: -3}


def test_mean_magic_method():
    """Test mean magic method on Unitful objects"""
    mass_unit = Unit(scale=0, dim={SI.kg: 1})
    masses = Unitful(val=jnp.array([10.0, 20.0, 30.0, 40.0]), unit=mass_unit)

    result = masses.mean()

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 25.0)  # (10+20+30+40)/4 = 25
    assert result.unit.dim == {SI.kg: 1}


def test_mean_overload_unitful():
    """Test mean function with Unitful objects"""
    # Create time values with millisecond scale
    time_unit = Unit(scale=-3, dim={SI.s: 1})  # milliseconds
    times = Unitful(val=jnp.array([100.0, 200.0, 300.0, 400.0, 500.0]), unit=time_unit)

    result = jnp.mean(times)  # type: ignore

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 300.0e-3)  # 300 ms = 0.3 s
    assert result.unit.dim == {SI.s: 1}


def test_mean_overload_jax_arrays():
    """Test mean function with regular JAX arrays"""
    array = jnp.array([2.5, 7.1, 4.8, 9.3, 6.2])

    result = jnp.mean(array)

    expected = (2.5 + 7.1 + 4.8 + 9.3 + 6.2) / 5
    assert jnp.allclose(result, expected)
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)


def test_mean_with_axis_parameter():
    """Test mean method with axis parameter on 2D Unitful array"""
    # Create a 3x4 array of distances
    distance_unit = Unit(scale=3, dim={SI.m: 1})  # kilometers
    distances = Unitful(
        val=jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]), unit=distance_unit
    )

    # Mean along axis 0 (rows)
    result = distances.mean(axis=0)

    assert isinstance(result, Unitful)
    expected_vals = jnp.array([5.0, 6.0, 7.0, 8.0])  # Column means
    assert jnp.allclose(result.value(), expected_vals * 1e3)  # Convert to meters
    assert result.unit.dim == {SI.m: 1}
    assert result.val.shape == (4,)  # type: ignore


def test_mean_with_fractional_and_negative_dimensions():
    """Test mean with complex fractional and negative dimensions"""
    # Create a unit: kg^(-2/3) * m^(4/5) * s^(-1)
    complex_unit = Unit(scale=-1, dim={SI.kg: Fraction(-2, 3), SI.m: Fraction(4, 5), SI.s: -1})
    values = Unitful(val=jnp.array([2.0, 4.0, 6.0, 8.0, 10.0]), unit=complex_unit)

    result = values.mean()

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 6.0e-1)  # mean=6.0, scale=-1 means *0.1
    assert result.unit.dim == {SI.kg: Fraction(-2, 3), SI.m: Fraction(4, 5), SI.s: -1}


def test_mean_preserves_all_unit_properties():
    """Test that mean preserves all unit properties including complex scales"""
    # Test with charge unit: A * s (Coulombs) at micro scale
    charge_unit = Unit(scale=-6, dim={SI.A: 1, SI.s: 1})
    charges = Unitful(val=jnp.array([10.0, 20.0, 30.0]), unit=charge_unit)

    result = charges.mean()  # type: ignore

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 20.0e-6)  # 20 microCoulombs
    assert result.unit.dim == {SI.A: 1, SI.s: 1}


def test_sum_magic_method():
    """Test sum magic method on Unitful objects"""
    energy_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -2})  # Joules
    energies = Unitful(val=jnp.array([10.0, 20.0, 30.0, 40.0]), unit=energy_unit)

    result = energies.sum()

    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 100.0)  # 10+20+30+40 = 100
    assert result.unit.dim == {SI.kg: 1, SI.m: 2, SI.s: -2}


def test_sum_overload_with_axis_and_scale():
    """Test sum function with axis parameter and unit scale conversion"""
    # Create power values with kilowatt scale
    power_unit = Unit(scale=3, dim={SI.kg: 1, SI.m: 2, SI.s: -3})  # kilowatts
    powers = Unitful(val=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), unit=power_unit)

    # Sum along axis 1 (rows)
    result = jnp.sum(powers, axis=1)  # type: ignore

    assert isinstance(result, Unitful)
    expected_vals = jnp.array([6.0, 15.0, 24.0])  # Row sums
    assert jnp.allclose(result.value(), expected_vals * 1e3)  # Convert to watts
    assert result.unit.dim == {SI.kg: 1, SI.m: 2, SI.s: -3}
    assert result.val.shape == (3,)  # type: ignore


def test_sum_with_fractional_dimensions_and_keepdims():
    """Test sum with complex fractional dimensions and keepdims parameter"""
    # Create a unit with mixed fractional dimensions: m^(5/4) * kg^(-1/2) * s^(3/7)
    complex_unit = Unit(scale=-2, dim={SI.m: Fraction(5, 4), SI.kg: Fraction(-1, 2), SI.s: Fraction(3, 7)})
    values = Unitful(val=jnp.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]), unit=complex_unit)

    # Sum along axis 0 with keepdims=True
    result = values.sum(axis=0, keepdims=True)

    assert isinstance(result, Unitful)
    expected_vals = jnp.array([[18.0, 24.0]])  # Column sums: [2+6+10, 4+8+12]
    assert jnp.allclose(result.value(), expected_vals * 1e-2)  # scale=-2 means *0.01
    assert result.unit.dim == {SI.m: Fraction(5, 4), SI.kg: Fraction(-1, 2), SI.s: Fraction(3, 7)}
    assert result.val.shape == (1, 2)  # type: ignore


def test_shape_scalar_python_int():
    """Test shape property with Python scalar integer"""
    time_unit = Unit(scale=0, dim={SI.s: 1})
    scalar_time = Unitful(val=5.0, unit=time_unit)

    assert scalar_time.shape == ()


def test_shape_scalar_python_float():
    """Test shape property with Python scalar float"""
    mass_unit = Unit(scale=0, dim={SI.kg: 1})
    scalar_mass = Unitful(val=2.5, unit=mass_unit)

    assert scalar_mass.shape == ()


def test_shape_scalar_python_complex():
    """Test shape property with Python scalar complex"""
    impedance_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -2})
    scalar_impedance = Unitful(val=3 + 4j, unit=impedance_unit)

    assert scalar_impedance.shape == ()


def test_shape_1d_array():
    """Test shape property with 1D JAX array"""
    distance_unit = Unit(scale=0, dim={SI.m: 1})
    distances = Unitful(val=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), unit=distance_unit)

    assert distances.shape == (5,)


def test_shape_2d_array():
    """Test shape property with 2D JAX array"""
    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})
    forces = Unitful(val=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), unit=force_unit)

    assert forces.shape == (2, 3)


def test_shape_3d_array():
    """Test shape property with 3D JAX array"""
    voltage_unit = Unit(scale=-3, dim={SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -1})  # millivolts
    voltages = Unitful(val=jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), unit=voltage_unit)

    assert voltages.shape == (2, 2, 2)


def test_shape_empty_array():
    """Test shape property with empty JAX array"""
    current_unit = Unit(scale=0, dim={SI.A: 1})
    empty_currents = Unitful(val=jnp.array([]), unit=current_unit)

    assert empty_currents.shape == (0,)


def test_shape_with_fractional_dimensions():
    """Test shape property with fractional unit dimensions"""
    fractional_unit = Unit(scale=2, dim={SI.m: Fraction(3, 2), SI.kg: Fraction(-1, 4)})
    values = Unitful(val=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), unit=fractional_unit)

    assert values.shape == (3, 3)


def test_dtype_float32_array():
    """Test dtype property with float32 JAX array"""
    temp_unit = Unit(scale=0, dim={SI.K: 1})
    temperatures = Unitful(val=jnp.array([300.0, 310.0, 320.0], dtype=jnp.float32), unit=temp_unit)

    assert temperatures.dtype == jnp.float32


def test_dtype_complex64_array():
    """Test dtype property with complex64 JAX array"""
    impedance_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -2})
    impedances = Unitful(val=jnp.array([1 + 2j, 3 + 4j], dtype=jnp.complex64), unit=impedance_unit)

    assert impedances.dtype == jnp.complex64


def test_dtype_scalar_raises_exception():
    """Test that accessing dtype on scalar values raises an exception"""
    energy_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -2})
    scalar_energy = Unitful(val=42.0, unit=energy_unit)

    with pytest.raises(Exception, match="Python scalar does not have dtype attribute"):
        _ = scalar_energy.dtype


def test_dtype_python_int_scalar_raises_exception():
    """Test that accessing dtype on Python int scalar raises an exception"""
    dimensionless_unit = Unit(scale=0, dim={})
    scalar_int = Unitful(val=5, unit=dimensionless_unit)

    with pytest.raises(Exception, match="Python scalar does not have dtype attribute"):
        _ = scalar_int.dtype


def test_dtype_python_complex_scalar_raises_exception():
    """Test that accessing dtype on Python complex scalar raises an exception"""
    impedance_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -2})
    scalar_complex = Unitful(val=3 + 4j, unit=impedance_unit)

    with pytest.raises(Exception, match="Python scalar does not have dtype attribute"):
        _ = scalar_complex.dtype


def test_ndim_scalar_python_types():
    """Test ndim property with Python scalar types"""
    # Test with Python int
    int_unit = Unit(scale=0, dim={SI.kg: 1})
    scalar_int = Unitful(val=42.0, unit=int_unit)
    assert scalar_int.ndim == 0

    # Test with Python float
    float_unit = Unit(scale=0, dim={SI.cd: 1})
    scalar_float = Unitful(val=3.14, unit=float_unit)
    assert scalar_float.ndim == 0

    # Test with Python complex
    complex_unit = Unit(scale=0, dim={SI.A: 1, SI.s: 1})
    scalar_complex = Unitful(val=2 + 3j, unit=complex_unit)
    assert scalar_complex.ndim == 0


def test_ndim_1d_array():
    """Test ndim property with 1D JAX array"""
    length_unit = Unit(scale=-2, dim={SI.m: 1})  # centimeters
    lengths = Unitful(val=jnp.array([10.0, 20.0, 30.0]), unit=length_unit)

    assert lengths.ndim == 1


def test_ndim_2d_array():
    """Test ndim property with 2D JAX array"""
    area_unit = Unit(scale=0, dim={SI.m: 2})
    areas = Unitful(val=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), unit=area_unit)

    assert areas.ndim == 2


def test_ndim_3d_array():
    """Test ndim property with 3D JAX array"""
    volume_unit = Unit(scale=3, dim={SI.m: 3})  # cubic kilometers
    volumes = Unitful(val=jnp.array([[[1.0]]]), unit=volume_unit)

    assert volumes.ndim == 3


def test_ndim_4d_array():
    """Test ndim property with 4D JAX array"""
    hypervolume_unit = Unit(scale=0, dim={SI.m: 4})
    hypervolumes = Unitful(val=jnp.ones((2, 3, 4, 5)), unit=hypervolume_unit)

    assert hypervolumes.ndim == 4


def test_ndim_with_complex_fractional_units():
    """Test ndim property with complex fractional unit dimensions"""
    complex_unit = Unit(scale=-1, dim={SI.kg: Fraction(2, 3), SI.m: Fraction(-5, 7), SI.s: Fraction(4, 9)})
    values = Unitful(val=jnp.array([[[1.0, 2.0]], [[3.0, 4.0]]]), unit=complex_unit)

    assert values.ndim == 3


def test_ndim_empty_1d_array():
    """Test ndim property with empty 1D array"""
    power_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -3})
    empty_powers = Unitful(val=jnp.array([]), unit=power_unit)

    assert empty_powers.ndim == 1


def test_size_scalar_python_types():
    """Test size property with Python scalar types"""
    # Test with Python int
    frequency_unit = Unit(scale=6, dim={SI.s: -1})  # megahertz
    scalar_freq = Unitful(val=100.0, unit=frequency_unit)
    assert scalar_freq.size == 1

    # Test with Python float
    resistance_unit = Unit(scale=3, dim={SI.kg: 1, SI.m: 2, SI.s: -3, SI.A: -2})  # kiloohms
    scalar_resistance = Unitful(val=4.7, unit=resistance_unit)
    assert scalar_resistance.size == 1

    # Test with Python complex
    admittance_unit = Unit(scale=-3, dim={SI.kg: -1, SI.m: -2, SI.s: 3, SI.A: 2})  # millisiemens
    scalar_admittance = Unitful(val=1 + 2j, unit=admittance_unit)
    assert scalar_admittance.size == 1


def test_size_1d_arrays():
    """Test size property with 1D JAX arrays of various sizes"""
    # Small array
    wavelength_unit = Unit(scale=-9, dim={SI.m: 1})  # nanometers
    small_wavelengths = Unitful(val=jnp.array([400.0, 500.0, 600.0]), unit=wavelength_unit)
    assert small_wavelengths.size == 3

    # Larger array
    time_unit = Unit(scale=-6, dim={SI.s: 1})  # microseconds
    times = Unitful(val=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), unit=time_unit)
    assert times.size == 10

    # Single element array
    charge_unit = Unit(scale=-19, dim={SI.A: 1, SI.s: 1})  # elementary charge scale
    single_charge = Unitful(val=jnp.array([1.602]), unit=charge_unit)
    assert single_charge.size == 1


def test_size_2d_arrays():
    """Test size property with 2D JAX arrays"""
    # 2x3 array
    flux_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -2, SI.A: -1})  # webers
    flux_2x3 = Unitful(val=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), unit=flux_unit)
    assert flux_2x3.size == 6

    # Square array
    conductance_unit = Unit(scale=-3, dim={SI.kg: -1, SI.m: -2, SI.s: 3, SI.A: 2})  # millisiemens
    square_array = Unitful(val=jnp.ones((4, 4)), unit=conductance_unit)
    assert square_array.size == 16


def test_size_3d_arrays():
    """Test size property with 3D JAX arrays"""
    # 2x3x4 array
    magnetic_field_unit = Unit(scale=-3, dim={SI.kg: 1, SI.s: -2, SI.A: -1})  # millitesla
    field_3d = Unitful(val=jnp.zeros((2, 3, 4)), unit=magnetic_field_unit)
    assert field_3d.size == 24

    # Cubic array
    density_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: -3})
    cubic_density = Unitful(val=jnp.ones((5, 5, 5)), unit=density_unit)
    assert cubic_density.size == 125


def test_size_higher_dimensional_arrays():
    """Test size property with higher dimensional arrays"""
    # 4D array
    hyperfield_unit = Unit(scale=0, dim={SI.kg: 2, SI.m: -1, SI.s: -4})
    hyperfield_4d = Unitful(val=jnp.ones((2, 3, 4, 5)), unit=hyperfield_unit)
    assert hyperfield_4d.size == 120

    # 5D array
    exotic_unit = Unit(scale=-12, dim={SI.kg: Fraction(1, 3), SI.cd: Fraction(-2, 5)})
    exotic_5d = Unitful(val=jnp.ones((2, 2, 2, 2, 2)), unit=exotic_unit)
    assert exotic_5d.size == 32


def test_size_empty_arrays():
    """Test size property with empty arrays"""
    # Empty 1D array
    luminosity_unit = Unit(scale=0, dim={SI.cd: 1})
    empty_1d = Unitful(val=jnp.array([]), unit=luminosity_unit)
    assert empty_1d.size == 0

    # Empty 2D array
    acceleration_unit = Unit(scale=0, dim={SI.m: 1, SI.s: -2})
    empty_2d = Unitful(val=jnp.zeros((0, 5)), unit=acceleration_unit)
    assert empty_2d.size == 0


def test_size_with_fractional_and_negative_dimensions():
    """Test size property with complex unit dimensions"""
    # Complex fractional and negative dimensions
    complex_unit = Unit(
        scale=7,
        dim={SI.kg: Fraction(-3, 4), SI.m: Fraction(5, 6), SI.s: -2, SI.A: Fraction(1, 7), SI.K: Fraction(-2, 9)},
    )
    complex_values = Unitful(val=jnp.ones((3, 7, 2)), unit=complex_unit)
    assert complex_values.size == 42


def test_properties_consistency_scalar():
    """Test that all properties are consistent for scalar values"""
    capacitance_unit = Unit(scale=-12, dim={SI.kg: -1, SI.m: -2, SI.s: 4, SI.A: 2})  # picofarads
    scalar_cap = Unitful(val=100.0, unit=capacitance_unit)

    assert scalar_cap.shape == ()
    assert scalar_cap.ndim == 0
    assert scalar_cap.size == 1
    # dtype should raise exception for scalars
    with pytest.raises(Exception):
        _ = scalar_cap.dtype


def test_properties_consistency_1d_array():
    """Test that all properties are consistent for 1D arrays"""
    inductance_unit = Unit(scale=-3, dim={SI.kg: 1, SI.m: 2, SI.s: -2, SI.A: -2})  # millihenries
    array_1d = Unitful(val=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32), unit=inductance_unit)

    assert array_1d.shape == (5,)
    assert array_1d.ndim == 1
    assert array_1d.size == 5
    assert array_1d.dtype == jnp.float32


def test_abs_overload_unitful_with_mixed_signs():
    """Test abs function with Unitful objects containing mixed positive/negative values"""
    # Create electric field unit: kg * m * s^(-3) * A^(-1) (Volts per meter)
    electric_field_unit = Unit(scale=3, dim={SI.kg: 1, SI.m: 1, SI.s: -3, SI.A: -1})  # kV/m
    fields = Unitful(val=jnp.array([-15.0, 0.0, 25.0, -8.5, 12.3]), unit=electric_field_unit)

    result = abs(fields)

    assert isinstance(result, Unitful)
    expected_values = jnp.array([15.0, 0.0, 25.0, 8.5, 12.3]) * 10**3
    assert jnp.allclose(result.value(), expected_values)
    assert result.unit.dim == {SI.kg: 1, SI.m: 1, SI.s: -3, SI.A: -1}


def test_abs_overload_unitful_with_fractional_dimensions():
    """Test abs function with Unitful objects containing fractional dimensions and negative values"""
    # Create a complex unit with fractional dimensions: kg^(2/3) * m^(-1/2) * s^(3/4)
    fractional_unit = Unit(scale=-2, dim={SI.kg: Fraction(2, 3), SI.m: Fraction(-1, 2), SI.s: Fraction(3, 4)})
    values = Unitful(val=jnp.array([[-3.7, 2.1], [0.0, -8.9], [5.2, -1.4]]), unit=fractional_unit)

    result = abs(values)

    assert isinstance(result, Unitful)
    expected_values = jnp.array([[3.7, 2.1], [0.0, 8.9], [5.2, 1.4]]) * 10 ** (-2)
    assert jnp.allclose(result.value(), expected_values)
    assert result.unit.dim == {SI.kg: Fraction(2, 3), SI.m: Fraction(-1, 2), SI.s: Fraction(3, 4)}


def test_aset_unitful_should_raise():
    """Test that aset method on Unitful raises an exception (intentionally disabled)"""
    # Create a Unitful object with force units
    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})
    force = Unitful(val=jnp.array([10.0, 20.0, 30.0]), unit=force_unit)

    # Attempting to use aset should raise an exception
    with pytest.raises(Exception, match="the aset-method is unsafe for Unitful internals"):
        force.aset("val", jnp.array([40.0, 50.0, 60.0]))


def test_aset_structure_containing_unitful():
    """Test that aset works on structures containing Unitful objects"""

    # Create a structure containing Unitful objects
    @autoinit
    class PhysicsData(TreeClass):
        force: Unitful
        mass: Unitful
        name: str

    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})
    mass_unit = Unit(scale=0, dim={SI.kg: 1})

    physics_data = PhysicsData(
        force=Unitful(val=jnp.array([100.0, 200.0]), unit=force_unit),
        mass=Unitful(val=jnp.array([5.0, 10.0]), unit=mass_unit),
        name="test_data",
    )

    # Using aset on the structure should work (this tests that aset works on containers)
    new_force = Unitful(val=jnp.array([300.0, 400.0]), unit=force_unit)
    new_data = physics_data.aset("force", new_force)

    assert jnp.allclose(new_data.force.value(), jnp.array([300.0, 400.0]))
    assert new_data.force.unit.dim == {SI.kg: 1, SI.m: 1, SI.s: -2}


def test_astype_unitful_float_to_complex():
    """Test astype with Unitful object converting float to complex dtype"""
    # Create a Unitful object with float values
    voltage_unit = Unit(scale=-3, dim={SI.kg: 1, SI.m: 2, SI.A: -1, SI.s: -3})  # millivolts
    voltage = Unitful(val=jnp.array([1.5, 2.7, 3.9]), unit=voltage_unit)

    # Convert to complex dtype
    result: Unitful = jnp.astype(voltage, jnp.complex64)  # type: ignore

    assert isinstance(result, Unitful)
    # Values should be converted to complex with zero imaginary part
    expected_vals = jnp.array([1.5 + 0j, 2.7 + 0j, 3.9 + 0j], dtype=jnp.complex64) * (10**-3)
    assert jnp.allclose(result.value(), expected_vals)
    # Units should be preserved
    assert result.unit.dim == {SI.kg: 1, SI.m: 2, SI.A: -1, SI.s: -3}
    # Check dtype conversion worked
    assert result.val.dtype == jnp.complex64  # type: ignore


def test_astype_jax_array():
    """Test astype with regular JAX array"""
    # Create a float array
    array = jnp.array([1.2, 3.7, 5.8, 9.1])

    # Convert to int32
    result = jnp.astype(array, jnp.int32)

    # Should return a regular JAX array with truncated values
    expected = jnp.array([1, 3, 5, 9], dtype=jnp.int32)
    assert jnp.array_equal(result, expected)
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)
    assert result.dtype == jnp.int32


def test_unitful_astype_method():
    """Test the Unitful.astype method directly on the class"""
    # Create a Unitful object with integer values
    energy_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 2, SI.s: -2})  # kilojoules
    energy = Unitful(val=jnp.array([10, 25, 40], dtype=jnp.float32), unit=energy_unit)

    # Use the instance method to convert to float64
    result = energy.astype(jnp.bfloat16)

    assert isinstance(result, Unitful)
    # Values should be converted to float
    expected_vals = jnp.array([10.0, 25.0, 40.0], dtype=jnp.bfloat16)
    assert jnp.allclose(result.value(), expected_vals)
    # Units should be preserved
    assert result.unit.dim == {SI.kg: 1, SI.m: 2, SI.s: -2}
    # Check dtype conversion worked
    assert result.val.dtype == jnp.bfloat16  # type: ignore
    # Original should be unchanged
    assert energy.val.dtype == jnp.float32  # type: ignore


def test_squeeze_unitful_with_numpy_array_val():
    """Test squeeze on Unitful with numpy array value"""
    # shape (1, 3, 1) -> squeeze -> (3,)
    time = s * np.array([[[1.0], [2.0], [3.0]]])
    assert isinstance(time, Unitful)
    assert isinstance(time.val, np.ndarray)
    assert time.val.shape == (1, 3, 1)

    result = squeeze(time)

    assert isinstance(result, Unitful)
    assert isinstance(result.val, np.ndarray)
    assert result.val.shape == (3,)
    assert np.allclose(result.value(), np.array([1.0, 2.0, 3.0]))
    assert result.unit.dim == time.unit.dim


def test_squeeze_unitful_with_jax_array_val():
    """Test squeeze on Unitful with jax Array value"""
    arr = jnp.array([[[1.0, 2.0, 3.0]]])
    time = s * arr
    assert isinstance(time.val, jax.Array)
    assert time.val.shape == (1, 1, 3)

    result = squeeze(time)

    assert isinstance(result, Unitful)
    assert isinstance(result.val, jax.Array)
    assert result.val.shape == (3,)
    assert jnp.allclose(result.value(), jnp.array([1.0, 2.0, 3.0]))
    assert result.unit.dim == time.unit.dim


def test_squeeze_unitful_python_scalar_raises():
    """Python scalar value in Unitful should raise in squeeze"""
    unit = Unit(scale=0, dim={SI.s: 1})
    x = Unitful(val=2.0, unit=unit)  # python float as val

    with pytest.raises(TypeError, match="python scalar"):
        squeeze(x)


def test_squeeze_jax_array():
    """Test squeeze on plain jax Array"""
    x = jnp.ones((1, 2, 1))
    result = squeeze(x)

    assert isinstance(result, jax.Array)
    assert result.shape == (2,)
    assert jnp.allclose(result, jnp.ones((2,)))


def test_squeeze_numpy_number_and_array():
    """Test squeeze on numpy scalar and numpy array"""
    # numpy scalar: squeeze no-op, still scalar (np.generic)
    s_np = np.float64(3.0)
    r_scalar = squeeze(s_np)
    assert isinstance(r_scalar, np.generic)
    assert r_scalar == np.float64(3.0)

    # numpy array
    arr = np.array([[1.0, 2.0, 3.0]])
    r_arr = squeeze(arr)
    assert isinstance(r_arr, np.ndarray)
    assert r_arr.shape == (3,)
    assert np.allclose(r_arr, np.array([1.0, 2.0, 3.0]))


def test_reshape_unitful_with_numpy_array_val():
    """Test reshape on Unitful with numpy array value"""
    # shape (2, 3) -> reshape (3, 2)
    a = s * np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert isinstance(a, Unitful)
    assert isinstance(a.val, np.ndarray)
    assert a.val.shape == (2, 3)

    result = reshape(a, (3, 2))

    assert isinstance(result, Unitful)
    assert isinstance(result.val, np.ndarray)
    assert result.val.shape == (3, 2)
    assert np.allclose(
        result.value(),
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    assert result.unit.dim == a.unit.dim


def test_reshape_unitful_python_scalar_raises():
    """Python scalar value in Unitful should raise in reshape"""
    unit = Unit(scale=0, dim={SI.m: 1})
    x = Unitful(val=3.0, unit=unit)

    with pytest.raises(TypeError, match="python scalar"):
        reshape(x, (1,))


def test_reshape_jax_array():
    """Test reshape on plain jax Array"""
    x = jnp.array([[1.0, 2.0, 3.0]])
    result = reshape(x, (3, 1))

    assert isinstance(result, jax.Array)
    assert result.shape == (3, 1)
    assert jnp.allclose(result, jnp.array([[1.0], [2.0], [3.0]]))


def test_reshape_numpy_number_and_array():
    """Test reshape on numpy scalar and numpy array"""
    # numpy scalar -> asarray + reshape -> ndarray
    s_np = np.float64(5.0)
    r_scalar = reshape(s_np, (1,))
    assert isinstance(r_scalar, np.ndarray)
    assert r_scalar.shape == (1,)
    assert np.allclose(r_scalar, np.array([5.0]))

    # numpy array
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    r_arr = reshape(arr, (4,))
    assert isinstance(r_arr, np.ndarray)
    assert r_arr.shape == (4,)
    assert np.allclose(r_arr, np.array([1.0, 2.0, 3.0, 4.0]))


def test_argmax_magic_method():
    """Test argmax magic method on Unitful objects"""
    temperature_unit = Unit(scale=0, dim={SI.K: 1})
    temperatures = Unitful(val=jnp.array([3.0, 250.0, 400.0, 275.0, 325.0], dtype=jnp.float32), unit=temperature_unit)

    result = temperatures.argmax()

    assert isinstance(result, Unitful)
    assert isinstance(result.val, jax.Array)
    assert result.val == jnp.array(2)


def test_argmax_overload_unitful():
    """Test argmax function with Unitful objects"""
    pressure_unit = Unit(scale=3, dim={SI.kg: 1, SI.m: -1, SI.s: -2})
    pressures = Unitful(val=jnp.array([101.0, 95.5, 110.2, 88.7, 105.3]), unit=pressure_unit)

    result = jnp.argmax(pressures)  # type: ignore

    assert isinstance(result, Unitful)
    assert isinstance(result.val, jax.Array)
    assert result == jnp.array(2)


def test_argmax_with_axis_parameter():
    """Test argmax method with axis parameter on 2D Unitful array"""
    force_unit = Unit(scale=0, dim={SI.kg: 1, SI.m: 1, SI.s: -2})  # Newtons
    forces = Unitful(val=jnp.array([[2.0, 8.0, 5.0], [7.0, 3.0, 9.0]]), unit=force_unit)

    # Argmax along axis 1 (rows)
    result = forces.argmax(axis=1)

    assert isinstance(result, Unitful)
    expected_indices = jnp.array([1, 2])
    assert jnp.all(result.val == expected_indices)


def test_argmax_unitfil_with_numpy_arrays():
    """Test argmax function with Unitful objects containing NumPy arrays"""
    pressure_unit = Unit(scale=3, dim={SI.kg: 1, SI.m: -1, SI.s: -2})
    pressures = Unitful(val=np.array([101.0, 95.5, 110.2, 88.7, 105.3]), unit=pressure_unit)

    result = pressures.argmax()

    assert isinstance(result, Unitful)
    assert isinstance(result.val, np.integer)
    assert result == 2


def test_argmax_unitful_with_StaticScalar():
    """Test argmax function with Unitful objects containing StaticScalar"""

    speed_unit = Unit(scale=0, dim={SI.m: 1, SI.s: -1})
    speeds_float = Unitful(val=100.0, unit=speed_unit)
    speeds_np0darray = Unitful(val=np.array(100.0), unit=speed_unit)
    speeds_jax0darray = Unitful(val=jnp.array(100.0), unit=speed_unit)

    result_float = speeds_float.argmax()
    result_np0darray = speeds_np0darray.argmax()
    result_jax0darray = speeds_jax0darray.argmax()

    assert isinstance(result_float, Unitful)
    assert isinstance(result_np0darray, Unitful)
    assert isinstance(result_jax0darray, Unitful)

    assert isinstance(result_float.val, np.number)
    assert isinstance(result_np0darray.val, np.integer)
    assert isinstance(result_jax0darray.val, jax.Array)

    assert result_float == 0
    assert result_np0darray == 0
    assert result_jax0darray == 0


def test_argmax_with_negative_nan_inf():
    """Test argmax behavior with negative, NaN, and Inf values"""
    u = Unit(scale=0, dim={SI.K: 1})
    array1 = Unitful(val=jnp.array([-1.0, jnp.nan, 5.0, jnp.inf]), unit=u)
    array2 = Unitful(val=np.array([-1.0, 3.4, 5.0, np.inf]), unit=u)
    result1 = array1.argmax()
    result2 = array2.argmax()

    # jnp.argmax([-1, nan, 5, inf]) → 3
    assert result1.val == jnp.array(1)
    assert result2.val == jnp.array(3)


def test_argmax_overload_jax_arrays():
    """Test argmax function with regular JAX arrays"""
    array = jnp.array([3.4, 7.1, 2.8, 9.3, 6.2])

    result = jnp.argmax(array)

    assert result == jnp.array(3)
    assert isinstance(result, jax.Array)


def test_argmax_overload_numpy_arrays():
    """Test argmax function with regular NumPy arrays"""
    array = np.array([3.4, 7.1, 2.8, 9.3, 6.2])

    result = array.argmax()

    expected_index = 3
    assert result == np.array(expected_index)
    assert isinstance(result, np.integer)


def test_argmax_untiful_jitted():
    """Test argmax function within a JIT-compiled function when input as unitful"""
    length_unit = Unit(scale=-2, dim={SI.m: 1})
    lengths = Unitful(val=jnp.array([12.0, 25.5, 18.3, 30.1, 22.7]), unit=length_unit)

    def fn(x: Unitful) -> Unitful:
        return x.argmax()

    jitted_fn = jax.jit(fn)
    result = jitted_fn(lengths)

    assert isinstance(result, Unitful)
    assert isinstance(result.val, jax.Array)
    assert result.val == jnp.array(3)


def test_argmax_jax_arrays_jitted():
    """Test argmax function within a JIT-compiled function when input as jax array"""
    array = jnp.array([3.4, 7.1, 2.8, 9.3, 6.2])

    def fn(x: Unitful) -> Unitful:
        return x.argmax()

    jitted_fn = jax.jit(fn)
    result = jitted_fn(array)

    assert isinstance(result, jax.Array)
    assert result == jnp.array(3)


def test_argmax_jitted_static():
    speed_unit = Unit(scale=0, dim={SI.m: 1, SI.s: -1})
    x = Unitful(val=jnp.asarray([1.0, 100.0, 213.0]), unit=speed_unit, static_arr=np.array([1.0, 99.0, 212.0]))

    def fn(x: Unitful) -> Unitful:
        if is_currently_compiling() and not unitful.STATIC_OPTIM_STOP_FLAG:
            assert x.static_arr is not None
        result = x.argmax()
        if is_currently_compiling() and not unitful.STATIC_OPTIM_STOP_FLAG:
            assert result.static_arr is not None
        return result

    jitted_fn = jax.jit(fn)
    result = jitted_fn(x)

    assert isinstance(result, Unitful)
    assert isinstance(result.val, jax.Array)
    assert isinstance(result.static_arr, np.integer)
    assert result.val == jnp.array(2)


def test_argmin_magic_method():
    """Test argmin magic method on Unitful objects"""
    temperature_unit = Unit(scale=0, dim={SI.K: 1})
    temperatures = Unitful(val=jnp.array([3.0, 250.0, 400.0, 275.0, 325.0], dtype=jnp.float32), unit=temperature_unit)

    result = temperatures.argmin()

    assert isinstance(result, Unitful)
    assert isinstance(result.val, jax.Array)
    assert result.val == jnp.array(0)


def test_argmin_overload_unitful():
    """Test argmin function with Unitful objects"""
    pressure_unit = Unit(scale=3, dim={SI.kg: 1, SI.m: -1, SI.s: -2})
    pressures = Unitful(val=jnp.array([101.0, 95.5, 110.2, 88.7, 105.3]), unit=pressure_unit)

    result = jnp.argmin(pressures)  # type: ignore

    assert isinstance(result, Unitful)
    assert isinstance(result.val, jax.Array)
    assert result.val == jnp.array(3)


def test_argmin_with_negative_nan_inf():
    """Test argmin behavior with negative, NaN, and Inf values"""
    u = Unit(scale=0, dim={SI.K: 1})
    array1 = Unitful(val=jnp.array([-1.0, jnp.nan, 5.0, jnp.inf]), unit=u)
    array2 = Unitful(val=np.array([-1.0, 3.4, 5.0, -np.inf]), unit=u)
    result1 = array1.argmin()
    result2 = array2.argmin()

    # jnp.argmax([-1, nan, 5, inf]) → 3
    assert result1.val == jnp.array(1)
    assert result2.val == jnp.array(3)


def test_cond_unitful_numpy_operand_python_branch_true():
    """All operand.val are numpy -> should use Python cond and only execute true branch."""
    time = s * np.array([1.0, 2.0, 3.0])
    operand = [time]

    called = {"t": 0, "f": 0}

    def true_fun(ops):
        called["t"] += 1
        x = ops[0]
        return Unitful(val=np.squeeze(x.value()) + 100.0, unit=x.unit)

    def false_fun(ops):
        called["f"] += 1
        x = ops[0]
        return squeeze(x)

    out = cond(True, true_fun, false_fun, operand)

    assert called["t"] == 1
    assert called["f"] == 0

    assert isinstance(out, Unitful)
    assert isinstance(out.val, np.ndarray)
    assert out.val.shape == (3,)
    assert np.allclose(out.value(), np.array([101.0, 102.0, 103.0]))
    assert out.unit.dim == time.unit.dim


def test_cond_unitful_numpy_operand_python_branch_false():
    """All operand.val are numpy -> should use Python cond and only execute false branch."""
    time = s * np.array([[1.0], [2.0], [3.0]])
    operand = [time]

    called = {"t": 0, "f": 0}

    def true_fun(ops):
        called["t"] += 1
        x = ops[0]
        return Unitful(val=np.squeeze(x.value()) + 999.0, unit=x.unit)

    def false_fun(ops):
        called["f"] += 1
        x = ops[0]
        return squeeze(x)

    out = cond(np.bool_(False), true_fun, false_fun, operand)

    assert called["t"] == 0
    assert called["f"] == 1

    assert isinstance(out, Unitful)
    assert isinstance(out.val, np.ndarray)
    assert out.val.shape == (3,)
    assert np.allclose(out.value(), np.array([1.0, 2.0, 3.0]))
    assert out.unit.dim == time.unit.dim


def test_cond_unitful_jax_operand_uses_lax_cond():
    """Operand contains jax.Array -> should use JAX cond and return jax.Array."""
    time = s * jnp.array([[1.0], [2.0], [3.0]])
    operand = [time]

    def true_fun(ops):
        x = ops[0]
        return squeeze(x)

    def false_fun(ops):
        x = ops[0]
        return Unitful(val=squeeze(x.value()) + 100.0, unit=x.unit)

    out = cond(jnp.array(True), true_fun, false_fun, operand)

    assert isinstance(out, Unitful)
    assert isinstance(out.val, jax.Array)
    assert out.val.shape == (3,)
    assert jnp.allclose(out.value(), jnp.array([1.0, 2.0, 3.0]))
    assert out.unit.dim == time.unit.dim


def test_cond_unitful_jit_with_jax_operand_runs():
    """cond should work under jax.jit when operand is JAX (pred is tracer)."""
    time = s * jnp.array([[1.0], [2.0], [3.0]])
    operand = [time]

    def true_fun(ops):
        x = ops[0]
        return squeeze(x)

    def false_fun(ops):
        x = ops[0]
        return Unitful(val=squeeze(x.value()) + 5.0, unit=x.unit)

    @jax.jit
    def f(pred, ops):
        return cond(pred, true_fun, false_fun, ops)

    out = f(jnp.array(False), operand)

    assert isinstance(out, Unitful)
    assert isinstance(out.val, jax.Array)
    assert out.val.shape == (3,)
    assert jnp.allclose(out.value(), jnp.array([6.0, 7.0, 8.0]))
    assert out.unit.dim == time.unit.dim


def test_cond_promote_unitful_scale_to_empty_unit():
    """Mixed Unitful / non-Unitful branches with scaled units are promoted to Unitful with EMPTY_UNIT."""
    unit_km = Unit(scale=3, dim={SI.m: 1})
    distance = Unitful(val=jnp.array([[1.0], [23.0], [3.0]]), unit=unit_km)
    operand = [distance]

    def true_fun(ops):
        x = ops[0]
        return squeeze(x)

    def false_fun(ops):
        x = ops[0]
        return squeeze(x.value()) + 10.0  # non-Unitful

    out = cond(jnp.array(False), true_fun, false_fun, operand)

    assert isinstance(out, Unitful)
    assert isinstance(out.val, jax.Array)
    assert out.val.shape == (3,)
    assert jnp.allclose(out.value(), jnp.array([1010.0, 23010.0, 3010.0]))
    assert out.unit.dim == {}


def test_cond_promote_false_nonunitful_to_unitful_leafwise():
    """false branch returns non-Unitful, true returns Unitful -> false should be promoted."""
    time = s * jnp.array([[1.0], [2.0], [3.0]])
    operand = [time]

    def true_fun(ops):
        x = ops[0]
        return squeeze(x)

    def false_fun(ops):
        x = ops[0]
        return squeeze(x.value()) + 10.0  # non-Unitful

    out = cond(jnp.array(False), true_fun, false_fun, operand)

    assert isinstance(out, Unitful)
    assert isinstance(out.val, jax.Array)
    assert out.val.shape == (3,)
    assert jnp.allclose(out.value(), jnp.array([11.0, 12.0, 13.0]))
    assert out.unit.dim == {}


def test_cond_promote_true_nonunitful_to_unitful_leafwise():
    """true returns non-Unitful, false returns Unitful -> true should be promoted."""
    time = s * jnp.array([[1.0], [2.0], [3.0]])
    operand = [time]

    def true_fun(ops):
        x = ops[0]
        return squeeze(x.value()) + 100.0  # non-Unitful

    def false_fun(ops):
        x = ops[0]
        return squeeze(x)

    out = cond(jnp.array(True), true_fun, false_fun, operand)

    assert isinstance(out, Unitful)
    assert isinstance(out.val, jax.Array)
    assert out.val.shape == (3,)
    assert jnp.allclose(out.value(), jnp.array([101.0, 102.0, 103.0]))
    assert out.unit.dim == {}


def test_cond_promote_partial_pytree_leafwise():
    """pytree outputs, only some leaves differ -> promote only those leaves."""
    time = s * jnp.array([[1.0], [2.0], [3.0]])
    operand = [time]

    def true_fun(ops):
        x = ops[0]
        # (Unitful, non-Unitful scalar)
        return (squeeze(x), jnp.array(7.0))

    def false_fun(ops):
        x = ops[0]
        # (non-Unitful array, non-Unitful scalar)
        return (squeeze(x.value()) + 1.0, jnp.array(7.0))

    out = cond(jnp.array(False), true_fun, false_fun, operand)

    assert isinstance(out, tuple) and len(out) == 2
    assert isinstance(out[0], Unitful)
    assert isinstance(out[0].val, jax.Array)
    assert out[0].val.shape == (3,)
    assert jnp.allclose(out[0].value(), jnp.array([2.0, 3.0, 4.0]))
    assert out[0].unit.dim == {}
    assert isinstance(out[1], Unitful)
    assert jnp.allclose(out[1].value(), 7.0)


def test_cond_both_unitful_keeps_unit_dim():
    """Both branches return Unitful -> output should keep the unit dim (not empty)."""
    time = s * jnp.array([[1.0], [2.0], [3.0]])
    operand = [time]

    def true_fun(ops):
        x = ops[0]
        return squeeze(x)

    def false_fun(ops):
        x = ops[0]
        return Unitful(val=squeeze(x.value()) + 5.0, unit=x.unit)

    out = cond(jnp.array(True), true_fun, false_fun, operand)

    assert isinstance(out, Unitful)
    assert isinstance(out.val, jax.Array)
    assert out.val.shape == (3,)
    assert jnp.allclose(out.value(), jnp.array([1.0, 2.0, 3.0]))
    assert out.unit.dim == time.unit.dim


def test_cond_eager_only_executes_selected_branch_numpy_operand():
    """If pred/operand are non-jax, should use python branch and not execute the other."""
    time = s * np.array([[1.0], [2.0], [3.0]])
    operand = [time]

    def true_fun(_):
        raise RuntimeError("true branch should NOT run")

    def false_fun(ops):
        x = ops[0]
        return Unitful(val=np.squeeze(x.value()) + 2.0, unit=x.unit)

    out = cond(False, true_fun, false_fun, operand)

    assert isinstance(out, Unitful)
    assert isinstance(out.val, (np.ndarray, np.number))
    assert np.allclose(out.value(), np.array([3.0, 4.0, 5.0]))
    assert out.unit.dim == time.unit.dim


def test_cond_jit_promote_false_nonunitful_runs():
    """Under jax.jit, mixed Unitful/non-Unitful outputs should still work via promote."""
    time = s * jnp.array([[1.0], [2.0], [3.0]])
    operand = [time]

    def true_fun(ops):
        x = ops[0]
        return squeeze(x)

    def false_fun(ops):
        x = ops[0]
        return squeeze(x.value()) + 10.0

    @jax.jit
    def f(pred, ops):
        return cond(pred, true_fun, false_fun, ops)

    out = f(jnp.array(False), operand)

    assert isinstance(out, Unitful)
    assert isinstance(out.val, jax.Array)
    assert out.val.shape == (3,)
    assert jnp.allclose(out.value(), jnp.array([11.0, 12.0, 13.0]))
    assert out.unit.dim == {}


def test_cond_incompatible_pytrees_raise_typeerror():
    """If branches return incompatible pytrees (e.g. scalar vs tuple), should raise TypeError."""
    time = s * jnp.array([[1.0], [2.0], [3.0]])
    operand = [time]

    def true_fun(ops):
        x = ops[0]
        return squeeze(x)

    def false_fun(ops):
        x = ops[0]
        return (squeeze(x.val),)  # tuple, incompatible with Unitful leaf

    with pytest.raises(TypeError):
        _ = cond(jnp.array(True), true_fun, false_fun, operand)


# todo: raised error

# def test_cond_with_add_and_large_scale_units():
#     """cond + add with large scale difference should keep correct unit behavior."""
#     unit_m = Unit(scale=0, dim={SI.m: 1})
#     distance = Unitful(val=jnp.array([[1.0], [2.0], [3.0]]), unit=unit_m)
#     operand = [distance]

#     def true_fun(ops):
#         x = ops[0]
#         # very large add
#         return add(x, Unitful(val=jnp.array([[400.0], [22.0], [3.0]]), unit=unit_m))  # type: ignore

#     def false_fun(ops):
#         x = ops[0]
#         # very small add
#         return add(x, Unitful(val=jnp.array([[1.0], [2.0], [3.0]]), unit=unit_m))  # type: ignore

#     out = cond(jnp.array(True), true_fun, false_fun, operand)

#     assert isinstance(out, Unitful)
#     assert isinstance(out.val, jax.Array)
#     assert out.val.shape == (3,)
#     assert jnp.allclose(out.value(), jnp.array([401.0, 24.0, 6.0]) + 1e12, rtol=1e-6)
#     assert out.unit.dim == {SI.m: 1}
