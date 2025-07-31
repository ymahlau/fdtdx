import jax
import plum
import pytest
from fdtdx.units.unitful import SI, Hz, Unitful, multiply, s, ms, m_per_s
import jax.numpy as jnp


def test_multiply_unitful_unitful_same_dimensions():
    """Test multiplication of two Unitful objects with same dimensions"""
    time1 = s * 2
    time2 = 3 * s
    
    result = multiply(time1, time2)
    
    assert jnp.allclose(result.val, 6.0)
    assert result.unit.scale == 0
    assert result.unit.dim == {SI.s: 2}


def test_multiply_unitful_unitful_different_dimensions():
    """Test multiplication of two Unitful objects with different dimensions"""
    time_val = 5 * s
    freq_val = 10 * Hz
    
    result = multiply(time_val, freq_val)
    
    assert jnp.allclose(result.val, 50.0)
    assert result.unit.scale == 0
    assert result.unit.dim == {}  # s^1 * s^-1 = dimensionless


def test_multiply_unitful_unitful_different_scales():
    """Test multiplication of Unitful objects with different scales"""
    time1 = 2 * s    # scale 0
    time2 = 3 * ms   # scale -3
    
    result = multiply(time1, time2)
    
    assert jnp.allclose(result.val, 6.0)
    assert result.unit.scale == -3
    assert result.unit.dim == {SI.s: 2}


def test_multiply_arraylike_unitful():
    """Test multiplication of ArrayLike with Unitful"""
    scalar = 5.0
    time_val = 3 * s
    
    result: Unitful = jnp.multiply(scalar, time_val)
    
    assert jnp.allclose(result.val, 15.0)
    assert result.unit.scale == 0
    assert result.unit.dim == {SI.s: 1}


def test_multiply_arraylike_arraylike():
    """Test multiplication of two ArrayLike objects"""
    arr1 = jnp.array([1.0, 2.0, 3.0])
    arr2 = jnp.array([4.0, 5.0, 6.0])
    
    result = multiply(arr1, arr2)
    
    expected = jnp.array([4.0, 10.0, 18.0])
    assert jnp.allclose(result, expected)


def test_multiply_not_implemented():
    """Test that multiply raises NotFoundLookupError for unsupported types"""
    with pytest.raises(plum.NotFoundLookupError):
        multiply("string", "string")  # type: ignore


def test_multiply_with_arrays():
    """Test multiplication with array values"""
    time_array = s * jnp.array([1.0, 2.0, 3.0])
    freq_array: Unitful = jnp.array([2.0, 3.0, 4.0]) * Hz
    assert isinstance(freq_array, Unitful)
    assert isinstance(time_array, Unitful)
    result = multiply(time_array, freq_array)
    
    expected_val = jnp.array([2.0, 6.0, 12.0])
    assert jnp.allclose(result.val, expected_val)
    assert result.unit.scale == 0
    assert result.unit.dim == {}  # dimensionless
    
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
    
    assert jnp.allclose(res1.val, res2.val)
    assert res1.unit.scale == -3
    assert res2.unit.scale == -6
