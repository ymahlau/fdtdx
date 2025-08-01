import jax
import plum
import pytest
from fdtdx.core.fraction import Fraction
from fdtdx.units.unitful import (
    SI, 
    Unit, 
    Unitful, 
    add, 
    multiply, 
    remainder, 
    subtract,
    eq,
    lt,
    gt,
    le,
    ne,
    ge,
)
from fdtdx.units.composite import Hz, s, ms, m_per_s 
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
    
    result: Unitful = jnp.multiply(scalar, time_val)  # type: ignore
    
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
    freq_array: Unitful = jnp.array([2.0, 3.0, 4.0]) * Hz   # type: ignore
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


def test_add_same_units():
    """Test addition of two Unitful objects with the same dimensions."""
    # Create two lengths in meters
    unit_m = Unit(scale=0, dim={SI.m: 1})
    u1 = Unitful(val=jnp.array(5.0), unit=unit_m)
    u2 = Unitful(val=jnp.array(3.0), unit=unit_m)
    
    result = jnp.add(u1, u2)  # type: ignore
    
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, 8.0)
    assert result.unit.dim == {SI.m: 1}
    assert result.unit.scale == 0


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
    assert jnp.allclose(result.val, 6.0)
    assert result.unit.dim == {SI.kg: 1}
    assert result.unit.scale == 0


def test_subtract_different_units_raises_error():
    """Test that subtraction of Unitful objects with different dimensions raises ValueError."""
    # Create temperature and current units
    unit_K = Unit(scale=0, dim={SI.K: 1})
    unit_A = Unit(scale=0, dim={SI.A: 1})
    u1 = Unitful(val=jnp.array(300.0), unit=unit_K)
    u2 = Unitful(val=jnp.array(2.0), unit=unit_A)
    
    with pytest.raises(ValueError):
        subtract(u1, u2)


def test_remainder_same_units():
    """Test remainder operation of two Unitful objects with the same dimensions."""
    # Create two times in seconds
    unit_s = Unit(scale=0, dim={SI.s: 1})
    u1 = Unitful(val=jnp.array(10.0), unit=unit_s)
    u2 = Unitful(val=jnp.array(3.0), unit=unit_s)
    
    result = jnp.remainder(u1, u2)  # type: ignore
    
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.val, 1.0)  # 10 % 3 = 1
    assert result.unit.dim == {SI.s: 1}
    assert result.unit.scale == 0


def test_remainder_different_units_raises_error():
    """Test that remainder of Unitful objects with different dimensions raises ValueError."""
    # Create amount of substance and luminous intensity units
    unit_mol = Unit(scale=0, dim={SI.mol: 1})
    unit_cd = Unit(scale=0, dim={SI.cd: 1})
    u1 = Unitful(val=jnp.array(7.0), unit=unit_mol)
    u2 = Unitful(val=jnp.array(2.0), unit=unit_cd)
    
    with pytest.raises(ValueError):
        remainder(u1, u2)


def test_le_magic_method():
    """Test less than or equal magic method of Unitful objects"""
    time1 = 2 * s
    time2 = 3 * s
    result = time1 <= time2
    assert jnp.allclose(result, True)


def test_le_same_units_success():
    """Test less than or equal with same units"""
    unit_m = Unit(scale=0, dim={SI.m: 1})
    u1 = Unitful(val=jnp.array(5.0), unit=unit_m)
    u2 = Unitful(val=jnp.array(5.0), unit=unit_m)
    result = jnp.less_equal(u1, u2)  # type: ignore
    assert jnp.allclose(result, True)


def test_le_different_units_raises_error():
    """Test that le with different units raises ValueError"""
    unit_m = Unit(scale=0, dim={SI.m: 1})
    unit_s = Unit(scale=0, dim={SI.s: 1})
    u1 = Unitful(val=jnp.array(5.0), unit=unit_m)
    u2 = Unitful(val=jnp.array(3.0), unit=unit_s)
    with pytest.raises(ValueError):
        le(u1, u2)


def test_lt_magic_method():
    """Test less than magic method of Unitful objects"""
    time1 = 2 * s
    time2 = 3 * s
    result = time1 < time2
    assert jnp.allclose(result, True)


def test_lt_same_units_success():
    """Test less than with same units"""
    unit_kg = Unit(scale=0, dim={SI.kg: 1})
    u1 = Unitful(val=jnp.array(4.0), unit=unit_kg)
    u2 = Unitful(val=jnp.array(10.0), unit=unit_kg)
    result = jnp.less(u1, u2)   # type: ignore
    assert jnp.allclose(result, True)


def test_lt_different_units_raises_error():
    """Test that lt with different units raises ValueError"""
    unit_K = Unit(scale=0, dim={SI.K: 1})
    unit_A = Unit(scale=0, dim={SI.A: 1})
    u1 = Unitful(val=jnp.array(300.0), unit=unit_K)
    u2 = Unitful(val=jnp.array(2.0), unit=unit_A)
    with pytest.raises(ValueError):
        lt(u1, u2)


def test_eq_magic_method():
    """Test equality magic method of Unitful objects"""
    time1 = 5 * s
    time2 = 5 * s
    result = time1 == time2
    assert jnp.allclose(result, True)


def test_eq_same_units_success():
    """Test equality with same units"""
    unit_mol = Unit(scale=0, dim={SI.mol: 1})
    u1 = Unitful(val=jnp.array(7.0), unit=unit_mol)
    u2 = Unitful(val=jnp.array(7.0), unit=unit_mol)
    result = jnp.equal(u1, u2)  # type: ignore
    assert jnp.allclose(result, True)


def test_eq_different_units_raises_error():
    """Test that eq with different units raises ValueError"""
    unit_mol = Unit(scale=0, dim={SI.mol: 1})
    unit_cd = Unit(scale=0, dim={SI.cd: 1})
    u1 = Unitful(val=jnp.array(7.0), unit=unit_mol)
    u2 = Unitful(val=jnp.array(7.0), unit=unit_cd)
    with pytest.raises(ValueError):
        eq(u1, u2)


def test_neq_magic_method():
    """Test not equal magic method of Unitful objects"""
    time1 = 5 * s
    time2 = 3 * s
    result = time1 != time2
    assert jnp.allclose(result, True)


def test_neq_same_units_success():
    """Test not equal with same units"""
    unit_cd = Unit(scale=0, dim={SI.cd: 1})
    u1 = Unitful(val=jnp.array(5.0), unit=unit_cd)
    u2 = Unitful(val=jnp.array(3.0), unit=unit_cd)
    result = jnp.not_equal(u1, u2)  # type: ignore
    assert jnp.allclose(result, True)


def test_neq_different_units_raises_error():
    """Test that neq with different units raises ValueError"""
    unit_m = Unit(scale=0, dim={SI.m: 1})
    unit_kg = Unit(scale=0, dim={SI.kg: 1})
    u1 = Unitful(val=jnp.array(5.0), unit=unit_m)
    u2 = Unitful(val=jnp.array(5.0), unit=unit_kg)
    with pytest.raises(ValueError):
        ne(u1, u2)


def test_ge_magic_method():
    """Test greater than or equal magic method of Unitful objects"""
    time1 = 3 * s
    time2 = 2 * s
    result = time1 >= time2
    assert jnp.allclose(result, True)


def test_ge_same_units_success():
    """Test greater than or equal with same units"""
    unit_A = Unit(scale=0, dim={SI.A: 1})
    u1 = Unitful(val=jnp.array(8.0), unit=unit_A)
    u2 = Unitful(val=jnp.array(8.0), unit=unit_A)
    result = jnp.greater_equal(u1, u2)  # type: ignore
    assert jnp.allclose(result, True)


def test_ge_different_units_raises_error():
    """Test that ge with different units raises ValueError"""
    unit_s = Unit(scale=0, dim={SI.s: 1})
    unit_m = Unit(scale=0, dim={SI.m: 1})
    u1 = Unitful(val=jnp.array(10.0), unit=unit_s)
    u2 = Unitful(val=jnp.array(3.0), unit=unit_m)
    with pytest.raises(ValueError):
        ge(u1, u2)


def test_gt_magic_method():
    """Test greater than magic method of Unitful objects"""
    time1 = 5 * s
    time2 = 3 * s
    result = time1 > time2
    assert jnp.allclose(result, True)


def test_gt_same_units_success():
    """Test greater than with same units"""
    unit_K = Unit(scale=0, dim={SI.K: 1})
    u1 = Unitful(val=jnp.array(400.0), unit=unit_K)
    u2 = Unitful(val=jnp.array(300.0), unit=unit_K)
    result = jnp.greater(u1, u2)  # type: ignore
    assert jnp.allclose(result, True)


def test_gt_different_units_raises_error():
    """Test that gt with different units raises ValueError"""
    unit_kg = Unit(scale=0, dim={SI.kg: 1})
    unit_A = Unit(scale=0, dim={SI.A: 1})
    u1 = Unitful(val=jnp.array(15.0), unit=unit_kg)
    u2 = Unitful(val=jnp.array(5.0), unit=unit_A)
    with pytest.raises(ValueError):
        gt(u1, u2)


def test_multiply_unitful_with_fractional_dimensions():
    """Test multiplication creating and using fractional dimensions"""
    # Create a unit with m^(1/2) dimension (like square root of area)
    unit_sqrt_m = Unit(scale=0, dim={SI.m: Fraction(1, 2)})
    u1 = Unitful(val=jnp.array(4.0), unit=unit_sqrt_m)
    u2 = Unitful(val=jnp.array(3.0), unit=unit_sqrt_m)
    
    # Multiplying m^(1/2) * m^(1/2) should give m^1
    result = multiply(u1, u2)
    
    assert jnp.allclose(result.val, 12.0)
    assert result.unit.scale == 0
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
    
    assert jnp.allclose(result.val, 16.0)
    assert result.unit.scale == 0
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
    
    assert jnp.allclose(result.val, 35.0)
    assert result.unit.scale == 0
    assert result.unit.dim == {}  # 3/4 + (-3/4) = 0, so dimension is removed


def test_add_same_fractional_dimensions():
    """Test addition of unitful objects with the same fractional dimensions"""
    # Create two units with kg^(2/5) dimension
    unit_frac_kg = Unit(scale=0, dim={SI.kg: Fraction(2, 5)})
    u1 = Unitful(val=jnp.array(10.0), unit=unit_frac_kg)
    u2 = Unitful(val=jnp.array(15.0), unit=unit_frac_kg)
    
    result = add(u1, u2)
    
    assert jnp.allclose(result.val, 25.0)
    assert result.unit.dim == {SI.kg: Fraction(2, 5)}
    assert result.unit.scale == 0


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
    
    assert jnp.allclose(result_equal, True)
    assert jnp.allclose(result_not_equal, False)
    
    # Also test that the units are properly preserved
    assert u1.unit.dim == {SI.A: Fraction(7, 11)}
    assert u2.unit.dim == {SI.A: Fraction(7, 11)}

