import jax
import plum
import pytest
from fdtdx.core.fraction import Fraction
from fdtdx.units.unitful import (
    SI, 
    Unit, 
    Unitful, 
    add,
    matmul, 
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
    
    assert jnp.allclose(result.value(), 6.0)
    assert result.unit.dim == {SI.s: 2}


def test_multiply_unitful_unitful_different_dimensions():
    """Test multiplication of two Unitful objects with different dimensions"""
    time_val = 5 * s
    freq_val = 10 * Hz
    
    result = multiply(time_val, freq_val)
    
    assert jnp.allclose(result.value(), 50.0)
    assert result.unit.dim == {}  # s^1 * s^-1 = dimensionless


def test_multiply_unitful_unitful_different_scales():
    """Test multiplication of Unitful objects with different scales"""
    time1 = 2 * s    # scale 0
    time2 = 3 * ms   # scale -3
    
    result = multiply(time1, time2)
    
    assert jnp.allclose(result.value(), 6.0e-3)
    assert result.unit.dim == {SI.s: 2}


def test_multiply_arraylike_unitful():
    """Test multiplication of ArrayLike with Unitful"""
    scalar = 5.0
    time_val = 3 * s
    
    result: Unitful = jnp.multiply(scalar, time_val)  # type: ignore
    
    assert jnp.allclose(result.value(), 15.0)
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
    assert jnp.allclose(result.value(), expected_val)
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
    
    assert jnp.allclose(res1.value(), res2.value()*1e3)


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


def test_remainder_same_units():
    """Test remainder operation of two Unitful objects with the same dimensions."""
    # Create two times in seconds
    unit_s = Unit(scale=0, dim={SI.s: 1})
    u1 = Unitful(val=jnp.array(10.0), unit=unit_s)
    u2 = Unitful(val=jnp.array(3.0), unit=unit_s)
    
    result = jnp.remainder(u1, u2)  # type: ignore
    
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 1.0)  # 10 % 3 = 1
    assert result.unit.dim == {SI.s: 1}


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
    
    assert jnp.allclose(result_equal, True)
    assert jnp.allclose(result_not_equal, False)
    
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
    u2 = Unitful(val=jnp.array(4.0), unit=time)       # 4 seconds
    
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
    time = Unitful(val=jnp.array(6.0), unit=time_unit)          # 6 s
    
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
    expected_values = jnp.array([-1.0, 2.0, -3.0, 4.0])
    assert jnp.allclose(result.val, expected_values)  # Check the raw values
    assert result.unit.dim == voltages.unit.dim
    assert result.unit.scale == voltages.unit.scale


def test_abs_scalar_unitful_negative():
    """Test __abs__ method with negative scalar Unitful object"""
    charge_unit = Unit(scale=-6, dim={SI.A: 1, SI.s: 1})  # microCoulombs
    charge = Unitful(val=jnp.array(-15.0), unit=charge_unit)
    
    result = abs(charge)
    
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 15.0e-6)  # 15 microCoulombs
    assert result.unit.dim == {SI.A: 1, SI.s: 1}
    assert result.unit.scale == charge.unit.scale


def test_abs_array_unitful_mixed_signs():
    """Test __abs__ method with array Unitful object containing mixed positive/negative values"""
    # Create power unit: kg * m^2 * s^(-3) (Watts)
    power_unit = Unit(scale=3, dim={SI.kg: 1, SI.m: 2, SI.s: -3})  # kiloWatts
    powers = Unitful(val=jnp.array([-2.5, 0.0, 3.7, -1.2, 4.8]), unit=power_unit)
    
    result = abs(powers)
    
    assert isinstance(result, Unitful)
    expected_values = jnp.array([2.5, 0.0, 3.7, 1.2, 4.8])
    assert jnp.allclose(result.val, expected_values)
    assert result.unit.dim == {SI.kg: 1, SI.m: 2, SI.s: -3}
    assert result.unit.scale == 3


def test_matmul_magic_method_same_units():
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
    # Force^2 units: (kg*m*s^-2)^2 = kg^2*m^2*s^-4
    assert result.unit.dim == {SI.kg: 2, SI.m: 2, SI.s: -4}


def test_matmul_overload_different_scales():
    """Test matmul function with Unitful objects having different scales but same dimensions"""
    # Create matrices with time units at different scales
    time_unit_s = Unit(scale=0, dim={SI.s: 1})    # seconds
    time_unit_ms = Unit(scale=-3, dim={SI.s: 1})  # milliseconds
    
    matrix1 = Unitful(val=jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=time_unit_s)
    matrix2 = Unitful(val=jnp.array([[500.0, 1000.0], [1500.0, 2000.0]]), unit=time_unit_ms)
    
    result = matmul(matrix1, matrix2)
    
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