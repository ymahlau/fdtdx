import math
import jax
import plum
import pytest
from fdtdx.core.fraction import Fraction
from fdtdx.functional.numpy import sqrt
from fdtdx.typing import SI
from fdtdx.units.unitful import Unit, Unitful
from fdtdx.units.composite import Hz, s, ms, m_per_s 
import jax.numpy as jnp

def test_sqrt_unitful_even_integer_dimensions():
    """Test sqrt with Unitful object having even integer dimensions"""
    # Create a unit with m^2 dimension (area)
    area_unit = Unit(scale=0, dim={SI.m: 2})
    area = Unitful(val=jnp.array(16.0), unit=area_unit)
    
    result = sqrt(area)
    
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 4.0)  # sqrt(16) = 4
    assert result.unit.dim == {SI.m: 1}  # sqrt(m^2) = m^1


def test_sqrt_unitful_odd_integer_dimensions():
    """Test sqrt with Unitful object having odd integer dimensions"""
    # Create a unit with kg^3 dimension 
    mass_cube_unit = Unit(scale=0, dim={SI.kg: 3})
    mass_cube = Unitful(val=jnp.array(8.0), unit=mass_cube_unit)
    
    result = jax.lax.sqrt(mass_cube)  # type: ignore
    
    assert isinstance(result, Unitful)
    assert jnp.allclose(result.value(), 2.828427124746190)  # sqrt(8) ≈ 2.828
    assert result.unit.dim == {SI.kg: Fraction(3, 2)}  # sqrt(kg^3) = kg^(3/2)


def test_sqrt_unitful_fractional_dimensions_and_odd_scale():
    """Test sqrt with Unitful object having fractional dimensions and odd scale"""
    # Create a unit with m^(2/3) dimension and scale=3 (factor of 1000)
    fractional_unit = Unit(scale=3, dim={SI.m: Fraction(2, 3)})
    value = Unitful(val=jnp.array(4.0), unit=fractional_unit)
    
    result = jnp.sqrt(value)  # type: ignore
    
    assert isinstance(result, Unitful)
    # sqrt(4 * 1000) * sqrt(10) = 2 * sqrt(10000) = 2 * 100 = 200 (approximately)
    # But scale adjustment: sqrt(4) * sqrt(10) with scale floor(3/2) = 1
    # So: sqrt(4) * sqrt(10) * 10^1 = 2 * 3.162 * 10 ≈ 63.24
    expected_value = 2.0 * math.sqrt(10) * 10  # 2 * sqrt(10) * 10^1
    assert jnp.allclose(result.value(), expected_value)
    assert result.unit.dim == {SI.m: Fraction(1, 3)}  # sqrt(m^(2/3)) = m^(1/3)
    
def test_sqrt_jax_array():
    """Test sqrt with regular JAX array (non-Unitful)"""
    # Test that the overloaded sqrt still works with regular JAX arrays
    array = jnp.array([4.0, 9.0, 16.0, 25.0])
    
    result = sqrt(array)
    
    # Expected: [sqrt(4), sqrt(9), sqrt(16), sqrt(25)] = [2, 3, 4, 5]
    expected = jnp.array([2.0, 3.0, 4.0, 5.0])
    assert jnp.allclose(result, expected)
    # Should return a regular JAX array, not a Unitful object
    assert isinstance(result, jax.Array)
    assert not isinstance(result, Unitful)
    