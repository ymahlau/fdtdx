import pytest
import jax.numpy as jnp
from fdtdx.units.composite import m, s
from fdtdx.units.unitful import Unitful
from fdtdx.constants import wavelength_to_period, c


def test_wavelength_to_period_basic_calculation():
    """Test basic wavelength to period conversion with known values.
    
    This test verifies that the function correctly converts a wavelength
    to its corresponding time period using the speed of light constant.
    Uses a common optical wavelength (1550 nm) as test case.
    """
    # Test with 1550 nm wavelength (common in optical communications)
    wavelength = 1550e-9 * m  # 1550 nanometers
    
    # Calculate period using the function
    period = wavelength_to_period(wavelength)
    
    # Expected period = wavelength / c = 1550e-9 / 299792458.0
    expected_period_value = 1550e-9 / 299792458.0
    expected_period = expected_period_value * s
    
    # Verify the result has correct units (time)
    assert period.unit.dim == expected_period.unit.dim, "Period should have time units"
    
    # Verify the numerical value is correct (within floating point precision)
    assert abs(period.value() - expected_period.value()) < 1e-20, \
        f"Expected period {expected_period.value()}, got {period.value()}"


def test_wavelength_to_period_unit_consistency():
    """Test that the function maintains proper unit algebra.
    
    This test verifies that the division of length units by velocity units
    correctly produces time units, and that the Unitful arithmetic is
    handled properly throughout the calculation.
    """
    # Test with a simple wavelength value
    wavelength = 3.0 * m  # 3 meters
    
    # Calculate period
    period = wavelength_to_period(wavelength)
    
    # Manually calculate expected result: wavelength / c
    # c has units of m/s, so wavelength/c should have units of s
    expected_period = wavelength / c
    
    # Verify unit dimensions match
    assert period.unit.dim == expected_period.unit.dim, \
        "Period units should match manual calculation"
    
    # Verify the values are identical
    assert abs(period.value() - expected_period.value()) < 1e-15, \
        "Period value should match manual calculation"
    
    # Verify that the result has time units (seconds)
    # The unit dimension should contain only time with power 1
    from fdtdx.units.typing import SI
    assert SI.s in period.unit.dim, "Period should contain time dimension"
    assert period.unit.dim[SI.s] == 1, "Time dimension should have power 1"
    assert len(period.unit.dim) == 1, "Period should only have time dimension"