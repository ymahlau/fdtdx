import pytest
import jax.numpy as jnp
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.units.unitful import Unit, Unitful
from fdtdx.units.typing import SI
from fdtdx.units import Hz, s, m
from fdtdx import constants


def test_wavecharacter_frequency_initialization_with_phase_shift():
    """Test WaveCharacter initialization with frequency and custom phase shift"""
    # Create a 1 GHz frequency with π/4 phase shift
    frequency_val = 1e9 * Hz  # 1 GHz
    phase_shift = jnp.pi / 4
    x = 1.0 / frequency_val
    
    wave_char = WaveCharacter(frequency=frequency_val, phase_shift=phase_shift)
    
    # Check that frequency is preserved
    assert jnp.allclose(wave_char.get_frequency().value(), 1e9)
    assert wave_char.get_frequency().unit.dim == {SI.s: -1}
    
    # Check that phase shift is preserved
    assert jnp.allclose(wave_char.phase_shift, jnp.pi / 4)
    
    # Check that period and wavelength are correctly calculated
    expected_period = 1e-9  # 1/1e9 = 1 ns
    assert jnp.allclose(wave_char.get_period().value(), expected_period)
    assert wave_char.get_period().unit.dim == {SI.s: 1}
    
    # Check wavelength calculation: λ = c/f
    expected_wavelength = constants.c.value() / 1e9  # ~0.3 m
    assert jnp.allclose(wave_char.get_wavelength().value(), expected_wavelength, rtol=1e-10)
    assert wave_char.get_wavelength().unit.dim == {SI.m: 1}


def test_wavecharacter_wavelength_initialization_infrared():
    """Test WaveCharacter initialization with infrared wavelength"""
    # Create a 1550 nm wavelength (common telecom wavelength)
    wavelength_val = 1550e-9 * m  # 1550 nm
    
    wave_char = WaveCharacter(wavelength=wavelength_val)
    
    # Check that wavelength is preserved
    assert jnp.allclose(wave_char.get_wavelength().value(), 1550e-9)
    assert wave_char.get_wavelength().unit.dim == {SI.m: 1}
    
    # Check default phase shift
    assert wave_char.phase_shift == 0.0
    
    # Check frequency calculation: f = c/λ
    expected_frequency = constants.c.value() / (1550e-9)  # ~193.5 THz
    assert jnp.allclose(wave_char.get_frequency().value(), expected_frequency, rtol=1e-10)
    assert wave_char.get_frequency().unit.dim == {SI.s: -1}
    
    # Check period calculation: T = λ/c
    expected_period = (1550e-9) / constants.c.value()  # ~5.17 fs
    assert jnp.allclose(wave_char.get_period().value(), expected_period, rtol=1e-10)
    assert wave_char.get_period().unit.dim == {SI.s: 1}


def test_wavecharacter_period_initialization_microseconds():
    """Test WaveCharacter initialization with period in microseconds"""
    # Create a 10 μs period (100 kHz)
    period_val = 10e-6 * s  # 10 microseconds
    
    wave_char = WaveCharacter(period=period_val)
    
    # Check that period is preserved
    assert jnp.allclose(wave_char.get_period().value(), 10e-6)
    assert wave_char.get_period().unit.dim == {SI.s: 1}
    
    # Check frequency calculation: f = 1/T
    expected_frequency = 1.0 / (10e-6)  # 100 kHz
    assert jnp.allclose(wave_char.get_frequency().value(), expected_frequency)
    assert wave_char.get_frequency().unit.dim == {SI.s: -1}
    
    # Check wavelength calculation: λ = c*T
    expected_wavelength = constants.c.value() * (10e-6)  # ~3000 m
    assert jnp.allclose(wave_char.get_wavelength().value(), expected_wavelength, rtol=1e-10)
    assert wave_char.get_wavelength().unit.dim == {SI.m: 1}


def test_wavecharacter_invalid_multiple_parameters_raises_error():
    """Test that providing multiple wave parameters raises an exception"""
    frequency_val = 1e6 * Hz  # 1 MHz
    wavelength_val = 300.0 * m  # 300 m
    period_val = 1e-6 * s  # 1 μs
    
    # Test frequency + wavelength
    with pytest.raises(Exception, match="Need to set exactly one of Period, Frequency or Wavelength"):
        WaveCharacter(frequency=frequency_val, wavelength=wavelength_val)
    
    # Test frequency + period
    with pytest.raises(Exception, match="Need to set exactly one of Period, Frequency or Wavelength"):
        WaveCharacter(frequency=frequency_val, period=period_val)
    
    # Test wavelength + period
    with pytest.raises(Exception, match="Need to set exactly one of Period, Frequency or Wavelength"):
        WaveCharacter(wavelength=wavelength_val, period=period_val)
    
    # Test all three parameters
    with pytest.raises(Exception, match="Need to set exactly one of Period, Frequency or Wavelength"):
        WaveCharacter(frequency=frequency_val, wavelength=wavelength_val, period=period_val)
    
    # Test no parameters
    with pytest.raises(Exception, match="Need to set exactly one of Period, Frequency or Wavelength"):
        WaveCharacter()


def test_wavecharacter_wrong_units_raises_error():
    """Test that providing parameters with wrong units raises appropriate errors"""
    # Test frequency with wrong units (using seconds instead of Hz)
    wrong_frequency = 1.0 * s  # Should be Hz (s^-1)
    with pytest.raises(AssertionError, match="Please specify frequency in Hz"):
        WaveCharacter(frequency=wrong_frequency)
    
    # Test wavelength with wrong units (using seconds instead of meters)
    wrong_wavelength = 1.0 * s  # Should be meters
    with pytest.raises(AssertionError, match="Please specify wavelength in meter"):
        WaveCharacter(wavelength=wrong_wavelength)
    
    # Test period with wrong units (using meters instead of seconds)
    wrong_period = 1.0 * m  # Should be seconds
    with pytest.raises(AssertionError, match="Please specify period in seconds"):
        WaveCharacter(period=wrong_period)
    
    # Test period with frequency units (Hz instead of seconds)
    wrong_period_hz = 1.0 * Hz  # Should be seconds, not Hz
    with pytest.raises(AssertionError, match="Please specify period in seconds"):
        WaveCharacter(period=wrong_period_hz)


def test_wavecharacter_consistency_across_all_conversions():
    """Test that all three initialization methods produce consistent results"""
    # Define a test case: 2.4 GHz (common WiFi frequency)
    test_frequency = 2.4e9  # Hz
    test_period = 1.0 / test_frequency  # seconds
    test_wavelength = constants.c / (test_frequency * Hz)  # meters
    
    # Create WaveCharacter objects using each initialization method
    wave_from_freq = WaveCharacter(frequency=test_frequency * Hz)
    wave_from_period = WaveCharacter(period=test_period * s)
    wave_from_wavelength = WaveCharacter(wavelength=test_wavelength)
    
    # All should have the same frequency
    assert jnp.allclose(
        wave_from_freq.get_frequency().value(),
        wave_from_period.get_frequency().value(),
        rtol=1e-12
    )
    assert jnp.allclose(
        wave_from_freq.get_frequency().value(),
        wave_from_wavelength.get_frequency().value(),
        rtol=1e-12
    )
    
    # All should have the same period
    assert jnp.allclose(
        wave_from_freq.get_period().value(),
        wave_from_period.get_period().value(),
        rtol=1e-12
    )
    assert jnp.allclose(
        wave_from_freq.get_period().value(),
        wave_from_wavelength.get_period().value(),
        rtol=1e-12
    )
    
    # All should have the same wavelength
    assert jnp.allclose(
        wave_from_freq.get_wavelength().value(),
        wave_from_period.get_wavelength().value(),
        rtol=1e-12
    )
    assert jnp.allclose(
        wave_from_freq.get_wavelength().value(),
        wave_from_wavelength.get_wavelength().value(),
        rtol=1e-12
    )
    
    # All should have the expected units
    for wave_char in [wave_from_freq, wave_from_period, wave_from_wavelength]:
        assert wave_char.get_frequency().unit.dim == {SI.s: -1}
        assert wave_char.get_period().unit.dim == {SI.s: 1}
        assert wave_char.get_wavelength().unit.dim == {SI.m: 1}
        assert wave_char.phase_shift == 0.0  # Default phase shift