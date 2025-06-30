import math

import pytest

from fdtdx import WaveCharacter, constants


class TestWaveCharacter:
    """Test cases for WaveCharacter dataclass."""

    def test_initialization_with_period(self):
        """Test initialization with period only."""
        period = 1e-9  # 1 nanosecond
        wave = WaveCharacter(period=period)

        assert wave.get_period() == period
        assert wave.get_wavelength() == period * constants.c
        assert wave.get_frequency() == 1.0 / period
        assert wave.phase_shift == 0.0

    def test_initialization_with_wavelength(self):
        """Test initialization with wavelength only."""
        wavelength = 1e-6  # 1 micrometer
        wave = WaveCharacter(wavelength=wavelength)

        assert wave.get_wavelength() == wavelength
        assert wave.get_period() == wavelength / constants.c
        assert wave.get_frequency() == constants.c / wavelength
        assert wave.phase_shift == 0.0

    def test_initialization_with_frequency(self):
        """Test initialization with frequency only."""
        frequency = 1e9  # 1 GHz
        wave = WaveCharacter(frequency=frequency)

        assert wave.get_frequency() == frequency
        assert wave.get_period() == 1.0 / frequency
        assert abs(wave.get_wavelength() - constants.c / frequency) < 1e-6
        assert wave.phase_shift == 0.0

    def test_initialization_with_phase_shift(self):
        """Test initialization with custom phase shift."""
        frequency = 1e9
        phase_shift = math.pi / 4
        wave = WaveCharacter(frequency=frequency, phase_shift=phase_shift)

        assert wave.get_frequency() == frequency
        assert wave.phase_shift == phase_shift

    def test_initialization_with_zero_values(self):
        """Test initialization with edge case values."""
        # Test with very small positive frequency
        frequency = 1e-12
        wave = WaveCharacter(frequency=frequency)
        assert wave.get_frequency() == frequency

        # Test with very large frequency
        frequency = 1e15
        wave = WaveCharacter(frequency=frequency)
        assert wave.get_frequency() == frequency

    def test_conversion_consistency_period_wavelength(self):
        """Test that conversions between period and wavelength are consistent."""
        period = 2e-9
        wave1 = WaveCharacter(period=period)
        wave2 = WaveCharacter(wavelength=wave1.get_wavelength())

        assert abs(wave1.get_period() - wave2.get_period()) < 1e-15
        assert abs(wave1.get_wavelength() - wave2.get_wavelength()) < 1e-10
        assert abs(wave1.get_frequency() - wave2.get_frequency()) < 1e6

    def test_conversion_consistency_period_frequency(self):
        """Test that conversions between period and frequency are consistent."""
        period = 1e-6
        wave1 = WaveCharacter(period=period)
        wave2 = WaveCharacter(frequency=wave1.get_frequency())

        assert abs(wave1.get_period() - wave2.get_period()) < 1e-15
        assert abs(wave1.get_wavelength() - wave2.get_wavelength()) < 1e-10
        assert abs(wave1.get_frequency() - wave2.get_frequency()) < 1e3

    def test_conversion_consistency_wavelength_frequency(self):
        """Test that conversions between wavelength and frequency are consistent."""
        wavelength = 0.5e-6  # 500 nm
        wave1 = WaveCharacter(wavelength=wavelength)
        wave2 = WaveCharacter(frequency=wave1.get_frequency())

        assert abs(wave1.get_period() - wave2.get_period()) < 1e-21
        assert abs(wave1.get_wavelength() - wave2.get_wavelength()) < 1e-15
        assert abs(wave1.get_frequency() - wave2.get_frequency()) < 1e9

    def test_physical_relationships(self):
        """Test that physical relationships hold: c = λ * f and T = 1/f."""
        frequency = 3e8  # 300 MHz
        wave = WaveCharacter(frequency=frequency)

        # Test c = λ * f
        calculated_c = wave.get_wavelength() * wave.get_frequency()
        assert abs(calculated_c - constants.c) < 1e-6

        # Test T = 1/f
        calculated_period = 1.0 / wave.get_frequency()
        assert abs(calculated_period - wave.get_period()) < 1e-15

    def test_initialization_no_parameters(self):
        """Test that initialization without any wave parameter raises exception."""
        with pytest.raises(Exception):
            WaveCharacter()

    def test_initialization_multiple_parameters(self):
        """Test that initialization with multiple wave parameters raises exception."""
        with pytest.raises(Exception):
            WaveCharacter(period=1e-9, frequency=1e9)

        with pytest.raises(Exception):
            WaveCharacter(wavelength=1e-6, frequency=1e9)

        with pytest.raises(Exception):
            WaveCharacter(period=1e-9, wavelength=1e-6)

        with pytest.raises(Exception):
            WaveCharacter(period=1e-9, wavelength=1e-6, frequency=1e9)

    def test_initialization_with_negative_phase_shift(self):
        """Test initialization with negative phase shift."""
        frequency = 1e9
        phase_shift = -math.pi / 2
        wave = WaveCharacter(frequency=frequency, phase_shift=phase_shift)

        assert wave.get_frequency() == frequency
        assert wave.phase_shift == phase_shift

    def test_initialization_with_large_phase_shift(self):
        """Test initialization with phase shift larger than 2π."""
        frequency = 1e9
        phase_shift = 3 * math.pi
        wave = WaveCharacter(frequency=frequency, phase_shift=phase_shift)

        assert wave.get_frequency() == frequency
        assert wave.phase_shift == phase_shift

    def test_getter_methods_return_correct_types(self):
        """Test that getter methods return float values."""
        wave = WaveCharacter(frequency=1e9)

        assert isinstance(wave.get_period(), float)
        assert isinstance(wave.get_wavelength(), float)
        assert isinstance(wave.get_frequency(), float)

    def test_visible_light_wavelength(self):
        """Test with typical visible light wavelength."""
        wavelength = 550e-9  # Green light, 550 nm
        wave = WaveCharacter(wavelength=wavelength)

        expected_frequency = constants.c / wavelength
        expected_period = 1.0 / expected_frequency

        assert abs(wave.get_frequency() - expected_frequency) < 1e9
        assert abs(wave.get_period() - expected_period) < 1e-21

    def test_radio_frequency(self):
        """Test with typical radio frequency."""
        frequency = 100e6  # 100 MHz FM radio
        wave = WaveCharacter(frequency=frequency)

        expected_wavelength = constants.c / frequency
        expected_period = 1.0 / frequency

        assert abs(wave.get_wavelength() - expected_wavelength) < 1e-6
        assert abs(wave.get_period() - expected_period) < 1e-15

    def test_microwave_period(self):
        """Test with microwave period."""
        period = 1e-10  # 100 picoseconds
        wave = WaveCharacter(period=period)

        expected_frequency = 1.0 / period
        expected_wavelength = constants.c / expected_frequency

        assert abs(wave.get_frequency() - expected_frequency) < 1e6
        assert abs(wave.get_wavelength() - expected_wavelength) < 1e-8
