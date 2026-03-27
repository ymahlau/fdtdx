import math

import pytest

from fdtdx import WaveCharacter, constants

pytestmark = pytest.mark.unit


class TestWaveCharacter:
    """Test cases for WaveCharacter dataclass."""

    def test_initialization_with_period(self):
        period = 1e-9  # 1 nanosecond
        wave = WaveCharacter(period=period)

        assert wave.get_period() == period
        assert wave.get_wavelength() == period * constants.c
        assert wave.get_frequency() == 1.0 / period
        assert wave.phase_shift == 0.0

    def test_initialization_with_wavelength(self):
        wavelength = 1e-6  # 1 micrometer
        wave = WaveCharacter(wavelength=wavelength)

        assert wave.get_wavelength() == wavelength
        assert wave.get_period() == wavelength / constants.c
        assert wave.get_frequency() == constants.c / wavelength

    def test_initialization_with_frequency(self):
        frequency = 1e9  # 1 GHz
        wave = WaveCharacter(frequency=frequency)

        assert wave.get_frequency() == frequency
        assert wave.get_period() == 1.0 / frequency
        assert abs(wave.get_wavelength() - constants.c / frequency) < 1e-6

    def test_phase_shift(self):
        wave = WaveCharacter(frequency=1e9, phase_shift=math.pi / 4)
        assert wave.phase_shift == math.pi / 4

    def test_no_parameters_raises(self):
        with pytest.raises(Exception):
            WaveCharacter()

    def test_multiple_parameters_raises(self):
        with pytest.raises(Exception):
            WaveCharacter(period=1e-9, frequency=1e9)
        with pytest.raises(Exception):
            WaveCharacter(wavelength=1e-6, frequency=1e9)
        with pytest.raises(Exception):
            WaveCharacter(period=1e-9, wavelength=1e-6)
        with pytest.raises(Exception):
            WaveCharacter(period=1e-9, wavelength=1e-6, frequency=1e9)

    def test_physical_relationships(self):
        """c = lambda * f and T = 1/f."""
        wave = WaveCharacter(frequency=3e8)

        calculated_c = wave.get_wavelength() * wave.get_frequency()
        assert abs(calculated_c - constants.c) < 1e-6

        calculated_period = 1.0 / wave.get_frequency()
        assert abs(calculated_period - wave.get_period()) < 1e-15
