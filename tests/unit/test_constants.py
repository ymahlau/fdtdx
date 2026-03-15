"""Unit tests for fdtdx.constants module."""

import math

from fdtdx import constants


class TestFundamentalConstants:
    """Tests for fundamental physical constants."""

    def test_speed_of_light_value(self):
        """Test that speed of light has correct value."""
        assert constants.c == 299792458.0

    def test_vacuum_permeability_value(self):
        """Test that vacuum permeability has correct value."""
        expected = 4e-7 * math.pi
        assert abs(constants.mu0 - expected) < 1e-20

    def test_vacuum_permittivity_value(self):
        """Test that vacuum permittivity has correct value."""
        expected = 1.0 / (constants.mu0 * constants.c**2)
        assert abs(constants.eps0 - expected) < 1e-30

    def test_free_space_impedance_value(self):
        """Test that free space impedance has correct value."""
        expected = constants.mu0 * constants.c
        assert abs(constants.eta0 - expected) < 1e-10


class TestConstantRelationships:
    """Tests for relationships between physical constants."""

    def test_maxwell_relation(self):
        """Test that c = 1/sqrt(mu0 * eps0) (Maxwell's relation)."""
        calculated_c = 1.0 / math.sqrt(constants.mu0 * constants.eps0)
        assert abs(calculated_c - constants.c) < 1e-6

    def test_impedance_relation(self):
        """Test that eta0 = sqrt(mu0 / eps0)."""
        calculated_eta0 = math.sqrt(constants.mu0 / constants.eps0)
        assert abs(calculated_eta0 - constants.eta0) < 1e-6


class TestRelativePermittivities:
    """Tests for material relative permittivities."""

    def test_permittivity_values(self):
        """Test that permittivities have expected values."""
        assert constants.relative_permittivity_air == 1.0
        assert constants.relative_permittivity_substrate == 2.1025
        assert constants.relative_permittivity_polymer == 2.368521
        assert constants.relative_permittivity_silicon == 12.25
        assert constants.relative_permittivity_silica == 2.25
        assert constants.relative_permittivity_SZ_2080 == 2.1786
        assert constants.relative_permittivity_ma_N_1400_series == 2.6326
        assert constants.relative_permittivity_bacteria == 1.96
        assert constants.relative_permittivity_water == 1.737
        assert constants.relative_permittivity_fused_silica == 2.13685924
        assert constants.relative_permittivity_coated_silica == 1.69
        assert constants.relative_permittivity_resin == 2.202256
        assert constants.relative_permittivity_ormo_prime == 1.817104

    def test_all_permittivities_at_least_one(self):
        """Test that all relative permittivities are >= 1 (physical requirement)."""
        permittivities = [
            constants.relative_permittivity_air,
            constants.relative_permittivity_substrate,
            constants.relative_permittivity_polymer,
            constants.relative_permittivity_silicon,
            constants.relative_permittivity_silica,
            constants.relative_permittivity_SZ_2080,
            constants.relative_permittivity_ma_N_1400_series,
            constants.relative_permittivity_bacteria,
            constants.relative_permittivity_water,
            constants.relative_permittivity_fused_silica,
            constants.relative_permittivity_coated_silica,
            constants.relative_permittivity_resin,
            constants.relative_permittivity_ormo_prime,
        ]
        assert all(p >= 1.0 for p in permittivities)


class TestShardString:
    """Tests for SHARD_STR constant."""

    def test_shard_str_value(self):
        """Test SHARD_STR has correct value."""
        assert constants.SHARD_STR == "shard"


class TestWavelengthToPeriod:
    """Tests for wavelength_to_period function."""

    def test_wavelength_to_period_calculation(self):
        """Test wavelength_to_period calculation."""
        wavelength = 1.55e-6  # 1550 nm (telecom wavelength)
        expected = wavelength / constants.c
        result = constants.wavelength_to_period(wavelength)
        assert abs(result - expected) < 1e-30

    def test_wavelength_to_period_proportional(self):
        """Test that period is proportional to wavelength."""
        wavelength1 = 1e-6
        wavelength2 = 2e-6
        period1 = constants.wavelength_to_period(wavelength1)
        period2 = constants.wavelength_to_period(wavelength2)
        assert abs(period2 / period1 - 2.0) < 1e-10
