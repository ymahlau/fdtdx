"""Unit tests for objects/sources/linear_polarization.py.

Tests for GaussianPlaneSource and UniformPlaneSource classes.
These are unit tests that use mocks where possible to avoid full simulation.
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.linear_polarization import (
    GaussianPlaneSource,
    UniformPlaneSource,
)


@pytest.fixture
def micro_config():
    """Minimal simulation config for source testing."""
    return SimulationConfig(
        time=100e-15,
        resolution=100e-9,
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )


@pytest.fixture
def jax_key():
    """JAX random key for tests."""
    return jax.random.PRNGKey(42)


class TestGaussianPlaneSource:
    """Unit tests for GaussianPlaneSource."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
        )
        assert source.std == 1 / 3
        assert source.radius == 1e-6
        assert source.direction == "-"
        assert source.normalize_by_energy is True

    def test_initialization_custom_std(self):
        """Test initialization with custom std."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="+",
            radius=2e-6,
            std=0.5,
        )
        assert source.std == 0.5
        assert source.radius == 2e-6

    def test_initialization_with_polarization(self):
        """Test initialization with E polarization vector."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            fixed_E_polarization_vector=(1, 0, 0),
        )
        assert source.fixed_E_polarization_vector == (1, 0, 0)

    def test_initialization_with_h_polarization(self):
        """Test initialization with H polarization vector."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            fixed_H_polarization_vector=(0, 1, 0),
        )
        assert source.fixed_H_polarization_vector == (0, 1, 0)

    def test_normalize_by_energy_false(self):
        """Test with normalize_by_energy=False."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            normalize_by_energy=False,
        )
        assert source.normalize_by_energy is False

    def test_gauss_profile_shape(self):
        """Test _gauss_profile returns correct shape."""
        profile = GaussianPlaneSource._gauss_profile(
            width=8,
            height=8,
            axis=2,
            center=(4.0, 4.0),
            radii=(3.0, 3.0),
            std=1 / 3,
        )
        assert profile.shape == (8, 8, 1)

    def test_gauss_profile_normalized(self):
        """Test _gauss_profile sums to 1."""
        profile = GaussianPlaneSource._gauss_profile(
            width=16,
            height=16,
            axis=2,
            center=(8.0, 8.0),
            radii=(6.0, 6.0),
            std=1 / 3,
        )
        assert jnp.isclose(profile.sum(), 1.0, atol=1e-5)

    def test_gauss_profile_centered(self):
        """Test _gauss_profile is centered correctly."""
        profile = GaussianPlaneSource._gauss_profile(
            width=9,
            height=9,
            axis=2,
            center=(4.0, 4.0),
            radii=(3.0, 3.0),
            std=1 / 3,
        )
        # Center should have highest value
        center_val = profile[4, 4, 0]
        corner_val = profile[0, 0, 0]
        assert center_val > corner_val

    def test_gauss_profile_different_axes(self):
        """Test _gauss_profile with different axis positions."""
        # Axis at position 0
        profile0 = GaussianPlaneSource._gauss_profile(
            width=8,
            height=8,
            axis=0,
            center=(4.0, 4.0),
            radii=(3.0, 3.0),
            std=1 / 3,
        )
        assert profile0.shape == (1, 8, 8)

        # Axis at position 1
        profile1 = GaussianPlaneSource._gauss_profile(
            width=8,
            height=8,
            axis=1,
            center=(4.0, 4.0),
            radii=(3.0, 3.0),
            std=1 / 3,
        )
        assert profile1.shape == (8, 1, 8)

    def test_place_on_grid(self, micro_config, jax_key):
        """Test placing source on grid."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            fixed_E_polarization_vector=(1, 0, 0),
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        assert placed.grid_shape == (8, 8, 1)
        assert placed.propagation_axis == 2

    def test_propagation_axis_detection(self, micro_config, jax_key):
        """Test propagation axis is detected from grid shape."""
        # Z-propagating source (shape has 1 in z)
        source_z = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
        )
        placed_z = source_z.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        assert placed_z.propagation_axis == 2

    def test_horizontal_vertical_axes(self, micro_config, jax_key):
        """Test horizontal and vertical axis calculation."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        # For propagation axis 2: horizontal = 0, vertical = 1
        assert placed.horizontal_axis == 0
        assert placed.vertical_axis == 1


class TestUniformPlaneSource:
    """Unit tests for UniformPlaneSource."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        source = UniformPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        assert source.amplitude == 1.0
        assert source.direction == "-"

    def test_initialization_custom_amplitude(self):
        """Test initialization with custom amplitude."""
        source = UniformPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="+",
            amplitude=2.5,
        )
        assert source.amplitude == 2.5

    def test_initialization_with_polarization(self):
        """Test initialization with polarization vector."""
        source = UniformPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            fixed_E_polarization_vector=(0, 1, 0),
        )
        assert source.fixed_E_polarization_vector == (0, 1, 0)

    def test_place_on_grid(self, micro_config, jax_key):
        """Test placing source on grid."""
        source = UniformPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            fixed_E_polarization_vector=(1, 0, 0),
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        assert placed.grid_shape == (8, 8, 1)

    def test_direction_positive(self):
        """Test source with positive direction."""
        source = UniformPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="+",
        )
        assert source.direction == "+"

    def test_direction_negative(self):
        """Test source with negative direction."""
        source = UniformPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        assert source.direction == "-"


class TestLinearlyPolarizedPlaneSourceApply:
    """Tests for apply method and get_EH_variation with mocks."""

    @patch("fdtdx.objects.sources.linear_polarization.compute_energy")
    @patch("fdtdx.objects.sources.linear_polarization.calculate_time_offset_yee")
    def test_get_EH_variation_with_normalization(self, mock_time_offset, mock_compute_energy, micro_config, jax_key):
        """Test get_EH_variation with energy normalization."""
        # Setup mocks
        mock_time_offset.return_value = (
            jnp.zeros((3, 8, 8, 1)),  # time_offset_E
            jnp.zeros((3, 8, 8, 1)),  # time_offset_H
        )
        mock_compute_energy.return_value = jnp.ones((8, 8, 1))

        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            fixed_E_polarization_vector=(1, 0, 0),
            normalize_by_energy=True,
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        inv_perm_mu = 1.0

        applied = placed.apply(jax_key, inv_perm, inv_perm_mu)
        E, H, time_E, time_H = applied._E, applied._H, applied._time_offset_E, applied._time_offset_H

        # Verify shapes
        assert E.shape == (3, 8, 8, 1)
        assert H.shape == (3, 8, 8, 1)
        assert time_E.shape == (3, 8, 8, 1)
        assert time_H.shape == (3, 8, 8, 1)

        # Verify compute_energy was called (normalization enabled)
        mock_compute_energy.assert_called()

    @patch("fdtdx.objects.sources.linear_polarization.calculate_time_offset_yee")
    def test_get_EH_variation_without_normalization(self, mock_time_offset, micro_config, jax_key):
        """Test get_EH_variation without energy normalization."""
        mock_time_offset.return_value = (
            jnp.zeros((3, 8, 8, 1)),
            jnp.zeros((3, 8, 8, 1)),
        )

        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            fixed_E_polarization_vector=(1, 0, 0),
            normalize_by_energy=False,
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)

        applied = placed.apply(jax_key, inv_perm, 1.0)
        E, H = applied._E, applied._H

        assert E.shape == (3, 8, 8, 1)
        assert H.shape == (3, 8, 8, 1)


class TestUniformSourceAmplitude:
    """Tests for UniformPlaneSource _get_amplitude_raw method."""

    def test_get_amplitude_raw_uniform(self, micro_config, jax_key):
        """Test that UniformPlaneSource produces uniform amplitude."""
        source = UniformPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            amplitude=2.0,
            fixed_E_polarization_vector=(1, 0, 0),
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

        center = jnp.array([4.0, 4.0])
        profile = placed._get_amplitude_raw(center)

        # Should be uniform with amplitude * 1
        assert jnp.allclose(profile, 2.0)
        assert profile.shape == (8, 8, 1)


class TestAnisotropicPermeability:
    """Tests for anisotropic permeability handling."""

    @patch("fdtdx.objects.sources.linear_polarization.calculate_time_offset_yee")
    def test_anisotropic_inv_permeability_slicing(self, mock_time_offset, micro_config, jax_key):
        """Test that anisotropic inv_permeability is sliced correctly."""
        mock_time_offset.return_value = (
            jnp.zeros((3, 8, 8, 1)),
            jnp.zeros((3, 8, 8, 1)),
        )

        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            fixed_E_polarization_vector=(1, 0, 0),
            normalize_by_energy=False,
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

        # Anisotropic permeability (3 components)
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        inv_perm_mu = jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * 0.5

        # Should not raise
        applied = placed.apply(jax_key, inv_perm, inv_perm_mu)
        E = applied._E
        assert E.shape == (3, 8, 8, 1)
