"""Unit tests for objects/sources/linear_polarization.py.

Tests for GaussianPlaneSource and UniformPlaneSource classes.
These are unit tests that use mocks where possible to avoid full simulation.
"""

import warnings
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import RectilinearGrid, UniformGrid
from fdtdx.core.misc import gaussian_amplitude
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
        grid=UniformGrid(spacing=100e-9),
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
        assert source.radius == 1e-6
        assert source.direction == "-"
        assert source.normalize_by_energy is True

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

    def test_gaussian_amplitude_peak_at_center(self):
        """gaussian_amplitude peaks at 1 at the center and decays away from it."""
        coords = jnp.arange(9.0)
        t0, t1 = jnp.meshgrid(coords, coords, indexing="ij")
        profile = gaussian_amplitude(t0, t1, radius=3.0, center=(4.0, 4.0))
        assert jnp.isclose(profile[4, 4], 1.0)
        assert float(profile[4, 4]) > float(profile[0, 0])

    def test_gaussian_amplitude_1e_radius(self):
        """At r = radius the amplitude is exp(-1) (radius = 1/e amplitude radius)."""
        profile = gaussian_amplitude(jnp.array([3.0]), jnp.array([0.0]), radius=3.0, center=(0.0, 0.0))
        assert jnp.isclose(profile[0], jnp.exp(-1.0), atol=1e-6)

    def test_gaussian_amplitude_smooth_tail_no_aperture(self):
        """Unified convention has no hard aperture — amplitude stays > 0 past r = radius."""
        profile = gaussian_amplitude(jnp.array([5.0]), jnp.array([0.0]), radius=3.0, center=(0.0, 0.0))
        assert float(profile[0]) > 0.0

    def test_gaussian_amplitude_shape(self):
        """Output shape matches the input coordinate grids."""
        coords = jnp.arange(8.0)
        t0, t1 = jnp.meshgrid(coords, coords, indexing="ij")
        profile = gaussian_amplitude(t0, t1, radius=3.0)
        assert profile.shape == (8, 8)

    def test_truncation_warning_when_plane_too_small(self):
        """A wide beam in a small source plane warns about >1% edge truncation."""
        config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=2e-7), backend="cpu")
        # plane is 8 * 200 nm = 1.6 um wide -> nearest edge at 0.8 um = 0.4 * radius -> ~85% amplitude
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="+",
            radius=2e-6,
            fixed_E_polarization_vector=(1, 0, 0),
        )
        with pytest.warns(UserWarning, match="truncates the beam"):
            source.place_on_grid(((0, 8), (0, 8), (0, 1)), config, jax.random.PRNGKey(0))

    def test_no_truncation_warning_when_well_sized(self):
        """A beam whose tails vanish before the plane edge does not warn."""
        config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=2e-7), backend="cpu")
        # plane is 40 * 200 nm = 8 um wide -> nearest edge at 4 um ~= 13 * radius -> ~0% amplitude
        source = GaussianPlaneSource(
            partial_grid_shape=(40, 40, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="+",
            radius=3e-7,
            fixed_E_polarization_vector=(1, 0, 0),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            source.place_on_grid(((0, 40), (0, 40), (0, 1)), config, jax.random.PRNGKey(0))
        assert not any("truncates the beam" in str(w.message) for w in caught)

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

    def test_nonuniform_grid_apply_uses_physical_gaussian_profile(self, jax_key):
        """Normal-incidence Gaussian sources sample profiles on rectilinear cell centers."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0]),
            y_edges=jnp.asarray([0.0, 2.0, 5.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )
        config = SimulationConfig(time=1e-8, grid=grid, backend="cpu")
        source = GaussianPlaneSource(
            partial_grid_shape=(2, 2, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=3.0,
            normalize_by_energy=False,
            fixed_E_polarization_vector=(1, 0, 0),
        )
        placed = source.place_on_grid(((0, 2), (0, 2), (0, 1)), config, jax_key)

        applied = placed.apply(jax_key, jnp.ones((1, 2, 2, 1)), 1.0)

        assert applied._E.shape == (3, 2, 2, 1)
        assert applied._H.shape == (3, 2, 2, 1)
        assert applied._time_offset_E.shape == (3, 2, 2, 1)
        assert jnp.isclose(jnp.linalg.norm(applied._E), jnp.linalg.norm(applied._H))

    def test_nonuniform_tilted_apply_uses_physical_projection(self, jax_key):
        """Tilted non-uniform plane sources project profiles in physical coordinates."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),
            y_edges=jnp.asarray([0.0, 1.0, 2.0, 4.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )
        config = SimulationConfig(time=1e-8, grid=grid, backend="cpu")
        source = GaussianPlaneSource(
            partial_grid_shape=(3, 3, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=3.0,
            azimuth_angle=15.0,
            elevation_angle=5.0,
            normalize_by_energy=False,
            fixed_E_polarization_vector=(1, 0, 0),
        )
        placed = source.place_on_grid(((0, 3), (0, 3), (0, 1)), config, jax_key)

        applied = placed.apply(jax_key, jnp.ones((1, 3, 3, 1)), 1.0)

        assert applied._E.shape == (3, 3, 3, 1)
        assert applied._H.shape == (3, 3, 3, 1)
        assert applied._time_offset_E.shape == (3, 3, 3, 1)
        assert jnp.any(applied._E[1] != 0) or jnp.any(applied._E[2] != 0)


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

    def test_apply_uniform_source_on_nonuniform_grid(self, jax_key):
        """Normal-incidence uniform plane sources can use rectilinear Yee offsets."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0]),
            y_edges=jnp.asarray([0.0, 2.0, 5.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )
        config = SimulationConfig(time=1e-8, grid=grid, backend="cpu")
        source = UniformPlaneSource(
            partial_grid_shape=(2, 2, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            normalize_by_energy=False,
            fixed_E_polarization_vector=(1, 0, 0),
        )
        placed = source.place_on_grid(((0, 2), (0, 2), (0, 1)), config, jax_key)

        applied = placed.apply(jax_key, jnp.ones((1, 2, 2, 1)), 1.0)

        assert applied._E.shape == (3, 2, 2, 1)
        assert applied._H.shape == (3, 2, 2, 1)
        assert applied._time_offset_E.shape == (3, 2, 2, 1)

    def test_nonuniform_tilted_uniform_source_preserves_constant_profile(self, jax_key):
        """Physical tilted projection must not distort a constant source profile."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 0.4, 1.1, 2.0]),
            y_edges=jnp.asarray([0.0, 0.6, 1.0, 2.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )
        config = SimulationConfig(time=1e-8, grid=grid, backend="cpu")
        source = UniformPlaneSource(
            partial_grid_shape=(3, 3, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            amplitude=2.0,
            azimuth_angle=20.0,
            elevation_angle=7.0,
            normalize_by_energy=False,
            fixed_E_polarization_vector=(1, 0, 0),
        )
        placed = source.place_on_grid(((0, 3), (0, 3), (0, 1)), config, jax_key)

        applied = placed.apply(jax_key, jnp.ones((1, 3, 3, 1)), 1.0)

        active = jnp.abs(applied._E) > 1e-6
        normalized = jnp.where(active, applied._E / applied._E[:, :1, :1, :], 1.0)
        assert jnp.allclose(normalized[active], 1.0, atol=1e-6)


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
