"""Unit tests for objects/sources/tfsf.py.

Tests for TFSFPlaneSource (Total-Field/Scattered-Field) base class.
These are unit tests that use mocks where possible to avoid full simulation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.linear_polarization import GaussianPlaneSource


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


@pytest.fixture
def placed_gaussian_source(micro_config, jax_key):
    """Create a placed GaussianPlaneSource for testing TFSF methods."""
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
    # Apply to set up _E, _H, _time_offset_E, _time_offset_H
    inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
    applied = placed.apply(jax_key, inv_perm, 1.0)
    return applied


class TestTFSFPlaneSourceProperties:
    """Tests for TFSFPlaneSource properties."""

    def test_azimuth_radians(self):
        """Test azimuth angle conversion to radians."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            azimuth_angle=45.0,
        )
        expected = np.deg2rad(45.0)
        assert np.isclose(source.azimuth_radians, expected)

    def test_elevation_radians(self):
        """Test elevation angle conversion to radians."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            elevation_angle=30.0,
        )
        expected = np.deg2rad(30.0)
        assert np.isclose(source.elevation_radians, expected)

    def test_max_angle_random_offset_radians(self):
        """Test max_angle_random_offset conversion to radians."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            max_angle_random_offset=10.0,
        )
        expected = np.deg2rad(10.0)
        assert np.isclose(source.max_angle_random_offset_radians, expected)

    def test_max_vertical_offset_grid(self, micro_config, jax_key):
        """Test max_vertical_offset conversion to grid units."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            max_vertical_offset=500e-9,  # Same as resolution
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        # At 100nm resolution, 500nm offset = 5 grid points
        assert np.isclose(placed.max_vertical_offset_grid, 5.0)

    def test_max_horizontal_offset_grid(self, micro_config, jax_key):
        """Test max_horizontal_offset conversion to grid units."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            max_horizontal_offset=200e-9,
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        # At 100nm resolution, 200nm offset = 2 grid points
        assert np.isclose(placed.max_horizontal_offset_grid, 2.0)


class TestTFSFPlaneSourceRandomization:
    """Tests for randomization methods."""

    def test_get_center_no_offset(self, micro_config, jax_key):
        """Test _get_center with no random offset."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            max_horizontal_offset=0.0,
            max_vertical_offset=0.0,
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        center = placed._get_center(jax_key)

        # With no offset, center should be at grid center
        # Grid shape is (8, 8, 1), so center is at (3.5, 3.5)
        assert center.shape == (2,)
        assert jnp.allclose(center, jnp.array([3.5, 3.5]), atol=0.1)

    def test_get_center_with_offset(self, micro_config, jax_key):
        """Test _get_center with random offset."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            max_horizontal_offset=100e-9,  # 1 grid point
            max_vertical_offset=100e-9,
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        center = placed._get_center(jax_key)

        # Center should be within 1 grid point of (3.5, 3.5)
        assert center.shape == (2,)
        assert jnp.abs(center[0] - 3.5) <= 1.5
        assert jnp.abs(center[1] - 3.5) <= 1.5

    def test_get_azimuth_elevation_no_offset(self, micro_config, jax_key):
        """Test _get_azimuth_elevation with no random offset."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            radius=1e-6,
            azimuth_angle=0.0,
            elevation_angle=0.0,
            max_angle_random_offset=0.0,
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        azimuth, elevation = placed._get_azimuth_elevation(jax_key)

        # With no offset, angles should be 0
        assert jnp.isclose(azimuth, 0.0)
        assert jnp.isclose(elevation, 0.0)

    def test_get_random_parts(self, micro_config, jax_key):
        """Test _get_random_parts returns all components."""
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
        center, azimuth, elevation = placed._get_random_parts(jax_key)

        assert center.shape == (2,)
        assert azimuth.shape == ()
        assert elevation.shape == ()


class TestTFSFPlaneSourceUpdateE:
    """Tests for update_E method."""

    def test_update_E_modifies_field(self, placed_gaussian_source):
        """Test that update_E modifies the E field."""
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed_gaussian_source.update_E(E, inv_perm, 1.0, time_step, inverse=False)

        # E should be modified at source location
        assert not jnp.allclose(E_updated, E)

    def test_update_E_inverse(self, placed_gaussian_source):
        """Test that update_E with inverse=True applies opposite update."""
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_forward = placed_gaussian_source.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        E_inverse = placed_gaussian_source.update_E(E, inv_perm, 1.0, time_step, inverse=True)

        # Forward and inverse should have opposite signs at source location
        diff_forward = E_forward - E
        diff_inverse = E_inverse - E
        # The sign of the update should be opposite
        assert not jnp.allclose(diff_forward, diff_inverse)

    def test_update_E_without_apply_raises(self, micro_config, jax_key):
        """Test that update_E raises if apply() wasn't called."""
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
        # Don't call apply()

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        # When apply() isn't called, internal fields are Null objects
        with pytest.raises(Exception):
            placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)

    def test_update_E_direction_positive(self, micro_config, jax_key):
        """Test update_E with positive direction."""
        source = GaussianPlaneSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="+",  # Positive direction
            radius=1e-6,
            fixed_E_polarization_vector=(1, 0, 0),
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        applied = placed.apply(jax_key, inv_perm, 1.0)

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = applied.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        assert not jnp.allclose(E_updated, E)

    def test_update_E_fully_anisotropic_permittivity(self, placed_gaussian_source):
        """Test update_E with fully anisotropic permittivity (9-component tensor)."""
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        # 9-component inv_permittivities triggers the fully anisotropic branch
        inv_perm_anisotropic = jnp.ones((9, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed_gaussian_source.update_E(E, inv_perm_anisotropic, 1.0, time_step, inverse=False)

        assert not jnp.allclose(E_updated, E)

    def test_update_E_fully_anisotropic_inverse(self, placed_gaussian_source):
        """Test update_E inverse=True with fully anisotropic permittivity."""
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm_anisotropic = jnp.ones((9, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_forward = placed_gaussian_source.update_E(E, inv_perm_anisotropic, 1.0, time_step, inverse=False)
        E_inverse = placed_gaussian_source.update_E(E, inv_perm_anisotropic, 1.0, time_step, inverse=True)

        diff_forward = E_forward - E
        diff_inverse = E_inverse - E
        assert not jnp.allclose(diff_forward, diff_inverse)


class TestTFSFPlaneSourceUpdateH:
    """Tests for update_H method."""

    def test_update_H_modifies_field(self, placed_gaussian_source):
        """Test that update_H modifies the H field."""
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_updated = placed_gaussian_source.update_H(H, inv_perm, 1.0, time_step, inverse=False)

        # H should be modified at source location
        assert not jnp.allclose(H_updated, H)

    def test_update_H_inverse(self, placed_gaussian_source):
        """Test that update_H with inverse=True applies opposite update."""
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_forward = placed_gaussian_source.update_H(H, inv_perm, 1.0, time_step, inverse=False)
        H_inverse = placed_gaussian_source.update_H(H, inv_perm, 1.0, time_step, inverse=True)

        diff_forward = H_forward - H
        diff_inverse = H_inverse - H
        assert not jnp.allclose(diff_forward, diff_inverse)

    def test_update_H_without_apply_raises(self, micro_config, jax_key):
        """Test that update_H raises if apply() wasn't called."""
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

        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        # When apply() isn't called, internal fields are Null objects
        with pytest.raises(Exception):
            placed.update_H(H, inv_perm, 1.0, time_step, inverse=False)

    def test_update_H_anisotropic_permeability(self, micro_config, jax_key):
        """Test update_H with anisotropic permeability array."""
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
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        # Anisotropic permeability (3 components)
        inv_perm_mu = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        applied = placed.apply(jax_key, inv_perm, inv_perm_mu)

        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        # Should handle anisotropic permeability
        H_updated = applied.update_H(H, inv_perm, inv_perm_mu, time_step, inverse=False)
        assert not jnp.allclose(H_updated, H)

    def test_update_H_fully_anisotropic_permeability(self, placed_gaussian_source):
        """Test update_H with fully anisotropic permeability (9-component tensor)."""
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        # 9-component inv_permeabilities triggers the fully anisotropic branch
        inv_perm_mu_anisotropic = jnp.ones((9, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_updated = placed_gaussian_source.update_H(H, inv_perm, inv_perm_mu_anisotropic, time_step, inverse=False)

        assert not jnp.allclose(H_updated, H)

    def test_update_H_fully_anisotropic_inverse(self, placed_gaussian_source):
        """Test update_H inverse=True with fully anisotropic permeability."""
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        inv_perm_mu_anisotropic = jnp.ones((9, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_forward = placed_gaussian_source.update_H(H, inv_perm, inv_perm_mu_anisotropic, time_step, inverse=False)
        H_inverse = placed_gaussian_source.update_H(H, inv_perm, inv_perm_mu_anisotropic, time_step, inverse=True)

        diff_forward = H_forward - H
        diff_inverse = H_inverse - H
        assert not jnp.allclose(diff_forward, diff_inverse)


class TestTFSFPlaneSourceApply:
    """Tests for apply method."""

    def test_apply_sets_internal_fields(self, micro_config, jax_key):
        """Test that apply() sets _E, _H, _time_offset_E, _time_offset_H."""
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

        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        applied = placed.apply(jax_key, inv_perm, 1.0)

        # Check internal fields are set
        assert applied._E is not None
        assert applied._H is not None
        assert applied._time_offset_E is not None
        assert applied._time_offset_H is not None

    def test_apply_field_shapes(self, micro_config, jax_key):
        """Test that apply() creates fields with correct shapes."""
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

        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        applied = placed.apply(jax_key, inv_perm, 1.0)

        # Fields should have shape (3, 8, 8, 1) for the source region
        assert applied._E.shape == (3, 8, 8, 1)
        assert applied._H.shape == (3, 8, 8, 1)
        assert applied._time_offset_E.shape == (3, 8, 8, 1)
        assert applied._time_offset_H.shape == (3, 8, 8, 1)
