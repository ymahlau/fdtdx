"""Unit tests for objects/sources/dipole.py.

Tests for PointDipoleSource: initialization, field updates, inverse behavior,
and arbitrary-orientation via azimuth/elevation angles.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.dipole import PointDipoleSource


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


class TestPointDipoleSourceInitialization:
    """Tests for PointDipoleSource construction."""

    def test_default_electric_dipole(self):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=2,
        )
        assert source.polarization == 2
        assert source.source_type == "electric"
        assert source.amplitude == 1.0
        assert source.azimuth_angle == 0.0
        assert source.elevation_angle == 0.0

    def test_magnetic_dipole(self):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=0,
            source_type="magnetic",
        )
        assert source.source_type == "magnetic"

    def test_custom_amplitude(self):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=1,
            amplitude=5.0,
        )
        assert source.amplitude == 5.0

    def test_invalid_source_type(self):
        with pytest.raises(ValueError, match="source_type must be electric or magnetic"):
            PointDipoleSource(
                partial_grid_shape=(1, 1, 1),
                wave_character=WaveCharacter(wavelength=1e-6),
                source_type="test",
                polarization=0,
            )

    def test_invalid_polarization(self):
        with pytest.raises(ValueError, match="polarization must be 0, 1, or 2"):
            PointDipoleSource(
                partial_grid_shape=(1, 1, 1),
                wave_character=WaveCharacter(wavelength=1e-6),
                polarization=3,
            )

    def test_all_polarization_axes(self):
        for axis in (0, 1, 2):
            source = PointDipoleSource(
                partial_grid_shape=(1, 1, 1),
                wave_character=WaveCharacter(wavelength=1e-6),
                polarization=axis,
            )
            assert source.polarization == axis

    def test_with_angles(self):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=2,
            azimuth_angle=30.0,
            elevation_angle=15.0,
        )
        assert source.azimuth_angle == 30.0
        assert source.elevation_angle == 15.0


class TestPointDipoleSourceUpdateE:
    """Tests for electric dipole update_E behavior."""

    def _make_placed(self, micro_config, jax_key, polarization=2, **kwargs):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=polarization,
            **kwargs,
        )
        return source.place_on_grid(
            grid_slice_tuple=((4, 5), (4, 5), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

    def test_electric_dipole_modifies_E(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        # Only the polarization component at the source cell should be modified
        assert not jnp.allclose(E_updated[2, 4, 4, 4], 0.0)
        # Other components should remain zero
        assert jnp.allclose(E_updated[0, 4, 4, 4], 0.0)
        assert jnp.allclose(E_updated[1, 4, 4, 4], 0.0)

    def test_electric_dipole_does_not_modify_H(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_updated = placed.update_H(H, inv_perm, 1.0, time_step, inverse=False)
        assert jnp.allclose(H_updated, H)

    def test_inverse_reverses_update(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_fwd = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        E_back = placed.update_E(E_fwd, inv_perm, 1.0, time_step, inverse=True)
        assert jnp.allclose(E_back, E, atol=1e-6)

    def test_amplitude_scaling(self, micro_config, jax_key):
        placed_1 = self._make_placed(micro_config, jax_key, amplitude=1.0)
        placed_2 = self._make_placed(micro_config, jax_key, amplitude=2.0)
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E1 = placed_1.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        E2 = placed_2.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        assert jnp.allclose(E2[2, 4, 4, 4], 2.0 * E1[2, 4, 4, 4])

    def test_only_modifies_source_cell(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        # Zero out the source cell — rest should still be zero
        E_check = E_updated.at[2, 4, 4, 4].set(0.0)
        assert jnp.allclose(E_check, 0.0)


class TestPointDipoleSourceUpdateH:
    """Tests for magnetic dipole update_H behavior."""

    def _make_placed(self, micro_config, jax_key, polarization=2, **kwargs):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=polarization,
            source_type="magnetic",
            **kwargs,
        )
        return source.place_on_grid(
            grid_slice_tuple=((4, 5), (4, 5), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

    def test_magnetic_dipole_modifies_H(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_updated = placed.update_H(H, inv_perm, 1.0, time_step, inverse=False)
        assert not jnp.allclose(H_updated[2, 4, 4, 4], 0.0)
        assert jnp.allclose(H_updated[0, 4, 4, 4], 0.0)
        assert jnp.allclose(H_updated[1, 4, 4, 4], 0.0)

    def test_magnetic_dipole_does_not_modify_E(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        assert jnp.allclose(E_updated, E)

    def test_magnetic_inverse_reverses_update(self, micro_config, jax_key):
        placed = self._make_placed(micro_config, jax_key)
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_fwd = placed.update_H(H, inv_perm, 1.0, time_step, inverse=False)
        H_back = placed.update_H(H_fwd, inv_perm, 1.0, time_step, inverse=True)
        assert jnp.allclose(H_back, H, atol=1e-6)


class TestPointDipoleArbitraryOrientation:
    """Tests for dipole with azimuth/elevation angles."""

    def _make_placed(self, micro_config, jax_key, polarization=2, **kwargs):
        source = PointDipoleSource(
            partial_grid_shape=(1, 1, 1),
            wave_character=WaveCharacter(wavelength=1e-6),
            polarization=polarization,
            **kwargs,
        )
        return source.place_on_grid(
            grid_slice_tuple=((4, 5), (4, 5), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

    def test_zero_angles_matches_axis_aligned(self, micro_config, jax_key):
        """azimuth=0, elevation=0 should be identical to the axis-aligned case."""
        placed_plain = self._make_placed(micro_config, jax_key, polarization=2)
        placed_angled = self._make_placed(micro_config, jax_key, polarization=2, azimuth_angle=0.0, elevation_angle=0.0)

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E1 = placed_plain.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        E2 = placed_angled.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        assert jnp.allclose(E1, E2, atol=1e-7)

    def test_angled_dipole_injects_multiple_components(self, micro_config, jax_key):
        """A tilted dipole should inject into more than one field component."""
        placed = self._make_placed(micro_config, jax_key, polarization=2, azimuth_angle=45.0)

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        # The z-component should still be modified
        assert not jnp.allclose(E_updated[2, 4, 4, 4], 0.0)
        # At least one other component should also be modified (tilted away from z)
        other_nonzero = not jnp.allclose(E_updated[0, 4, 4, 4], 0.0) or not jnp.allclose(E_updated[1, 4, 4, 4], 0.0)
        assert other_nonzero

    def test_angled_dipole_preserves_unit_norm(self, micro_config, jax_key):
        """The orientation vector should be unit-norm regardless of angles."""
        placed = self._make_placed(micro_config, jax_key, polarization=1, azimuth_angle=30.0, elevation_angle=20.0)
        orientation = placed._orientation
        assert jnp.allclose(jnp.linalg.norm(orientation), 1.0, atol=1e-6)

    def test_angled_dipole_inverse_reverses(self, micro_config, jax_key):
        """Forward + inverse should cancel for a tilted dipole."""
        placed = self._make_placed(micro_config, jax_key, polarization=0, azimuth_angle=60.0, elevation_angle=25.0)

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_fwd = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        E_back = placed.update_E(E_fwd, inv_perm, 1.0, time_step, inverse=True)
        assert jnp.allclose(E_back, E, atol=1e-6)

    def test_angled_magnetic_dipole(self, micro_config, jax_key):
        """Tilted magnetic dipole should inject into multiple H components."""
        placed = self._make_placed(micro_config, jax_key, polarization=2, azimuth_angle=45.0, source_type="magnetic")

        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_updated = placed.update_H(H, inv_perm, 1.0, time_step, inverse=False)
        assert not jnp.allclose(H_updated[2, 4, 4, 4], 0.0)
        other_nonzero = not jnp.allclose(H_updated[0, 4, 4, 4], 0.0) or not jnp.allclose(H_updated[1, 4, 4, 4], 0.0)
        assert other_nonzero

    def test_90_degree_azimuth_rotates_to_adjacent_axis(self, micro_config, jax_key):
        """A 90-degree azimuth from z should rotate entirely into an adjacent axis."""
        placed = self._make_placed(micro_config, jax_key, polarization=2, azimuth_angle=90.0)

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        # z-component should be ~zero after 90-degree rotation away from z
        assert jnp.allclose(E_updated[2, 4, 4, 4], 0.0, atol=1e-6)
        # The injection should have gone into one of the other axes
        total_other = jnp.abs(E_updated[0, 4, 4, 4]) + jnp.abs(E_updated[1, 4, 4, 4])
        assert total_other > 0

    def test_only_modifies_source_cell_with_angles(self, micro_config, jax_key):
        """An angled dipole should still only modify the single source cell."""
        placed = self._make_placed(micro_config, jax_key, polarization=1, azimuth_angle=35.0, elevation_angle=20.0)

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        # Zero out the source cell for all components — rest should be zero
        E_check = E_updated.at[:, 4, 4, 4].set(0.0)
        assert jnp.allclose(E_check, 0.0)

    def test_45_degree_equal_split(self, micro_config, jax_key):
        """A 45-degree azimuth rotation should split energy equally between two axes."""
        placed = self._make_placed(micro_config, jax_key, polarization=2, azimuth_angle=45.0)
        orientation = placed._orientation

        # The z-component and the rotated-into component should have equal magnitude
        # (cos(45°) = sin(45°) ≈ 0.707)
        z_mag = jnp.abs(orientation[2])
        # Find which horizontal axis got the energy
        h_mag = jnp.sqrt(orientation[0] ** 2 + orientation[1] ** 2)
        assert jnp.allclose(z_mag, h_mag, atol=1e-5)
        assert jnp.allclose(z_mag, np.cos(np.deg2rad(45.0)), atol=1e-5)
