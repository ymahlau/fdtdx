"""Unit tests for objects/sources/dipole.py.

Tests for PointDipoleSource: initialization, field updates, and inverse behavior.
"""

import jax
import jax.numpy as jnp
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
