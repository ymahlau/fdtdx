"""Unit tests for objects/sources/source.py.

Tests for Source, DirectionalPlaneSourceBase, and HardConstantAmplitudePlanceSource.
These are unit tests that test the base classes and their methods.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.switch import OnOffSwitch
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.sources.source import (
    HardConstantAmplitudePlanceSource,
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


class TestHardConstantAmplitudePlanceSource:
    """Unit tests for HardConstantAmplitudePlanceSource."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        assert source.amplitude == 1.0
        assert source.fixed_E_polarization_vector is None
        assert source.fixed_H_polarization_vector is None

    def test_initialization_custom_amplitude(self):
        """Test initialization with custom amplitude."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            amplitude=3.0,
        )
        assert source.amplitude == 3.0

    def test_initialization_with_polarization(self):
        """Test initialization with polarization vectors."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            fixed_E_polarization_vector=(1, 0, 0),
            fixed_H_polarization_vector=(0, 1, 0),
        )
        assert source.fixed_E_polarization_vector == (1, 0, 0)
        assert source.fixed_H_polarization_vector == (0, 1, 0)

    def test_place_on_grid(self, micro_config, jax_key):
        """Test placing source on grid."""
        source = HardConstantAmplitudePlanceSource(
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

    def test_update_E_modifies_field(self, micro_config, jax_key):
        """Test that update_E modifies the E field."""
        source = HardConstantAmplitudePlanceSource(
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

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)

        # E should be modified at source location
        assert not jnp.allclose(E_updated, E)

    def test_update_E_inverse_returns_unchanged(self, micro_config, jax_key):
        """Test that update_E with inverse=True returns unchanged field."""
        source = HardConstantAmplitudePlanceSource(
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

        E = jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * 5.0
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=True)

        # With inverse=True, E should be unchanged
        assert jnp.allclose(E_updated, E)

    def test_update_H_modifies_field(self, micro_config, jax_key):
        """Test that update_H modifies the H field."""
        source = HardConstantAmplitudePlanceSource(
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

        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_updated = placed.update_H(H, inv_perm, 1.0, time_step, inverse=False)

        # H should be modified at source location
        assert not jnp.allclose(H_updated, H)

    def test_update_H_inverse_returns_unchanged(self, micro_config, jax_key):
        """Test that update_H with inverse=True returns unchanged field."""
        source = HardConstantAmplitudePlanceSource(
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

        H = jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * 5.0
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        H_updated = placed.update_H(H, inv_perm, 1.0, time_step, inverse=True)

        # With inverse=True, H should be unchanged
        assert jnp.allclose(H_updated, H)

    def test_update_with_static_amplitude_factor(self, micro_config, jax_key):
        """Test update with static_amplitude_factor."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            amplitude=1.0,
            static_amplitude_factor=2.0,
            fixed_E_polarization_vector=(1, 0, 0),
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        inv_perm = jnp.ones((1, 8, 8, 8), dtype=jnp.float32)
        time_step = jnp.array(10)

        E_updated = placed.update_E(E, inv_perm, 1.0, time_step, inverse=False)
        assert not jnp.allclose(E_updated, E)

    def test_sample_time_signal_accepts_explicit_config_before_placement(self, micro_config):
        """Test source time-signal sampling with an explicit config before placement."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )

        time, signal = source.sample_time_signal(config=micro_config)

        assert time.shape == signal.shape
        assert time.shape == (micro_config.time_steps_total,)

    @pytest.mark.parametrize(
        "method_name",
        [
            "sample_time_signal",
            "frequency_spectrum",
            "plot_time_signal_and_spectrum",
        ],
    )
    def test_time_signal_methods_require_config_or_placement(self, method_name):
        """Test that unplaced source time-signal methods require an explicit config."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )

        with pytest.raises(ValueError, match="place_objects"):
            getattr(source, method_name)()

    def test_sample_time_signal_matches_explicit_and_stored_config(self, micro_config, jax_key):
        """Test explicit config and placed source config give the same sampled signal."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

        time_explicit, signal_explicit = placed.sample_time_signal(config=micro_config)
        time_stored, signal_stored = placed.sample_time_signal()

        assert np.allclose(time_stored, time_explicit)
        assert np.allclose(signal_stored, signal_explicit)

    def test_frequency_spectrum_accepts_explicit_config_before_placement(self, micro_config):
        """Test source spectrum sampling with an explicit config before placement."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )

        frequencies, spectrum = source.frequency_spectrum(config=micro_config, normalize=False)

        assert frequencies.shape == spectrum.shape
        assert frequencies.shape == (micro_config.time_steps_total // 2 + 1,)
        assert np.max(spectrum) > 0.0

    def test_plot_time_signal_and_spectrum_accepts_explicit_config_and_axes(self, micro_config):
        """Test source plotting forwards explicit config and caller-provided axes."""
        import matplotlib.pyplot as plt

        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        fig, axs = plt.subplots(1, 2)

        result_fig = source.plot_time_signal_and_spectrum(
            config=micro_config,
            axs=axs,
            time_range="full",
            frequency_range="full",
        )

        assert result_fig is fig
        assert len(axs[0].lines) == 1
        assert len(axs[1].lines) == 1
        plt.close("all")


class TestDirectionalPlaneSourceBaseProperties:
    """Tests for DirectionalPlaneSourceBase axis properties."""

    def test_propagation_axis_z(self, micro_config, jax_key):
        """Test propagation axis when z-dimension is 1."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),  # Z is 1
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        assert placed.propagation_axis == 2

    def test_propagation_axis_y(self, micro_config, jax_key):
        """Test propagation axis when y-dimension is 1."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 1, 8),  # Y is 1
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (4, 5), (0, 8)),
            config=micro_config,
            key=jax_key,
        )
        assert placed.propagation_axis == 1

    def test_propagation_axis_x(self, micro_config, jax_key):
        """Test propagation axis when x-dimension is 1."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(1, 8, 8),  # X is 1
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((4, 5), (0, 8), (0, 8)),
            config=micro_config,
            key=jax_key,
        )
        assert placed.propagation_axis == 0

    def test_horizontal_vertical_axes_z_propagation(self, micro_config, jax_key):
        """Test horizontal/vertical axes for z-propagation."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )
        # prop_axis=2: horizontal=(2+1)%3=0, vertical=(2+2)%3=1
        assert placed.horizontal_axis == 0
        assert placed.vertical_axis == 1

    def test_horizontal_vertical_axes_y_propagation(self, micro_config, jax_key):
        """Test horizontal/vertical axes for y-propagation."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 1, 8),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (4, 5), (0, 8)),
            config=micro_config,
            key=jax_key,
        )
        # prop_axis=1: horizontal=(1+1)%3=2, vertical=(1+2)%3=0
        assert placed.horizontal_axis == 2
        assert placed.vertical_axis == 0

    def test_horizontal_vertical_axes_x_propagation(self, micro_config, jax_key):
        """Test horizontal/vertical axes for x-propagation."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(1, 8, 8),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((4, 5), (0, 8), (0, 8)),
            config=micro_config,
            key=jax_key,
        )
        # prop_axis=0: horizontal=(0+1)%3=1, vertical=(0+2)%3=2
        assert placed.horizontal_axis == 1
        assert placed.vertical_axis == 2


class TestSourceOnOffSwitch:
    """Tests for Source on/off switch functionality."""

    def test_is_on_at_time_step(self, micro_config, jax_key):
        """Test is_on_at_time_step method."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            switch=OnOffSwitch(interval=1),  # On every step
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

        # Should be on at step 0
        assert placed.is_on_at_time_step(jnp.array(0))

    def test_switch_with_interval(self, micro_config, jax_key):
        """Test source with interval switch."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
            switch=OnOffSwitch(interval=2),  # On every 2 steps
        )
        placed = source.place_on_grid(
            grid_slice_tuple=((0, 8), (0, 8), (4, 5)),
            config=micro_config,
            key=jax_key,
        )

        # Should have internal arrays set
        assert placed._is_on_at_time_step_arr is not None
        assert placed._time_step_to_on_idx is not None


class TestSourceWaveCharacter:
    """Tests for Source wave character."""

    def test_wave_character_wavelength(self):
        """Test source with specific wavelength."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6),
            direction="-",
        )
        assert source.wave_character.wavelength == 1.55e-6

    def test_wave_character_with_phase(self):
        """Test source with phase shift."""
        source = HardConstantAmplitudePlanceSource(
            partial_grid_shape=(8, 8, 1),
            wave_character=WaveCharacter(wavelength=1.55e-6, phase_shift=jnp.pi / 2),
            direction="-",
        )
        assert jnp.isclose(source.wave_character.phase_shift, jnp.pi / 2)
