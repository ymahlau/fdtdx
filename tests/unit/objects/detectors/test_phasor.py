"""Tests for objects/detectors/phasor.py - Phasor detector."""

import jax.numpy as jnp
import pytest

from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.detectors.phasor import PhasorDetector


@pytest.fixture
def single_frequency():
    """Single frequency wave character (optical frequency)."""
    return [WaveCharacter(wavelength=1e-6)]  # 1 micron


@pytest.fixture
def multiple_frequencies():
    """Multiple frequency wave characters."""
    return [
        WaveCharacter(wavelength=1e-6),
        WaveCharacter(wavelength=1.5e-6),
        WaveCharacter(frequency=3e14),
    ]


class TestPhasorDetectorShapes:
    """Tests for PhasorDetector shape and initialization."""

    def test_shape_dtype_single_frequency(self, simulation_config, small_grid_slice, random_key, single_frequency):
        """Test shape calculation for single frequency."""
        detector = PhasorDetector(wave_characters=single_frequency)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert "phasor" in shape_dtype
        # 1 frequency, 6 components, 8x8x8 grid
        assert shape_dtype["phasor"].shape == (1, 6, 8, 8, 8)
        assert shape_dtype["phasor"].dtype == jnp.complex64

    def test_shape_dtype_multiple_frequencies(
        self, simulation_config, small_grid_slice, random_key, multiple_frequencies
    ):
        """Test shape calculation for multiple frequencies."""
        detector = PhasorDetector(wave_characters=multiple_frequencies)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        # 3 frequencies, 6 components, 8x8x8 grid
        assert shape_dtype["phasor"].shape == (3, 6, 8, 8, 8)

    def test_shape_dtype_reduced_volume(self, simulation_config, small_grid_slice, random_key, single_frequency):
        """Test shape calculation with reduced volume."""
        detector = PhasorDetector(wave_characters=single_frequency, reduce_volume=True)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        # 1 frequency, 6 components, no spatial dimensions
        assert shape_dtype["phasor"].shape == (1, 6)

    def test_shape_dtype_subset_components(self, simulation_config, small_grid_slice, random_key, single_frequency):
        """Test shape with subset of components."""
        detector = PhasorDetector(wave_characters=single_frequency, components=("Ex", "Ey"))
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        # 1 frequency, 2 components, 8x8x8 grid
        assert shape_dtype["phasor"].shape == (1, 2, 8, 8, 8)

    def test_shape_dtype_complex128(self, simulation_config, small_grid_slice, random_key, single_frequency):
        """Test with complex128 dtype."""
        detector = PhasorDetector(wave_characters=single_frequency, dtype=jnp.complex128)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert shape_dtype["phasor"].dtype == jnp.complex128

    def test_init_state(self, simulation_config, small_grid_slice, random_key, single_frequency):
        """Test state initialization."""
        detector = PhasorDetector(wave_characters=single_frequency)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        state = detector.init_state()

        assert "phasor" in state
        # Only 1 latent time step for phasor detector
        assert state["phasor"].shape == (1, 1, 6, 8, 8, 8)


class TestPhasorDetectorUpdate:
    """Tests for PhasorDetector update method."""

    def test_update_computes_phasor(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        single_frequency,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test that update computes phasor from E and H fields."""
        detector = PhasorDetector(wave_characters=single_frequency)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        state = detector.init_state()

        new_state = detector.update(
            time_step=jnp.array(0),
            E=constant_E_field,
            H=constant_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        # Phasor should be complex
        assert jnp.iscomplexobj(new_state["phasor"])
        # Should have non-zero values with non-zero input
        assert jnp.any(jnp.abs(new_state["phasor"]) > 0)

    def test_update_accumulates(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        single_frequency,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test that phasor accumulates over time steps."""
        detector = PhasorDetector(wave_characters=single_frequency)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        state = detector.init_state()

        # First update
        state = detector.update(
            time_step=jnp.array(0),
            E=constant_E_field,
            H=constant_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )
        first_value = state["phasor"].copy()

        # Second update
        state = detector.update(
            time_step=jnp.array(1),
            E=constant_E_field,
            H=constant_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )
        second_value = state["phasor"]

        # Phase-rotated accumulation: two updates produce a different value than one.
        assert not jnp.allclose(first_value, second_value)

    def test_update_accumulates_same_time_step(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        single_frequency,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Two updates at the same time step exactly double the phasor magnitude."""
        detector = PhasorDetector(wave_characters=single_frequency)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        state = detector.init_state()

        # Single update at t=0
        state_after_one = detector.update(
            time_step=jnp.array(0),
            E=constant_E_field,
            H=constant_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        # Second update at the same t=0
        state_after_two = detector.update(
            time_step=jnp.array(0),
            E=constant_E_field,
            H=constant_H_field,
            state=state_after_one,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        # True accumulation means the second call adds to the first, so value ≈ 2×
        assert jnp.allclose(state_after_two["phasor"], 2 * state_after_one["phasor"], atol=1e-5)

    def test_update_with_reduce_volume(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        single_frequency,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test update with volume reduction."""
        detector = PhasorDetector(wave_characters=single_frequency, reduce_volume=True)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        state = detector.init_state()

        new_state = detector.update(
            time_step=jnp.array(0),
            E=constant_E_field,
            H=constant_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        # Should be reduced to (1, 1, 6) - 1 latent step, 1 freq, 6 components
        assert new_state["phasor"].shape == (1, 1, 6)

    def test_update_multiple_frequencies(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        multiple_frequencies,
        sinusoidal_E_field,
        sinusoidal_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test update with multiple frequencies."""
        detector = PhasorDetector(wave_characters=multiple_frequencies, reduce_volume=True)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        state = detector.init_state()

        new_state = detector.update(
            time_step=jnp.array(0),
            E=sinusoidal_E_field,
            H=sinusoidal_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        # Should have 3 frequencies
        assert new_state["phasor"].shape[1] == 3

    def test_update_zero_field_zero_phasor(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        single_frequency,
        inv_permittivity,
        inv_permeability,
    ):
        """Test that zero fields give zero phasor."""
        detector = PhasorDetector(wave_characters=single_frequency, reduce_volume=True)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        state = detector.init_state()

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)

        new_state = detector.update(
            time_step=jnp.array(0),
            E=E,
            H=H,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        assert jnp.allclose(jnp.abs(new_state["phasor"]), 0.0)


class TestPhasorDetectorConfiguration:
    """Tests for PhasorDetector configuration options."""

    def test_invalid_dtype_raises(self, single_frequency):
        """Test that invalid dtype raises error."""
        with pytest.raises(Exception, match="Invalid dtype"):
            PhasorDetector(wave_characters=single_frequency, dtype=jnp.float32)

    def test_default_components(self, single_frequency):
        """Test default component configuration."""
        detector = PhasorDetector(wave_characters=single_frequency)
        assert detector.components == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

    def test_plot_default_false(self, single_frequency):
        """Test plot defaults to False for phasor detector."""
        detector = PhasorDetector(wave_characters=single_frequency)
        assert detector.plot is False

    def test_reduce_volume_default_false(self, single_frequency):
        """Test reduce_volume defaults to False."""
        detector = PhasorDetector(wave_characters=single_frequency)
        assert detector.reduce_volume is False

    def test_dtype_default_complex64(self, single_frequency):
        """Test dtype defaults to complex64."""
        detector = PhasorDetector(wave_characters=single_frequency)
        assert detector.dtype == jnp.complex64


class TestPhasorDetectorAngularFrequencies:
    """Tests for angular frequency calculation."""

    def test_angular_frequencies_from_wavelength(self, simulation_config, small_grid_slice, random_key):
        """Test angular frequency calculation from wavelength."""
        wc = [WaveCharacter(wavelength=1e-6)]  # 1 micron = 3e14 Hz
        detector = PhasorDetector(wave_characters=wc)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        angular_freq = detector._angular_frequencies
        # omega = 2 * pi * f = 2 * pi * c / lambda
        expected = 2 * jnp.pi * 3e14  # approximately
        assert jnp.isclose(angular_freq[0], expected, rtol=0.01)

    def test_angular_frequencies_from_frequency(self, simulation_config, small_grid_slice, random_key):
        """Test angular frequency calculation from frequency."""
        freq = 1e14  # Hz
        wc = [WaveCharacter(frequency=freq)]
        detector = PhasorDetector(wave_characters=wc)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        angular_freq = detector._angular_frequencies
        expected = 2 * jnp.pi * freq
        assert jnp.isclose(angular_freq[0], expected)


class TestPhasorDetectorLatentTimeSteps:
    """Tests for latent time step behavior."""

    def test_num_latent_time_steps_always_one(self, simulation_config, small_grid_slice, random_key, single_frequency):
        """Test that phasor detector always has 1 latent time step."""
        detector = PhasorDetector(wave_characters=single_frequency)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        # Phasor accumulates all time into single array
        assert detector._num_latent_time_steps() == 1
