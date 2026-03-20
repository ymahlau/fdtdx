"""Tests for objects/detectors/field.py - Field detector."""

import jax.numpy as jnp

from fdtdx.objects.detectors.field import FieldDetector


class TestFieldDetectorShapes:
    """Tests for FieldDetector shape and initialization."""

    def test_shape_dtype_full_volume(self, simulation_config, small_grid_slice, random_key):
        """Test shape calculation for full volume recording."""
        detector = FieldDetector()
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert "fields" in shape_dtype
        # 6 components (Ex, Ey, Ez, Hx, Hy, Hz), 8x8x8 grid
        assert shape_dtype["fields"].shape == (6, 8, 8, 8)
        assert shape_dtype["fields"].dtype == jnp.float32

    def test_shape_dtype_reduced_volume(self, simulation_config, small_grid_slice, random_key):
        """Test shape calculation for reduced volume."""
        detector = FieldDetector(reduce_volume=True)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert shape_dtype["fields"].shape == (6,)

    def test_shape_dtype_subset_components(self, simulation_config, small_grid_slice, random_key):
        """Test shape with subset of components."""
        detector = FieldDetector(components=("Ex", "Ey"))
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert shape_dtype["fields"].shape == (2, 8, 8, 8)

    def test_init_state(self, simulation_config, small_grid_slice, random_key):
        """Test state initialization."""
        detector = FieldDetector()
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        state = detector.init_state()

        assert "fields" in state
        num_time_steps = detector.num_time_steps_recorded
        assert state["fields"].shape == (num_time_steps, 6, 8, 8, 8)


class TestFieldDetectorUpdate:
    """Tests for FieldDetector update method."""

    def test_update_records_all_components(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test that update records all 6 field components."""
        detector = FieldDetector()
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

        # E field is all 1.0, H field is all 0.5
        recorded = new_state["fields"][0]  # First time step
        assert jnp.allclose(recorded[0], 1.0)  # Ex
        assert jnp.allclose(recorded[1], 1.0)  # Ey
        assert jnp.allclose(recorded[2], 1.0)  # Ez
        assert jnp.allclose(recorded[3], 0.5)  # Hx
        assert jnp.allclose(recorded[4], 0.5)  # Hy
        assert jnp.allclose(recorded[5], 0.5)  # Hz

    def test_update_records_subset_components(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test recording only Ex and Hz components."""
        detector = FieldDetector(components=("Ex", "Hz"))
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

        recorded = new_state["fields"][0]
        assert recorded.shape[0] == 2  # Only 2 components
        assert jnp.allclose(recorded[0], 1.0)  # Ex
        assert jnp.allclose(recorded[1], 0.5)  # Hz

    def test_update_with_reduce_volume(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test update with volume reduction."""
        detector = FieldDetector(reduce_volume=True)
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

        recorded = new_state["fields"][0]
        assert recorded.shape == (6,)  # Reduced to 6 values
        # Mean of constant fields
        assert jnp.allclose(recorded[:3], 1.0)  # E components
        assert jnp.allclose(recorded[3:], 0.5)  # H components

    def test_update_multiple_time_steps(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        inv_permittivity,
        inv_permeability,
    ):
        """Test updating at multiple time steps."""
        detector = FieldDetector(reduce_volume=True, components=("Ex",))
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        state = detector.init_state()

        # Time step 0: E = 1.0
        E1 = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        H1 = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        state = detector.update(jnp.array(0), E1, H1, state, inv_permittivity, inv_permeability)

        # Time step 1: E = 2.0
        E2 = jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * 2.0
        H2 = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        state = detector.update(jnp.array(1), E2, H2, state, inv_permittivity, inv_permeability)

        assert jnp.isclose(state["fields"][0, 0], 1.0)  # t=0
        assert jnp.isclose(state["fields"][1, 0], 2.0)  # t=1

    def test_update_with_sinusoidal_field(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        sinusoidal_E_field,
        sinusoidal_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test update with sinusoidal field pattern."""
        detector = FieldDetector()
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

        recorded = new_state["fields"][0]
        # Check that the recorded Ex matches the input
        assert jnp.allclose(recorded[0], sinusoidal_E_field[0])


class TestFieldDetectorConfiguration:
    """Tests for FieldDetector configuration options."""

    def test_default_components(self):
        """Test default component configuration."""
        detector = FieldDetector()
        assert detector.components == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

    def test_custom_components_E_only(self):
        """Test with E field components only."""
        detector = FieldDetector(components=("Ex", "Ey", "Ez"))
        assert len(detector.components) == 3

    def test_custom_components_H_only(self):
        """Test with H field components only."""
        detector = FieldDetector(components=("Hx", "Hy", "Hz"))
        assert len(detector.components) == 3

    def test_single_component(self):
        """Test with single component."""
        detector = FieldDetector(components=("Ex",))
        assert len(detector.components) == 1

    def test_reduce_volume_default_false(self):
        """Test reduce_volume defaults to False."""
        detector = FieldDetector()
        assert detector.reduce_volume is False


class TestFieldDetectorPlaneSlice:
    """Tests for FieldDetector with plane/line slices."""

    def test_plane_slice_shape(self, simulation_config, plane_grid_slice, random_key):
        """Test shape with 2D plane slice."""
        detector = FieldDetector()
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert shape_dtype["fields"].shape == (6, 8, 8, 1)

    def test_line_slice_shape(self, simulation_config, line_grid_slice, random_key):
        """Test shape with 1D line slice."""
        detector = FieldDetector()
        detector = detector.place_on_grid(line_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert shape_dtype["fields"].shape == (6, 8, 1, 1)
