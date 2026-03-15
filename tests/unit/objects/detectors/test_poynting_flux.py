"""Tests for objects/detectors/poynting_flux.py - Poynting flux detector."""

import jax.numpy as jnp
import pytest

from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector


class TestPoyntingFluxDetectorShapes:
    """Tests for PoyntingFluxDetector shape and initialization."""

    def test_shape_dtype_reduced_volume(self, simulation_config, plane_grid_slice, random_key):
        """Test shape calculation for reduced volume (default)."""
        detector = PoyntingFluxDetector(direction="+")
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert "poynting_flux" in shape_dtype
        assert shape_dtype["poynting_flux"].shape == (1,)
        assert shape_dtype["poynting_flux"].dtype == jnp.float32

    def test_shape_dtype_full_spatial(self, simulation_config, plane_grid_slice, random_key):
        """Test shape calculation with full spatial distribution."""
        detector = PoyntingFluxDetector(direction="+", reduce_volume=False)
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert shape_dtype["poynting_flux"].shape == (8, 8, 1)

    def test_shape_dtype_all_components_reduced(self, simulation_config, plane_grid_slice, random_key):
        """Test shape with all components and reduced volume."""
        detector = PoyntingFluxDetector(direction="+", keep_all_components=True, reduce_volume=True)
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert shape_dtype["poynting_flux"].shape == (3,)

    def test_shape_dtype_all_components_full(self, simulation_config, plane_grid_slice, random_key):
        """Test shape with all components and full spatial."""
        detector = PoyntingFluxDetector(direction="+", keep_all_components=True, reduce_volume=False)
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert shape_dtype["poynting_flux"].shape == (3, 8, 8, 1)

    def test_init_state(self, simulation_config, plane_grid_slice, random_key):
        """Test state initialization."""
        detector = PoyntingFluxDetector(direction="+")
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)

        state = detector.init_state()

        assert "poynting_flux" in state
        num_time_steps = detector.num_time_steps_recorded
        assert state["poynting_flux"].shape == (num_time_steps, 1)


class TestPoyntingFluxDetectorPropagationAxis:
    """Tests for propagation axis detection."""

    def test_propagation_axis_z(self, simulation_config, plane_grid_slice, random_key):
        """Test propagation axis detection for z-normal plane."""
        # plane_grid_slice is (8, 8, 1) - z-axis has size 1
        detector = PoyntingFluxDetector(direction="+")
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)

        assert detector.propagation_axis == 2

    def test_propagation_axis_x(self, simulation_config, random_key):
        """Test propagation axis detection for x-normal plane."""
        x_plane_slice = ((0, 1), (0, 8), (0, 8))
        detector = PoyntingFluxDetector(direction="+")
        detector = detector.place_on_grid(x_plane_slice, simulation_config, random_key)

        assert detector.propagation_axis == 0

    def test_propagation_axis_y(self, simulation_config, random_key):
        """Test propagation axis detection for y-normal plane."""
        y_plane_slice = ((0, 8), (0, 1), (0, 8))
        detector = PoyntingFluxDetector(direction="+")
        detector = detector.place_on_grid(y_plane_slice, simulation_config, random_key)

        assert detector.propagation_axis == 1

    def test_fixed_propagation_axis(self, simulation_config, small_grid_slice, random_key):
        """Test fixed propagation axis override."""
        detector = PoyntingFluxDetector(direction="+", fixed_propagation_axis=1)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        assert detector.propagation_axis == 1

    def test_invalid_fixed_propagation_axis_raises(self, simulation_config, small_grid_slice, random_key):
        """Test invalid fixed propagation axis raises error."""
        detector = PoyntingFluxDetector(direction="+", fixed_propagation_axis=5)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        with pytest.raises(Exception, match="Invalid"):
            _ = detector.propagation_axis

    def test_multiple_size_one_axes_raises(self, simulation_config, line_grid_slice, random_key):
        """Test that multiple size-1 axes raises error without fixed axis."""
        # line_grid_slice is (8, 1, 1) - two axes have size 1
        detector = PoyntingFluxDetector(direction="+")
        detector = detector.place_on_grid(line_grid_slice, simulation_config, random_key)

        with pytest.raises(Exception, match="Invalid poynting flux detector shape"):
            _ = detector.propagation_axis


class TestPoyntingFluxDetectorUpdate:
    """Tests for PoyntingFluxDetector update method."""

    def test_update_computes_flux(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test that update computes Poynting flux."""
        detector = PoyntingFluxDetector(direction="+")
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = detector.init_state()

        new_state = detector.update(
            time_step=jnp.array(0),
            E=constant_E_field,
            H=constant_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        # Poynting flux should be computed (E cross H)
        assert "poynting_flux" in new_state
        # Value depends on cross product of constant fields

    def test_update_direction_negates(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        sinusoidal_E_field,
        sinusoidal_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test that negative direction negates the flux."""
        detector_pos = PoyntingFluxDetector(direction="+")
        detector_pos = detector_pos.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state_pos = detector_pos.init_state()

        detector_neg = PoyntingFluxDetector(direction="-")
        detector_neg = detector_neg.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state_neg = detector_neg.init_state()

        new_state_pos = detector_pos.update(
            jnp.array(0),
            sinusoidal_E_field,
            sinusoidal_H_field,
            state_pos,
            inv_permittivity,
            inv_permeability,
        )
        new_state_neg = detector_neg.update(
            jnp.array(0),
            sinusoidal_E_field,
            sinusoidal_H_field,
            state_neg,
            inv_permittivity,
            inv_permeability,
        )

        # Flux should be negated
        assert jnp.allclose(
            new_state_pos["poynting_flux"][0],
            -new_state_neg["poynting_flux"][0],
        )

    def test_update_zero_field_zero_flux(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        inv_permittivity,
        inv_permeability,
    ):
        """Test that zero fields give zero flux."""
        detector = PoyntingFluxDetector(direction="+")
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)
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

        assert jnp.allclose(new_state["poynting_flux"][0], 0.0)

    def test_update_all_components(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        sinusoidal_E_field,
        sinusoidal_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test update with all vector components."""
        detector = PoyntingFluxDetector(direction="+", keep_all_components=True, reduce_volume=True)
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = detector.init_state()

        new_state = detector.update(
            time_step=jnp.array(0),
            E=sinusoidal_E_field,
            H=sinusoidal_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        # Should have 3 components
        assert new_state["poynting_flux"][0].shape == (3,)

    def test_update_multiple_time_steps(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        inv_permittivity,
        inv_permeability,
    ):
        """Test updating at multiple time steps."""
        detector = PoyntingFluxDetector(direction="+")
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = detector.init_state()

        # Time step 0
        E1 = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        H1 = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        state = detector.update(jnp.array(0), E1, H1, state, inv_permittivity, inv_permeability)

        # Time step 1: doubled fields
        E2 = jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * 2.0
        H2 = jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * 2.0
        state = detector.update(jnp.array(1), E2, H2, state, inv_permittivity, inv_permeability)

        # Both time steps should have values recorded
        # (Note: specific values depend on cross product implementation)


class TestPoyntingFluxDetectorConfiguration:
    """Tests for PoyntingFluxDetector configuration options."""

    def test_direction_required(self):
        """Test that direction is required."""
        # This should work with direction specified
        detector = PoyntingFluxDetector(direction="+")
        assert detector.direction == "+"

        detector = PoyntingFluxDetector(direction="-")
        assert detector.direction == "-"

    def test_reduce_volume_default_true(self):
        """Test reduce_volume defaults to True."""
        detector = PoyntingFluxDetector(direction="+")
        assert detector.reduce_volume is True

    def test_keep_all_components_default_false(self):
        """Test keep_all_components defaults to False."""
        detector = PoyntingFluxDetector(direction="+")
        assert detector.keep_all_components is False

    def test_fixed_propagation_axis_default_none(self):
        """Test fixed_propagation_axis defaults to None."""
        detector = PoyntingFluxDetector(direction="+")
        assert detector.fixed_propagation_axis is None
