"""Tests for objects/detectors/energy.py - Energy detector."""

import jax.numpy as jnp
import pytest

from fdtdx.objects.detectors.energy import EnergyDetector


class TestEnergyDetectorShapes:
    """Tests for EnergyDetector shape and initialization."""

    def test_shape_dtype_full_volume(self, simulation_config, small_grid_slice, random_key):
        """Test shape calculation for full volume recording."""
        detector = EnergyDetector()
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert "energy" in shape_dtype
        assert shape_dtype["energy"].shape == (8, 8, 8)
        assert shape_dtype["energy"].dtype == jnp.float32

    def test_shape_dtype_reduced_volume(self, simulation_config, small_grid_slice, random_key):
        """Test shape calculation for reduced volume."""
        detector = EnergyDetector(reduce_volume=True)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert shape_dtype["energy"].shape == (1,)

    def test_shape_dtype_as_slices(self, simulation_config, small_grid_slice, random_key):
        """Test shape calculation for slice mode."""
        detector = EnergyDetector(as_slices=True)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        shape_dtype = detector._shape_dtype_single_time_step()

        assert "XY Plane" in shape_dtype
        assert "XZ Plane" in shape_dtype
        assert "YZ Plane" in shape_dtype
        assert shape_dtype["XY Plane"].shape == (8, 8)
        assert shape_dtype["XZ Plane"].shape == (8, 8)
        assert shape_dtype["YZ Plane"].shape == (8, 8)

    def test_slices_and_reduce_raises(self, simulation_config, small_grid_slice, random_key):
        """Test that both as_slices and reduce_volume raises error."""
        detector = EnergyDetector(as_slices=True, reduce_volume=True)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        with pytest.raises(Exception, match="Cannot both reduce volume and save slices"):
            detector._shape_dtype_single_time_step()

    def test_init_state(self, simulation_config, small_grid_slice, random_key):
        """Test state initialization."""
        detector = EnergyDetector()
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)

        state = detector.init_state()

        assert "energy" in state
        num_time_steps = detector.num_time_steps_recorded
        assert state["energy"].shape == (num_time_steps, 8, 8, 8)


class TestEnergyDetectorUpdate:
    """Tests for EnergyDetector update method."""

    def test_update_computes_energy(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test that update computes energy from E and H fields."""
        detector = EnergyDetector()
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

        # Energy should be non-negative
        assert jnp.all(new_state["energy"][0] >= 0)
        # With constant non-zero fields, energy should be non-zero
        assert jnp.any(new_state["energy"][0] > 0)

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
        detector = EnergyDetector(reduce_volume=True)
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

        # Should be a single value (total energy)
        assert new_state["energy"][0].shape == (1,)
        assert new_state["energy"][0, 0] > 0

    def test_update_as_slices_mean(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test update with slice mode using mean aggregation."""
        detector = EnergyDetector(as_slices=True, aggregate="mean")
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

        assert "XY Plane" in new_state
        assert "XZ Plane" in new_state
        assert "YZ Plane" in new_state
        assert new_state["XY Plane"][0].shape == (8, 8)

    def test_update_zero_field_zero_energy(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        inv_permittivity,
        inv_permeability,
    ):
        """Test that zero fields give zero energy."""
        detector = EnergyDetector(reduce_volume=True)
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

        assert jnp.isclose(new_state["energy"][0, 0], 0.0)

    def test_update_multiple_time_steps(
        self,
        simulation_config,
        small_grid_slice,
        random_key,
        inv_permittivity,
        inv_permeability,
    ):
        """Test updating at multiple time steps."""
        detector = EnergyDetector(reduce_volume=True)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        state = detector.init_state()

        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)

        # Time step 0: low energy
        E1 = jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * 0.5
        state = detector.update(jnp.array(0), E1, H, state, inv_permittivity, inv_permeability)

        # Time step 1: higher energy
        E2 = jnp.ones((3, 8, 8, 8), dtype=jnp.float32) * 2.0
        state = detector.update(jnp.array(1), E2, H, state, inv_permittivity, inv_permeability)

        # Higher E field should give higher energy
        assert state["energy"][1, 0] > state["energy"][0, 0]


class TestEnergyDetectorConfiguration:
    """Tests for EnergyDetector configuration options."""

    def test_default_configuration(self):
        """Test default configuration values."""
        detector = EnergyDetector()
        assert detector.as_slices is False
        assert detector.reduce_volume is False
        assert detector.x_slice is None
        assert detector.y_slice is None
        assert detector.z_slice is None
        assert detector.aggregate is None

    def test_slice_positions(self):
        """Test slice position configuration."""
        detector = EnergyDetector(
            as_slices=True,
            x_slice=1e-6,
            y_slice=2e-6,
            z_slice=3e-6,
        )
        assert detector.x_slice == 1e-6
        assert detector.y_slice == 2e-6
        assert detector.z_slice == 3e-6

    def test_aggregate_mean(self):
        """Test aggregate mean configuration."""
        detector = EnergyDetector(as_slices=True, aggregate="mean")
        assert detector.aggregate == "mean"


class TestEnergyDetectorEdgeCases:
    """Edge case tests for EnergyDetector."""

    def test_single_point_detector(
        self,
        simulation_config,
        point_grid_slice,
        random_key,
        inv_permittivity,
        inv_permeability,
    ):
        """Test with single point detector."""
        detector = EnergyDetector()
        detector = detector.place_on_grid(point_grid_slice, simulation_config, random_key)
        state = detector.init_state()

        E = jnp.ones((3, 8, 8, 8), dtype=jnp.float32)
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)

        new_state = detector.update(
            time_step=jnp.array(0),
            E=E,
            H=H,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        # Should be a single point
        assert new_state["energy"][0].shape == (1, 1, 1)

    def test_plane_detector(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Test with plane detector."""
        detector = EnergyDetector()
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

        assert new_state["energy"][0].shape == (8, 8, 1)
