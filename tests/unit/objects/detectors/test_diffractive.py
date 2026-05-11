"""Tests for objects/detectors/diffractive.py - DiffractiveDetector."""

import jax.numpy as jnp
import pytest

from fdtdx.objects.detectors.diffractive import DiffractiveDetector


class TestDiffractiveDetectorInit:
    """Tests for DiffractiveDetector initialization and validation."""

    def test_valid_complex64_dtype(self):
        """complex64 dtype is accepted."""
        det = DiffractiveDetector(frequencies=[3e14], direction="+", dtype=jnp.complex64)
        assert det.dtype == jnp.complex64

    def test_valid_complex128_dtype(self):
        """complex128 dtype is accepted."""
        det = DiffractiveDetector(frequencies=[3e14], direction="+", dtype=jnp.complex128)
        assert det.dtype == jnp.complex128

    def test_float32_dtype_raises(self):
        """Non-complex dtype raises during post-init validation."""
        with pytest.raises(Exception, match="Invalid dtype"):
            DiffractiveDetector(frequencies=[3e14], direction="+", dtype=jnp.float32)

    def test_float64_dtype_raises(self):
        """float64 dtype raises during post-init validation."""
        with pytest.raises(Exception, match="Invalid dtype"):
            DiffractiveDetector(frequencies=[3e14], direction="+", dtype=jnp.float64)

    def test_default_order_is_zeroth(self):
        """Default diffraction order is (0, 0)."""
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        assert det.orders == ((0, 0),)

    def test_custom_orders(self):
        """Custom diffraction orders are stored correctly."""
        orders = ((0, 0), (1, 0), (-1, 0), (0, 1))
        det = DiffractiveDetector(frequencies=[3e14], direction="+", orders=orders)
        assert len(det.orders) == 4

    def test_direction_plus(self):
        """Positive direction is stored correctly."""
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        assert det.direction == "+"

    def test_direction_minus(self):
        """Negative direction is stored correctly."""
        det = DiffractiveDetector(frequencies=[3e14], direction="-")
        assert det.direction == "-"

    def test_multiple_frequencies(self):
        """Multiple frequencies are stored correctly."""
        freqs = [1e14, 2e14, 3e14]
        det = DiffractiveDetector(frequencies=freqs, direction="+")
        assert len(det.frequencies) == 3

    def test_empty_frequencies_raises(self):
        """Empty frequency list should raise ValueError."""
        with pytest.raises(ValueError, match="at least one frequency"):
            DiffractiveDetector(frequencies=[], direction="+")

    def test_negative_frequency_raises(self):
        """Negative frequency should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            DiffractiveDetector(frequencies=[-3e14], direction="+")

    def test_zero_frequency_raises(self):
        """Zero frequency should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            DiffractiveDetector(frequencies=[0.0], direction="+")

    def test_empty_orders_raises(self):
        """Empty orders list should raise ValueError."""
        with pytest.raises(ValueError, match="at least one diffraction order"):
            DiffractiveDetector(frequencies=[3e14], direction="+", orders=())


class TestDiffractiveDetectorPropagationAxis:
    """Tests for the propagation_axis property."""

    def test_propagation_axis_z(self, simulation_config, plane_grid_slice, random_key):
        """Plane perpendicular to z-axis → propagation_axis == 2."""
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        assert det.propagation_axis == 2

    def test_propagation_axis_x(self, simulation_config, random_key):
        """Plane perpendicular to x-axis → propagation_axis == 0."""
        x_plane = ((0, 1), (0, 8), (0, 8))
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        det = det.place_on_grid(x_plane, simulation_config, random_key)
        assert det.propagation_axis == 0

    def test_propagation_axis_y(self, simulation_config, random_key):
        """Plane perpendicular to y-axis → propagation_axis == 1."""
        y_plane = ((0, 8), (0, 1), (0, 8))
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        det = det.place_on_grid(y_plane, simulation_config, random_key)
        assert det.propagation_axis == 1

    def test_non_plane_shape_raises(self, simulation_config, small_grid_slice, random_key):
        """Volume detector (no single-cell dimension) raises."""
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        det = det.place_on_grid(small_grid_slice, simulation_config, random_key)
        with pytest.raises(Exception, match="Invalid diffractive detector shape"):
            _ = det.propagation_axis


class TestDiffractiveDetectorShapeDtype:
    """Tests for _shape_dtype_single_time_step and related."""

    def test_single_freq_single_order_shape(self, simulation_config, plane_grid_slice, random_key):
        """Single frequency, default order → shape (1, 1)."""
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        shape_dtype = det._shape_dtype_single_time_step()

        assert "diffractive" in shape_dtype
        assert shape_dtype["diffractive"].shape == (1, 1)

    def test_multi_freq_multi_order_shape(self, simulation_config, plane_grid_slice, random_key):
        """Multiple frequencies and orders → shape (num_freqs, num_orders)."""
        det = DiffractiveDetector(
            frequencies=[3e14, 2e14, 1e14],
            direction="+",
            orders=((0, 0), (1, 0), (-1, 0)),
        )
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        shape_dtype = det._shape_dtype_single_time_step()

        assert shape_dtype["diffractive"].shape == (3, 3)

    def test_output_dtype_is_complex64(self, simulation_config, plane_grid_slice, random_key):
        """Default dtype → output ShapeDtypeStruct dtype is complex64."""
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        shape_dtype = det._shape_dtype_single_time_step()

        assert shape_dtype["diffractive"].dtype == jnp.complex64

    def test_latent_time_steps_always_one(self, simulation_config, plane_grid_slice, random_key):
        """DiffractiveDetector always accumulates into 1 time slot."""
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        assert det._num_latent_time_steps() == 1

    def test_init_state_shape(self, simulation_config, plane_grid_slice, random_key):
        """init_state creates array with shape (1, num_freqs, num_orders)."""
        det = DiffractiveDetector(frequencies=[3e14, 2e14], direction="+", orders=((0, 0), (1, 0)))
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()

        assert "diffractive" in state
        assert state["diffractive"].shape == (1, 2, 2)

    def test_init_state_is_complex(self, simulation_config, plane_grid_slice, random_key):
        """init_state creates complex-valued array."""
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()

        assert jnp.iscomplexobj(state["diffractive"])


class TestDiffractiveDetectorUpdate:
    """Tests for the update method."""

    def test_update_returns_complex_state(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """update() computes complex diffractive amplitudes."""
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()

        new_state = det.update(
            time_step=jnp.array(0),
            E=constant_E_field,
            H=constant_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        assert "diffractive" in new_state
        assert jnp.iscomplexobj(new_state["diffractive"])

    def test_update_zero_fields_zero_amplitude(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        inv_permittivity,
        inv_permeability,
    ):
        """Zero fields produce zero diffractive amplitude."""
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()

        E = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)
        H = jnp.zeros((3, 8, 8, 8), dtype=jnp.float32)

        new_state = det.update(
            time_step=jnp.array(0),
            E=E,
            H=H,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        assert jnp.allclose(jnp.abs(new_state["diffractive"]), 0.0)

    def test_update_preserves_shape(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """update() output shape matches (1, num_freqs, num_orders)."""
        det = DiffractiveDetector(frequencies=[3e14, 2e14], direction="+", orders=((0, 0), (1, 0)))
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()

        new_state = det.update(
            time_step=jnp.array(0),
            E=constant_E_field,
            H=constant_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        assert new_state["diffractive"].shape == (1, 2, 2)

    def test_direction_negates_amplitude(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        sinusoidal_E_field,
        sinusoidal_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Negative direction negates the computed diffractive amplitude."""
        det_pos = DiffractiveDetector(frequencies=[3e14], direction="+")
        det_pos = det_pos.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state_pos = det_pos.init_state()

        det_neg = DiffractiveDetector(frequencies=[3e14], direction="-")
        det_neg = det_neg.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state_neg = det_neg.init_state()

        new_pos = det_pos.update(
            jnp.array(0),
            sinusoidal_E_field,
            sinusoidal_H_field,
            state_pos,
            inv_permittivity,
            inv_permeability,
        )
        new_neg = det_neg.update(
            jnp.array(0),
            sinusoidal_E_field,
            sinusoidal_H_field,
            state_neg,
            inv_permittivity,
            inv_permeability,
        )

        assert jnp.allclose(new_pos["diffractive"], -new_neg["diffractive"])

    def test_update_multiple_orders(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        sinusoidal_E_field,
        sinusoidal_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Orders (1, 0) and (-1, 0) can differ from (0, 0) with non-trivial fields."""
        det = DiffractiveDetector(
            frequencies=[3e14],
            direction="+",
            orders=((0, 0), (1, 0), (-1, 0)),
        )
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()

        new_state = det.update(
            time_step=jnp.array(0),
            E=sinusoidal_E_field,
            H=sinusoidal_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        assert new_state["diffractive"].shape == (1, 1, 3)

    def test_update_nonzero_with_nonzero_fields(
        self,
        simulation_config,
        plane_grid_slice,
        random_key,
        sinusoidal_E_field,
        sinusoidal_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """Non-zero fields produce non-zero diffractive amplitude."""
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        det = det.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = det.init_state()

        new_state = det.update(
            time_step=jnp.array(0),
            E=sinusoidal_E_field,
            H=sinusoidal_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        assert jnp.any(jnp.abs(new_state["diffractive"]) > 0)

    def test_update_x_propagation_plane(
        self,
        simulation_config,
        random_key,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """update() works on x-normal planes (FFT axes fixed to (1, 2))."""
        x_plane = ((0, 1), (0, 8), (0, 8))
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        det = det.place_on_grid(x_plane, simulation_config, random_key)
        state = det.init_state()

        new_state = det.update(
            time_step=jnp.array(0),
            E=constant_E_field,
            H=constant_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        assert "diffractive" in new_state
        assert new_state["diffractive"].shape == (1, 1, 1)

    def test_update_y_propagation_plane(
        self,
        simulation_config,
        random_key,
        constant_E_field,
        constant_H_field,
        inv_permittivity,
        inv_permeability,
    ):
        """update() works on y-normal planes (FFT axes fixed to (1, 2))."""
        y_plane = ((0, 8), (0, 1), (0, 8))
        det = DiffractiveDetector(frequencies=[3e14], direction="+")
        det = det.place_on_grid(y_plane, simulation_config, random_key)
        state = det.init_state()

        new_state = det.update(
            time_step=jnp.array(0),
            E=constant_E_field,
            H=constant_H_field,
            state=state,
            inv_permittivity=inv_permittivity,
            inv_permeability=inv_permeability,
        )

        assert "diffractive" in new_state
        assert new_state["diffractive"].shape == (1, 1, 1)
