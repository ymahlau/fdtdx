"""Unit tests for objects/boundaries/perfectly_matched_layer.py.

Tests PerfectlyMatchedLayer initialization, properties, and profile computation.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer


@pytest.fixture
def micro_config():
    return SimulationConfig(
        time=100e-15,
        resolution=50e-9,
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )


@pytest.fixture
def jax_key():
    return jax.random.PRNGKey(0)


def make_pml(axis=0, direction="-", thickness=10, **kwargs):
    """Create a PML with given axis/direction and optional overrides."""
    shape_list = [None, None, None]
    shape_list[axis] = thickness
    return PerfectlyMatchedLayer(
        axis=axis,
        partial_grid_shape=tuple(shape_list),
        direction=direction,
        **kwargs,
    )


def place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 20)):
    """Place a PML on a grid matching the given volume shape."""
    axis = pml.axis
    direction = pml.direction
    thickness = pml.partial_grid_shape[axis]

    # Build grid_slice_tuple: PML occupies the boundary slice along its axis
    slices = [[0, volume_shape[i]] for i in range(3)]
    if direction == "-":
        slices[axis] = [0, thickness]
    else:
        slices[axis] = [volume_shape[axis] - thickness, volume_shape[axis]]

    grid_slice_tuple = tuple(tuple(s) for s in slices)
    return pml.place_on_grid(grid_slice_tuple=grid_slice_tuple, config=micro_config, key=jax_key)


class TestPMLPostInit:
    """Tests for PerfectlyMatchedLayer.__post_init__ default values."""

    def test_alpha_start_default(self):
        pml = make_pml()
        assert pml.alpha_start is not None
        assert pml.alpha_start > 0

    def test_alpha_end_default_zero(self):
        pml = make_pml()
        assert pml.alpha_end == 0.0

    def test_alpha_order_default(self):
        pml = make_pml()
        assert pml.alpha_order == 1.0

    def test_kappa_start_default(self):
        pml = make_pml()
        assert pml.kappa_start == 1.0

    def test_kappa_end_default(self):
        pml = make_pml()
        assert pml.kappa_end == 1.0

    def test_kappa_order_default(self):
        pml = make_pml()
        assert pml.kappa_order == 3.0

    def test_sigma_start_default_zero(self):
        pml = make_pml()
        assert pml.sigma_start == 0.0

    def test_sigma_order_default(self):
        pml = make_pml()
        assert pml.sigma_order == 3.0

    def test_sigma_end_none_before_place_on_grid(self):
        # sigma_end is only computed in place_on_grid
        pml = make_pml()
        assert pml.sigma_end is None

    def test_provided_values_not_overwritten(self):
        pml = make_pml(
            alpha_start=0.5,
            alpha_end=0.1,
            alpha_order=2.0,
            kappa_start=1.5,
            kappa_end=2.0,
            kappa_order=2.0,
            sigma_start=0.1,
            sigma_end=1.0,
            sigma_order=2.0,
        )
        assert pml.alpha_start == 0.5
        assert pml.alpha_end == 0.1
        assert pml.alpha_order == 2.0
        assert pml.kappa_start == 1.5
        assert pml.kappa_end == 2.0
        assert pml.kappa_order == 2.0
        assert pml.sigma_start == 0.1
        assert pml.sigma_end == 1.0
        assert pml.sigma_order == 2.0


class TestPMLPlaceOnGrid:
    """Tests for PerfectlyMatchedLayer.place_on_grid."""

    def test_sigma_end_computed_when_none(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=10)
        placed = place_pml(pml, micro_config, jax_key)
        assert placed.sigma_end is not None
        assert placed.sigma_end > 0

    def test_sigma_end_preserved_when_provided(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=10, sigma_end=999.0)
        placed = place_pml(pml, micro_config, jax_key)
        assert placed.sigma_end == 999.0

    def test_grid_shape_set_correctly(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=10)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 20))
        assert placed.grid_shape == (10, 20, 20)

    def test_grid_shape_axis1(self, micro_config, jax_key):
        pml = make_pml(axis=1, direction="-", thickness=8)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(20, 30, 20))
        assert placed.grid_shape == (20, 8, 20)

    def test_grid_shape_axis2_plus(self, micro_config, jax_key):
        pml = make_pml(axis=2, direction="+", thickness=6)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(20, 20, 30))
        assert placed.grid_shape == (20, 20, 6)


class TestPMLDescriptiveName:
    """Tests for PerfectlyMatchedLayer.descriptive_name."""

    @pytest.mark.parametrize(
        "axis,direction,expected",
        [
            (0, "-", "min_x"),
            (0, "+", "max_x"),
            (1, "-", "min_y"),
            (1, "+", "max_y"),
            (2, "-", "min_z"),
            (2, "+", "max_z"),
        ],
    )
    def test_descriptive_name(self, micro_config, jax_key, axis, direction, expected):
        pml = make_pml(axis=axis, direction=direction, thickness=8)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 30, 30))
        assert placed.descriptive_name == expected


class TestPMLThickness:
    """Tests for PerfectlyMatchedLayer.thickness property."""

    def test_thickness_matches_grid_shape(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=12)
        placed = place_pml(pml, micro_config, jax_key)
        assert placed.thickness == 12

    def test_thickness_axis1(self, micro_config, jax_key):
        pml = make_pml(axis=1, direction="-", thickness=7)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(20, 30, 20))
        assert placed.thickness == 7


class TestPMLComputeProfile:
    """Tests for PerfectlyMatchedLayer._compute_pml_profile."""

    def test_profile_shape_matches_grid_shape(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=10)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 20))
        profileE, profileH = placed._compute_pml_profile(0.0, 1.0, 3.0, jnp.float32)
        assert profileE.shape == (10, 20, 20)
        assert profileH.shape == (10, 20, 20)

    def test_profile_dtype(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=10)
        placed = place_pml(pml, micro_config, jax_key)
        profileE, profileH = placed._compute_pml_profile(0.0, 1.0, 3.0, jnp.float32)
        assert profileE.dtype == jnp.float32

    def test_min_direction_profile_decreasing(self, micro_config, jax_key):
        """For min boundary, profile should be highest at index 0 (outer edge) and decay inward."""
        pml = make_pml(axis=0, direction="-", thickness=10)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 5, 5))
        profileE, _ = placed._compute_pml_profile(0.0, 1.0, 3.0, jnp.float32)
        # Take 1D slice along axis 0
        profile_1d = profileE[:, 0, 0]
        # Outer edge (index 0) should be highest, inner (index -1) should be lowest
        assert float(profile_1d[0]) >= float(profile_1d[-1])

    def test_plus_direction_profile_increasing(self, micro_config, jax_key):
        """For max boundary, profile should increase from inner to outer edge."""
        pml = make_pml(axis=0, direction="+", thickness=10)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 5, 5))
        profileE, _ = placed._compute_pml_profile(0.0, 1.0, 3.0, jnp.float32)
        profile_1d = profileE[:, 0, 0]
        # Inner edge (index 0) should be lowest, outer edge (index -1) should be highest
        assert float(profile_1d[0]) <= float(profile_1d[-1])

    def test_uniform_profile_when_same_values(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=8)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(20, 5, 5))
        profileE, _ = placed._compute_pml_profile(1.0, 1.0, 3.0, jnp.float32)
        assert jnp.allclose(profileE, jnp.ones_like(profileE))

    def test_profile_axis1_shape(self, micro_config, jax_key):
        pml = make_pml(axis=1, direction="+", thickness=6)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(15, 20, 10))
        profileE, profileH = placed._compute_pml_profile(0.0, 1.0, 1.0, jnp.float32)
        assert profileE.shape == (15, 6, 10)

    def test_profile_axis2_shape(self, micro_config, jax_key):
        pml = make_pml(axis=2, direction="-", thickness=5)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(10, 10, 20))
        profileE, profileH = placed._compute_pml_profile(0.0, 1.0, 1.0, jnp.float32)
        assert profileE.shape == (10, 10, 5)


class TestPMLModifyArrays:
    """Tests for PerfectlyMatchedLayer.modify_arrays."""

    def _make_simulation_arrays(self, volume_shape):
        """Create zero simulation arrays for testing."""
        alpha = jnp.zeros((6, *volume_shape), dtype=jnp.float32)
        kappa = jnp.ones((6, *volume_shape), dtype=jnp.float32)
        sigma = jnp.zeros((6, *volume_shape), dtype=jnp.float32)
        electric_conductivity = jnp.zeros(volume_shape, dtype=jnp.float32)
        magnetic_conductivity = jnp.zeros(volume_shape, dtype=jnp.float32)
        return alpha, kappa, sigma, electric_conductivity, magnetic_conductivity

    def test_modify_arrays_returns_dict_with_correct_keys(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=10, sigma_end=1e6)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 20))
        alpha, kappa, sigma, ec, mc = self._make_simulation_arrays((30, 20, 20))
        result = placed.modify_arrays(alpha, kappa, sigma, ec, mc)
        assert set(result.keys()) == {"alpha", "kappa", "sigma", "electric_conductivity", "magnetic_conductivity"}

    def test_modify_arrays_sigma_nonzero_in_pml_region(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=10, sigma_end=1e6)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 20))
        alpha, kappa, sigma, ec, mc = self._make_simulation_arrays((30, 20, 20))
        result = placed.modify_arrays(alpha, kappa, sigma, ec, mc)
        # sigma should be modified in the PML region (axis 0 component)
        pml_sigma = result["sigma"][0, :10, :, :]
        assert jnp.any(pml_sigma != 0.0)

    def test_modify_arrays_sigma_zero_outside_pml(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=10, sigma_end=1e6)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 20))
        alpha, kappa, sigma, ec, mc = self._make_simulation_arrays((30, 20, 20))
        result = placed.modify_arrays(alpha, kappa, sigma, ec, mc)
        # sigma should be zero outside PML region
        outside_sigma = result["sigma"][0, 10:, :, :]
        assert jnp.all(outside_sigma == 0.0)

    def test_modify_arrays_max_boundary(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="+", thickness=10, sigma_end=1e6)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 20))
        alpha, kappa, sigma, ec, mc = self._make_simulation_arrays((30, 20, 20))
        result = placed.modify_arrays(alpha, kappa, sigma, ec, mc)
        pml_sigma = result["sigma"][0, 20:, :, :]
        assert jnp.any(pml_sigma != 0.0)

    def test_conductivity_arrays_unchanged(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=10, sigma_end=1e6)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 20))
        alpha, kappa, sigma, ec, mc = self._make_simulation_arrays((30, 20, 20))
        result = placed.modify_arrays(alpha, kappa, sigma, ec, mc)
        assert jnp.allclose(result["electric_conductivity"], ec)
        assert jnp.allclose(result["magnetic_conductivity"], mc)
