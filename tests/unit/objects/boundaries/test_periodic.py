"""Unit tests for BlochBoundary used as periodic boundary (bloch_vector=0).

Tests BlochBoundary initialization, properties, and slice computation
when used as a periodic boundary (zero Bloch vector).
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.objects.boundaries.bloch import BlochBoundary


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


def make_periodic(axis=0, direction="-"):
    """Create a BlochBoundary with zero bloch_vector (periodic) for given axis/direction."""
    shape_list = [None, None, None]
    shape_list[axis] = 1
    return BlochBoundary(
        axis=axis,
        partial_grid_shape=tuple(shape_list),
        direction=direction,
        bloch_vector=(0.0, 0.0, 0.0),
    )


def place_periodic(pb, micro_config, jax_key, volume_shape=(30, 20, 20), grid_slice_tuple=None):
    """Place a BlochBoundary on a grid matching the given volume shape."""
    if grid_slice_tuple is None:
        axis = pb.axis
        direction = pb.direction

        slices = [[0, volume_shape[i]] for i in range(3)]
        if direction == "-":
            slices[axis] = [0, 1]
        else:
            slices[axis] = [volume_shape[axis] - 1, volume_shape[axis]]

        grid_slice_tuple = tuple(tuple(s) for s in slices)
    return pb.place_on_grid(grid_slice_tuple=grid_slice_tuple, config=micro_config, key=jax_key)


class TestPeriodicThickness:
    """Tests for BlochBoundary.thickness when used as periodic."""

    def test_thickness_always_one(self):
        pb = make_periodic(axis=0, direction="-")
        assert pb.thickness == 1

    def test_thickness_one_for_all_axes(self):
        for axis in range(3):
            for direction in ("-", "+"):
                pb = make_periodic(axis=axis, direction=direction)
                assert pb.thickness == 1


class TestPeriodicDescriptiveName:
    """Tests for BlochBoundary.descriptive_name when used as periodic."""

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
        pb = make_periodic(axis=axis, direction=direction)
        placed = place_periodic(pb, micro_config, jax_key, volume_shape=(30, 30, 30))
        assert placed.descriptive_name == expected


class TestPeriodicBoundarySlice:
    """Tests for BlochBoundary.boundary_slice property."""

    def test_plus_direction_axis0_slice(self, micro_config, jax_key):
        """For + direction, boundary_slice should cover the first cell along axis."""
        pb = make_periodic(axis=0, direction="+")
        placed = place_periodic(pb, micro_config, jax_key, volume_shape=(30, 20, 20))
        bslice = placed.boundary_slice
        # For "+" direction: start of boundary cell
        assert bslice[0] == slice(29, 30)

    def test_minus_direction_axis0_slice(self, micro_config, jax_key):
        """For - direction, boundary_slice should cover the last cell of boundary."""
        pb = make_periodic(axis=0, direction="-")
        placed = place_periodic(pb, micro_config, jax_key, volume_shape=(30, 20, 20))
        bslice = placed.boundary_slice
        assert bslice[0] == slice(0, 1)

    def test_axis1_boundary_slice(self, micro_config, jax_key):
        pb = make_periodic(axis=1, direction="+")
        placed = place_periodic(pb, micro_config, jax_key, volume_shape=(20, 30, 20))
        bslice = placed.boundary_slice
        assert bslice[1] == slice(29, 30)

    def test_axis2_boundary_slice(self, micro_config, jax_key):
        pb = make_periodic(axis=2, direction="-")
        placed = place_periodic(pb, micro_config, jax_key, volume_shape=(20, 20, 30))
        bslice = placed.boundary_slice
        assert bslice[2] == slice(0, 1)


class TestPeriodicOppositeSlice:
    """Tests for BlochBoundary.opposite_slice property."""

    def test_plus_direction_opposite_is_last_cell(self, micro_config, jax_key):
        """For + direction, opposite_slice should be the last cell of the boundary volume."""
        pb = make_periodic(axis=0, direction="+")
        placed = place_periodic(pb, micro_config, jax_key, volume_shape=(30, 20, 20))
        oslice = placed.opposite_slice
        assert oslice[0] == slice(29, 30)

    def test_minus_direction_opposite_is_first_cell(self, micro_config, jax_key):
        """For - direction, opposite_slice should be the first cell of the boundary volume."""
        pb = make_periodic(axis=0, direction="-")
        placed = place_periodic(pb, micro_config, jax_key, volume_shape=(30, 20, 20))
        oslice = placed.opposite_slice
        assert oslice[0] == slice(0, 1)

    def test_axis1_opposite_slice(self, micro_config, jax_key):
        pb = make_periodic(axis=1, direction="-")
        placed = place_periodic(pb, micro_config, jax_key, volume_shape=(20, 30, 20))
        oslice = placed.opposite_slice
        assert oslice[1] == slice(0, 1)

    def test_axis2_opposite_slice(self, micro_config, jax_key):
        pb = make_periodic(axis=2, direction="+")
        placed = place_periodic(pb, micro_config, jax_key, volume_shape=(20, 20, 30))
        oslice = placed.opposite_slice
        assert oslice[2] == slice(29, 30)

    def test_boundary_and_opposite_same_for_single_cell_boundary(self, micro_config, jax_key):
        """Since the boundary volume is 1-cell wide, both slices refer to the same logical region."""
        pb = make_periodic(axis=0, direction="+")
        placed = place_periodic(pb, micro_config, jax_key, volume_shape=(30, 20, 20))
        bslice = placed.boundary_slice
        oslice = placed.opposite_slice
        # Both cover the single cell at position 29
        assert bslice[0] == oslice[0]


class TestPeriodicApplyFieldReset:
    """Tests for BlochBoundary.apply_field_reset when used as periodic."""

    def test_no_op_for_single_cell_boundary(self, micro_config, jax_key):
        """For a 1-cell thick boundary, boundary_slice == grid_slice, so reset is a no-op."""
        pb = make_periodic(axis=0, direction="+")
        placed = place_periodic(pb, micro_config, jax_key, volume_shape=(10, 10, 10))
        E = jnp.arange(3 * 10 * 10 * 10, dtype=jnp.float32).reshape(3, 10, 10, 10)
        result = placed.apply_field_reset({"E": E})
        assert jnp.allclose(result["E"], E)

    def test_returns_only_provided_field_keys(self, micro_config, jax_key):
        pb = make_periodic(axis=0, direction="-")
        placed = place_periodic(pb, micro_config, jax_key, volume_shape=(10, 10, 10))
        result = placed.apply_field_reset({"H": jnp.ones((3, 10, 10, 10))})
        assert set(result.keys()) == {"H"}

    def test_all_field_keys_passed_through(self, micro_config, jax_key):
        pb = make_periodic(axis=1, direction="+")
        placed = place_periodic(pb, micro_config, jax_key, volume_shape=(10, 10, 10))
        E = jnp.ones((3, 10, 10, 10))
        H = jnp.ones((3, 10, 10, 10))
        result = placed.apply_field_reset({"E": E, "H": H})
        assert set(result.keys()) == {"E", "H"}

    def test_positive_direction_copies_first_cell_across_region(self, micro_config, jax_key):
        """For + direction with a multi-cell grid_slice, boundary_slice is the first
        cell of grid_slice. apply_field_reset should broadcast that cell's values
        across the entire grid_slice region."""
        pb = make_periodic(axis=0, direction="+")
        # Place with a 2-cell grid_slice: x=0:2
        placed = place_periodic(
            pb,
            micro_config,
            jax_key,
            grid_slice_tuple=((0, 2), (0, 10), (0, 10)),
        )
        # Non-uniform E field so we can verify the copy direction
        E = jnp.ones((3, 10, 10, 10))
        E = E.at[:, 0, :, :].set(1.0)
        E = E.at[:, 1, :, :].set(5.0)
        H = jnp.full((3, 10, 10, 10), 2.0)
        result = placed.apply_field_reset({"E": E, "H": H})
        # boundary_slice is x=0 (first cell of grid_slice for + direction).
        # That value (1.0) should be broadcast across x=0:2, overwriting x=1.
        assert jnp.allclose(result["E"][:, 0:2, :, :], 1.0)
        assert jnp.allclose(result["H"][:, 0:2, :, :], 2.0)
        # Region outside grid_slice should be unchanged
        assert jnp.allclose(result["E"][:, 2:, :, :], 1.0)

    def test_negative_direction_copies_last_cell_across_region(self, micro_config, jax_key):
        """For - direction with a multi-cell grid_slice, boundary_slice is the last
        cell of grid_slice. apply_field_reset should broadcast that cell's values
        across the entire grid_slice region."""
        pb = make_periodic(axis=1, direction="-")
        # Place with a 2-cell grid_slice: y=8:10
        placed = place_periodic(
            pb,
            micro_config,
            jax_key,
            grid_slice_tuple=((0, 10), (8, 10), (0, 10)),
        )
        # Non-uniform E field so we can verify the copy direction
        E = jnp.ones((3, 10, 10, 10))
        E = E.at[:, :, 8, :].set(7.0)
        E = E.at[:, :, 9, :].set(3.0)
        H = jnp.full((3, 10, 10, 10), 2.0)
        result = placed.apply_field_reset({"E": E, "H": H})
        # boundary_slice is y=9 (last cell of grid_slice for - direction).
        # That value (3.0) should be broadcast across y=8:10, overwriting y=8.
        assert jnp.allclose(result["E"][:, :, 8:10, :], 3.0)
        assert jnp.allclose(result["H"][:, :, 8:10, :], 2.0)
        # Region outside grid_slice should be unchanged
        assert jnp.allclose(result["E"][:, :, :8, :], 1.0)


class TestNeedsComplexFields:
    """Tests for BlochBoundary.needs_complex_fields property."""

    def test_zero_vector_does_not_need_complex(self):
        pb = make_periodic(axis=0, direction="-")
        assert pb.needs_complex_fields is False

    def test_nonzero_component_on_axis_needs_complex(self):
        shape_list = [1, None, None]
        pb = BlochBoundary(
            axis=0,
            partial_grid_shape=tuple(shape_list),
            direction="-",
            bloch_vector=(1.0, 0.0, 0.0),
        )
        assert pb.needs_complex_fields is True

    def test_nonzero_component_off_axis_does_not_need_complex(self):
        shape_list = [1, None, None]
        pb = BlochBoundary(
            axis=0,
            partial_grid_shape=tuple(shape_list),
            direction="-",
            bloch_vector=(0.0, 1.0, 0.0),
        )
        assert pb.needs_complex_fields is False
