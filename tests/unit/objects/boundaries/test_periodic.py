"""Unit tests for objects/boundaries/periodic.py.

Tests PeriodicBoundary initialization, properties, and slice computation.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.objects.boundaries.periodic import PeriodicBoundary


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
    """Create a PeriodicBoundary with given axis/direction."""
    shape_list = [None, None, None]
    shape_list[axis] = 1
    return PeriodicBoundary(
        axis=axis,
        partial_grid_shape=tuple(shape_list),
        direction=direction,
    )


def place_periodic(pb, micro_config, jax_key, volume_shape=(30, 20, 20)):
    """Place a PeriodicBoundary on a grid matching the given volume shape."""
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
    """Tests for PeriodicBoundary.thickness."""

    def test_thickness_always_one(self):
        pb = make_periodic(axis=0, direction="-")
        assert pb.thickness == 1

    def test_thickness_one_for_all_axes(self):
        for axis in range(3):
            for direction in ("-", "+"):
                pb = make_periodic(axis=axis, direction=direction)
                assert pb.thickness == 1


class TestPeriodicDescriptiveName:
    """Tests for PeriodicBoundary.descriptive_name."""

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
    """Tests for PeriodicBoundary.boundary_slice property."""

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
    """Tests for PeriodicBoundary.opposite_slice property."""

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
