"""Unit tests for objects/boundaries/pec.py.

Tests PerfectElectricConductor initialization and properties.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.objects.boundaries.pec import PerfectElectricConductor


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


def make_pec(axis=0, direction="-"):
    """Create a PerfectElectricConductor with given axis/direction."""
    shape_list = [None, None, None]
    shape_list[axis] = 1
    return PerfectElectricConductor(
        axis=axis,
        partial_grid_shape=tuple(shape_list),
        direction=direction,
    )


def place_pec(pec, micro_config, jax_key, volume_shape=(30, 20, 20)):
    """Place a PerfectElectricConductor on a grid matching the given volume shape."""
    axis = pec.axis
    direction = pec.direction

    slices = [[0, volume_shape[i]] for i in range(3)]
    if direction == "-":
        slices[axis] = [0, 1]
    else:
        slices[axis] = [volume_shape[axis] - 1, volume_shape[axis]]

    grid_slice_tuple = tuple(tuple(s) for s in slices)
    return pec.place_on_grid(grid_slice_tuple=grid_slice_tuple, config=micro_config, key=jax_key)


class TestPecThickness:
    """Tests for PerfectElectricConductor.thickness."""

    def test_thickness_always_one(self):
        pec = make_pec(axis=0, direction="-")
        assert pec.thickness == 1

    def test_thickness_one_for_all_axes(self):
        for axis in range(3):
            for direction in ("-", "+"):
                pec = make_pec(axis=axis, direction=direction)
                assert pec.thickness == 1


class TestPecDescriptiveName:
    """Tests for PerfectElectricConductor.descriptive_name."""

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
        pec = make_pec(axis=axis, direction=direction)
        placed = place_pec(pec, micro_config, jax_key, volume_shape=(30, 30, 30))
        assert placed.descriptive_name == expected
