"""Unit tests for objects/boundaries/pmc.py.

Tests PerfectMagneticConductor initialization, properties, and tangential components.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.objects.boundaries.pmc import PerfectMagneticConductor


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


def make_pmc(axis=0, direction="-"):
    """Create a PerfectMagneticConductor with given axis/direction."""
    shape_list = [None, None, None]
    shape_list[axis] = 1
    return PerfectMagneticConductor(
        axis=axis,
        partial_grid_shape=tuple(shape_list),
        direction=direction,
    )


def place_pmc(pmc, micro_config, jax_key, volume_shape=(30, 20, 20)):
    """Place a PerfectMagneticConductor on a grid matching the given volume shape."""
    axis = pmc.axis
    direction = pmc.direction

    slices = [[0, volume_shape[i]] for i in range(3)]
    if direction == "-":
        slices[axis] = [0, 1]
    else:
        slices[axis] = [volume_shape[axis] - 1, volume_shape[axis]]

    grid_slice_tuple = tuple(tuple(s) for s in slices)
    return pmc.place_on_grid(grid_slice_tuple=grid_slice_tuple, config=micro_config, key=jax_key)


class TestPmcThickness:
    """Tests for PerfectMagneticConductor.thickness."""

    def test_thickness_always_one(self):
        pmc = make_pmc(axis=0, direction="-")
        assert pmc.thickness == 1

    def test_thickness_one_for_all_axes(self):
        for axis in range(3):
            for direction in ("-", "+"):
                pmc = make_pmc(axis=axis, direction=direction)
                assert pmc.thickness == 1


class TestPmcDescriptiveName:
    """Tests for PerfectMagneticConductor.descriptive_name."""

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
        pmc = make_pmc(axis=axis, direction=direction)
        placed = place_pmc(pmc, micro_config, jax_key, volume_shape=(30, 30, 30))
        assert placed.descriptive_name == expected


class TestPmcTangentialComponents:
    """Tests for PerfectMagneticConductor.tangential_components."""

    def test_x_axis_tangential_is_hy_hz(self):
        pmc = make_pmc(axis=0, direction="-")
        assert pmc.tangential_components == (1, 2)

    def test_y_axis_tangential_is_hx_hz(self):
        pmc = make_pmc(axis=1, direction="-")
        assert pmc.tangential_components == (0, 2)

    def test_z_axis_tangential_is_hx_hy(self):
        pmc = make_pmc(axis=2, direction="-")
        assert pmc.tangential_components == (0, 1)

    def test_direction_does_not_affect_tangential(self):
        for axis in range(3):
            pmc_min = make_pmc(axis=axis, direction="-")
            pmc_max = make_pmc(axis=axis, direction="+")
            assert pmc_min.tangential_components == pmc_max.tangential_components
