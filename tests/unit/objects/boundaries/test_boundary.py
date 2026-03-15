"""Unit tests for objects/boundaries/boundary.py.

Tests BaseBoundary abstract class interface via PerfectlyMatchedLayer
(used as a concrete implementation of BaseBoundary).
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


def make_pml(axis, direction, thickness=10):
    shape_list = [None, None, None]
    shape_list[axis] = thickness
    return PerfectlyMatchedLayer(
        axis=axis,
        partial_grid_shape=tuple(shape_list),
        direction=direction,
        sigma_end=1.0,  # provide explicit value to avoid needing resolution
    )


def place_pml(pml, config, key, volume_shape=(30, 20, 25)):
    axis = pml.axis
    direction = pml.direction
    thickness = pml.partial_grid_shape[axis]
    slices = [[0, volume_shape[i]] for i in range(3)]
    if direction == "-":
        slices[axis] = [0, thickness]
    else:
        slices[axis] = [volume_shape[axis] - thickness, volume_shape[axis]]
    grid_slice_tuple = tuple(tuple(s) for s in slices)
    return pml.place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)


class TestInterfaceGridShape:
    """Tests for BaseBoundary.interface_grid_shape()."""

    def test_axis0_interface_shape(self, micro_config, jax_key):
        """Axis 0: interface is 1 × Ny × Nz."""
        pml = make_pml(axis=0, direction="-", thickness=10)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 25))
        shape = placed.interface_grid_shape()
        assert shape == (1, 20, 25)

    def test_axis1_interface_shape(self, micro_config, jax_key):
        """Axis 1: interface is Nx × 1 × Nz."""
        pml = make_pml(axis=1, direction="-", thickness=8)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 25))
        shape = placed.interface_grid_shape()
        assert shape == (30, 1, 25)

    def test_axis2_interface_shape(self, micro_config, jax_key):
        """Axis 2: interface is Nx × Ny × 1."""
        pml = make_pml(axis=2, direction="-", thickness=6)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 25))
        shape = placed.interface_grid_shape()
        assert shape == (30, 20, 1)

    def test_plus_direction_same_shape_as_minus(self, micro_config, jax_key):
        pml_minus = make_pml(axis=0, direction="-", thickness=10)
        pml_plus = make_pml(axis=0, direction="+", thickness=10)
        placed_minus = place_pml(pml_minus, micro_config, jax_key, volume_shape=(30, 20, 25))
        placed_plus = place_pml(pml_plus, micro_config, jax_key, volume_shape=(30, 20, 25))
        assert placed_minus.interface_grid_shape() == placed_plus.interface_grid_shape()


class TestInterfaceSliceTuple:
    """Tests for BaseBoundary.interface_slice_tuple()."""

    def test_axis0_minus_direction(self, micro_config, jax_key):
        """For - direction, interface is at the inner edge (last cell of PML)."""
        pml = make_pml(axis=0, direction="-", thickness=10)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 20))
        st = placed.interface_slice_tuple()
        # direction "-": last cell of the PML region along axis 0 = index 9..10
        assert st[0] == (9, 10)

    def test_axis0_plus_direction(self, micro_config, jax_key):
        """For + direction, interface is at the inner edge (first cell of PML)."""
        pml = make_pml(axis=0, direction="+", thickness=10)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 20))
        st = placed.interface_slice_tuple()
        # direction "+": first cell of the PML region along axis 0 = index 20..21
        assert st[0] == (20, 21)

    def test_axis1_minus_direction(self, micro_config, jax_key):
        pml = make_pml(axis=1, direction="-", thickness=8)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(20, 30, 20))
        st = placed.interface_slice_tuple()
        assert st[1] == (7, 8)

    def test_axis2_plus_direction(self, micro_config, jax_key):
        pml = make_pml(axis=2, direction="+", thickness=6)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(20, 20, 30))
        st = placed.interface_slice_tuple()
        assert st[2] == (24, 25)

    def test_returns_tuple_of_length_3(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=10)
        placed = place_pml(pml, micro_config, jax_key)
        st = placed.interface_slice_tuple()
        assert len(st) == 3


class TestInterfaceSlice:
    """Tests for BaseBoundary.interface_slice()."""

    def test_axis0_minus_returns_slice(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=10)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 20))
        s = placed.interface_slice()
        assert isinstance(s[0], slice)
        assert s[0] == slice(9, 10)

    def test_axis0_plus_returns_slice(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="+", thickness=10)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 20))
        s = placed.interface_slice()
        assert s[0] == slice(20, 21)

    def test_interface_slice_length_3(self, micro_config, jax_key):
        pml = make_pml(axis=0, direction="-", thickness=10)
        placed = place_pml(pml, micro_config, jax_key)
        s = placed.interface_slice()
        assert len(s) == 3

    def test_non_axis_slices_unchanged(self, micro_config, jax_key):
        """Non-boundary axes should keep the original full grid slice."""
        pml = make_pml(axis=0, direction="-", thickness=10)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(30, 20, 25))
        s = placed.interface_slice()
        assert s[1] == slice(0, 20)
        assert s[2] == slice(0, 25)

    def test_axis1_minus_slice(self, micro_config, jax_key):
        pml = make_pml(axis=1, direction="-", thickness=8)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(20, 30, 15))
        s = placed.interface_slice()
        assert s[1] == slice(7, 8)

    def test_axis2_plus_slice(self, micro_config, jax_key):
        pml = make_pml(axis=2, direction="+", thickness=5)
        placed = place_pml(pml, micro_config, jax_key, volume_shape=(10, 10, 25))
        s = placed.interface_slice()
        assert s[2] == slice(20, 21)
