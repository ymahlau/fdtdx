"""Unit tests for objects/static_material/cylinder.py.

Tests Cylinder shape generation and material mapping.

Note: get_voxel_mask_for_shape returns a mask with size 1 along the fiber
(extrusion) axis. The cross-section dimensions match the grid, but the
fiber-axis dimension is 1 (relying on broadcasting for application).
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.materials import Material
from fdtdx.objects.static_material.cylinder import Cylinder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return SimulationConfig(
        time=100e-15,
        resolution=50e-9,
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=None,
    )


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def two_materials():
    return {
        "air": Material(permittivity=1.0),
        "si": Material(permittivity=12.25),
    }


def _make_cylinder(materials, radius=150e-9, axis=2, material_name="si"):
    return Cylinder(
        materials=materials,
        radius=radius,
        axis=axis,
        material_name=material_name,
    )


def _place(cylinder, config, key, slices=((0, 20), (0, 20), (0, 10))):
    return cylinder.place_on_grid(grid_slice_tuple=slices, config=config, key=key)


# ---------------------------------------------------------------------------
# Axis properties
# ---------------------------------------------------------------------------


class TestAxisProperties:
    """Tests for horizontal_axis and vertical_axis properties."""

    def test_horizontal_axis_for_axis0(self, two_materials):
        cyl = _make_cylinder(two_materials, axis=0)
        assert cyl.horizontal_axis == 1

    def test_horizontal_axis_for_axis1(self, two_materials):
        cyl = _make_cylinder(two_materials, axis=1)
        assert cyl.horizontal_axis == 0

    def test_horizontal_axis_for_axis2(self, two_materials):
        cyl = _make_cylinder(two_materials, axis=2)
        assert cyl.horizontal_axis == 0

    def test_vertical_axis_for_axis0(self, two_materials):
        cyl = _make_cylinder(two_materials, axis=0)
        assert cyl.vertical_axis == 2

    def test_vertical_axis_for_axis1(self, two_materials):
        cyl = _make_cylinder(two_materials, axis=1)
        assert cyl.vertical_axis == 2

    def test_vertical_axis_for_axis2(self, two_materials):
        cyl = _make_cylinder(two_materials, axis=2)
        assert cyl.vertical_axis == 1


# ---------------------------------------------------------------------------
# get_voxel_mask_for_shape
# ---------------------------------------------------------------------------


class TestGetVoxelMaskForShape:
    """The mask has size 1 along the fiber axis (extrusion is implicit/broadcast).

    Expected mask shapes:
      axis=0 → (1, grid_shape[1], grid_shape[2])
      axis=1 → (grid_shape[0], 1, grid_shape[2])
      axis=2 → (grid_shape[0], grid_shape[1], 1)
    """

    def test_mask_shape_axis2(self, config, key, two_materials):
        cyl = _make_cylinder(two_materials, axis=2, radius=150e-9)
        placed = _place(cyl, config, key, ((0, 20), (0, 16), (0, 10)))
        mask = placed.get_voxel_mask_for_shape()
        # fiber along z: cross-section in x-y, size 1 in z
        assert mask.shape == (20, 16, 1)

    def test_mask_shape_axis0(self, config, key, two_materials):
        cyl = _make_cylinder(two_materials, axis=0, radius=150e-9)
        placed = _place(cyl, config, key, ((0, 10), (0, 20), (0, 20)))
        mask = placed.get_voxel_mask_for_shape()
        # fiber along x: cross-section in y-z, size 1 in x
        assert mask.shape == (1, 20, 20)

    def test_mask_shape_axis1(self, config, key, two_materials):
        cyl = _make_cylinder(two_materials, axis=1, radius=150e-9)
        placed = _place(cyl, config, key, ((0, 20), (0, 10), (0, 20)))
        mask = placed.get_voxel_mask_for_shape()
        # fiber along y: cross-section in x-z, size 1 in y
        assert mask.shape == (20, 1, 20)

    def test_mask_is_bool(self, config, key, two_materials):
        cyl = _make_cylinder(two_materials, axis=2)
        placed = _place(cyl, config, key)
        mask = placed.get_voxel_mask_for_shape()
        assert mask.dtype == jnp.bool_

    def test_mask_has_true_values_for_large_radius(self, config, key, two_materials):
        """Radius larger than grid resolution produces interior True voxels."""
        # radius = 200nm, resolution = 50nm → grid_radius = 4
        cyl = _make_cylinder(two_materials, axis=2, radius=200e-9)
        placed = _place(cyl, config, key, ((0, 20), (0, 20), (0, 5)))
        mask = placed.get_voxel_mask_for_shape()
        assert bool(jnp.any(mask))

    def test_mask_has_false_values_for_small_radius(self, config, key, two_materials):
        """Small radius in large grid yields exterior False voxels."""
        cyl = _make_cylinder(two_materials, axis=2, radius=50e-9)
        placed = _place(cyl, config, key, ((0, 20), (0, 20), (0, 5)))
        mask = placed.get_voxel_mask_for_shape()
        assert bool(jnp.any(~mask))

    def test_fiber_axis_has_size_one(self, config, key, two_materials):
        """The fiber (extrusion) axis always has size 1 in the mask."""
        for axis, slices in [
            (0, ((0, 8), (0, 16), (0, 16))),
            (1, ((0, 16), (0, 8), (0, 16))),
            (2, ((0, 16), (0, 16), (0, 8))),
        ]:
            cyl = _make_cylinder(two_materials, axis=axis)
            placed = cyl.place_on_grid(slices, config, key)
            mask = placed.get_voxel_mask_for_shape()
            assert mask.shape[axis] == 1, f"Expected size 1 along fiber axis={axis}"


# ---------------------------------------------------------------------------
# get_material_mapping
# ---------------------------------------------------------------------------


class TestGetMaterialMapping:
    def test_shape_matches_grid(self, config, key, two_materials):
        cyl = _make_cylinder(two_materials, axis=2)
        placed = _place(cyl, config, key, ((0, 10), (0, 10), (0, 5)))
        mapping = placed.get_material_mapping()
        assert mapping.shape == placed.grid_shape

    def test_dtype_is_int(self, config, key, two_materials):
        cyl = _make_cylinder(two_materials, axis=2)
        placed = _place(cyl, config, key)
        mapping = placed.get_material_mapping()
        assert jnp.issubdtype(mapping.dtype, jnp.integer)

    def test_returns_correct_index_for_si(self, config, key, two_materials):
        """si (permittivity=12.25) > air (1.0) → sorted index 1."""
        cyl = _make_cylinder(two_materials, axis=2, material_name="si")
        placed = _place(cyl, config, key, ((0, 10), (0, 10), (0, 5)))
        mapping = placed.get_material_mapping()
        assert bool(jnp.all(mapping == 1))

    def test_returns_correct_index_for_air(self, config, key, two_materials):
        """air (permittivity=1.0) → sorted index 0."""
        cyl = Cylinder(
            materials=two_materials,
            radius=100e-9,
            axis=2,
            material_name="air",
        )
        placed = _place(cyl, config, key, ((0, 10), (0, 10), (0, 5)))
        mapping = placed.get_material_mapping()
        assert bool(jnp.all(mapping == 0))
