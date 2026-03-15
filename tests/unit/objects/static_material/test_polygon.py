"""Unit tests for objects/static_material/polygon.py.

Tests ExtrudedPolygon shape generation, material mapping, and axis properties.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.materials import Material
from fdtdx.objects.static_material.polygon import ExtrudedPolygon

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


def _square_vertices(half_side=100e-9):
    """Simple square polygon centered at origin."""
    h = half_side
    return np.array(
        [
            [-h, -h],
            [h, -h],
            [h, h],
            [-h, h],
        ]
    )


def _make_polygon(materials, axis=2, material_name="si", vertices=None):
    if vertices is None:
        vertices = _square_vertices()
    return ExtrudedPolygon(
        materials=materials,
        axis=axis,
        material_name=material_name,
        vertices=vertices,
    )


def _place(polygon, config, key, slices=((0, 20), (0, 20), (0, 10))):
    return polygon.place_on_grid(grid_slice_tuple=slices, config=config, key=key)


# ---------------------------------------------------------------------------
# Axis properties
# ---------------------------------------------------------------------------


class TestAxisProperties:
    def test_horizontal_axis_for_axis0(self, two_materials):
        poly = _make_polygon(two_materials, axis=0)
        assert poly.horizontal_axis == 1

    def test_horizontal_axis_for_axis1(self, two_materials):
        poly = _make_polygon(two_materials, axis=1)
        assert poly.horizontal_axis == 0

    def test_horizontal_axis_for_axis2(self, two_materials):
        poly = _make_polygon(two_materials, axis=2)
        assert poly.horizontal_axis == 0

    def test_vertical_axis_for_axis0(self, two_materials):
        poly = _make_polygon(two_materials, axis=0)
        assert poly.vertical_axis == 2

    def test_vertical_axis_for_axis1(self, two_materials):
        poly = _make_polygon(two_materials, axis=1)
        assert poly.vertical_axis == 2

    def test_vertical_axis_for_axis2(self, two_materials):
        poly = _make_polygon(two_materials, axis=2)
        assert poly.vertical_axis == 1


# ---------------------------------------------------------------------------
# centered_vertices
# ---------------------------------------------------------------------------


class TestCenteredVertices:
    def test_shape_preserved(self, config, key, two_materials):
        verts = _square_vertices()
        poly = _make_polygon(two_materials, axis=2, vertices=verts)
        placed = _place(poly, config, key)
        cv = placed.centered_vertices
        assert cv.shape == verts.shape

    def test_vertices_shifted_right(self, config, key, two_materials):
        """Centered vertices should all have non-negative x (after shifting by half real_shape)."""
        verts = _square_vertices(half_side=100e-9)  # vertices in [-100nm, 100nm]
        poly = _make_polygon(two_materials, axis=2, vertices=verts)
        # grid: 20x20x10, real: 20*50nm=1000nm along horizontal axis
        placed = _place(poly, config, key, ((0, 20), (0, 20), (0, 10)))
        cv = placed.centered_vertices
        # half real_shape for horizontal axis (axis=2 → horizontal=0 → x-dim=20 → 1000nm)
        # shift = 500nm; verts range from -100 to 100nm → shifted to 400..600nm
        assert bool(np.all(cv[:, 0] > 0))


# ---------------------------------------------------------------------------
# get_voxel_mask_for_shape
# ---------------------------------------------------------------------------


class TestGetVoxelMaskForShape:
    def test_mask_shape_matches_grid(self, config, key, two_materials):
        poly = _make_polygon(two_materials, axis=2)
        placed = _place(poly, config, key, ((0, 20), (0, 16), (0, 8)))
        mask = placed.get_voxel_mask_for_shape()
        assert mask.shape == (20, 16, 8)

    def test_mask_is_bool(self, config, key, two_materials):
        poly = _make_polygon(two_materials, axis=2)
        placed = _place(poly, config, key)
        mask = placed.get_voxel_mask_for_shape()
        assert mask.dtype == jnp.bool_

    def test_square_polygon_has_interior_voxels(self, config, key, two_materials):
        """A square polygon centered in the grid should have True voxels."""
        verts = _square_vertices(half_side=100e-9)  # 2 grid units each side
        poly = _make_polygon(two_materials, axis=2, vertices=verts)
        placed = _place(poly, config, key, ((0, 20), (0, 20), (0, 10)))
        mask = placed.get_voxel_mask_for_shape()
        assert bool(jnp.any(mask))

    def test_mask_uniform_along_extrusion_axis(self, config, key, two_materials):
        """All slices along the extrusion axis should be identical."""
        poly = _make_polygon(two_materials, axis=2)
        placed = _place(poly, config, key, ((0, 20), (0, 20), (0, 8)))
        mask = placed.get_voxel_mask_for_shape()
        for z in range(mask.shape[2]):
            assert jnp.array_equal(mask[:, :, z], mask[:, :, 0])

    def test_axis0_extrusion(self, config, key, two_materials):
        """Polygon extruded along x-axis: result shape should match grid."""
        poly = _make_polygon(two_materials, axis=0)
        placed = _place(poly, config, key, ((0, 10), (0, 20), (0, 20)))
        mask = placed.get_voxel_mask_for_shape()
        assert mask.shape == (10, 20, 20)

    def test_mask_uniform_along_axis0_extrusion(self, config, key, two_materials):
        """Slices along x-axis (extrusion axis=0) should be identical."""
        poly = _make_polygon(two_materials, axis=0)
        placed = _place(poly, config, key, ((0, 8), (0, 20), (0, 20)))
        mask = placed.get_voxel_mask_for_shape()
        for x in range(mask.shape[0]):
            assert jnp.array_equal(mask[x, :, :], mask[0, :, :])


# ---------------------------------------------------------------------------
# get_material_mapping
# ---------------------------------------------------------------------------


class TestGetMaterialMapping:
    def test_shape_matches_grid(self, config, key, two_materials):
        poly = _make_polygon(two_materials, axis=2)
        placed = _place(poly, config, key, ((0, 10), (0, 12), (0, 6)))
        mapping = placed.get_material_mapping()
        assert mapping.shape == (10, 12, 6)

    def test_dtype_is_int(self, config, key, two_materials):
        poly = _make_polygon(two_materials, axis=2)
        placed = _place(poly, config, key)
        mapping = placed.get_material_mapping()
        assert jnp.issubdtype(mapping.dtype, jnp.integer)

    def test_correct_index_for_si(self, config, key, two_materials):
        """si (permittivity=12.25) > air → index 1."""
        poly = _make_polygon(two_materials, axis=2, material_name="si")
        placed = _place(poly, config, key, ((0, 10), (0, 10), (0, 5)))
        mapping = placed.get_material_mapping()
        assert bool(jnp.all(mapping == 1))

    def test_correct_index_for_air(self, config, key, two_materials):
        """air (permittivity=1.0) → index 0."""
        poly = ExtrudedPolygon(
            materials=two_materials,
            axis=2,
            material_name="air",
            vertices=_square_vertices(),
        )
        placed = _place(poly, config, key, ((0, 10), (0, 10), (0, 5)))
        mapping = placed.get_material_mapping()
        assert bool(jnp.all(mapping == 0))

    def test_mapping_is_uniform(self, config, key, two_materials):
        """Every voxel gets the same material index."""
        poly = _make_polygon(two_materials, axis=2)
        placed = _place(poly, config, key)
        mapping = placed.get_material_mapping()
        assert bool(jnp.all(mapping == mapping[0, 0, 0]))
