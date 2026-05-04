"""Unit tests for objects/static_material/polygon.py.

Tests ExtrudedPolygon shape generation, material mapping, and axis properties.
"""

import gdstk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import RectilinearGrid
from fdtdx.materials import Material
from fdtdx.objects.static_material.polygon import (
    ExtrudedPolygon,
    extruded_polygon_from_gds,
    extruded_polygon_from_gds_path,
)

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
# centered convention
# ---------------------------------------------------------------------------


class TestCenteredConvention:
    """Vertices are given centered at origin; polygon is placed centered in the grid region."""

    def test_symmetric_polygon_produces_symmetric_mask_axis2(self, config, key, two_materials):
        """A square centered at origin yields an x/y-symmetric mask."""
        verts = _square_vertices(half_side=100e-9)
        poly = _make_polygon(two_materials, axis=2, vertices=verts)
        placed = _place(poly, config, key, ((0, 20), (0, 20), (0, 10)))
        mask = placed.get_voxel_mask_for_shape()
        slice_xy = np.array(mask[:, :, 0])
        assert np.array_equal(slice_xy, slice_xy[::-1, :]), "mask should be symmetric along x"
        assert np.array_equal(slice_xy, slice_xy[:, ::-1]), "mask should be symmetric along y"

    def test_polygon_center_aligns_with_grid_center(self, config, key, two_materials):
        """Bounding box center of the True region should coincide with the grid center."""
        # 300nm = 6 cells; 20-cell grid center is between cells 9 and 10 (0-indexed)
        verts = _square_vertices(half_side=150e-9)
        poly = _make_polygon(two_materials, axis=2, vertices=verts)
        placed = _place(poly, config, key, ((0, 20), (0, 20), (0, 10)))
        mask = placed.get_voxel_mask_for_shape()
        slice_xy = np.array(mask[:, :, 0])
        rows = np.where(slice_xy.any(axis=1))[0]
        cols = np.where(slice_xy.any(axis=0))[0]
        center_row = (rows[0] + rows[-1] + 1) / 2.0
        center_col = (cols[0] + cols[-1] + 1) / 2.0
        assert abs(center_row - 10.0) <= 1.0, f"row center {center_row} not near grid center 10"
        assert abs(center_col - 10.0) <= 1.0, f"col center {center_col} not near grid center 10"

    def test_symmetric_polygon_produces_symmetric_mask_axis0(self, config, key, two_materials):
        """Symmetry check for extrusion along x-axis."""
        verts = _square_vertices(half_side=100e-9)
        poly = _make_polygon(two_materials, axis=0, vertices=verts)
        placed = _place(poly, config, key, ((0, 10), (0, 20), (0, 20)))
        mask = placed.get_voxel_mask_for_shape()
        slice_yz = np.array(mask[0, :, :])
        assert np.array_equal(slice_yz, slice_yz[::-1, :]), "mask should be symmetric along y"
        assert np.array_equal(slice_yz, slice_yz[:, ::-1]), "mask should be symmetric along z"

    def test_no_auto_centering_of_offset_polygon(self, config, key, two_materials):
        """Vertices NOT centered at origin should NOT be auto-centered — caller is responsible."""
        # Shift a square 200nm off-center in x
        offset = 200e-9
        verts = _square_vertices(half_side=100e-9) + np.array([offset, 0.0])
        poly = _make_polygon(two_materials, axis=2, vertices=verts)
        placed = _place(poly, config, key, ((0, 20), (0, 20), (0, 10)))
        mask = placed.get_voxel_mask_for_shape()
        slice_xy = np.array(mask[:, :, 0])
        # The mask should NOT be symmetric in x
        assert not np.array_equal(slice_xy, slice_xy[::-1, :])


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

    def test_nonuniform_grid_uses_explicit_cell_centers(self, key, two_materials):
        """Polygon masks evaluate the shape at rectilinear cell centers."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0]),
            y_edges=jnp.asarray([0.0, 1.0, 3.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )
        config = SimulationConfig(time=1e-8, resolution=1.0, grid=grid, backend="cpu")
        poly = _make_polygon(two_materials, axis=2, vertices=_square_vertices(half_side=0.8))
        placed = _place(poly, config, key, ((0, 2), (0, 2), (0, 1)))

        mask = placed.get_voxel_mask_for_shape()

        assert mask.shape == (2, 2, 1)
        assert bool(jnp.any(mask))


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


# ---------------------------------------------------------------------------
# extruded_polygon_from_gds / extruded_polygon_from_gds_path
# ---------------------------------------------------------------------------


@pytest.fixture
def square_lib():
    """In-memory gdstk Library with a 200nm square on layer 1."""
    half = 0.1  # 0.1 µm = 100 nm
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("TOP")
    cell.add(
        gdstk.Polygon(
            [(-half, -half), (half, -half), (half, half), (-half, half)],
            layer=1,
            datatype=0,
        )
    )
    return lib


@pytest.fixture
def square_gds(square_lib, tmp_path):
    """Write the square library to a .gds file and return its path."""
    path = tmp_path / "test.gds"
    square_lib.write_gds(str(path))
    return path


@pytest.mark.unit
class TestExtrudedPolygonFromGds:
    """Tests for extruded_polygon_from_gds (accepts a gdstk.Library)."""

    def test_returns_extruded_polygon(self, square_lib, two_materials):
        result = extruded_polygon_from_gds(
            square_lib, "TOP", layer=1, axis=2, material_name="si", materials=two_materials
        )
        assert isinstance(result, ExtrudedPolygon)

    def test_vertices_shape(self, square_lib, two_materials):
        result = extruded_polygon_from_gds(
            square_lib, "TOP", layer=1, axis=2, material_name="si", materials=two_materials
        )
        assert result.vertices.shape == (4, 2)

    def test_vertices_centred_around_origin(self, square_lib, two_materials):
        """Vertices should be symmetric: max ≈ +100 nm, min ≈ -100 nm."""
        result = extruded_polygon_from_gds(
            square_lib, "TOP", layer=1, axis=2, material_name="si", materials=two_materials
        )
        expected_half = 100e-9
        assert np.allclose(result.vertices.max(axis=0), [expected_half, expected_half])
        assert np.allclose(result.vertices.min(axis=0), [-expected_half, -expected_half])

    def test_bad_cell_name_raises(self, square_lib, two_materials):
        with pytest.raises(ValueError, match="Cell 'MISSING'"):
            extruded_polygon_from_gds(
                square_lib, "MISSING", layer=1, axis=2, material_name="si", materials=two_materials
            )

    def test_bad_layer_raises(self, square_lib, two_materials):
        with pytest.raises(ValueError, match="layer=99"):
            extruded_polygon_from_gds(square_lib, "TOP", layer=99, axis=2, material_name="si", materials=two_materials)

    def test_polygon_index_out_of_range_raises(self, square_lib, two_materials):
        with pytest.raises(IndexError, match="polygon_index=5"):
            extruded_polygon_from_gds(
                square_lib, "TOP", layer=1, polygon_index=5, axis=2, material_name="si", materials=two_materials
            )


@pytest.mark.unit
class TestExtrudedPolygonFromGdsPath:
    """Tests for extruded_polygon_from_gds_path (accepts a file path)."""

    def test_returns_extruded_polygon(self, square_gds, two_materials):
        result = extruded_polygon_from_gds_path(
            square_gds, "TOP", layer=1, axis=2, material_name="si", materials=two_materials
        )
        assert isinstance(result, ExtrudedPolygon)

    def test_vertices_match_library_function(self, square_lib, square_gds, two_materials):
        """Path variant must produce the same vertices as the library variant."""
        kwargs = {"layer": 1, "axis": 2, "material_name": "si", "materials": two_materials}
        from_lib = extruded_polygon_from_gds(square_lib, "TOP", **kwargs)
        from_path = extruded_polygon_from_gds_path(square_gds, "TOP", **kwargs)
        assert np.allclose(from_lib.vertices, from_path.vertices)

    def test_bad_cell_name_raises(self, square_gds, two_materials):
        with pytest.raises(ValueError, match="Cell 'MISSING'"):
            extruded_polygon_from_gds_path(
                square_gds, "MISSING", layer=1, axis=2, material_name="si", materials=two_materials
            )
