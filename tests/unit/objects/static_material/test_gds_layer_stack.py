"""Unit tests for objects/static_material/gds_layer_stack.py."""

import gdstk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.materials import Material
from fdtdx.objects.static_material.gds_layer_stack import GDSLayerObject, GDSLayerSpec, gds_layer_stack
from fdtdx.objects.static_material.static import SimulationVolume

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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _square_polygon(half_side=100e-9):
    """Square polygon centered at origin in GDS coords."""
    h = half_side
    return np.array([[-h, -h], [h, -h], [h, h], [-h, h]])


def _make_layer_obj(materials, polygons=None, gds_center=(0.0, 0.0), axis=2, thickness=200e-9, material_name="si"):
    if polygons is None:
        polygons = [_square_polygon()]
    return GDSLayerObject(
        materials=materials,
        polygons=polygons,
        gds_center=gds_center,
        material_name=material_name,
        axis=axis,
        thickness=thickness,
    )


def _place(obj, config, key, slices=((0, 20), (0, 20), (0, 4))):
    return obj.place_on_grid(grid_slice_tuple=slices, config=config, key=key)


# ---------------------------------------------------------------------------
# Axis properties
# ---------------------------------------------------------------------------


class TestAxisProperties:
    def test_horizontal_axis_axis0(self, two_materials):
        obj = _make_layer_obj(two_materials, axis=0)
        assert obj.horizontal_axis == 1

    def test_horizontal_axis_axis1(self, two_materials):
        obj = _make_layer_obj(two_materials, axis=1)
        assert obj.horizontal_axis == 0

    def test_horizontal_axis_axis2(self, two_materials):
        obj = _make_layer_obj(two_materials, axis=2)
        assert obj.horizontal_axis == 0

    def test_vertical_axis_axis0(self, two_materials):
        obj = _make_layer_obj(two_materials, axis=0)
        assert obj.vertical_axis == 2

    def test_vertical_axis_axis1(self, two_materials):
        obj = _make_layer_obj(two_materials, axis=1)
        assert obj.vertical_axis == 2

    def test_vertical_axis_axis2(self, two_materials):
        obj = _make_layer_obj(two_materials, axis=2)
        assert obj.vertical_axis == 1


# ---------------------------------------------------------------------------
# Geometry size hint
# ---------------------------------------------------------------------------


class TestGeometrySizeHint:
    def test_extrusion_axis_returns_thickness_axis0(self, two_materials):
        thickness = 150e-9
        obj = _make_layer_obj(two_materials, axis=0, thickness=thickness)
        hint = obj.get_geometry_size_hint()
        assert hint[0] == pytest.approx(thickness)

    def test_extrusion_axis_returns_thickness_axis1(self, two_materials):
        thickness = 200e-9
        obj = _make_layer_obj(two_materials, axis=1, thickness=thickness)
        hint = obj.get_geometry_size_hint()
        assert hint[1] == pytest.approx(thickness)

    def test_extrusion_axis_returns_thickness_axis2(self, two_materials):
        thickness = 300e-9
        obj = _make_layer_obj(two_materials, axis=2, thickness=thickness)
        hint = obj.get_geometry_size_hint()
        assert hint[2] == pytest.approx(thickness)

    def test_cross_section_axes_are_none_axis2(self, two_materials):
        obj = _make_layer_obj(two_materials, axis=2)
        hint = obj.get_geometry_size_hint()
        assert hint[0] is None
        assert hint[1] is None

    def test_cross_section_axes_are_none_axis0(self, two_materials):
        obj = _make_layer_obj(two_materials, axis=0)
        hint = obj.get_geometry_size_hint()
        assert hint[1] is None
        assert hint[2] is None

    def test_cross_section_axes_are_none_axis1(self, two_materials):
        obj = _make_layer_obj(two_materials, axis=1)
        hint = obj.get_geometry_size_hint()
        assert hint[0] is None
        assert hint[2] is None


# ---------------------------------------------------------------------------
# get_voxel_mask_for_shape
# ---------------------------------------------------------------------------


class TestGetVoxelMaskForShape:
    def test_mask_shape_matches_grid(self, config, key, two_materials):
        obj = _make_layer_obj(two_materials, axis=2)
        placed = _place(obj, config, key, slices=((0, 20), (0, 20), (0, 4)))
        mask = placed.get_voxel_mask_for_shape()
        assert mask.shape == (20, 20, 4)

    def test_mask_is_bool(self, config, key, two_materials):
        obj = _make_layer_obj(two_materials, axis=2)
        placed = _place(obj, config, key)
        mask = placed.get_voxel_mask_for_shape()
        assert mask.dtype == jnp.bool_

    def test_single_polygon_has_interior_voxels(self, config, key, two_materials):
        """A 200nm square polygon with gds_center=(0,0) near the simulation center should have True voxels.

        Grid: 20x20x4 cells at 50nm resolution = 1µm×1µm cross-section.
        Polygon: ±100nm square centered at GDS origin, gds_center=(0,0) maps GDS origin to grid center.
        """
        polygons = [_square_polygon(half_side=100e-9)]
        obj = GDSLayerObject(
            materials=two_materials,
            polygons=polygons,
            gds_center=(0.0, 0.0),
            material_name="si",
            axis=2,
            thickness=200e-9,
        )
        placed = _place(obj, config, key, slices=((0, 20), (0, 20), (0, 4)))
        mask = placed.get_voxel_mask_for_shape()
        assert bool(jnp.any(mask)), "Expected some True voxels inside the 200nm square polygon"

    def test_mask_uniform_along_extrusion_axis(self, config, key, two_materials):
        """All z-slices should be identical for extrusion along axis=2."""
        obj = _make_layer_obj(two_materials, axis=2)
        placed = _place(obj, config, key, slices=((0, 20), (0, 20), (0, 4)))
        mask = placed.get_voxel_mask_for_shape()
        for z in range(mask.shape[2]):
            assert jnp.array_equal(mask[:, :, z], mask[:, :, 0]), f"z-slice {z} differs from slice 0"

    def test_two_polygons_unioned(self, config, key, two_materials):
        """Two non-overlapping squares should both appear as True in the mask.

        Place two 100nm squares offset ±300nm along horizontal axis (x).
        In a 20x20 grid at 50nm resolution = 1µm wide, ±300nm puts squares
        at indices ~4 and ~16.
        """
        # gds_center=(0,0) means GDS origin maps to grid center (cell 10 of 20)
        # Square 1: centered at x=-300nm in GDS → cell ~4 in grid
        # Square 2: centered at x=+300nm in GDS → cell ~16 in grid
        offset = 300e-9
        sq1 = _square_polygon(half_side=100e-9) + np.array([-offset, 0.0])
        sq2 = _square_polygon(half_side=100e-9) + np.array([+offset, 0.0])
        obj = GDSLayerObject(
            materials=two_materials,
            polygons=[sq1, sq2],
            gds_center=(0.0, 0.0),
            material_name="si",
            axis=2,
            thickness=200e-9,
        )
        placed = _place(obj, config, key, slices=((0, 20), (0, 20), (0, 4)))
        mask = placed.get_voxel_mask_for_shape()

        # Check True voxels appear in the low-x region (cells 0-6) and high-x region (cells 14-19)
        low_x_region = np.array(mask[:7, :, 0])
        high_x_region = np.array(mask[14:, :, 0])
        assert low_x_region.any(), "Expected True voxels in the left polygon region"
        assert high_x_region.any(), "Expected True voxels in the right polygon region"

    def test_empty_polygon_list_all_false(self, config, key, two_materials):
        """No polygons → mask should be all False."""
        obj = GDSLayerObject(
            materials=two_materials,
            polygons=[],
            gds_center=(0.0, 0.0),
            material_name="si",
            axis=2,
            thickness=200e-9,
        )
        placed = _place(obj, config, key, slices=((0, 20), (0, 20), (0, 4)))
        mask = placed.get_voxel_mask_for_shape()
        assert bool(jnp.all(~mask)), "Expected all-False mask for empty polygon list"


# ---------------------------------------------------------------------------
# get_material_mapping
# ---------------------------------------------------------------------------


class TestGetMaterialMapping:
    def test_shape_matches_grid(self, config, key, two_materials):
        obj = _make_layer_obj(two_materials, axis=2)
        placed = _place(obj, config, key, slices=((0, 20), (0, 20), (0, 4)))
        mapping = placed.get_material_mapping()
        assert mapping.shape == (20, 20, 4)

    def test_dtype_is_int(self, config, key, two_materials):
        obj = _make_layer_obj(two_materials, axis=2)
        placed = _place(obj, config, key)
        mapping = placed.get_material_mapping()
        assert jnp.issubdtype(mapping.dtype, jnp.integer)

    def test_correct_index_for_si(self, config, key, two_materials):
        """si (permittivity=12.25) > air (1.0) → sorted index 1."""
        obj = _make_layer_obj(two_materials, axis=2, material_name="si")
        placed = _place(obj, config, key, slices=((0, 20), (0, 20), (0, 4)))
        mapping = placed.get_material_mapping()
        assert bool(jnp.all(mapping == 1))

    def test_mapping_uniform(self, config, key, two_materials):
        """Every voxel gets the same material index."""
        obj = _make_layer_obj(two_materials, axis=2)
        placed = _place(obj, config, key)
        mapping = placed.get_material_mapping()
        assert bool(jnp.all(mapping == mapping[0, 0, 0]))


# ---------------------------------------------------------------------------
# gds_layer_stack
# ---------------------------------------------------------------------------


@pytest.fixture
def square_lib():
    """In-memory gdstk Library with a 200nm square on layer 1, datatype 0."""
    half = 0.1  # 0.1 µm = 100 nm in GDS units (unit=1e-6)
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
def sim_volume():
    return SimulationVolume(name="volume")


def _spec(gds_layer=1, material_name="si", thickness=200e-9, z_base=0.0, name=None):
    return GDSLayerSpec(gds_layer=gds_layer, material_name=material_name, thickness=thickness, z_base=z_base, name=name)


def _stack(lib, sim_volume, materials, specs):
    return gds_layer_stack(
        gds_source=lib,
        cell_name="TOP",
        layers=specs,
        materials=materials,
        simulation_volume=sim_volume,
    )


@pytest.mark.unit
class TestGdsLayerStack:
    def test_returns_correct_number_of_objects(self, square_lib, sim_volume, two_materials):
        """1 GDSLayerSpec → 1 GDSLayerObject returned."""
        objects, _ = _stack(square_lib, sim_volume, two_materials, [_spec()])
        assert len(objects) == 1

    def test_returns_correct_number_of_constraints(self, square_lib, sim_volume, two_materials):
        """1 GDSLayerSpec → 2 constraints (z position + xy size)."""
        _, constraints = _stack(square_lib, sim_volume, two_materials, [_spec()])
        assert len(constraints) == 2

    def test_object_is_gds_layer_object(self, square_lib, sim_volume, two_materials):
        """Returned object must be a GDSLayerObject instance."""
        objects, _ = _stack(square_lib, sim_volume, two_materials, [_spec()])
        assert isinstance(objects[0], GDSLayerObject)

    def test_missing_cell_raises(self, square_lib, sim_volume, two_materials):
        """Requesting a nonexistent cell name raises ValueError."""
        with pytest.raises(ValueError):
            gds_layer_stack(
                gds_source=square_lib,
                cell_name="NONEXISTENT",
                layers=[_spec()],
                materials=two_materials,
                simulation_volume=sim_volume,
            )

    def test_empty_layer_produces_empty_polygons(self, square_lib, sim_volume, two_materials):
        """A spec referencing a layer with no shapes produces an object with an empty polygon list."""
        objects, _ = _stack(square_lib, sim_volume, two_materials, [_spec(gds_layer=99)])
        assert len(objects[0].polygons) == 0

    def test_custom_name_used(self, square_lib, sim_volume, two_materials):
        """Spec with an explicit name → object.name matches."""
        objects, _ = _stack(square_lib, sim_volume, two_materials, [_spec(name="my_layer")])
        assert objects[0].name == "my_layer"

    def test_auto_name_generated(self, square_lib, sim_volume, two_materials):
        """Spec with name=None → auto-generated name encodes layer and datatype."""
        objects, _ = _stack(square_lib, sim_volume, two_materials, [_spec(name=None)])
        assert objects[0].name == "gds_1_0"
