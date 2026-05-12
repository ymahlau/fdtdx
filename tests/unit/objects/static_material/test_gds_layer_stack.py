"""Unit tests for objects/static_material/gds_layer_stack.py."""

import gdstk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import RectilinearGrid, UniformGrid
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.materials import Material
from fdtdx.objects.detectors.mode import ModeOverlapDetector
from fdtdx.objects.sources.mode import ModePlaneSource
from fdtdx.objects.static_material.gds_layer_stack import (
    GDSLayerObject,
    GDSLayerSpec,
    GDSPortSpec,
    detectors_from_gds_ports,
    gds_layer_stack,
    sources_from_gds_ports,
)
from fdtdx.objects.static_material.static import SimulationVolume

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return SimulationConfig(
        time=100e-15,
        grid=UniformGrid(spacing=50e-9),
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=None,
    )


@pytest.fixture
def rectilinear_config():
    """Config backed by an explicit RectilinearGrid (non-uniform-aware path)."""
    res = 50e-9
    grid = RectilinearGrid.uniform(shape=(20, 20, 4), spacing=res)
    return SimulationConfig(
        time=100e-15,
        grid=grid,
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

        Grid: 20x20x4 cells at 50nm resolution = 1µmx1µm cross-section.
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

    def test_axis_not_2_raises(self, config, key, two_materials):
        """axis != 2 must raise ValueError since GDS has no z-coordinate."""
        obj = _make_layer_obj(two_materials, axis=0)
        placed = _place(obj, config, key, slices=((0, 4), (0, 20), (0, 20)))
        with pytest.raises(ValueError, match="axis=2"):
            placed.get_voxel_mask_for_shape()


# ---------------------------------------------------------------------------
# get_voxel_mask_for_shape — RectilinearGrid path
# ---------------------------------------------------------------------------


class TestGetVoxelMaskRectilinearGrid:
    def test_mask_shape_matches_grid(self, rectilinear_config, key, two_materials):
        obj = _make_layer_obj(two_materials, axis=2)
        placed = _place(obj, rectilinear_config, key, slices=((0, 20), (0, 20), (0, 4)))
        mask = placed.get_voxel_mask_for_shape()
        assert mask.shape == (20, 20, 4)

    def test_mask_is_bool(self, rectilinear_config, key, two_materials):
        obj = _make_layer_obj(two_materials, axis=2)
        placed = _place(obj, rectilinear_config, key)
        mask = placed.get_voxel_mask_for_shape()
        assert mask.dtype == jnp.bool_

    def test_single_polygon_has_interior_voxels(self, rectilinear_config, key, two_materials):
        obj = GDSLayerObject(
            materials=two_materials,
            polygons=[_square_polygon(half_side=100e-9)],
            gds_center=(0.0, 0.0),
            material_name="si",
            axis=2,
            thickness=200e-9,
        )
        placed = _place(obj, rectilinear_config, key, slices=((0, 20), (0, 20), (0, 4)))
        mask = placed.get_voxel_mask_for_shape()
        assert bool(jnp.any(mask))

    def test_empty_polygon_list_all_false(self, rectilinear_config, key, two_materials):
        obj = GDSLayerObject(
            materials=two_materials,
            polygons=[],
            gds_center=(0.0, 0.0),
            material_name="si",
            axis=2,
            thickness=200e-9,
        )
        placed = _place(obj, rectilinear_config, key, slices=((0, 20), (0, 20), (0, 4)))
        mask = placed.get_voxel_mask_for_shape()
        assert bool(jnp.all(~mask))

    def test_mask_uniform_along_extrusion_axis(self, rectilinear_config, key, two_materials):
        obj = _make_layer_obj(two_materials, axis=2)
        placed = _place(obj, rectilinear_config, key, slices=((0, 20), (0, 20), (0, 4)))
        mask = placed.get_voxel_mask_for_shape()
        for z in range(mask.shape[2]):
            assert jnp.array_equal(mask[:, :, z], mask[:, :, 0])

    def test_rectilinear_matches_uniform_for_uniform_spacing(self, config, rectilinear_config, key, two_materials):
        """With identical uniform spacing, both grid paths should yield the same mask."""
        obj = GDSLayerObject(
            materials=two_materials,
            polygons=[_square_polygon(half_side=100e-9)],
            gds_center=(0.0, 0.0),
            material_name="si",
            axis=2,
            thickness=200e-9,
        )
        slices = ((0, 20), (0, 20), (0, 4))
        placed_uniform = _place(obj, config, key, slices=slices)
        placed_rect = _place(obj, rectilinear_config, key, slices=slices)
        mask_uniform = placed_uniform.get_voxel_mask_for_shape()
        mask_rect = placed_rect.get_voxel_mask_for_shape()
        assert jnp.array_equal(mask_uniform, mask_rect)


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
        gds_center=(0.0, 0.0),
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
                gds_center=(0.0, 0.0),
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


# ---------------------------------------------------------------------------
# Boolean etch (etch_by)
# ---------------------------------------------------------------------------


@pytest.fixture
def etch_lib():
    """Library with a 600nm square on layer 1 and a 200nm square etch hole on layer 2."""
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("TOP")
    big = 0.3  # 600nm half-side in µm
    hole = 0.1  # 200nm half-side in µm
    cell.add(
        gdstk.Polygon(
            [(-big, -big), (big, -big), (big, big), (-big, big)],
            layer=1,
            datatype=0,
        )
    )
    cell.add(
        gdstk.Polygon(
            [(-hole, -hole), (hole, -hole), (hole, hole), (-hole, hole)],
            layer=2,
            datatype=0,
        )
    )
    return lib


@pytest.mark.unit
class TestEtchBy:
    def test_etch_reduces_polygon_count(self, etch_lib, sim_volume, two_materials):
        """Etching layer 2 from layer 1 should produce more polygons than the original (ring shape)."""
        spec_no_etch = GDSLayerSpec(gds_layer=1, material_name="si", thickness=200e-9, z_base=0.0)
        spec_etch = GDSLayerSpec(gds_layer=1, material_name="si", thickness=200e-9, z_base=0.0, etch_by=((2, 0),))
        objects_no_etch, _ = gds_layer_stack(
            gds_source=etch_lib,
            cell_name="TOP",
            layers=[spec_no_etch],
            materials=two_materials,
            simulation_volume=sim_volume,
            gds_center=(0.0, 0.0),
        )
        objects_etch, _ = gds_layer_stack(
            gds_source=etch_lib,
            cell_name="TOP",
            layers=[spec_etch],
            materials=two_materials,
            simulation_volume=sim_volume,
            gds_center=(0.0, 0.0),
        )
        assert len(objects_etch[0].polygons[0]) > len(objects_no_etch[0].polygons[0])

    def test_etch_empty_removes_nothing(self, square_lib, sim_volume, two_materials):
        """etch_by=[] (empty) should behave identically to no etch."""
        spec_with = GDSLayerSpec(gds_layer=1, material_name="si", thickness=200e-9, z_base=0.0, etch_by=())
        objects, _ = gds_layer_stack(
            gds_source=square_lib,
            cell_name="TOP",
            layers=[spec_with],
            materials=two_materials,
            simulation_volume=sim_volume,
            gds_center=(0.0, 0.0),
        )
        assert len(objects[0].polygons) == 1

    def test_etch_nonexistent_layer_keeps_original(self, square_lib, sim_volume, two_materials):
        """etch_by referencing a layer with no polygons leaves the original unchanged."""
        spec = GDSLayerSpec(gds_layer=1, material_name="si", thickness=200e-9, z_base=0.0, etch_by=((99, 0),))
        objects, _ = gds_layer_stack(
            gds_source=square_lib,
            cell_name="TOP",
            layers=[spec],
            materials=two_materials,
            simulation_volume=sim_volume,
            gds_center=(0.0, 0.0),
        )
        assert len(objects[0].polygons) == 1

    def test_etch_non_overlapping_leaves_polygon_unchanged(self, sim_volume, two_materials):
        """Etching with a non-overlapping polygon leaves the original shape intact."""
        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        cell = lib.new_cell("TOP")
        cell.add(gdstk.Polygon([(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)], layer=1, datatype=0))
        # Etch polygon placed far away — no geometric overlap
        cell.add(gdstk.Polygon([(1.0, 1.0), (1.5, 1.0), (1.5, 1.5), (1.0, 1.5)], layer=2, datatype=0))
        spec = GDSLayerSpec(gds_layer=1, material_name="si", thickness=200e-9, z_base=0.0, etch_by=((2, 0),))
        objects, _ = gds_layer_stack(
            gds_source=lib,
            cell_name="TOP",
            layers=[spec],
            materials=two_materials,
            simulation_volume=sim_volume,
            gds_center=(0.0, 0.0),
        )
        assert len(objects[0].polygons) == 1
        assert len(objects[0].polygons[0]) == 4


# ---------------------------------------------------------------------------
# gdsfactory integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGdsLayerStackFromComponent:
    def test_requires_gdsfactory(self, sim_volume, two_materials):
        """Without gdsfactory installed, ImportError is raised with a helpful message."""
        from unittest.mock import patch

        from fdtdx.objects.static_material.gds_layer_stack import gds_layer_stack_from_component

        class _FakeComponent:
            name = "FAKE"

            def write_gds(self, path):
                raise RuntimeError("should not be called")

        with patch.dict("sys.modules", {"gdsfactory": None}):
            with pytest.raises(ImportError, match="gdsfactory"):
                gds_layer_stack_from_component(
                    component=_FakeComponent(),
                    layers=[_spec()],
                    materials=two_materials,
                    simulation_volume=sim_volume,
                    gds_center=(0.0, 0.0),
                )

    def test_with_gdsfactory_if_available(self, sim_volume, two_materials):
        """If gdsfactory is installed, the function returns GDSLayerObjects."""
        gf = pytest.importorskip("gdsfactory")
        from fdtdx.objects.static_material.gds_layer_stack import gds_layer_stack_from_component

        c = gf.Component("TEST")
        c.add_polygon(
            [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
            layer=(1, 0),
        )
        objects, constraints = gds_layer_stack_from_component(
            component=c,
            cell_name="TEST",
            layers=[_spec(gds_layer=1)],
            materials=two_materials,
            simulation_volume=sim_volume,
            gds_center=(0.0, 0.0),
        )
        assert len(objects) == 1
        assert isinstance(objects[0], GDSLayerObject)
        assert len(constraints) == 2


# ---------------------------------------------------------------------------
# sources_from_gds_ports / detectors_from_gds_ports
# ---------------------------------------------------------------------------


@pytest.fixture
def port_lib():
    """Library with a 100nm-wide port marker on layer 10 centred at GDS x=0."""
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("TOP")
    # Thin rectangle: x ∈ [-0.05, 0.05] µm, y ∈ [-0.2, 0.2] µm
    cell.add(
        gdstk.Polygon(
            [(-0.05, -0.2), (0.05, -0.2), (0.05, 0.2), (-0.05, 0.2)],
            layer=10,
            datatype=0,
        )
    )
    return lib


@pytest.fixture
def vol_with_x_size():
    """SimulationVolume with partial_real_shape set on axis 0 (1 µm wide)."""
    return SimulationVolume(name="volume", partial_real_shape=(1e-6, None, None))


@pytest.fixture
def wave_char():
    return WaveCharacter(wavelength=1.55e-6)


@pytest.mark.unit
class TestSourcesFromGdsPorts:
    def test_returns_correct_count(self, port_lib, vol_with_x_size, wave_char):
        """One port polygon → one ModePlaneSource."""
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        sources, _constraints = sources_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_character=wave_char,
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        assert len(sources) == 1
        assert isinstance(sources[0], ModePlaneSource)

    def test_four_constraints_per_port(self, port_lib, vol_with_x_size, wave_char):
        """Each port should produce 4 constraints: propagation position, transverse center, height size, height center."""
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        _, constraints = sources_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_character=wave_char,
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        assert len(constraints) == 4

    def test_propagation_axis_2_raises(self, port_lib, vol_with_x_size, wave_char):
        """propagation_axis=2 must raise ValueError."""
        spec = GDSPortSpec(gds_layer=10, propagation_axis=2)
        with pytest.raises(ValueError, match="propagation_axis"):
            sources_from_gds_ports(
                gds_source=port_lib,
                cell_name="TOP",
                port_specs=[spec],
                wave_character=wave_char,
                simulation_volume=vol_with_x_size,
                gds_center=(0.0, 0.0),
            )

    def test_missing_vol_size_raises(self, port_lib, wave_char):
        """Unset partial_real_shape on propagation axis must raise ValueError."""
        vol_no_size = SimulationVolume(name="volume")
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        with pytest.raises(ValueError, match="partial_real_shape"):
            sources_from_gds_ports(
                gds_source=port_lib,
                cell_name="TOP",
                port_specs=[spec],
                wave_character=wave_char,
                simulation_volume=vol_no_size,
                gds_center=(0.0, 0.0),
            )

    def test_empty_port_layer_returns_empty(self, port_lib, vol_with_x_size, wave_char):
        """Spec referencing a layer with no polygons → empty lists."""
        spec = GDSPortSpec(gds_layer=99, propagation_axis=0)
        sources, constraints = sources_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_character=wave_char,
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        assert sources == []
        assert constraints == []

    def test_name_prefix_applied(self, port_lib, vol_with_x_size, wave_char):
        """Custom name_prefix is reflected in the source name."""
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0, name_prefix="in")
        sources, _ = sources_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_character=wave_char,
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        assert sources[0].name == "in_0"


@pytest.mark.unit
class TestDetectorsFromGdsPorts:
    def test_returns_mode_overlap_detector(self, port_lib, vol_with_x_size, wave_char):
        """One port polygon → one ModeOverlapDetector."""
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        detectors, _constraints = detectors_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_characters=[wave_char],
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        assert len(detectors) == 1
        assert isinstance(detectors[0], ModeOverlapDetector)

    def test_four_constraints_per_detector(self, port_lib, vol_with_x_size, wave_char):
        """Each port should produce 4 constraints: propagation position, transverse center, height size, height center."""
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        _, constraints = detectors_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_characters=[wave_char],
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        assert len(constraints) == 4

    def test_propagation_axis_2_raises(self, port_lib, vol_with_x_size, wave_char):
        """propagation_axis=2 must raise ValueError."""
        spec = GDSPortSpec(gds_layer=10, propagation_axis=2)
        with pytest.raises(ValueError, match="propagation_axis"):
            detectors_from_gds_ports(
                gds_source=port_lib,
                cell_name="TOP",
                port_specs=[spec],
                wave_characters=[wave_char],
                simulation_volume=vol_with_x_size,
                gds_center=(0.0, 0.0),
            )

    def test_missing_vol_size_raises(self, port_lib, wave_char):
        """Unset partial_real_shape on propagation axis must raise ValueError."""
        vol_no_size = SimulationVolume(name="volume")
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        with pytest.raises(ValueError, match="partial_real_shape"):
            detectors_from_gds_ports(
                gds_source=port_lib,
                cell_name="TOP",
                port_specs=[spec],
                wave_characters=[wave_char],
                simulation_volume=vol_no_size,
                gds_center=(0.0, 0.0),
            )

    def test_empty_port_layer_returns_empty(self, port_lib, vol_with_x_size, wave_char):
        """Spec referencing a layer with no polygons → empty lists."""
        spec = GDSPortSpec(gds_layer=99, propagation_axis=0)
        detectors, constraints = detectors_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_characters=[wave_char],
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        assert detectors == []
        assert constraints == []

    def test_name_prefix_applied(self, port_lib, vol_with_x_size, wave_char):
        """Custom name_prefix is reflected in the detector name."""
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0, name_prefix="out")
        detectors, _ = detectors_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_characters=[wave_char],
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        assert detectors[0].name == "out_0"
