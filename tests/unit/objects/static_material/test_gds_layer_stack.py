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
from fdtdx.objects.object import RealCoordinateConstraint
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


def _square_polygon_at(center, half_side=100e-9):
    return _square_polygon(half_side=half_side) + np.asarray(center)


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


def _rectilinear_config(x_edges, y_edges, z_edges=None):
    if z_edges is None:
        z_edges = jnp.linspace(0.0, 200e-9, 5)
    grid = RectilinearGrid(x_edges=x_edges, y_edges=y_edges, z_edges=z_edges)
    return SimulationConfig(time=100e-15, grid=grid, backend="cpu", dtype=jnp.float32, gradient_config=None)


def _gds_mask_xy(materials, config, key, polygons, gds_center, slices=((0, 20), (0, 20), (0, 4))):
    obj = _make_layer_obj(materials, polygons=polygons, gds_center=gds_center, axis=2)
    return np.array(_place(obj, config, key, slices=slices).get_voxel_mask_for_shape()[:, :, 0])


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
# get_voxel_mask_for_shape — sidewall_angle (trapezoidal extrusion)
# ---------------------------------------------------------------------------


def _sidewall_config(spacing, nz):
    """Config with a tall-enough z stack to resolve a sidewall taper."""
    return SimulationConfig(
        time=100e-15,
        grid=UniformGrid(spacing=spacing),
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=None,
    )


def _half_width_per_slice(mask, axis_h, axis_v):
    """Measured half-width (metres-agnostic, in cells) of a centred square per z-slice.

    Returns the count of True cells along the central horizontal row of each z-slice.
    """
    nz = mask.shape[2]
    mid_v = mask.shape[axis_v] // 2
    widths = []
    for z in range(nz):
        row = np.array(mask[:, mid_v, z])
        widths.append(int(row.sum()))
    return np.array(widths)


class TestSidewallAngle:
    """Trapezoidal extrusion: each z-slice is eroded/dilated by (z - z_ref) * tan(90deg - angle)."""

    SPACING = 25e-9
    NZ = 20  # 500 nm tall stack
    HALF = 300e-9  # 600 nm square footprint

    def _build(self, materials, key, sidewall_angle, reference_plane="bottom"):
        config = _sidewall_config(self.SPACING, self.NZ)
        obj = GDSLayerObject(
            materials=materials,
            polygons=[_square_polygon(half_side=self.HALF)],
            gds_center=(0.0, 0.0),
            material_name="si",
            axis=2,
            thickness=self.NZ * self.SPACING,
            sidewall_angle=sidewall_angle,
            reference_plane=reference_plane,
        )
        n_xy = 40  # 1 µm cross-section at 25 nm
        slices = ((0, n_xy), (0, n_xy), (0, self.NZ))
        placed = obj.place_on_grid(grid_slice_tuple=slices, config=config, key=key)
        return placed.get_voxel_mask_for_shape()

    def test_vertical_angle_is_uniform(self, key, two_materials):
        """sidewall_angle=90 (vertical) reproduces the plain vertical extrusion (all slices identical)."""
        mask = self._build(two_materials, key, sidewall_angle=90.0)
        for z in range(mask.shape[2]):
            assert jnp.array_equal(mask[:, :, z], mask[:, :, 0])

    def test_angle_below_90_tapers_inward_from_bottom(self, key, two_materials):
        """Angle < 90, reference_plane='bottom': width shrinks monotonically with z."""
        angle = 75.0  # degrees from substrate (15 deg off vertical)
        mask = self._build(two_materials, key, sidewall_angle=angle, reference_plane="bottom")
        widths = _half_width_per_slice(mask, 0, 1)
        # bottom slice is the widest, top the narrowest, monotone non-increasing
        assert widths[0] >= widths[-1]
        assert widths[-1] < widths[0]
        assert np.all(np.diff(widths) <= 0)

    @pytest.mark.parametrize("angle", [89.0, 80.0, 75.0, 60.0])
    @pytest.mark.parametrize("reference_plane", ["bottom", "middle", "top"])
    def test_slice_width_matches_analytic_trapezoid(self, key, two_materials, angle, reference_plane):
        """Each z-slice's central-row width matches the analytic trapezoid to within one cell PER EDGE.

        Independent ground truth (a different method from the implementation): the sidewall shrinks the
        footprint half-width to ``HALF - offset(z)`` with ``offset(z) = (z_center - z_ref) * tan(90 - angle)``,
        so the analytic solid-cell count across a central row is ``count(|cell_centre| < HALF - offset(z))``.

        The bound is ``|measured - analytic| <= 2`` full-width (one cell per edge), and it is TIGHT rather
        than slack: the implementation offsets the wall with a Euclidean distance transform whose distances
        are sampled cell-centre-to-cell-centre, so an edge cell reads the nearest void centre one pitch away
        instead of the continuous boundary half a pitch away -- a fixed <=half-cell bias per edge relative to
        a *continuous* trapezoid.  Bit-exact ("total") agreement with a continuous model is therefore not
        attainable for a discrete-EDT staircase; it would require comparing against a second distance
        transform, i.e. re-deriving the implementation inside the test.  Empirically this bound is saturated
        (dev = 2) only at steep, non-physical angles; at a foundry-realistic near-vertical wall (89 deg) the
        offset is far below one pitch across the whole stack and the agreement is exact (dev = 0).
        """
        mask = np.asarray(self._build(two_materials, key, sidewall_angle=angle, reference_plane=reference_plane))
        n_xy = mask.shape[0]
        cc = (np.arange(n_xy) + 0.5) * self.SPACING - n_xy * self.SPACING / 2.0  # cell centres about the polygon centre
        z_centers = (np.arange(self.NZ) + 0.5) * self.SPACING
        z_ref = {"bottom": z_centers[0], "top": z_centers[-1], "middle": 0.5 * (z_centers[0] + z_centers[-1])}[
            reference_plane
        ]
        tan = np.tan(np.deg2rad(90.0 - angle))
        mid = n_xy // 2
        for z in range(self.NZ):
            half_z = self.HALF - (z_centers[z] - z_ref) * tan
            expected = int(np.count_nonzero(np.abs(cc) < half_z))  # analytic point-in-interval count
            measured = int(mask[:, mid, z].sum())
            assert abs(measured - expected) <= 2, (  # <= 1 cell per edge; == 0 at near-vertical (89 deg)
                f"angle={angle}, plane={reference_plane}, z={z}: width {measured} cells vs analytic {expected}"
            )

    def test_reference_plane_top_keeps_top_footprint(self, key, two_materials):
        """reference_plane='top' keeps the *top* footprint nominal; the base is wider.

        For an angle < 90 the slab still narrows toward the top; changing the
        reference plane only shifts which z-plane equals the input polygon, so here the bottom is
        dilated and the top is nominal -> width still decreases with z, but every slice is wider than
        the matching 'bottom'-reference slice.
        """
        angle = 75.0  # degrees from substrate
        mask_top = self._build(two_materials, key, sidewall_angle=angle, reference_plane="top")
        mask_bot = self._build(two_materials, key, sidewall_angle=angle, reference_plane="bottom")
        w_top = _half_width_per_slice(mask_top, 0, 1)
        w_bot = _half_width_per_slice(mask_bot, 0, 1)
        # still narrows toward the top
        assert w_top[-1] <= w_top[0]
        # 'top' reference is uniformly wider-or-equal than 'bottom' reference (footprint shifted up)
        assert np.all(w_top >= w_bot)
        # the top slice of the 'top' reference equals the nominal bottom slice of 'bottom' reference
        assert abs(int(w_top[-1]) - int(w_bot[0])) <= 2

    def test_reference_plane_middle_symmetric(self, key, two_materials):
        """reference_plane='middle': the mid-slice is nominal, base wider, top narrower, symmetric."""
        angle = 75.0  # degrees from substrate
        mask = self._build(two_materials, key, sidewall_angle=angle, reference_plane="middle")
        widths = _half_width_per_slice(mask, 0, 1)
        mid = self.NZ // 2
        # monotone narrowing with z; mid-slice ~ halfway between the extremes
        assert widths[0] >= widths[mid] >= widths[-1]
        assert abs((int(widths[0]) - int(widths[-1])) - 2 * (int(widths[0]) - int(widths[mid]))) <= 3

    def test_angle_above_90_dilates(self, key, two_materials):
        """Angle > 90 (re-entrant / undercut): width grows with z from the bottom reference."""
        angle = 105.0  # degrees from substrate (15 deg past vertical, re-entrant)
        mask = self._build(two_materials, key, sidewall_angle=angle, reference_plane="bottom")
        widths = _half_width_per_slice(mask, 0, 1)
        assert widths[-1] >= widths[0]

    def test_single_slice_middle_keeps_nominal_width(self, key, two_materials):
        """A 1-cell-tall layer with reference_plane='middle' keeps the nominal footprint (offset ~ 0)."""
        cfg = _sidewall_config(self.SPACING, 1)
        n_xy = 40

        def build(angle):
            obj = GDSLayerObject(
                materials=two_materials,
                polygons=[_square_polygon(half_side=self.HALF)],
                gds_center=(0.0, 0.0),
                material_name="si",
                axis=2,
                thickness=self.SPACING,
                sidewall_angle=angle,
                reference_plane="middle",
            )
            placed = obj.place_on_grid(grid_slice_tuple=((0, n_xy), (0, n_xy), (0, 1)), config=cfg, key=key)
            return np.array(placed.get_voxel_mask_for_shape())

        # a single mid-plane slice must match the vertical footprint regardless of angle
        assert np.array_equal(build(70.0), build(90.0))

    def test_invalid_angle_raises(self, key, two_materials):
        """An angle outside (0, 180) degrees is rejected (validated eagerly in __post_init__)."""
        with pytest.raises(ValueError, match="degrees"):
            self._build(two_materials, key, sidewall_angle=200.0)

    def test_invalid_reference_plane_raises(self, key, two_materials):
        """An unknown reference_plane is rejected eagerly (not silently treated as 'middle')."""
        with pytest.raises(ValueError, match="reference_plane"):
            self._build(two_materials, key, sidewall_angle=80.0, reference_plane="center")

    def test_spec_threads_sidewall_through_stack(self, square_lib, sim_volume, two_materials):
        """GDSLayerSpec.sidewall_angle / reference_plane are forwarded onto the GDSLayerObject."""
        spec = GDSLayerSpec(
            gds_layer=1,
            material_name="si",
            thickness=200e-9,
            sidewall_angle=82.0,
            reference_plane="middle",
        )
        objects, _ = gds_layer_stack(
            gds_source=square_lib,
            cell_name="TOP",
            layers=[spec],
            materials=two_materials,
            simulation_volume=sim_volume,
            gds_center=(0.0, 0.0),
        )
        assert objects[0].sidewall_angle == pytest.approx(82.0)
        assert objects[0].reference_plane == "middle"


@pytest.mark.unit
class TestSidewallVsTidy3D:
    """Cross-check the discrete sidewall mask footprint against Tidy3D.PolySlab geometry.

    Tidy3D is a hard dependency of fdtdx, but we still guard the import so the test degrades
    gracefully if it is ever made optional. This compares the *geometry* (which cells lie inside the
    slanted slab at a given height) rather than n_eff, keeping the test fast and solver-free; the
    n_eff agreement (max|Δ| ~= 0.005 at 90/88/86 deg) is checked by the in-tree example
    examples/validate_sidewall_neff.py.
    """

    def test_footprint_matches_polyslab_cross_sections(self, key, two_materials):
        td = pytest.importorskip("tidy3d")

        spacing = 20e-9
        nz = 20
        half = 400e-9  # 800 nm square
        gds_angle = 82.0  # degrees from substrate (8 deg off vertical)
        polyslab_angle = np.deg2rad(90.0 - gds_angle)  # Tidy3D PolySlab: deviation-from-vertical (rad)

        config = SimulationConfig(
            time=100e-15,
            grid=UniformGrid(spacing=spacing),
            backend="cpu",
            dtype=jnp.float32,
            gradient_config=None,
        )
        n_xy = 60
        obj = GDSLayerObject(
            materials=two_materials,
            polygons=[_square_polygon(half_side=half)],
            gds_center=(0.0, 0.0),
            material_name="si",
            axis=2,
            thickness=nz * spacing,
            sidewall_angle=gds_angle,
            reference_plane="bottom",
        )
        placed = obj.place_on_grid(grid_slice_tuple=((0, n_xy), (0, n_xy), (0, nz)), config=config, key=key)
        mask = np.array(placed.get_voxel_mask_for_shape())

        # Build the equivalent Tidy3D PolySlab and query ITS OWN geometry (inside()), rather than
        # re-deriving the taper. Tidy3D coordinates are dimensionless (µm by convention), so the whole
        # comparison is done in microns; the taper relation is scale-free.
        um = 1e6
        h = half * um
        verts = [(-h, -h), (h, -h), (h, h), (-h, h)]
        slab = td.PolySlab(
            vertices=verts,
            slab_bounds=(0.0, nz * spacing * um),
            axis=2,
            sidewall_angle=float(polyslab_angle),
            reference_plane="bottom",
        )

        # Cross-section centres (microns); PolySlab.inside meshes the 1D coord vectors into a 3D grid.
        x_centers = ((np.arange(n_xy) + 0.5) * spacing - n_xy * spacing / 2) * um
        y_centers = x_centers.copy()
        z_centers = (np.arange(nz) + 0.5) * spacing * um
        pitch_um = spacing * um

        X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
        inside_td = np.asarray(slab.inside(X, Y, Z))  # (n_xy, n_xy, nz)

        max_edge_err_cells = 0.0
        for kz, z in enumerate(z_centers):
            inside_row = inside_td[:, n_xy // 2, kz]
            mask_row = mask[:, n_xy // 2, kz]
            disagree = np.where(mask_row != inside_row)[0]
            # disagreement is only allowed within one cell of the slanted edge; the analytical
            # half-width locates the edge to measure the discretization error against.
            half_z = h - z * np.tan(polyslab_angle)
            for idx in disagree:
                edge_dist_cells = (abs(x_centers[idx]) - half_z) / pitch_um
                max_edge_err_cells = max(max_edge_err_cells, abs(edge_dist_cells))

        assert max_edge_err_cells <= 1.0, (
            f"sidewall footprint disagrees with Tidy3D PolySlab by {max_edge_err_cells:.2f} cells"
        )


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

    def test_polygon_position_with_nonzero_gds_center(self, key, two_materials):
        """A shifted GDS center is evaluated relative to the placed object center."""
        n, res = 20, 50e-9
        edges_xy = jnp.linspace(0.0, n * res, n + 1)
        cfg = _rectilinear_config(edges_xy, edges_xy)
        gds_center = (400e-9, 500e-9)
        mask_xy = _gds_mask_xy(
            two_materials,
            cfg,
            key,
            polygons=[_square_polygon_at(gds_center, half_side=100e-9)],
            gds_center=gds_center,
        )

        expected = np.zeros((20, 20), dtype=bool)
        expected[8:12, 8:12] = True
        np.testing.assert_array_equal(mask_xy, expected)

    def test_rectilinear_matches_uniform_for_nonzero_gds_center(self, config, key, two_materials):
        """RectilinearGrid and UniformGrid paths produce the same mask for a nonzero gds_center."""
        n, res = 20, 50e-9
        edges_xy = jnp.linspace(0.0, n * res, n + 1)
        rect_cfg = _rectilinear_config(edges_xy, edges_xy)
        gds_center = (400e-9, 500e-9)
        polygons = [_square_polygon_at(gds_center, half_side=100e-9)]
        slices = ((0, 20), (0, 20), (0, 4))
        mask_uniform = _gds_mask_xy(two_materials, config, key, polygons, gds_center, slices=slices)
        mask_rect = _gds_mask_xy(two_materials, rect_cfg, key, polygons, gds_center, slices=slices)
        np.testing.assert_array_equal(mask_rect, mask_uniform)

    def test_polygon_position_on_nonuniform_grid(self, key, two_materials):
        """Cell-center sampling uses real rectilinear coordinates, not a uniform index grid."""
        coarse_edges = np.linspace(0.0, 800e-9, 11)
        fine_edges = np.linspace(800e-9, 1000e-9, 11)[1:]
        edges_x = jnp.asarray(np.concatenate([coarse_edges, fine_edges]))
        edges_y = jnp.linspace(0.0, 20 * 50e-9, 21)
        cfg = _rectilinear_config(edges_x, edges_y)
        gds_center = (500e-9, 500e-9)
        mask_xy = _gds_mask_xy(
            two_materials,
            cfg,
            key,
            polygons=[_square_polygon_at(gds_center, half_side=50e-9)],
            gds_center=gds_center,
        )

        expected = np.zeros((20, 20), dtype=bool)
        expected[6, 9:11] = True
        np.testing.assert_array_equal(mask_xy, expected)


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

    def test_partial_grid_shape_one_cell_along_propagation_axis(self, port_lib, vol_with_x_size, wave_char):
        """Source must be exactly 1 cell thick along the propagation axis, unconstrained on others."""
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        sources, _ = sources_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_character=wave_char,
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        assert sources[0].partial_grid_shape[0] == 1
        assert sources[0].partial_grid_shape[1] is None
        assert sources[0].partial_grid_shape[2] is None

    def test_transverse_width_matches_port_polygon(self, port_lib, vol_with_x_size, wave_char):
        """partial_real_shape on the transverse axis equals the port polygon bounding-box width.

        Port polygon y ∈ [-200nm, 200nm] → transverse_width = 400nm.
        """
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        sources, _ = sources_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_character=wave_char,
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        assert sources[0].partial_real_shape[1] == pytest.approx(400e-9)
        assert sources[0].partial_real_shape[0] is None
        assert sources[0].partial_real_shape[2] is None

    def test_propagation_position_from_gds_centroid(self, port_lib, vol_with_x_size, wave_char):
        """RealCoordinateConstraint positions source at port centroid in simulation coordinates.

        Port centroid x=0 in GDS, gds_center=(0,0), vol_size=1µm → sim_prop_pos = 0µm (centered coordinates).
        """
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        _, constraints = sources_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_character=wave_char,
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        prop_constraint = constraints[0]
        assert isinstance(prop_constraint, RealCoordinateConstraint)
        assert prop_constraint.axes == (0,)
        assert prop_constraint.coordinates[0] == pytest.approx(0.0)

    def test_gds_center_offset_shifts_position(self, port_lib, wave_char):
        """Shifting gds_center shifts the source position by the same amount.

        Port centroid x=0, gds_center=(200nm, 0), vol_size=1µm → sim_prop_pos = 0 - 200 = -200nm
        """
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        vol = SimulationVolume(name="volume", partial_real_shape=(1e-6, None, None))
        _, constraints = sources_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_character=wave_char,
            simulation_volume=vol,
            gds_center=(200e-9, 0.0),
        )
        assert constraints[0].coordinates[0] == pytest.approx(-200e-9)


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

    def test_partial_grid_shape_one_cell_along_propagation_axis(self, port_lib, vol_with_x_size, wave_char):
        """Detector must be exactly 1 cell thick along the propagation axis, unconstrained on others."""
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        detectors, _ = detectors_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_characters=[wave_char],
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        assert detectors[0].partial_grid_shape[0] == 1
        assert detectors[0].partial_grid_shape[1] is None
        assert detectors[0].partial_grid_shape[2] is None

    def test_transverse_width_matches_port_polygon(self, port_lib, vol_with_x_size, wave_char):
        """partial_real_shape on the transverse axis equals the port polygon bounding-box width.

        Port polygon y ∈ [-200nm, 200nm] → transverse_width = 400nm.
        """
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        detectors, _ = detectors_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_characters=[wave_char],
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        assert detectors[0].partial_real_shape[1] == pytest.approx(400e-9)
        assert detectors[0].partial_real_shape[0] is None
        assert detectors[0].partial_real_shape[2] is None

    def test_propagation_position_from_gds_centroid(self, port_lib, vol_with_x_size, wave_char):
        """RealCoordinateConstraint positions detector at port centroid in simulation coordinates.

        Port centroid x=0 in GDS, gds_center=(0,0), vol_size=1µm → sim_prop_pos = 0µm (centered coordinates).
        """
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        _, constraints = detectors_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_characters=[wave_char],
            simulation_volume=vol_with_x_size,
            gds_center=(0.0, 0.0),
        )
        prop_constraint = constraints[0]
        assert isinstance(prop_constraint, RealCoordinateConstraint)
        assert prop_constraint.axes == (0,)
        assert prop_constraint.coordinates[0] == pytest.approx(0.0)

    def test_gds_center_offset_shifts_position(self, port_lib, wave_char):
        """Shifting gds_center shifts the detector position by the same amount.

        Port centroid x=0, gds_center=(200nm, 0), vol_size=1µm → sim_prop_pos = 0 -200 = -200nm.
        """
        spec = GDSPortSpec(gds_layer=10, propagation_axis=0)
        vol = SimulationVolume(name="volume", partial_real_shape=(1e-6, None, None))
        _, constraints = detectors_from_gds_ports(
            gds_source=port_lib,
            cell_name="TOP",
            port_specs=[spec],
            wave_characters=[wave_char],
            simulation_volume=vol,
            gds_center=(200e-9, 0.0),
        )
        assert constraints[0].coordinates[0] == pytest.approx(-200e-9)
