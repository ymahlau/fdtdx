import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx import constants
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import (
    QuasiUniformGrid,
    RectilinearGrid,
    calculate_spatial_offsets_yee,
    calculate_time_offset_yee,
    polygon_to_mask,
)
from fdtdx.fdtd.initialization import resolve_object_constraints
from fdtdx.materials import Material
from fdtdx.objects.object import GridCoordinateConstraint
from fdtdx.objects.static_material.static import SimulationVolume, UniformMaterialObject


class TestRectilinearGrid:
    """Tests for the canonical rectilinear grid representation."""

    def test_uniform_constructor_stores_edges(self):
        """Uniform grids are represented as ordinary rectilinear edge arrays."""
        grid = RectilinearGrid.uniform(shape=(2, 3, 4), spacing=0.5, origin=(1.0, 2.0, 3.0))

        assert grid.shape == (2, 3, 4)
        assert np.allclose(np.asarray(grid.x_edges), [1.0, 1.5, 2.0])
        assert np.allclose(np.asarray(grid.y_edges), [2.0, 2.5, 3.0, 3.5])
        assert np.allclose(np.asarray(grid.z_edges), [3.0, 3.5, 4.0, 4.5, 5.0])
        assert grid.is_uniform
        assert grid.uniform_spacing == 0.5

    def test_custom_constructor_stores_explicit_edges(self):
        """Custom grids are realized edge arrays, not automatic meshing policies."""
        grid = RectilinearGrid.custom(
            x_edges=jnp.asarray([0.0, 1.0, 2.5]),
            y_edges=jnp.asarray([0.0, 2.0]),
            z_edges=jnp.asarray([0.0, 0.5, 1.5]),
        )

        assert grid.shape == (2, 1, 2)
        assert np.allclose(np.asarray(grid.dx), [1.0, 1.5])
        assert not grid.is_uniform

    def test_nonuniform_grid_metrics(self):
        """Non-uniform grids derive widths, centers, extents, areas, and volumes from edges."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0]),
            y_edges=jnp.asarray([0.0, 2.0, 5.0]),
            z_edges=jnp.asarray([0.0, 4.0, 6.0]),
        )
        slice_tuple = ((0, 2), (0, 2), (0, 2))

        assert grid.shape == (2, 2, 2)
        assert np.allclose(np.asarray(grid.dx), [1.0, 2.0])
        assert np.allclose(np.asarray(grid.centers(1)), [1.0, 3.5])
        assert grid.min_spacing == 1.0
        assert not grid.is_uniform
        assert grid.slice_extent(slice_tuple) == (3.0, 5.0, 6.0)
        assert np.allclose(np.asarray(grid.face_area(axis=0, slice_tuple=slice_tuple)), [[[8.0, 4.0], [12.0, 6.0]]])
        assert np.allclose(
            np.asarray(grid.cell_volume(slice_tuple)),
            [[[8.0, 4.0], [12.0, 6.0]], [[16.0, 8.0], [24.0, 12.0]]],
        )

    def test_uniform_spacing_raises_for_nonuniform_grid(self):
        """Scalar-resolution compatibility paths must fail loudly for non-uniform grids."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0]),
            y_edges=jnp.asarray([0.0, 1.0, 2.0]),
            z_edges=jnp.asarray([0.0, 1.0, 2.0]),
        )

        with pytest.raises(ValueError, match="requires a uniform grid"):
            _ = grid.uniform_spacing

    def test_coord_to_index_snapping_rules(self):
        """Coordinate snapping is centralized so placement code does not open-code searchsorted."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),
            y_edges=jnp.asarray([0.0, 1.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )

        assert grid.coord_to_index(0, 2.2, snap="nearest") == 2
        assert grid.coord_to_index(0, 2.2, snap="lower") == 1
        assert grid.coord_to_index(0, 2.2, snap="upper") == 2

    def test_bounds_for_center_uses_physical_interval_centers(self):
        """Center snapping compares physical interval centers, not index centers."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),
            y_edges=jnp.asarray([0.0, 1.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )

        assert grid.bounds_for_center(axis=0, center=3.6, size=2) == (1, 3)

    def test_bounds_for_anchor_uses_physical_anchor_positions(self):
        """Relative placement can target physical lower/center/upper anchors."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),
            y_edges=jnp.asarray([0.0, 1.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )

        assert grid.anchor_coordinate(axis=0, bounds=(1, 3), position=-1.0) == 1.0
        assert grid.anchor_coordinate(axis=0, bounds=(1, 3), position=1.0) == 6.0
        assert grid.bounds_for_anchor(axis=0, size=1, anchor=3.1, position=-1.0) == (2, 3)

    def test_cfl_time_step_uses_min_spacing_per_axis(self):
        """The rectilinear CFL limit uses the smallest spacing on each axis."""
        grid = RectilinearGrid(
            x_edges=jnp.asarray([0.0, 2.0, 5.0]),
            y_edges=jnp.asarray([0.0, 1.0]),
            z_edges=jnp.asarray([0.0, 4.0]),
        )

        expected = 0.99 / (constants.c * np.sqrt(1 / 2.0**2 + 1 / 1.0**2 + 1 / 4.0**2))
        assert np.isclose(grid.cfl_time_step(0.99), expected)

    def test_is_uniform_uses_relative_tolerance_at_physical_scale(self):
        """Uniformity is detected relatively, so it is correct at nanometre scales and large sizes."""
        base = 25e-9  # photonic scale: absolute widths are ~1e-8, comparable to np.allclose's atol

        # A genuinely non-uniform nm-scale grid must NOT be classified uniform (the regression).
        idx = np.arange(40)
        widths = base * (1.0 + 0.05 * np.abs(idx - 19.5) / 19.5)  # 5% variation
        non_uniform = RectilinearGrid.custom(
            x_edges=jnp.asarray(np.concatenate([[0.0], np.cumsum(widths)]), dtype=jnp.float32),
            y_edges=jnp.arange(3) * base,
            z_edges=jnp.arange(3) * base,
        )
        assert not non_uniform.is_uniform

        # Truly uniform grids stay uniform at both fine spacing and large cell counts (float32
        # edge jitter must not flip them to non-uniform, which would break uniform_spacing()).
        assert RectilinearGrid.uniform((4000, 2, 2), base).is_uniform
        assert jnp.isclose(RectilinearGrid.uniform((40, 40, 40), base).uniform_spacing, base)

    @staticmethod
    def _symmetric_edges(widths):
        """Edge array from a palindromic width profile (mirror-symmetric about the center)."""
        widths = jnp.asarray(widths)
        return jnp.concatenate([jnp.zeros(1), jnp.cumsum(widths)])

    def test_reduce_symmetric_keeps_upper_half(self):
        """Reduction halves the symmetric axis and keeps the upper-half cell widths."""
        # palindromic widths about the center: [3, 2, 1, 1, 2, 3]
        x_edges = self._symmetric_edges([3.0, 2.0, 1.0, 1.0, 2.0, 3.0])
        grid = RectilinearGrid.custom(
            x_edges=x_edges,
            y_edges=jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
            z_edges=jnp.asarray([0.0, 0.5, 1.5]),
        )

        reduced = grid.reduce_symmetric((-1, 0, 0))

        # only x is reduced (6 -> 3); other axes unchanged
        assert reduced.shape == (3, 4, 2)
        assert np.allclose(np.asarray(reduced.cell_widths(0)), [1.0, 2.0, 3.0])
        assert np.allclose(np.asarray(reduced.cell_widths(0)), np.asarray(grid.cell_widths(0))[3:])
        assert np.allclose(np.asarray(reduced.y_edges), np.asarray(grid.y_edges))
        assert np.allclose(np.asarray(reduced.z_edges), np.asarray(grid.z_edges))

    def test_reduce_symmetric_no_symmetry_is_noop(self):
        """symmetry=(0, 0, 0) returns an equivalent grid."""
        grid = RectilinearGrid.custom(
            x_edges=self._symmetric_edges([2.0, 1.0, 1.0, 2.0]),
            y_edges=jnp.asarray([0.0, 1.0, 2.0]),
            z_edges=jnp.asarray([0.0, 1.0, 2.0]),
        )

        reduced = grid.reduce_symmetric((0, 0, 0))

        assert reduced.shape == grid.shape
        for axis in range(3):
            assert np.allclose(np.asarray(reduced.edges(axis)), np.asarray(grid.edges(axis)))

    def test_reduce_symmetric_odd_cell_count_raises(self):
        """An odd cell count on a symmetric axis cannot split down the middle."""
        grid = RectilinearGrid.custom(
            x_edges=jnp.asarray([0.0, 1.0, 2.0, 3.0]),  # 3 cells (odd)
            y_edges=jnp.asarray([0.0, 1.0, 2.0]),
            z_edges=jnp.asarray([0.0, 1.0, 2.0]),
        )

        with pytest.raises(ValueError, match="even number"):
            grid.reduce_symmetric((-1, 0, 0))

    def test_reduce_symmetric_asymmetric_widths_raise(self):
        """A non-palindromic width profile cannot be mirror-reconstructed."""
        grid = RectilinearGrid.custom(
            x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0, 10.0]),  # widths 1,2,3,4 (not symmetric)
            y_edges=jnp.asarray([0.0, 1.0, 2.0]),
            z_edges=jnp.asarray([0.0, 1.0, 2.0]),
        )

        with pytest.raises(ValueError, match="mirror-symmetric"):
            grid.reduce_symmetric((-1, 0, 0))


class TestQuasiUniformGrid:
    """Tests for QuasiUniformGrid.

    Covers:
    - Construction validation
    - resolve() edge arrays and parity guard
    - Per-axis convenience methods
    - Integration with SimulationConfig
    - Constraint placement via resolve_object_constraints
    """

    # ---------------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------------

    def test_construction_stores_spacings(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        assert g.dx == pytest.approx(10e-9)
        assert g.dy == pytest.approx(20e-9)
        assert g.dz == pytest.approx(30e-9)

    def test_construction_default_center_is_origin(self):
        g = QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=10e-9)
        assert g.center == (0, 0, 0)

    def test_construction_custom_center(self):
        g = QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=10e-9, center=(1e-6, 2e-6, 3e-6))
        assert g.center == (1e-6, 2e-6, 3e-6)

    def test_construction_nonpositive_dx_raises(self):
        with pytest.raises(ValueError, match="dx"):
            QuasiUniformGrid(dx=0.0, dy=10e-9, dz=10e-9)

    def test_construction_negative_dy_raises(self):
        with pytest.raises(ValueError, match="dy"):
            QuasiUniformGrid(dx=10e-9, dy=-5e-9, dz=10e-9)

    def test_construction_negative_dz_raises(self):
        with pytest.raises(ValueError, match="dz"):
            QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=-1e-9)

    # ---------------------------------------------------------------------------
    # resolve() — edge arrays
    # ---------------------------------------------------------------------------

    def test_resolve_returns_rectilinear_grid(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=40e-9)
        r = g.resolve((4, 6, 8))
        assert isinstance(r, RectilinearGrid)

    def test_resolve_shape_matches_request(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=40e-9)
        r = g.resolve((4, 6, 8))
        assert r.shape == (4, 6, 8)

    def test_resolve_x_edges_centered_at_origin(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=40e-9)
        r = g.resolve((4, 6, 8))
        x = np.asarray(r.x_edges)
        assert x[0] == pytest.approx(-20e-9)  # -4*10e-9/2
        assert x[-1] == pytest.approx(+20e-9)
        assert (x[0] + x[-1]) / 2 == pytest.approx(0.0)

    def test_resolve_y_edges_centered_at_origin(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=40e-9)
        r = g.resolve((4, 6, 8))
        y = np.asarray(r.y_edges)
        assert y[0] == pytest.approx(-60e-9)  # -6*20e-9/2
        assert y[-1] == pytest.approx(+60e-9)

    def test_resolve_z_edges_centered_at_origin(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=40e-9)
        r = g.resolve((4, 6, 8))
        z = np.asarray(r.z_edges)
        assert z[0] == pytest.approx(-160e-9)  # -8*40e-9/2
        assert z[-1] == pytest.approx(+160e-9)

    def test_resolve_x_spacing_is_uniform(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=40e-9)
        r = g.resolve((4, 6, 8))
        dx = np.diff(np.asarray(r.x_edges))
        assert np.allclose(dx, 10e-9)

    def test_resolve_axes_independent(self):
        """Each axis uses its own spacing, not a shared one."""
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=40e-9)
        r = g.resolve((4, 6, 8))
        assert np.allclose(np.diff(np.asarray(r.x_edges)), 10e-9)
        assert np.allclose(np.diff(np.asarray(r.y_edges)), 20e-9)
        assert np.allclose(np.diff(np.asarray(r.z_edges)), 40e-9)

    def test_resolve_with_nonzero_center(self):
        cx, cy, cz = 1e-6, 2e-6, 3e-6
        g = QuasiUniformGrid(dx=100e-9, dy=100e-9, dz=100e-9, center=(cx, cy, cz))
        r = g.resolve((4, 4, 4))
        x = np.asarray(r.x_edges)
        assert (x[0] + x[-1]) / 2 == pytest.approx(cx)
        y = np.asarray(r.y_edges)
        assert (y[0] + y[-1]) / 2 == pytest.approx(cy)
        z = np.asarray(r.z_edges)
        assert (z[0] + z[-1]) / 2 == pytest.approx(cz)

    # ---------------------------------------------------------------------------
    # resolve() — parity guard
    # ---------------------------------------------------------------------------

    def test_resolve_odd_x_raises(self):
        g = QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=10e-9)
        with pytest.raises(ValueError, match="Axis 0"):
            g.resolve((3, 4, 4))

    def test_resolve_odd_y_raises(self):
        g = QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=10e-9)
        with pytest.raises(ValueError, match="Axis 1"):
            g.resolve((4, 5, 4))

    def test_resolve_odd_z_raises(self):
        g = QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=10e-9)
        with pytest.raises(ValueError, match="Axis 2"):
            g.resolve((4, 4, 7))

    def test_resolve_all_even_does_not_raise(self):
        g = QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=10e-9)
        g.resolve((2, 4, 6))  # should not raise

    # ---------------------------------------------------------------------------
    # is_uniform / min_spacing
    # ---------------------------------------------------------------------------

    def test_is_uniform_true_when_all_equal(self):
        g = QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=10e-9)
        assert g.is_uniform is True

    def test_is_uniform_false_when_unequal(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=10e-9)
        assert g.is_uniform is False

    def test_min_spacing_returns_smallest(self):
        g = QuasiUniformGrid(dx=30e-9, dy=10e-9, dz=20e-9)
        assert g.min_spacing == pytest.approx(10e-9)

    def test_axis_spacing_returns_per_axis(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        assert g.axis_spacing(0) == pytest.approx(10e-9)
        assert g.axis_spacing(1) == pytest.approx(20e-9)
        assert g.axis_spacing(2) == pytest.approx(30e-9)

    # ---------------------------------------------------------------------------
    # axis_extent / slice_extent
    # ---------------------------------------------------------------------------

    def test_axis_extent_correct(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        assert g.axis_extent(0, (0, 5)) == pytest.approx(5 * 10e-9)
        assert g.axis_extent(1, (2, 8)) == pytest.approx(6 * 20e-9)
        assert g.axis_extent(2, (1, 4)) == pytest.approx(3 * 30e-9)

    def test_slice_extent_correct(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        ex, ey, ez = g.slice_extent(((0, 4), (0, 6), (0, 8)))
        assert ex == pytest.approx(4 * 10e-9)
        assert ey == pytest.approx(6 * 20e-9)
        assert ez == pytest.approx(8 * 30e-9)

    # ---------------------------------------------------------------------------
    # coord_to_index / length_to_cell_count
    # ---------------------------------------------------------------------------

    def test_coord_to_index_at_center_is_zero(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        assert g.coord_to_index(0, 0.0) == 0
        assert g.coord_to_index(1, 0.0) == 0
        assert g.coord_to_index(2, 0.0) == 0

    def test_coord_to_index_positive_offset(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        assert g.coord_to_index(0, 50e-9) == 5  # 50e-9 / 10e-9
        assert g.coord_to_index(1, 60e-9) == 3  # 60e-9 / 20e-9

    def test_coord_to_index_negative_offset(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        assert g.coord_to_index(0, -30e-9) == -3

    def test_coord_to_index_snap_lower(self):
        g = QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=10e-9)
        assert g.coord_to_index(0, 15e-9, snap="lower") == 1  # floor(1.5)

    def test_coord_to_index_snap_upper(self):
        g = QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=10e-9)
        assert g.coord_to_index(0, 15e-9, snap="upper") == 2  # ceil(1.5)

    def test_coord_to_index_invalid_snap_raises(self):
        g = QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=10e-9)
        with pytest.raises(ValueError, match="snap"):
            g.coord_to_index(0, 0.0, snap="invalid")

    def test_length_to_cell_count(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        assert g.length_to_cell_count(0, 50e-9) == 5
        assert g.length_to_cell_count(1, 60e-9) == 3
        assert g.length_to_cell_count(2, 90e-9) == 3

    # ---------------------------------------------------------------------------
    # cell_volume / face_area
    # ---------------------------------------------------------------------------

    def test_cell_volume_shape_and_value(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        vol = g.cell_volume(((0, 2), (0, 3), (0, 4)))
        assert vol.shape == (2, 3, 4)
        assert jnp.allclose(vol, 10e-9 * 20e-9 * 30e-9)

    def test_face_area_normal_to_z(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        area = g.face_area(2, ((0, 2), (0, 3), (0, 4)))
        # transverse axes are 0 and 1 → dx * dy
        assert area.shape == (2, 3)
        assert jnp.allclose(area, 10e-9 * 20e-9)

    def test_face_area_normal_to_x(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        area = g.face_area(0, ((0, 2), (0, 3), (0, 4)))
        # transverse axes are 1 and 2 → dy * dz
        assert area.shape == (3, 4)
        assert jnp.allclose(area, 20e-9 * 30e-9)

    # ---------------------------------------------------------------------------
    # SimulationConfig integration
    # ---------------------------------------------------------------------------

    def test_config_time_step_uses_min_spacing(self):
        import math

        from fdtdx import constants

        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        config = SimulationConfig(grid=g, time=10e-15)
        courant = config.courant_factor / math.sqrt(3)
        expected = courant * 10e-9 / constants.c
        assert config.time_step_duration == pytest.approx(expected)

    def test_config_uniform_spacing_raises_when_anisotropic(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=10e-9)
        config = SimulationConfig(grid=g, time=10e-15)
        with pytest.raises(ValueError, match="no single uniform spacing"):
            config.uniform_spacing()

    def test_config_uniform_spacing_succeeds_when_isotropic(self):
        g = QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=10e-9)
        config = SimulationConfig(grid=g, time=10e-15)
        assert config.uniform_spacing() == pytest.approx(10e-9)

    def test_config_resolved_grid_none_before_placement(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        config = SimulationConfig(grid=g, time=10e-15)
        assert config.resolved_grid is None

    def test_config_resolve_grid_returns_rectilinear(self):
        g = QuasiUniformGrid(dx=10e-9, dy=20e-9, dz=30e-9)
        config = SimulationConfig(grid=g, time=10e-15)
        r = config.resolve_grid((4, 6, 8))
        assert isinstance(r, RectilinearGrid)
        assert r.shape == (4, 6, 8)

    # ---------------------------------------------------------------------------
    # resolve_object_constraints integration
    # ---------------------------------------------------------------------------

    @pytest.fixture
    def material(self):
        return Material()

    @pytest.fixture
    def anisotropic_config(self):
        """dx=dy=100 nm, dz=200 nm — genuinely anisotropic."""
        return SimulationConfig(
            grid=QuasiUniformGrid(dx=100e-9, dy=100e-9, dz=200e-9),
            time=10e-15,
        )

    @pytest.fixture
    def isotropic_quasi_config(self):
        """QuasiUniformGrid with equal spacings — treated as uniform by the grid guards."""
        return SimulationConfig(
            grid=QuasiUniformGrid(dx=100e-9, dy=100e-9, dz=100e-9),
            time=10e-15,
        )

    def test_placement_resolves_grid(self, anisotropic_config, material):
        """resolve_object_constraints pins a RectilinearGrid onto config.

        Using partial_grid_shape for the volume bypasses the real→cell conversion
        that would call coord_to_index with a length larger than the domain on a
        non-uniform grid.
        """
        volume = SimulationVolume(partial_grid_shape=(20, 20, 10))
        slices, errors = resolve_object_constraints(objects=[volume], constraints=[], config=anisotropic_config)
        assert errors[volume.name] is None
        assert slices[volume.name] == ((0, 20), (0, 20), (0, 10))

    def test_placement_object_sizes_in_cells(self, anisotropic_config, material):
        """Objects sized in metres resolve to different cell counts per axis.

        1 µm along z = 1e-6 / 200e-9 = 5 cells.
        """
        volume = SimulationVolume(partial_grid_shape=(20, 20, 10))
        slab = UniformMaterialObject(
            name="slab",
            partial_real_shape=(None, None, 1e-6),
            material=material,
        )
        constraints = [slab.place_relative_to(volume, axes=2, own_positions=-1, other_positions=-1)]
        slices, errors = resolve_object_constraints(
            objects=[volume, slab], constraints=constraints, config=anisotropic_config
        )
        assert errors["slab"] is None
        assert slices["slab"][2] == (0, 5)

    def test_placement_object_sizes_xy(self, anisotropic_config, material):
        """x/y sizes resolve using dx=dy=100e-9.

        1 µm along x/y = 1e-6 / 100e-9 = 10 cells.
        """
        volume = SimulationVolume(partial_grid_shape=(20, 20, 10))
        rod = UniformMaterialObject(
            name="rod",
            partial_real_shape=(1e-6, 1e-6, None),
            material=material,
        )
        constraints = [rod.place_at_center(volume, axes=(0, 1, 2))]
        slices, errors = resolve_object_constraints(
            objects=[volume, rod], constraints=constraints, config=anisotropic_config
        )
        assert errors["rod"] is None
        x0, x1 = slices["rod"][0]
        y0, y1 = slices["rod"][1]
        assert x1 - x0 == 10
        assert y1 - y0 == 10

    def test_placement_centered_domain(self, anisotropic_config, material):
        """A box filling the volume occupies [0, N] on every axis."""
        volume = SimulationVolume(partial_grid_shape=(20, 20, 10))
        box = UniformMaterialObject(
            name="box",
            partial_grid_shape=(20, 20, 10),
            material=material,
        )
        pos_c, size_c = box.same_position_and_size(volume)
        slices, errors = resolve_object_constraints(
            objects=[volume, box], constraints=[pos_c, size_c], config=anisotropic_config
        )
        assert errors["box"] is None
        assert slices["box"] == ((0, 20), (0, 20), (0, 10))

    def test_placement_grid_coordinate_constraint(self, isotropic_quasi_config, material):
        """GridCoordinateConstraint works when all spacings are equal.

        The non-uniform guard rejects index-space constraints on stretched grids;
        an isotropic QuasiUniformGrid passes as uniform.
        """
        volume = SimulationVolume(partial_grid_shape=(20, 20, 20))
        obj = UniformMaterialObject(
            name="obj",
            partial_grid_shape=(4, 4, 4),
            material=material,
        )
        c = GridCoordinateConstraint(object="obj", axes=(0, 2), sides=("-", "-"), coordinates=(6, 3))
        slices, errors = resolve_object_constraints(
            objects=[volume, obj], constraints=[c], config=isotropic_quasi_config
        )
        assert errors["obj"] is None
        assert slices["obj"][0] == (6, 10)
        assert slices["obj"][2] == (3, 7)

    def test_isotropic_quasi_uniform_matches_uniform_grid(self):
        """QuasiUniformGrid with equal spacings resolves identically to UniformGrid."""
        from fdtdx.core.grid import UniformGrid

        spacing = 100e-9
        shape = (4, 4, 4)
        quasi = QuasiUniformGrid(dx=spacing, dy=spacing, dz=spacing)
        uniform = UniformGrid(spacing=spacing)
        r_quasi = quasi.resolve(shape)
        r_uniform = uniform.resolve(shape)
        assert jnp.allclose(r_quasi.x_edges, r_uniform.x_edges)
        assert jnp.allclose(r_quasi.y_edges, r_uniform.y_edges)
        assert jnp.allclose(r_quasi.z_edges, r_uniform.z_edges)


def test_calculate_spatial_offsets_yee_basic():
    """Test basic functionality of calculate_spatial_offsets_yee"""
    offset_E, offset_H = calculate_spatial_offsets_yee()

    # Check shapes
    assert offset_E.shape == (3, 1, 1, 1, 3)
    assert offset_H.shape == (3, 1, 1, 1, 3)

    # Check E field offsets
    expected_E = jnp.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    assert jnp.allclose(offset_E.squeeze(), expected_E)

    # Check H field offsets
    expected_H = jnp.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    assert jnp.allclose(offset_H.squeeze(), expected_H)


def test_calculate_time_offset_yee_basic():
    """Test basic functionality of calculate_time_offset_yee"""
    center = jnp.array([1.0, 2.0])
    wave_vector = jnp.array([1.0, 0.0, 0.0])
    inv_permittivities = jnp.ones((4, 4, 1))
    inv_permeabilities = 1.0
    resolution = 0.1
    time_step_duration = 1e-15

    time_offset_E, time_offset_H = calculate_time_offset_yee(
        center, wave_vector, inv_permittivities, inv_permeabilities, resolution, time_step_duration
    )

    # Check shapes
    assert time_offset_E.shape == (3, 4, 4, 1)
    assert time_offset_H.shape == (3, 4, 4, 1)

    # Check that results are finite
    assert jnp.all(jnp.isfinite(time_offset_E))
    assert jnp.all(jnp.isfinite(time_offset_H))


def test_calculate_time_offset_yee_with_effective_index():
    """Test calculate_time_offset_yee with effective_index parameter"""
    center = jnp.array([0.0, 0.0])
    wave_vector = jnp.array([1.0, 0.0, 0.0])
    inv_permittivities = jnp.ones((2, 2, 1))
    inv_permeabilities = 1.0
    resolution = 0.1
    time_step_duration = 1e-15
    effective_index = 1.5

    time_offset_E, time_offset_H = calculate_time_offset_yee(
        center, wave_vector, inv_permittivities, inv_permeabilities, resolution, time_step_duration, effective_index
    )

    # Check shapes
    assert time_offset_E.shape == (3, 2, 2, 1)
    assert time_offset_H.shape == (3, 2, 2, 1)

    # Check that results are finite
    assert jnp.all(jnp.isfinite(time_offset_E))
    assert jnp.all(jnp.isfinite(time_offset_H))


def test_calculate_time_offset_yee_uniform_coordinates_match_scalar_path():
    """A uniform RectilinearGrid is a metric-equivalent replacement for scalar spacing."""
    spacing = 0.2
    shape = (3, 3, 1)
    center = jnp.array([1.0, 1.0])
    wave_vector = jnp.array([0.3, 0.4, -0.5])
    wave_vector = wave_vector / jnp.linalg.norm(wave_vector)
    inv_permittivities = jnp.ones(shape)
    inv_permeabilities = 1.0
    time_step_duration = 1e-15

    scalar_E, scalar_H = calculate_time_offset_yee(
        center=center,
        wave_vector=wave_vector,
        inv_permittivities=inv_permittivities,
        inv_permeabilities=inv_permeabilities,
        resolution=spacing,
        time_step_duration=time_step_duration,
    )
    grid_E, grid_H = calculate_time_offset_yee(
        center=center,
        wave_vector=wave_vector,
        inv_permittivities=inv_permittivities,
        inv_permeabilities=inv_permeabilities,
        resolution=spacing,
        time_step_duration=time_step_duration,
        coordinate_edges=(
            spacing * jnp.arange(shape[0] + 1),
            spacing * jnp.arange(shape[1] + 1),
            spacing * jnp.arange(shape[2] + 1),
        ),
        center_physical=jnp.array([center[0] * spacing, center[1] * spacing, 0.0]),
    )

    assert jnp.allclose(grid_E, scalar_E, rtol=1e-6, atol=0.2)
    assert jnp.allclose(grid_H, scalar_H, rtol=1e-6, atol=0.2)


def test_calculate_time_offset_yee_invalid_permittivity_shape():
    """Test that invalid permittivity shapes raise exceptions"""
    center = jnp.array([0.0, 0.0])
    wave_vector = jnp.array([1.0, 0.0, 0.0])
    inv_permeabilities = 1.0
    resolution = 0.1
    time_step_duration = 1e-15

    # Test with 1D array
    inv_permittivities_1d = jnp.ones((4,))
    with pytest.raises(Exception):
        calculate_time_offset_yee(
            center, wave_vector, inv_permittivities_1d, inv_permeabilities, resolution, time_step_duration
        )

    # Test with 4D array (but valid shape - triggers different code path)
    inv_permittivities_4d = jnp.ones((2, 2, 2, 2))
    with pytest.raises(Exception):
        calculate_time_offset_yee(
            center, wave_vector, inv_permittivities_4d, inv_permeabilities, resolution, time_step_duration
        )


def test_calculate_time_offset_yee_no_unit_spatial_axis():
    """Test that spatial shape without a unit axis raises exception."""
    center = jnp.array([0.0, 0.0])
    wave_vector = jnp.array([1.0, 0.0, 0.0])
    inv_permeabilities = 1.0
    resolution = 0.1
    time_step_duration = 1e-15

    # 3D array with no axis equal to 1
    inv_permittivities = jnp.ones((2, 3, 4))
    with pytest.raises(Exception, match="Expected one spatial axis to be one"):
        calculate_time_offset_yee(
            center, wave_vector, inv_permittivities, inv_permeabilities, resolution, time_step_duration
        )


def test_calculate_time_offset_yee_array_permeabilities():
    """Test calculate_time_offset_yee with array permeabilities"""
    center = jnp.array([0.0, 0.0])
    wave_vector = jnp.array([1.0, 0.0, 0.0])
    inv_permittivities = jnp.ones((3, 3, 1))
    inv_permeabilities = jnp.ones((3, 3, 1)) * 2.0
    resolution = 0.1
    time_step_duration = 1e-15

    time_offset_E, time_offset_H = calculate_time_offset_yee(
        center, wave_vector, inv_permittivities, inv_permeabilities, resolution, time_step_duration
    )

    # Check shapes
    assert time_offset_E.shape == (3, 3, 3, 1)
    assert time_offset_H.shape == (3, 3, 3, 1)

    # Check that results are finite
    assert jnp.all(jnp.isfinite(time_offset_E))
    assert jnp.all(jnp.isfinite(time_offset_H))


def test_calculate_time_offset_yee_anisotropic_with_polarization():
    """Test calculate_time_offset_yee with anisotropic materials and polarization weighting.

    For anisotropic materials, the effective permittivity should be weighted by the
    polarization direction: inv_eps_eff = |p_x|^2 * inv_eps_x + |p_y|^2 * inv_eps_y + |p_z|^2 * inv_eps_z
    """

    # Fail for now, until NotImplemented exception removed (grid.py line 85)
    with pytest.raises(Exception):
        center = jnp.array([1.0, 1.0])
        wave_vector = jnp.array([1.0, 0.0, 0.0])  # propagating along x
        resolution = 0.1
        time_step_duration = 1e-15

        # Anisotropic permittivity: shape (3, Nx, Ny, Nz)
        # eps_x = 4, eps_y = 9, eps_z = 16  -> inv_eps_x = 0.25, inv_eps_y = 1/9, inv_eps_z = 0.0625
        inv_perm_x = 0.25 * jnp.ones((3, 3, 1))
        inv_perm_y = (1 / 9) * jnp.ones((3, 3, 1))
        inv_perm_z = 0.0625 * jnp.ones((3, 3, 1))
        inv_permittivities = jnp.stack([inv_perm_x, inv_perm_y, inv_perm_z], axis=0)

        inv_permeabilities = 1.0

        # Test 1: Polarization along y (perpendicular to wave_vector)
        e_pol_y = jnp.array([0.0, 1.0, 0.0])
        h_pol_z = jnp.array([0.0, 0.0, 1.0])

        time_offset_E_y, _time_offset_H_y = calculate_time_offset_yee(
            center,
            wave_vector,
            inv_permittivities,
            inv_permeabilities,
            resolution,
            time_step_duration,
            e_polarization=e_pol_y,
            h_polarization=h_pol_z,
        )

        # Test 2: Polarization along z (perpendicular to wave_vector)
        e_pol_z = jnp.array([0.0, 0.0, 1.0])
        h_pol_y = jnp.array([0.0, 1.0, 0.0])

        time_offset_E_z, _time_offset_H_z = calculate_time_offset_yee(
            center,
            wave_vector,
            inv_permittivities,
            inv_permeabilities,
            resolution,
            time_step_duration,
            e_polarization=e_pol_z,
            h_polarization=h_pol_y,
        )

        # Check shapes
        assert time_offset_E_y.shape == (3, 3, 3, 1)
        assert time_offset_E_z.shape == (3, 3, 3, 1)

        # Check that results are finite
        assert jnp.all(jnp.isfinite(time_offset_E_y))
        assert jnp.all(jnp.isfinite(time_offset_E_z))

        # The time offsets should be different because different polarizations
        # see different effective permittivities in anisotropic media.
        # E polarized along y sees eps_y = 9 (n = 3)
        # E polarized along z sees eps_z = 16 (n = 4)
        # Higher refractive index -> slower velocity -> larger time offset magnitude
        assert not jnp.allclose(time_offset_E_y, time_offset_E_z)

        # For y-polarization, velocity is faster (n=3) so time offsets are smaller in magnitude
        # For z-polarization, velocity is slower (n=4) so time offsets are larger in magnitude
        # The ratio of time offsets should reflect the ratio of refractive indices (4/3)
        ratio = jnp.abs(time_offset_E_z).mean() / jnp.abs(time_offset_E_y).mean()
        expected_ratio = 4.0 / 3.0
        assert jnp.isclose(ratio, expected_ratio, rtol=0.01)


def test_calculate_time_offset_yee_anisotropic_requires_polarization():
    """Test that anisotropic materials raise an error when polarization is not provided."""

    # Fail for now, until NotImplemented exception removed (grid.py line 85)
    with pytest.raises(Exception):
        center = jnp.array([1.0, 1.0])
        wave_vector = jnp.array([1.0, 0.0, 0.0])
        resolution = 0.1
        time_step_duration = 1e-15

        # Anisotropic permittivity: shape (3, Nx, Ny, Nz)
        inv_perm_x = 0.25 * jnp.ones((3, 3, 1))
        inv_perm_y = (1 / 9) * jnp.ones((3, 3, 1))
        inv_perm_z = 0.0625 * jnp.ones((3, 3, 1))
        inv_permittivities = jnp.stack([inv_perm_x, inv_perm_y, inv_perm_z], axis=0)

        inv_permeabilities = 1.0

        # Without polarization, should raise ValueError for anisotropic materials
        with pytest.raises(ValueError, match="e_polarization is required"):
            calculate_time_offset_yee(
                center,
                wave_vector,
                inv_permittivities,
                inv_permeabilities,
                resolution,
                time_step_duration,
            )


def test_calculate_time_offset_yee_4d_permeability():
    """Test calculate_time_offset_yee with 4D anisotropic permeability raises NotImplementedError."""
    # This code path raises NotImplementedError until full anisotropic
    # permeability support is added (grid.py line 119).
    center = jnp.array([1.0, 1.0])
    wave_vector = jnp.array([1.0, 0.0, 0.0])
    resolution = 0.1
    time_step_duration = 1e-15

    # Isotropic permittivity (1D stack)
    inv_permittivities = jnp.ones((1, 3, 3, 1))

    # Anisotropic permeability: shape (3, Nx, Ny, Nz)
    inv_perm_x = 1.0 * jnp.ones((3, 3, 1))
    inv_perm_y = 0.5 * jnp.ones((3, 3, 1))
    inv_perm_z = 0.25 * jnp.ones((3, 3, 1))
    inv_permeabilities = jnp.stack([inv_perm_x, inv_perm_y, inv_perm_z], axis=0)

    e_pol = jnp.array([0.0, 1.0, 0.0])
    h_pol = jnp.array([0.0, 0.0, 1.0])

    with pytest.raises(Exception):
        calculate_time_offset_yee(
            center,
            wave_vector,
            inv_permittivities,
            inv_permeabilities,
            resolution,
            time_step_duration,
            e_polarization=e_pol,
            h_polarization=h_pol,
        )


def test_calculate_time_offset_yee_uses_coordinate_edges():
    """Explicit rectilinear coordinates produce physical travel offsets."""
    center = jnp.asarray([0.0, 0.0])
    wave_vector = jnp.asarray([1.0, 0.0, 0.0])
    inv_permittivities = jnp.ones((1, 2, 1, 1))
    time_step_duration = 1.0 / constants.c

    time_offset_E, _time_offset_H = calculate_time_offset_yee(
        center=center,
        wave_vector=wave_vector,
        inv_permittivities=inv_permittivities,
        inv_permeabilities=1.0,
        resolution=1.0,
        time_step_duration=time_step_duration,
        coordinate_edges=(
            jnp.asarray([0.0, 1.0, 3.0]),
            jnp.asarray([0.0, 1.0]),
            jnp.asarray([0.0, 1.0]),
        ),
        center_physical=jnp.asarray([0.0, 0.0, 0.0]),
    )

    assert jnp.allclose(time_offset_E[0, :, 0, 0], jnp.asarray([-0.5, -2.0]))


def test_calculate_time_offset_yee_4d_permeability_without_h_polarization():
    """Test that 4D isotropic permeability works without h_polarization (uses [0] slice)."""
    center = jnp.array([1.0, 1.0])
    wave_vector = jnp.array([1.0, 0.0, 0.0])
    resolution = 0.1
    time_step_duration = 1e-15

    # Isotropic permittivity (1D stack)
    inv_permittivities = jnp.ones((1, 3, 3, 1))

    # 4D permeability array (isotropic, shape (1, ...))
    inv_permeabilities = jnp.ones((1, 3, 3, 1))

    e_pol = jnp.array([0.0, 1.0, 0.0])

    time_offset_E, time_offset_H = calculate_time_offset_yee(
        center,
        wave_vector,
        inv_permittivities,
        inv_permeabilities,
        resolution,
        time_step_duration,
        e_polarization=e_pol,
        # h_polarization not provided - should still work
    )
    assert time_offset_E.shape == (3, 3, 3, 1)
    assert time_offset_H.shape == (3, 3, 3, 1)
    assert jnp.all(jnp.isfinite(time_offset_E))


def test_calculate_time_offset_yee_4d_permittivity_without_e_polarization():
    """Test that isotropic 4D permittivity works without e_polarization (uses [0] slice)."""
    center = jnp.array([1.0, 1.0])
    wave_vector = jnp.array([1.0, 0.0, 0.0])
    resolution = 0.1
    time_step_duration = 1e-15

    # Isotropic 4D permittivity (shape (1, Nx, Ny, Nz))
    inv_permittivities = jnp.ones((1, 3, 3, 1))

    inv_permeabilities = 1.0

    time_offset_E, time_offset_H = calculate_time_offset_yee(
        center,
        wave_vector,
        inv_permittivities,
        inv_permeabilities,
        resolution,
        time_step_duration,
        # e_polarization not provided - should still work
    )
    assert time_offset_E.shape == (3, 3, 3, 1)
    assert time_offset_H.shape == (3, 3, 3, 1)
    assert jnp.all(jnp.isfinite(time_offset_E))


def test_calculate_time_offset_yee_isotropic_4d_permeability_with_h_polarization():
    """Test 4D isotropic permeability with h_polarization executes h_pol_squared path."""
    center = jnp.array([1.0, 1.0])
    wave_vector = jnp.array([1.0, 0.0, 0.0])
    resolution = 0.1
    time_step_duration = 1e-15

    # Isotropic 4D permittivity
    inv_permittivities = jnp.ones((1, 3, 3, 1))

    # Isotropic 4D permeability (shape (1, ...) bypasses anisotropic check)
    inv_permeabilities = jnp.ones((1, 3, 3, 1)) * 0.5

    e_pol = jnp.array([0.0, 1.0, 0.0])
    h_pol = jnp.array([0.0, 0.0, 1.0])

    time_offset_E, time_offset_H = calculate_time_offset_yee(
        center,
        wave_vector,
        inv_permittivities,
        inv_permeabilities,
        resolution,
        time_step_duration,
        e_polarization=e_pol,
        h_polarization=h_pol,
    )

    assert time_offset_E.shape == (3, 3, 3, 1)
    assert time_offset_H.shape == (3, 3, 3, 1)
    assert jnp.all(jnp.isfinite(time_offset_E))
    assert jnp.all(jnp.isfinite(time_offset_H))


def test_polygon_to_mask_basic_square():
    """Test polygon_to_mask with a basic square polygon"""
    boundary = (0.0, 0.0, 2.0, 2.0)
    resolution = 0.5
    # Square polygon
    polygon_vertices = np.array(
        [
            [0.5, 0.5],
            [1.5, 0.5],
            [1.5, 1.5],
            [0.5, 1.5],
            [0.5, 0.5],  # Close the polygon
        ]
    )

    mask = polygon_to_mask(boundary, resolution, polygon_vertices)

    # Check that mask has correct shape
    expected_shape = (5, 5)  # (2.0-0.0)/0.5 + 1 = 5
    assert mask.shape == expected_shape

    # Check that mask is boolean
    assert mask.dtype == bool

    # Check that there are both True and False values
    assert np.any(mask)
    assert not np.all(mask)


def test_polygon_to_mask_triangle():
    """Test polygon_to_mask with a triangle polygon"""
    boundary = (0.0, 0.0, 3.0, 3.0)
    resolution = 1.0
    # Triangle polygon
    polygon_vertices = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.5, 2.0],
            [1.0, 1.0],  # Close the polygon
        ]
    )

    mask = polygon_to_mask(boundary, resolution, polygon_vertices)

    # Check that mask has correct shape
    expected_shape = (4, 4)  # (3.0-0.0)/1.0 + 1 = 4
    assert mask.shape == expected_shape

    # Check that mask is boolean
    assert mask.dtype == bool


def test_polygon_to_mask_empty_polygon():
    """Test polygon_to_mask with a very small polygon that might not contain grid points"""
    boundary = (0.0, 0.0, 2.0, 2.0)
    resolution = 1.0
    # Very small triangle
    polygon_vertices = np.array([[0.1, 0.1], [0.2, 0.1], [0.15, 0.2], [0.1, 0.1]])

    mask = polygon_to_mask(boundary, resolution, polygon_vertices)

    # Should still return a valid mask
    assert mask.shape == (3, 3)  # (2.0-0.0)/1.0 + 1 = 3
    assert mask.dtype == bool


def test_polygon_to_mask_invalid_vertices_shape():
    """Test polygon_to_mask with invalid vertex shapes"""
    boundary = (0.0, 0.0, 2.0, 2.0)
    resolution = 0.5

    # Test with 1D array
    polygon_vertices_1d = np.array([0.5, 0.5, 1.5, 1.5])
    with pytest.raises(AssertionError):
        polygon_to_mask(boundary, resolution, polygon_vertices_1d)

    # Test with 3D coordinates
    polygon_vertices_3d = np.array([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [1.5, 1.5, 0.0], [0.5, 1.5, 0.0]])
    with pytest.raises(AssertionError):
        polygon_to_mask(boundary, resolution, polygon_vertices_3d)


def test_polygon_to_mask_large_polygon():
    """Test polygon_to_mask with a polygon that covers the entire boundary"""
    boundary = (0.0, 0.0, 2.0, 2.0)
    resolution = 1.0
    # Large polygon that covers entire boundary
    polygon_vertices = np.array([[-1.0, -1.0], [3.0, -1.0], [3.0, 3.0], [-1.0, 3.0], [-1.0, -1.0]])

    mask = polygon_to_mask(boundary, resolution, polygon_vertices)

    # All points should be inside
    assert np.all(mask)
    assert mask.shape == (3, 3)
