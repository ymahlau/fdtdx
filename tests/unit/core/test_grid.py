import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx import constants
from fdtdx.core.grid import GridSpec, calculate_spatial_offsets_yee, calculate_time_offset_yee, polygon_to_mask


class TestGridSpec:
    """Tests for the canonical rectilinear grid representation."""

    def test_uniform_constructor_stores_edges(self):
        """Uniform grids are represented as ordinary rectilinear edge arrays."""
        grid = GridSpec.uniform(shape=(2, 3, 4), spacing=0.5, origin=(1.0, 2.0, 3.0))

        assert grid.shape == (2, 3, 4)
        assert np.allclose(np.asarray(grid.x_edges), [1.0, 1.5, 2.0])
        assert np.allclose(np.asarray(grid.y_edges), [2.0, 2.5, 3.0, 3.5])
        assert np.allclose(np.asarray(grid.z_edges), [3.0, 3.5, 4.0, 4.5, 5.0])
        assert grid.is_uniform
        assert grid.uniform_spacing == 0.5

    def test_nonuniform_grid_metrics(self):
        """Non-uniform grids derive widths, centers, extents, areas, and volumes from edges."""
        grid = GridSpec(
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
        grid = GridSpec(
            x_edges=jnp.asarray([0.0, 1.0, 3.0]),
            y_edges=jnp.asarray([0.0, 1.0, 2.0]),
            z_edges=jnp.asarray([0.0, 1.0, 2.0]),
        )

        with pytest.raises(ValueError, match="requires a uniform grid"):
            _ = grid.uniform_spacing

    def test_coord_to_index_snapping_rules(self):
        """Coordinate snapping is centralized so placement code does not open-code searchsorted."""
        grid = GridSpec(
            x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),
            y_edges=jnp.asarray([0.0, 1.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )

        assert grid.coord_to_index(0, 2.2, snap="nearest") == 2
        assert grid.coord_to_index(0, 2.2, snap="lower") == 1
        assert grid.coord_to_index(0, 2.2, snap="upper") == 2

    def test_bounds_for_center_uses_physical_interval_centers(self):
        """Center snapping compares physical interval centers, not index centers."""
        grid = GridSpec(
            x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),
            y_edges=jnp.asarray([0.0, 1.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )

        assert grid.bounds_for_center(axis=0, center=3.6, size=2) == (1, 3)

    def test_bounds_for_anchor_uses_physical_anchor_positions(self):
        """Relative placement can target physical lower/center/upper anchors."""
        grid = GridSpec(
            x_edges=jnp.asarray([0.0, 1.0, 3.0, 6.0]),
            y_edges=jnp.asarray([0.0, 1.0]),
            z_edges=jnp.asarray([0.0, 1.0]),
        )

        assert grid.anchor_coordinate(axis=0, bounds=(1, 3), position=-1.0) == 1.0
        assert grid.anchor_coordinate(axis=0, bounds=(1, 3), position=1.0) == 6.0
        assert grid.bounds_for_anchor(axis=0, size=1, anchor=3.1, position=-1.0) == (2, 3)

    def test_cfl_time_step_uses_min_spacing_per_axis(self):
        """The rectilinear CFL limit uses the smallest spacing on each axis."""
        grid = GridSpec(
            x_edges=jnp.asarray([0.0, 2.0, 5.0]),
            y_edges=jnp.asarray([0.0, 1.0]),
            z_edges=jnp.asarray([0.0, 4.0]),
        )

        expected = 0.99 / (constants.c * np.sqrt(1 / 2.0**2 + 1 / 1.0**2 + 1 / 4.0**2))
        assert np.isclose(grid.cfl_time_step(0.99), expected)


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
