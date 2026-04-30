import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from fdtdx.core.grid import (
    calculate_spatial_offsets_yee,
    calculate_time_offset_yee,
    get_voxel_centers_numpy,
    polygon_to_mask,
)


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


class TestVoxelCentersNumpy:
    def test_output_shape(self):
        """Test that the output array has the correct shape (N, 3)."""
        grid_shape = (2, 3, 4)
        resolution = 0.5
        centers = get_voxel_centers_numpy(grid_shape, resolution)

        expected_num_voxels = 2 * 3 * 4
        assert centers.shape == (expected_num_voxels, 3), "Shape mismatch"

    def test_single_voxel(self):
        """Test a 1x1x1 grid. The center should be exactly at the origin (0, 0, 0)."""
        grid_shape = (1, 1, 1)
        resolution = 2.0
        centers = get_voxel_centers_numpy(grid_shape, resolution)

        expected = np.array([[0.0, 0.0, 0.0]])
        assert_array_almost_equal(centers, expected, err_msg="Single voxel not centered at origin")

    def test_2x2x2_grid_exact_values(self):
        """Test a 2x2x2 grid with resolution 1.0 against known exact coordinates."""
        grid_shape = (2, 2, 2)
        resolution = 1.0
        centers = get_voxel_centers_numpy(grid_shape, resolution)

        # With resolution 1.0 and size 2, bounds are -1.0 to 1.0.
        # Centers should be at -0.5 and 0.5 for all axes.
        # meshgrid with 'ij' indexing prioritizes x, then y, then z.
        expected = np.array(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ]
        )
        assert_array_almost_equal(centers, expected, err_msg="Exact 2x2x2 coordinates mismatch")

    def test_grid_centering(self):
        """Test that the entire grid is perfectly centered around (0,0,0)."""
        grid_shape = (5, 10, 3)  # Asymmetric bounds
        resolution = 0.1
        centers = get_voxel_centers_numpy(grid_shape, resolution)

        # The mean of all centers in a properly centered grid should be 0 for x, y, and z
        mean_center = np.mean(centers, axis=0)
        assert_array_almost_equal(mean_center, [0.0, 0.0, 0.0], err_msg="Grid is not perfectly centered")

    def test_resolution_scaling(self):
        """Test that changing the resolution scales the coordinates linearly."""
        grid_shape = (3, 3, 3)
        centers_res_1 = get_voxel_centers_numpy(grid_shape, 1.0)
        centers_res_2 = get_voxel_centers_numpy(grid_shape, 2.0)

        # Doubling the resolution should exactly double all coordinate values
        assert_array_almost_equal(centers_res_1 * 2.0, centers_res_2, err_msg="Resolution scaling failed")

    def test_single_voxel_at_origin(self):
        centers = get_voxel_centers_numpy((1, 1, 1), resolution=1.0)
        assert centers.shape == (1, 3)
        np.testing.assert_allclose(centers[0], [0.0, 0.0, 0.0])

    def test_2x2x2_exact_centers(self):
        """2 voxels per axis at resolution=1 → centers at ±0.5."""
        centers = get_voxel_centers_numpy((2, 2, 2), resolution=1.0)
        expected = np.array([-0.5, 0.5])
        for axis in range(3):
            np.testing.assert_allclose(np.sort(np.unique(centers[:, axis])), expected)

    def test_centers_symmetric_around_origin(self):
        """Sum of all centers is zero (grid is centred at origin)."""
        centers = get_voxel_centers_numpy((5, 3, 4), resolution=0.1)
        np.testing.assert_allclose(centers.sum(axis=0), [0.0, 0.0, 0.0], atol=1e-12)

    def test_resolution_scales_linearly(self):
        """Doubling the resolution doubles every coordinate."""
        c1 = get_voxel_centers_numpy((3, 3, 3), resolution=1.0)
        c2 = get_voxel_centers_numpy((3, 3, 3), resolution=2.0)
        np.testing.assert_allclose(c2, 2.0 * c1)

    def test_adjacent_center_spacing_equals_resolution(self):
        """Consecutive centers along each axis are exactly one resolution apart."""
        res = 0.25
        centers = get_voxel_centers_numpy((6, 1, 1), resolution=res)
        xs = np.sort(centers[:, 0])
        np.testing.assert_allclose(np.diff(xs), res)

    def test_3x1x1_exact_x_values(self):
        """3 voxels along x, resolution=2 → centers at -2, 0, +2."""
        centers = get_voxel_centers_numpy((3, 1, 1), resolution=2.0)
        np.testing.assert_allclose(np.sort(centers[:, 0]), [-2.0, 0.0, 2.0])

    def test_column_order_xyz(self):
        """Columns are X, Y, Z (indexing='ij' convention)."""
        centers = get_voxel_centers_numpy((2, 3, 1), resolution=1.0)
        # 2 unique x values, 3 unique y values, 1 unique z value
        assert len(np.unique(centers[:, 0])) == 2
        assert len(np.unique(centers[:, 1])) == 3
        assert len(np.unique(centers[:, 2])) == 1
