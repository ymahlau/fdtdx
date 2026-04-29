from pathlib import Path

import jax.numpy as jnp
import meshio
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from fdtdx.core.grid import (
    build_spatial_lookup_grid_csr,
    calculate_points_in_mesh,
    calculate_spatial_offsets_yee,
    calculate_time_offset_yee,
    compute_lookup_grid_dims,
    exact_analytical_fractions,
    get_voxel_centers_numpy,
    points_in_mesh_csr,
    polygon_to_mask,
)
from fdtdx.objects.static_material.meshed import _extract_surface


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


# --- Helper constants for tests ---
# Use high sample count for MC to prevent flaky tests in CI/CD
MC_ARGS = {"mc_samples_per_voxel": 50000}
MC_TOLERANCE = 0.015
EXACT_TOLERANCE = 1e-6


# --- HELPER: Triangle Generators ---
# Ensure triangles are small enough to pass the broad-phase KDTree (r=1.5*res)
def plane_z(z_val, normal_dir=1):
    """Generates a small triangle at z=z_val. normal_dir=1 is +z, -1 is -z."""
    if normal_dir == 1:
        return [[-0.2, -0.2, z_val], [0.2, -0.2, z_val], [0.0, 0.2, z_val]]
    return [[0.0, 0.2, z_val], [0.2, -0.2, z_val], [-0.2, -0.2, z_val]]


def plane_x(x_val, normal_dir=1):
    if normal_dir == 1:
        return [[x_val, -0.2, -0.2], [x_val, 0.2, -0.2], [x_val, 0.0, 0.2]]
    return [[x_val, 0.0, 0.2], [x_val, 0.2, -0.2], [x_val, -0.2, -0.2]]


def plane_y(y_val, normal_dir=1):
    if normal_dir == 1:
        return [[-0.2, y_val, -0.2], [-0.2, y_val, 0.2], [0.2, y_val, 0.0]]
    return [[0.2, y_val, 0.0], [-0.2, y_val, 0.2], [-0.2, y_val, -0.2]]


class TestAnalyticalFractions:
    # ==========================================
    # CATEGORY 1: System & Broad-Phase Setup
    # ==========================================
    def test_1_empty_mesh_safety(self):
        """Useful for: Validating function doesn't crash on completely empty geometries."""
        grid_shape = (2, 2, 2)
        binary_inside = np.zeros(grid_shape, dtype=bool)
        vertices = np.empty((0, 3), dtype=np.float64)
        faces = np.empty((0, 3), dtype=np.int32)

        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy(grid_shape, 1.0), 1.0, binary_inside.flatten()
        )
        assert np.all(fractions == 0.0)

    def test_2_kd_tree_out_of_bounds(self):
        """Useful for: Ensuring the broad-phase KD-tree accurately culls far-away meshes."""
        grid_shape = (3, 3, 3)
        binary_inside = np.zeros(grid_shape, dtype=bool)
        # Voxel centers are at [-1,0,1] per axis. KD tree r=1.5. Vertex at 10 is far out of reach.
        vertices = np.array([[10, 10, 10], [11, 10, 10], [10, 11, 10]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy(grid_shape, 1.0), 1.0, binary_inside.flatten()
        )
        assert np.all(fractions == 0.0)

    def test_3_preservation_of_binary_inside(self):
        """Useful for: Checking that completely solid voxels are preserved as 1.0."""
        grid_shape = (2, 2, 2)
        binary_inside = np.ones(grid_shape, dtype=bool)
        # No mesh intersects
        vertices = np.array([[10, 10, 10], [11, 10, 10], [10, 11, 10]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy(grid_shape, 1.0), 1.0, binary_inside.flatten()
        )
        assert np.all(fractions == 1.0)

    def test_4_non_cubic_grid_dimensions(self):
        """Useful for: Validating correct un-flattening and 'ij' meshgrid indexing."""
        grid_shape = (2, 3, 4)
        binary_inside = np.zeros(grid_shape, dtype=bool)
        binary_inside[1, 2, 3] = True
        vertices = np.empty((0, 3), dtype=np.float64)
        faces = np.empty((0, 3), dtype=np.int32)

        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy(grid_shape, 1.0), 1.0, binary_inside.flatten()
        )
        assert fractions.shape == (24,)
        assert fractions.reshape(grid_shape)[1, 2, 3] == 1.0

    def test_5_different_resolutions(self):
        """Useful for: Ensuring alpha/distance scaling works with resolution != 1.0."""
        grid_shape = (1, 1, 1)
        res = 0.1
        binary_inside = np.zeros(grid_shape, dtype=bool)
        # A plane slicing exactly through the middle (z=0)
        vertices = np.array(plane_z(0.0, 1), dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy(grid_shape, res), res, binary_inside.flatten()
        )
        assert np.isclose(fractions[0], 0.5, atol=EXACT_TOLERANCE)

    # ==========================================
    # CATEGORY 2: K=1 Exact Analytical
    # ==========================================

    def test_6_k1_exact_half_x(self):
        """Useful for: Testing standard axis-aligned bisection along X."""
        vertices = np.array(plane_x(0.0, 1), dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1)
        )
        assert np.isclose(fractions[0], 0.5, atol=EXACT_TOLERANCE)

    def test_7_k1_diagonal_bisect(self):
        """Useful for: Testing a non-axis aligned plane exactly bisecting the voxel."""
        # Plane passing through origin with normal [1, 1, 0]
        vertices = np.array([[-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, 0.0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1)
        )
        assert np.isclose(fractions[0], 0.5, atol=EXACT_TOLERANCE)

    def test_8_k1_barely_clipping_corner(self):
        """Useful for: Testing numerical stability near 0.0 (volume ≈ 0.0001)."""
        # Voxel bounds are [-0.5, 0.5]. Plane at z = 0.45, normal +z. Points inside if z <= 0.45.
        vertices = np.array(plane_z(0.45, 1), dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1)
        )
        # Volume is 95% full (since everything below 0.45 is inside)
        assert np.isclose(fractions[0], 0.95, atol=EXACT_TOLERANCE)

    def test_9_k1_almost_empty(self):
        """Useful for: Testing analytical volume near exactly 0.0 without going negative."""
        # Plane at z = -0.45, normal -z. Inside if z >= -0.45.
        vertices = np.array(plane_z(-0.45, -1), dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1)
        )
        assert np.isclose(fractions[0], 0.95, atol=EXACT_TOLERANCE)

    def test_10_k1_aligned_with_boundary(self):
        """Useful for: Checking boundary face alignment. Plane exactly on face x=0.5."""
        vertices = np.array(plane_x(0.5, 1), dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1)
        )
        # Inside if x <= 0.5. Since voxel is [-0.5, 0.5], it's 100% inside.
        assert np.isclose(fractions[0], 1.0, atol=EXACT_TOLERANCE)

    def test_11_k1_batch_processing(self):
        """Useful for: Validating vectorization across multiple K=1 voxels simultaneously."""
        grid_shape = (3, 1, 1)  # Voxels centered at x=-1, x=0, x=1.
        binary_inside = np.zeros(grid_shape, dtype=bool)

        # Triangle at y=0 with x extent [-0.6, 0.6]. Two constraints must both hold:
        # (1) KDTree: centroid (0, 0, -0.17) is ≈1.01 from outermost centers — within r=1.5.
        # (2) AABB: expanded by res/2=0.5 gives x ∈ [-1.1, 1.1], covering voxel centers at ±1.
        vertices = np.array([[-0.6, 0.0, -0.5], [0.6, 0.0, -0.5], [0.0, 0.0, 0.5]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy(grid_shape, 1.0), 1.0, binary_inside.flatten()
        )
        # Every voxel is bisected by y=0, so each should be exactly 50%
        assert np.allclose(fractions, 0.5, atol=EXACT_TOLERANCE)

    # ==========================================
    # CATEGORY 3: K=2 Monte-Carlo Multi-Plane
    # ==========================================

    def test_12_k2_quarter_space(self):
        """Useful for: Baseline K=2 verification (25% volume)."""
        vertices = np.vstack([plane_x(0, 1), plane_y(0, 1)])
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1), **MC_ARGS
        )
        assert np.isclose(fractions[0], 0.25, atol=MC_TOLERANCE)

    def test_13_k2_three_quarter_space(self):
        """Useful for: Inverted normals for K=2 convex hulls."""
        # Using inverted normals. Points are inside if x>=0 AND y>=0. Still 25% of the voxel!
        vertices = np.vstack([plane_x(0, -1), plane_y(0, -1)])
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1), **MC_ARGS
        )
        assert np.isclose(fractions[0], 0.25, atol=MC_TOLERANCE)

    def test_14_k2_parallel_slice(self):
        """Useful for: Two parallel planes making a 'sandwich' slice out of the voxel."""
        # Plane 1: x = 0.25 (Normal +x, so inside is x <= 0.25)
        # Plane 2: x = -0.25 (Normal -x, so inside is x >= -0.25)
        # Net: x in [-0.25, 0.25]. Voxel width is 1.0. Therefore 50% volume.
        vertices = np.vstack([plane_x(0.25, 1), plane_x(-0.25, -1)])
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1), **MC_ARGS
        )
        assert np.isclose(fractions[0], 0.50, atol=MC_TOLERANCE)

    def test_15_k2_mutually_exclusive(self):
        """Useful for: Ensuring disjoint planes yield 0% volume safely."""
        # Plane 1: x <= -0.25, Plane 2: x >= 0.25. Intersection is empty.
        vertices = np.vstack([plane_x(-0.25, 1), plane_x(0.25, -1)])
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1), **MC_ARGS
        )
        assert fractions[0] == 0.0  # Should be exactly 0.

    def test_16_k2_almost_coplanar(self):
        """Useful for: Padding strategy robustness. Plane 3 is padded with Plane 2."""
        # Two planes that are nearly identical (essentially overlapping). Volume should be 50%.
        vertices = np.vstack([plane_z(0.0, 1), plane_z(0.0001, 1)])
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1), **MC_ARGS
        )
        assert np.isclose(fractions[0], 0.5, atol=MC_TOLERANCE)

    # ==========================================
    # CATEGORY 4: K=3 Monte-Carlo Multi-Plane
    # ==========================================

    def test_17_k3_eighth_space(self):
        """Useful for: Baseline K=3 verification (12.5% volume)."""
        vertices = np.vstack([plane_x(0, 1), plane_y(0, 1), plane_z(0, 1)])
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1), **MC_ARGS
        )
        assert np.isclose(fractions[0], 0.125, atol=MC_TOLERANCE)

    def test_18_k3_tetrahedron_corner(self):
        """Useful for: Evaluating 3 non-axis aligned planes defining a complex corner."""
        # Corner of voxel is at (-0.5, -0.5, -0.5).
        # Cut a small tetrahedron: x+y+z <= -1.2
        v1 = np.array([-0.2, -0.5, -0.5])
        v2 = np.array([-0.5, -0.2, -0.5])
        v3 = np.array([-0.5, -0.5, -0.2])
        # The plane through these 3 points cuts off a tiny corner.
        # Just to trigger K=3, we'll feed it this plane + 2 bounding planes.
        vertices = np.vstack(
            [
                [v1, v2, v3],  # Triangle 1
                plane_x(-0.49, -1),  # Triangle 2
                plane_y(-0.49, -1),  # Triangle 3
            ]
        )
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1), **MC_ARGS
        )
        assert fractions[0] > 0.0
        assert fractions[0] < 0.1  # Should be a very small fraction

    # ==========================================
    # CATEGORY 5: K > 3 Capping & Stress Tests
    # ==========================================

    def test_19_k4_capping_logic(self):
        """Useful for: Guaranteeing `counts = np.clip(counts, 1, 3)` prevents tensor shape errors."""
        vertices = np.vstack([plane_x(0.1, 1), plane_y(0.1, 1), plane_z(0.1, 1), plane_z(-0.1, -1)])
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int32)
        # The function evaluates the first 3 faces, ignores the 4th.
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1), **MC_ARGS
        )

        # 3 planes intersecting at 0.1, 0.1, 0.1. Vol of x<0.1, y<0.1, z<0.1 is 0.6 * 0.6 * 0.6 = 0.216
        assert np.isclose(fractions[0], 0.216, atol=MC_TOLERANCE)

    def test_20_k_many_highly_tessellated(self):
        """Useful for: Stress testing voxel index sorting and capping on dense meshes."""
        grid_shape = (1, 1, 1)
        binary_inside = np.zeros(grid_shape, dtype=bool)

        # Generate 10 random small triangles near the center
        np.random.seed(42)
        vertices = np.random.uniform(-0.1, 0.1, size=(30, 3))
        faces = np.arange(30, dtype=np.int32).reshape(10, 3)

        # Should not crash, should route to K >= 2, and yield a valid 0-1 fraction
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy(grid_shape, 1.0), 1.0, binary_inside.flatten()
        )
        assert 0.0 <= fractions[0] <= 1.0

    def test_21_mixed_k1_and_kmulti(self):
        """Useful for: Ensuring the batch splitting (K=1 vs K>=2) works perfectly together."""
        # Use a 5-voxel grid so the K=1 voxel (index 0, x=-2) and K=2 voxel (index 4, x=+2)
        # are 4 units apart. Each set of triangles is centred >3 units from the other voxel,
        # preventing cross-contamination through the KDTree broad-phase (r=1.5).
        grid_shape = (5, 1, 1)
        binary_inside = np.zeros(grid_shape, dtype=bool)

        # Voxel 0 (center x=-2): one y=0 plane → K=1 path → expected 0.5
        tri_voxel0 = np.array([[-2.2, 0.0, -0.2], [-2.2, 0.0, 0.2], [-1.8, 0.0, 0.0]])

        # Voxel 4 (center x=+2): two planes (y=0 and x=2) → K=2 path → expected 0.25
        tri_voxel4_a = np.array([[1.8, 0.0, -0.2], [1.8, 0.0, 0.2], [2.2, 0.0, 0.0]])  # y=0, normal +y
        tri_voxel4_b = np.array([[2.0, -0.2, -0.2], [2.0, 0.2, -0.2], [2.0, 0.0, 0.2]])  # x=2, normal +x

        vertices = np.vstack([tri_voxel0, tri_voxel4_a, tri_voxel4_b])
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)

        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy(grid_shape, 1.0), 1.0, binary_inside.flatten(), **MC_ARGS
        )

        # Voxel 0: K=1, plane y=0 bisects → 0.5 exactly
        assert np.isclose(fractions[0], 0.5, atol=EXACT_TOLERANCE)
        # Voxel 4: K=2, {y<=0} ∩ {x<=2} each cover half the voxel → 0.25
        assert np.isclose(fractions[4], 0.25, atol=MC_TOLERANCE)

    def test_22_completely_inside_bounding_box_but_outside_planes(self):
        """Useful for: Ensuring Monte Carlo correctly zeros out points that fail normal distance tests."""
        # Three planes defined such that the "inside" space (distance <= 0) is entirely outside the voxel.
        # E.g. Normal pointing outwards, plane at bounds.
        vertices = np.vstack(
            [
                plane_x(-0.5, 1),  # x <= -0.5
                plane_y(-0.5, 1),  # y <= -0.5
                plane_z(-0.5, 1),  # z <= -0.5
            ]
        )
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
        fractions = exact_analytical_fractions(
            vertices, faces, get_voxel_centers_numpy((1, 1, 1), 1.0), 1.0, np.zeros(1), **MC_ARGS
        )

        # The intersection is exactly the corner (-0.5, -0.5, -0.5) and outward. 0 volume in the voxel.
        assert fractions[0] == 0.0

    def test_loaded_mesh(self):
        mesh_path = Path(__file__).parent.parent.parent / "data" / "mesh" / "LabradorLowPoly.stl"
        mesh = meshio.read(str(mesh_path))
        centre = 0.5 * (mesh.points.min(axis=0) + mesh.points.max(axis=0))
        mesh.points = mesh.points - centre
        cells_and_types = [(b.type, b.data) for b in mesh.cells]
        faces = _extract_surface(cells_and_types)
        n = 100
        grid_shape = (n, n, n)
        resolution = 200 / n
        voxel_centers = get_voxel_centers_numpy(
            grid_shape=grid_shape,
            resolution=resolution,
        )
        arr = exact_analytical_fractions(
            vertices=mesh.points,
            faces=faces,
            query_points=voxel_centers,
            resolution=resolution,
        )

        # sanity checks
        assert not np.any(np.isnan(arr))
        assert not np.any(np.isinf(arr))
        assert np.sum(arr) > 0

        # the object in the mesh is a dog, where the center is inside the mesh
        center_idx = round(n / 2)
        center_value = arr.reshape(*grid_shape)[center_idx, center_idx, center_idx]
        assert center_value > 0


# ─── Mesh helpers ─────────────────────────────────────────────────────────────


def _unit_cube():
    """Watertight unit cube occupying [-0.5, 0.5]^3 (12 triangles)."""
    v = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],  # 0, 1
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],  # 2, 3
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],  # 4, 5
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],  # 6, 7
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom z=-0.5
            [4, 6, 5],
            [4, 7, 6],  # top    z=+0.5
            [0, 5, 1],
            [0, 4, 5],  # front  y=-0.5
            [2, 6, 7],
            [2, 7, 3],  # back   y=+0.5
            [0, 7, 4],
            [0, 3, 7],  # left   x=-0.5
            [1, 5, 6],
            [1, 6, 2],  # right  x=+0.5
        ],
        dtype=np.int32,
    )
    return v, f


def _unit_octahedron():
    """
    Watertight regular octahedron with all 6 vertices on the unit sphere.
    Analytically inside iff |x| + |y| + |z| < 1.
    """
    v = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],  # 0, 1
            [0, 1, 0],
            [0, -1, 0],  # 2, 3
            [0, 0, 1],
            [0, 0, -1],  # 4, 5
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],  # upper cap
            [0, 3, 5],
            [3, 1, 5],
            [1, 2, 5],
            [2, 0, 5],  # lower cap
        ],
        dtype=np.int32,
    )
    return v, f


# ─── get_voxel_centers_numpy ───────────────────────────────────────────────────


class TestGetVoxelCentersNumpy:
    def test_single_voxel_at_origin(self):
        centers = get_voxel_centers_numpy((1, 1, 1), resolution=1.0)
        assert centers.shape == (1, 3)
        np.testing.assert_allclose(centers[0], [0.0, 0.0, 0.0])

    def test_output_shape(self):
        nx, ny, nz = 3, 4, 5
        centers = get_voxel_centers_numpy((nx, ny, nz), resolution=0.5)
        assert centers.shape == (nx * ny * nz, 3)

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


# ─── compute_lookup_grid_dims ─────────────────────────────────────────────────


class TestComputeLookupGridDims:
    def _square_mesh(self):
        verts2d = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        return verts2d, faces

    def test_fixed_size_returns_exact_dims(self):
        v, f = self._square_mesh()
        gx, gy, _, _ = compute_lookup_grid_dims(v, f, fixed_lookup_grid_size=(7, 11))
        assert gx == 7 and gy == 11

    def test_fixed_size_bounds_are_vertex_minmax(self):
        v, f = self._square_mesh()
        _, _, mn, mx = compute_lookup_grid_dims(v, f, fixed_lookup_grid_size=(3, 3))
        np.testing.assert_allclose(mn, [0.0, 0.0])
        np.testing.assert_allclose(mx, [1.0, 1.0])

    def test_auto_dims_are_at_least_one(self):
        v, f = self._square_mesh()
        gx, gy, _, _ = compute_lookup_grid_dims(v, f, fixed_lookup_grid_size=None)
        assert gx >= 1 and gy >= 1

    def test_auto_bounds_match_vertex_minmax(self):
        v, f = self._square_mesh()
        _, _, mn, mx = compute_lookup_grid_dims(v, f, fixed_lookup_grid_size=None)
        np.testing.assert_allclose(mn, v.min(axis=0))
        np.testing.assert_allclose(mx, v.max(axis=0))

    def test_invalid_type_raises_value_error(self):
        v, f = self._square_mesh()
        with pytest.raises(ValueError):
            compute_lookup_grid_dims(v, f, fixed_lookup_grid_size=42)

    def test_auto_narrow_mesh_still_valid(self):
        """Very narrow mesh (width >> height) still gives positive grid dims."""
        v = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 0.1], [0.0, 0.1]])
        f = np.array([[0, 1, 2], [0, 2, 3]])
        gx, gy, _, _ = compute_lookup_grid_dims(v, f, fixed_lookup_grid_size=None)
        assert gx >= 1 and gy >= 1

    def test_auto_more_faces_gives_larger_or_equal_grid(self):
        """Auto-sizing should produce a finer grid when face density is higher."""
        v, f2 = self._square_mesh()
        f_many = np.tile(f2, (100, 1))  # duplicate faces, same bounds
        gx_few, gy_few, _, _ = compute_lookup_grid_dims(v, f2, None)
        gx_many, gy_many, _, _ = compute_lookup_grid_dims(v, f_many, None)
        # grid area should be >= (heuristic targets ~2-4 faces per cell)
        assert gx_many * gy_many >= gx_few * gy_few

    def test_fixed_size_1x1_works(self):
        v, f = self._square_mesh()
        gx, gy, _, _ = compute_lookup_grid_dims(v, f, fixed_lookup_grid_size=(1, 1))
        assert gx == 1 and gy == 1


# ─── build_spatial_lookup_grid_csr ────────────────────────────────────────────


class TestBuildSpatialLookupGridCSR:
    def test_offsets_length(self):
        v, f = _unit_cube()
        offsets, _, (gx, gy), _, _ = build_spatial_lookup_grid_csr(v, f)
        assert len(offsets) == gx * gy + 1

    def test_offsets_start_zero_end_total_indices(self):
        v, f = _unit_cube()
        offsets, indices, _, _, _ = build_spatial_lookup_grid_csr(v, f)
        assert offsets[0] == 0
        assert offsets[-1] == len(indices)

    def test_offsets_monotonically_nondecreasing(self):
        v, f = _unit_cube()
        offsets, _, _, _, _ = build_spatial_lookup_grid_csr(v, f)
        assert np.all(np.diff(offsets) >= 0)

    def test_every_face_referenced_at_least_once(self):
        v, f = _unit_cube()
        _, indices, _, _, _ = build_spatial_lookup_grid_csr(v, f)
        for fi in range(len(f)):
            assert fi in indices, f"Face {fi} missing from CSR indices"

    def test_fixed_grid_size_respected(self):
        v, f = _unit_cube()
        _, _, (gx, gy), _, _ = build_spatial_lookup_grid_csr(v, f, fixed_lookup_grid_size=(5, 7))
        assert gx == 5 and gy == 7

    def test_cell_size_positive(self):
        v, f = _unit_cube()
        _, _, _, _, cell_size = build_spatial_lookup_grid_csr(v, f)
        assert np.all(cell_size > 0)

    def test_min_b_matches_xy_vertex_min(self):
        v, f = _unit_cube()
        _, _, _, min_b, _ = build_spatial_lookup_grid_csr(v, f)
        np.testing.assert_allclose(min_b, v[:, :2].min(axis=0))

    def test_single_triangle_face_in_indices(self):
        """A non-degenerate single triangle must appear in the CSR indices."""
        v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.5]])
        f = np.array([[0, 1, 2]])
        _, indices, _, _, _ = build_spatial_lookup_grid_csr(v, f)
        assert 0 in indices

    def test_face_count_in_indices_matches_cell_coverage(self):
        """len(indices) equals sum of per-cell face counts from offsets."""
        v, f = _unit_cube()
        offsets, indices, _, _, _ = build_spatial_lookup_grid_csr(v, f)
        counts = np.diff(offsets)
        assert counts.sum() == len(indices)

    def test_indices_are_valid_face_indices(self):
        """All values in indices are in range [0, F)."""
        v, f = _unit_cube()
        _, indices, _, _, _ = build_spatial_lookup_grid_csr(v, f)
        assert np.all(indices >= 0)
        assert np.all(indices < len(f))


# ─── points_in_mesh_csr ───────────────────────────────────────────────────────


class TestPointsInMeshCSR:
    @pytest.fixture(scope="class")
    def cube_csr(self):
        v, f = _unit_cube()
        offsets, indices, grid_dims, min_b, cell_size = build_spatial_lookup_grid_csr(v, f)
        return v, f, offsets, indices, grid_dims, min_b, cell_size

    @pytest.fixture(scope="class")
    def oct_csr(self):
        v, f = _unit_octahedron()
        offsets, indices, grid_dims, min_b, cell_size = build_spatial_lookup_grid_csr(v, f)
        return v, f, offsets, indices, grid_dims, min_b, cell_size

    # ── unit cube: inside ─────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "pt",
        [
            [0.1, 0.2, 0.3],
            [0.0, 0.1, 0.2],
            [-0.2, 0.1, -0.3],
            [0.3, 0.1, -0.2],
            [0.0, 0.1, 0.0],
        ],
    )
    def test_cube_interior_points(self, cube_csr, pt):
        v, f, offsets, indices, gd, mb, cs = cube_csr
        result = points_in_mesh_csr(v, f, np.array([pt]), offsets, indices, gd, mb, cs)
        assert result[0], f"Expected inside for {pt}"

    # ── unit cube: outside ────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "pt",
        [
            [0.7, 0.0, 0.0],
            [-0.7, 0.0, 0.0],
            [0.0, 0.7, 0.0],
            [0.0, -0.7, 0.0],
            [0.0, 0.0, 0.7],
            [0.0, 0.0, -0.7],
            [2.0, 2.0, 2.0],
        ],
    )
    def test_cube_exterior_points(self, cube_csr, pt):
        v, f, offsets, indices, gd, mb, cs = cube_csr
        result = points_in_mesh_csr(v, f, np.array([pt]), offsets, indices, gd, mb, cs)
        assert not result[0], f"Expected outside for {pt}"

    def test_cube_dense_interior_grid(self, cube_csr):
        """All points at |coord| ≤ 0.4 are inside the unit cube."""
        v, f, offsets, indices, gd, mb, cs = cube_csr
        coords = np.linspace(-0.4, 0.4, 4)
        xx, yy, zz = np.meshgrid(coords, coords, coords)
        # Shift slightly to avoid aligning on face-diagonal edges
        q = np.column_stack([xx.ravel() + 0.01, yy.ravel() + 0.02, zz.ravel()])
        result = points_in_mesh_csr(v, f, q, offsets, indices, gd, mb, cs)
        assert np.all(result), f"Interior points misclassified: {np.where(~result)}"

    # ── octahedron: analytically |x|+|y|+|z| < 1 → inside ───────────────────

    @pytest.mark.parametrize(
        "pt",
        [
            [0.2, 0.2, 0.2],  # L1 = 0.6
            [0.1, 0.2, 0.3],  # L1 = 0.6
            [0.3, 0.1, 0.1],  # L1 = 0.5
        ],
    )
    def test_octahedron_interior_points(self, oct_csr, pt):
        v, f, offsets, indices, gd, mb, cs = oct_csr
        result = points_in_mesh_csr(v, f, np.array([pt]), offsets, indices, gd, mb, cs)
        assert result[0], f"Expected inside for L1={sum(abs(x) for x in pt):.2f}: {pt}"

    @pytest.mark.parametrize(
        "pt",
        [
            [0.4, 0.4, 0.4],  # L1 = 1.2
            [0.5, 0.3, 0.3],  # L1 = 1.1
            [0.7, 0.2, 0.2],  # L1 = 1.1
        ],
    )
    def test_octahedron_exterior_points(self, oct_csr, pt):
        v, f, offsets, indices, gd, mb, cs = oct_csr
        result = points_in_mesh_csr(v, f, np.array([pt]), offsets, indices, gd, mb, cs)
        assert not result[0], f"Expected outside for L1={sum(abs(x) for x in pt):.2f}: {pt}"

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_empty_query_returns_empty_bool_array(self, cube_csr):
        v, f, offsets, indices, gd, mb, cs = cube_csr
        result = points_in_mesh_csr(v, f, np.empty((0, 3)), offsets, indices, gd, mb, cs)
        assert result.shape == (0,)
        assert result.dtype == bool

    def test_empty_csr_cells_branch(self):
        """
        Two small faces far apart in a 4x4 grid leave centre cells empty.
        The total==0 branch inside the chunk loop must be reached without error.
        """
        v = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.1],  # face 0 - bottom-left
                [0.9, 0.9, 0.0],
                [1.0, 0.9, 0.0],
                [0.9, 1.0, 0.1],  # face 1 - top-right
            ],
            dtype=np.float64,
        )
        f = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        offsets, indices, grid_dims, min_b, cell_size = build_spatial_lookup_grid_csr(
            v, f, fixed_lookup_grid_size=(4, 4)
        )
        # Confirm some cells are truly empty
        assert np.any(np.diff(offsets) == 0), "Expected empty grid cells"
        # Query a point in the empty middle zone
        q = np.array([[0.5, 0.5, 0.5]])
        result = points_in_mesh_csr(v, f, q, offsets, indices, grid_dims, min_b, cell_size)
        # Mesh is not watertight; result is False (no crossings from empty cell)
        assert not result[0]

    def test_chunk_size_does_not_affect_result(self, cube_csr):
        """Results must be identical regardless of chunk_size."""
        v, f, offsets, indices, gd, mb, cs = cube_csr
        q = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.3, 0.1, -0.2],
                [0.8, 0.0, 0.0],
                [-0.8, 0.0, 0.0],
            ]
        )
        r_single = points_in_mesh_csr(v, f, q, offsets, indices, gd, mb, cs, chunk_size=100_000)
        r_chunked = points_in_mesh_csr(v, f, q, offsets, indices, gd, mb, cs, chunk_size=2)
        np.testing.assert_array_equal(r_single, r_chunked)

    def test_far_outside_point_clipped_to_boundary_cell(self, cube_csr):
        """Points far outside the mesh bounding box are clipped; result is False."""
        v, f, offsets, indices, gd, mb, cs = cube_csr
        q = np.array([[100.0, 100.0, 0.0], [-100.0, -100.0, 0.0]])
        result = points_in_mesh_csr(v, f, q, offsets, indices, gd, mb, cs)
        assert not np.any(result)


# ─── calculate_points_in_mesh ─────────────────────────────────────────────────


class TestCalculatePointsInMesh:
    def test_cube_interior_point(self):
        v, f = _unit_cube()
        result = calculate_points_in_mesh(v, f, np.array([[0.1, 0.2, 0.3]]))
        assert result[0]

    def test_cube_exterior_point(self):
        v, f = _unit_cube()
        result = calculate_points_in_mesh(v, f, np.array([[2.0, 2.0, 2.0]]))
        assert not result[0]

    def test_octahedron_inside_outside_pair(self):
        """Analytical criterion |x|+|y|+|z| < 1 separates the two points."""
        v, f = _unit_octahedron()
        q = np.array(
            [
                [0.1, 0.2, 0.3],  # L1 = 0.6 → inside
                [0.4, 0.4, 0.4],  # L1 = 1.2 → outside
            ]
        )
        result = calculate_points_in_mesh(v, f, q)
        assert result[0] and not result[1]

    def test_fixed_lookup_grid_same_as_auto(self):
        """Fixed and auto grid sizes give the same classification."""
        v, f = _unit_cube()
        q = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.8, 0.0, 0.0],
            ]
        )
        r_auto = calculate_points_in_mesh(v, f, q)
        r_fixed = calculate_points_in_mesh(v, f, q, fixed_lookup_grid_size=(4, 4))
        np.testing.assert_array_equal(r_auto, r_fixed)

    def test_empty_query_returns_empty_array(self):
        v, f = _unit_cube()
        result = calculate_points_in_mesh(v, f, np.empty((0, 3), dtype=np.float64))
        assert result.shape == (0,)
        assert result.dtype == bool

    def test_multiple_interior_exterior_mixed(self):
        """Mixed batch: verify each point independently."""
        v, f = _unit_cube()
        interior = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2], [-0.2, 0.1, -0.3]])
        exterior = np.array([[0.7, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, -0.7]])
        q = np.vstack([interior, exterior])
        result = calculate_points_in_mesh(v, f, q)
        assert np.all(result[:3]), "Interior points should be inside"
        assert np.all(~result[3:]), "Exterior points should be outside"

    def test_output_shape_matches_query_count(self):
        v, f = _unit_cube()
        q = np.zeros((13, 3))
        result = calculate_points_in_mesh(v, f, q)
        assert result.shape == (13,)
        assert result.dtype == bool
