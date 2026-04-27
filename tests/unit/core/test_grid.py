import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from fdtdx.core.grid import (
    _voxel_centers_numpy,
    calculate_spatial_offsets_yee,
    calculate_time_offset_yee,
    exact_analytical_fractions,
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
        centers = _voxel_centers_numpy(grid_shape, resolution)

        expected_num_voxels = 2 * 3 * 4
        assert centers.shape == (expected_num_voxels, 3), "Shape mismatch"

    def test_single_voxel(self):
        """Test a 1x1x1 grid. The center should be exactly at the origin (0, 0, 0)."""
        grid_shape = (1, 1, 1)
        resolution = 2.0
        centers = _voxel_centers_numpy(grid_shape, resolution)

        expected = np.array([[0.0, 0.0, 0.0]])
        assert_array_almost_equal(centers, expected, err_msg="Single voxel not centered at origin")

    def test_2x2x2_grid_exact_values(self):
        """Test a 2x2x2 grid with resolution 1.0 against known exact coordinates."""
        grid_shape = (2, 2, 2)
        resolution = 1.0
        centers = _voxel_centers_numpy(grid_shape, resolution)

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
        centers = _voxel_centers_numpy(grid_shape, resolution)

        # The mean of all centers in a properly centered grid should be 0 for x, y, and z
        mean_center = np.mean(centers, axis=0)
        assert_array_almost_equal(mean_center, [0.0, 0.0, 0.0], err_msg="Grid is not perfectly centered")

    def test_resolution_scaling(self):
        """Test that changing the resolution scales the coordinates linearly."""
        grid_shape = (3, 3, 3)
        centers_res_1 = _voxel_centers_numpy(grid_shape, 1.0)
        centers_res_2 = _voxel_centers_numpy(grid_shape, 2.0)

        # Doubling the resolution should exactly double all coordinate values
        assert_array_almost_equal(centers_res_1 * 2.0, centers_res_2, err_msg="Resolution scaling failed")


# --- Helper constants for tests ---
# Use high iterations for MC to prevent flaky tests in CI/CD
MC_ARGS = {"mc_batch_size": 5000, "mc_iterations": 10}
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


# ==========================================
# CATEGORY 1: System & Broad-Phase Setup
# ==========================================


def test_1_empty_mesh_safety():
    """Useful for: Validating function doesn't crash on completely empty geometries."""
    grid_shape = (2, 2, 2)
    binary_inside = np.zeros(grid_shape, dtype=bool)
    vertices = np.empty((0, 3), dtype=np.float64)
    faces = np.empty((0, 3), dtype=np.int32)

    fractions = exact_analytical_fractions(vertices, faces, grid_shape, 1.0, binary_inside)
    assert np.all(fractions == 0.0)


def test_2_kd_tree_out_of_bounds():
    """Useful for: Ensuring the broad-phase KD-tree accurately culls far-away meshes."""
    grid_shape = (3, 3, 3)
    binary_inside = np.zeros(grid_shape, dtype=bool)
    # Voxel centers are at [0..2], KD tree r=1.5. Vertex at 10 is far out of reach.
    vertices = np.array([[10, 10, 10], [11, 10, 10], [10, 11, 10]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    fractions = exact_analytical_fractions(vertices, faces, grid_shape, 1.0, binary_inside)
    assert np.all(fractions == 0.0)


def test_3_preservation_of_binary_inside():
    """Useful for: Checking that completely solid voxels are preserved as 1.0."""
    grid_shape = (2, 2, 2)
    binary_inside = np.ones(grid_shape, dtype=bool)
    # No mesh intersects
    vertices = np.array([[10, 10, 10], [11, 10, 10], [10, 11, 10]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    fractions = exact_analytical_fractions(vertices, faces, grid_shape, 1.0, binary_inside)
    assert np.all(fractions == 1.0)


def test_4_non_cubic_grid_dimensions():
    """Useful for: Validating correct un-flattening and 'ij' meshgrid indexing."""
    grid_shape = (2, 3, 4)
    binary_inside = np.zeros(grid_shape, dtype=bool)
    binary_inside[1, 2, 3] = True
    vertices = np.empty((0, 3), dtype=np.float64)
    faces = np.empty((0, 3), dtype=np.int32)

    fractions = exact_analytical_fractions(vertices, faces, grid_shape, 1.0, binary_inside)
    assert fractions.shape == (24,)
    assert fractions.reshape(grid_shape)[1, 2, 3] == 1.0


def test_5_different_resolutions():
    """Useful for: Ensuring alpha/distance scaling works with resolution != 1.0."""
    grid_shape = (1, 1, 1)
    res = 0.1
    binary_inside = np.zeros(grid_shape, dtype=bool)
    # A plane slicing exactly through the middle (z=0)
    vertices = np.array(plane_z(0.0, 1), dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    fractions = exact_analytical_fractions(vertices, faces, grid_shape, res, binary_inside)
    assert np.isclose(fractions[0], 0.5, atol=EXACT_TOLERANCE)


# ==========================================
# CATEGORY 2: K=1 Exact Analytical
# ==========================================


def test_6_k1_exact_half_x():
    """Useful for: Testing standard axis-aligned bisection along X."""
    vertices = np.array(plane_x(0.0, 1), dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)))
    assert np.isclose(fractions[0], 0.5, atol=EXACT_TOLERANCE)


def test_7_k1_diagonal_bisect():
    """Useful for: Testing a non-axis aligned plane exactly bisecting the voxel."""
    # Plane passing through origin with normal [1, 1, 0]
    vertices = np.array([[-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, 0.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)))
    assert np.isclose(fractions[0], 0.5, atol=EXACT_TOLERANCE)


def test_8_k1_barely_clipping_corner():
    """Useful for: Testing numerical stability near 0.0 (volume ≈ 0.0001)."""
    # Voxel bounds are [-0.5, 0.5]. Plane at z = 0.45, normal +z. Points inside if z <= 0.45.
    vertices = np.array(plane_z(0.45, 1), dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)))
    # Volume is 95% full (since everything below 0.45 is inside)
    assert np.isclose(fractions[0], 0.95, atol=EXACT_TOLERANCE)


def test_9_k1_almost_empty():
    """Useful for: Testing analytical volume near exactly 0.0 without going negative."""
    # Plane at z = -0.45, normal -z. Inside if z >= -0.45.
    vertices = np.array(plane_z(-0.45, -1), dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)))
    assert np.isclose(fractions[0], 0.95, atol=EXACT_TOLERANCE)


def test_10_k1_aligned_with_boundary():
    """Useful for: Checking boundary face alignment. Plane exactly on face x=0.5."""
    vertices = np.array(plane_x(0.5, 1), dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)))
    # Inside if x <= 0.5. Since voxel is [-0.5, 0.5], it's 100% inside.
    assert np.isclose(fractions[0], 1.0, atol=EXACT_TOLERANCE)


def test_11_k1_batch_processing():
    """Useful for: Validating vectorization across multiple K=1 voxels simultaneously."""
    grid_shape = (3, 1, 1)  # Voxels at x=0, x=1, x=2. Bounds [-0.5, 0.5], [0.5, 1.5], [1.5, 2.5]
    binary_inside = np.zeros(grid_shape, dtype=bool)

    # Large plane spanning all voxels at y=0. Inside if y <= 0.
    vertices = np.array([[-1.0, 0.0, -1.0], [3.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    fractions = exact_analytical_fractions(vertices, faces, grid_shape, 1.0, binary_inside)
    # Every voxel should be exactly 50%
    assert np.allclose(fractions, 0.5, atol=EXACT_TOLERANCE)


# ==========================================
# CATEGORY 3: K=2 Monte-Carlo Multi-Plane
# ==========================================


def test_12_k2_quarter_space():
    """Useful for: Baseline K=2 verification (25% volume)."""
    vertices = np.vstack([plane_x(0, 1), plane_y(0, 1)])
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)), **MC_ARGS)
    assert np.isclose(fractions[0], 0.25, atol=MC_TOLERANCE)


def test_13_k2_three_quarter_space():
    """Useful for: Inverted normals for K=2 convex hulls."""
    # Using inverted normals. Points are inside if x>=0 AND y>=0. Still 25% of the voxel!
    vertices = np.vstack([plane_x(0, -1), plane_y(0, -1)])
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)), **MC_ARGS)
    assert np.isclose(fractions[0], 0.25, atol=MC_TOLERANCE)


def test_14_k2_parallel_slice():
    """Useful for: Two parallel planes making a 'sandwich' slice out of the voxel."""
    # Plane 1: x = 0.25 (Normal +x, so inside is x <= 0.25)
    # Plane 2: x = -0.25 (Normal -x, so inside is x >= -0.25)
    # Net: x in [-0.25, 0.25]. Voxel width is 1.0. Therefore 50% volume.
    vertices = np.vstack([plane_x(0.25, 1), plane_x(-0.25, -1)])
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)), **MC_ARGS)
    assert np.isclose(fractions[0], 0.50, atol=MC_TOLERANCE)


def test_15_k2_mutually_exclusive():
    """Useful for: Ensuring disjoint planes yield 0% volume safely."""
    # Plane 1: x <= -0.25, Plane 2: x >= 0.25. Intersection is empty.
    vertices = np.vstack([plane_x(-0.25, 1), plane_x(0.25, -1)])
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)), **MC_ARGS)
    assert fractions[0] == 0.0  # Should be exactly 0.


def test_16_k2_almost_coplanar():
    """Useful for: Padding strategy robustness. Plane 3 is padded with Plane 2."""
    # Two planes that are nearly identical (essentially overlapping). Volume should be 50%.
    vertices = np.vstack([plane_z(0.0, 1), plane_z(0.0001, 1)])
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)), **MC_ARGS)
    assert np.isclose(fractions[0], 0.5, atol=MC_TOLERANCE)


# ==========================================
# CATEGORY 4: K=3 Monte-Carlo Multi-Plane
# ==========================================


def test_17_k3_eighth_space():
    """Useful for: Baseline K=3 verification (12.5% volume)."""
    vertices = np.vstack([plane_x(0, 1), plane_y(0, 1), plane_z(0, 1)])
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)), **MC_ARGS)
    assert np.isclose(fractions[0], 0.125, atol=MC_TOLERANCE)


def test_18_k3_tetrahedron_corner():
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
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)), **MC_ARGS)
    assert fractions[0] > 0.0
    assert fractions[0] < 0.1  # Should be a very small fraction


# ==========================================
# CATEGORY 5: K > 3 Capping & Stress Tests
# ==========================================


def test_19_k4_capping_logic():
    """Useful for: Guaranteeing `counts = np.clip(counts, 1, 3)` prevents tensor shape errors."""
    vertices = np.vstack([plane_x(0.1, 1), plane_y(0.1, 1), plane_z(0.1, 1), plane_z(-0.1, -1)])
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int32)
    # The function evaluates the first 3 faces, ignores the 4th.
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)), **MC_ARGS)

    # 3 planes intersecting at 0.1, 0.1, 0.1. Vol of x<0.1, y<0.1, z<0.1 is 0.6 * 0.6 * 0.6 = 0.216
    assert np.isclose(fractions[0], 0.216, atol=MC_TOLERANCE)


def test_20_k_many_highly_tessellated():
    """Useful for: Stress testing voxel index sorting and capping on dense meshes."""
    grid_shape = (1, 1, 1)
    binary_inside = np.zeros(grid_shape, dtype=bool)

    # Generate 10 random small triangles near the center
    np.random.seed(42)
    vertices = np.random.uniform(-0.1, 0.1, size=(30, 3))
    faces = np.arange(30, dtype=np.int32).reshape(10, 3)

    # Should not crash, should route to K >= 2, and yield a valid 0-1 fraction
    fractions = exact_analytical_fractions(vertices, faces, grid_shape, 1.0, binary_inside)
    assert 0.0 <= fractions[0] <= 1.0


def test_21_mixed_k1_and_kmulti():
    """Useful for: Ensuring the batch splitting (K=1 vs K>=2) works perfectly together."""
    grid_shape = (2, 1, 1)
    binary_inside = np.zeros(grid_shape, dtype=bool)

    # Voxel 0: [-0.5, 0.5]. Intersected by 1 plane (x=0.0). K=1 path. Expected: 0.5
    # Voxel 1: [0.5, 1.5]. Intersected by 2 planes (x=1.0, y=0.0). K=2 path. Expected: 0.25

    vertices = np.vstack(
        [
            [[-0.2, -0.2, -0.2], [0.2, -0.2, -0.2], [0.0, 0.2, 0.2]],  # Voxel 0 plane (x=0)
            [[1.0, -0.2, -0.2], [1.0, 0.2, -0.2], [1.0, 0.0, 0.2]],  # Voxel 1 plane 1 (x=1)
            [[0.8, 0.0, -0.2], [0.8, 0.0, 0.2], [1.2, 0.0, 0.0]],  # Voxel 1 plane 2 (y=0)
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)

    fractions = exact_analytical_fractions(vertices, faces, grid_shape, 1.0, binary_inside, **MC_ARGS)

    # Voxel 0 routes to K=1, Voxel 1 routes to Multi-Plane
    assert np.isclose(fractions[0], 0.5, atol=EXACT_TOLERANCE)
    assert np.isclose(fractions[1], 0.25, atol=MC_TOLERANCE)


def test_22_completely_inside_bounding_box_but_outside_planes():
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
    fractions = exact_analytical_fractions(vertices, faces, (1, 1, 1), 1.0, np.zeros((1, 1, 1)), **MC_ARGS)

    # The intersection is exactly the corner (-0.5, -0.5, -0.5) and outward. 0 volume in the voxel.
    assert fractions[0] == 0.0
