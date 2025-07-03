import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.grid import calculate_spatial_offsets_yee, calculate_time_offset_yee, polygon_to_mask


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

    # Test with 4D array
    inv_permittivities_4d = jnp.ones((2, 2, 2, 2))
    with pytest.raises(Exception):
        calculate_time_offset_yee(
            center, wave_vector, inv_permittivities_4d, inv_permeabilities, resolution, time_step_duration
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
