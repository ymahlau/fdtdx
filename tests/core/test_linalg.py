import jax.numpy as jnp
import pytest

from fdtdx.core.linalg import (
    get_orthogonal_vector,
    get_single_directional_rotation_matrix,
    get_wave_vector_raw,
    rotate_vector,
)


def test_get_wave_vector_raw_positive_direction():
    """Test wave vector generation with positive direction."""
    result = get_wave_vector_raw("+", 0)
    expected = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    assert jnp.allclose(result, expected)

    result = get_wave_vector_raw("+", 1)
    expected = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    assert jnp.allclose(result, expected)


def test_get_wave_vector_raw_negative_direction():
    """Test wave vector generation with negative direction."""
    result = get_wave_vector_raw("-", 2)
    expected = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)
    assert jnp.allclose(result, expected)


def test_get_wave_vector_raw_all_axes():
    """Test wave vector generation for all three axes."""
    for axis in range(3):
        pos_result = get_wave_vector_raw("+", axis)
        neg_result = get_wave_vector_raw("-", axis)

        # Check that only the specified axis is non-zero
        assert jnp.sum(jnp.abs(pos_result)) == 1.0
        assert jnp.sum(jnp.abs(neg_result)) == 1.0

        # Check correct signs
        assert pos_result[axis] == 1.0
        assert neg_result[axis] == -1.0


def test_get_orthogonal_vector_with_electric_field():
    """Test orthogonal vector computation using electric field."""
    v_E = jnp.array([1.0, 0.0, 0.0])
    wave_vector = jnp.array([0.0, 0.0, 1.0])
    result = get_orthogonal_vector(v_E=v_E, wave_vector=wave_vector)
    expected = jnp.array([0.0, 1.0, 0.0])  # k × E
    assert jnp.allclose(result, expected)


def test_get_orthogonal_vector_with_magnetic_field():
    """Test orthogonal vector computation using magnetic field."""
    v_H = jnp.array([0.0, 1.0, 0.0])
    wave_vector = jnp.array([0.0, 0.0, 1.0])
    result = get_orthogonal_vector(v_H=v_H, wave_vector=wave_vector)
    expected = jnp.array([1.0, 0.0, 0.0])  # H × k
    assert jnp.allclose(result, expected)


def test_get_orthogonal_vector_with_direction_and_axis():
    """Test orthogonal vector computation using direction and propagation axis."""
    v_E = jnp.array([1.0, 0.0, 0.0])
    result = get_orthogonal_vector(v_E=v_E, direction="+", propagation_axis=2)
    expected = jnp.array([0.0, 1.0, 0.0])
    assert jnp.allclose(result, expected)


def test_get_orthogonal_vector_invalid_inputs():
    """Test that invalid inputs raise exceptions."""
    # Both v_E and v_H provided
    with pytest.raises(Exception):
        get_orthogonal_vector(
            v_E=jnp.array([1.0, 0.0, 0.0]), v_H=jnp.array([0.0, 1.0, 0.0]), wave_vector=jnp.array([0.0, 0.0, 1.0])
        )

    # Neither v_E nor v_H provided
    with pytest.raises(Exception):
        get_orthogonal_vector(wave_vector=jnp.array([0.0, 0.0, 1.0]))

    # No wave vector information provided
    with pytest.raises(Exception):
        get_orthogonal_vector(v_E=jnp.array([1.0, 0.0, 0.0]))


def test_get_orthogonal_vector_missing_wave_vector_info():
    """Test exception when wave vector information is incomplete."""
    with pytest.raises(Exception):
        get_orthogonal_vector(
            v_E=jnp.array([1.0, 0.0, 0.0]),
            direction="+",
            # Missing propagation_axis
        )


def test_get_single_directional_rotation_matrix_x_axis():
    """Test rotation matrix generation around x-axis."""
    angle = jnp.pi / 2
    result = get_single_directional_rotation_matrix(0, angle)

    # Test rotating a vector around x-axis
    test_vector = jnp.array([0.0, 1.0, 0.0])
    rotated = result @ test_vector
    expected = jnp.array([0.0, 0.0, 1.0])
    assert jnp.allclose(rotated, expected, atol=1e-6)


def test_get_single_directional_rotation_matrix_y_axis():
    """Test rotation matrix generation around y-axis."""
    angle = jnp.pi / 2
    result = get_single_directional_rotation_matrix(1, angle)

    # Test rotating a vector around y-axis
    test_vector = jnp.array([1.0, 0.0, 0.0])
    rotated = result @ test_vector
    expected = jnp.array([0.0, 0.0, 1.0])
    assert jnp.allclose(rotated, expected, atol=1e-6)


def test_get_single_directional_rotation_matrix_z_axis():
    """Test rotation matrix generation around z-axis."""
    angle = jnp.pi / 2
    result = get_single_directional_rotation_matrix(2, angle)

    # Test rotating a vector around z-axis
    test_vector = jnp.array([1.0, 0.0, 0.0])
    rotated = result @ test_vector
    expected = jnp.array([0.0, 1.0, 0.0])
    assert jnp.allclose(rotated, expected, atol=1e-6)


def test_get_single_directional_rotation_matrix_identity():
    """Test that zero rotation gives identity behavior."""
    for axis in range(3):
        result = get_single_directional_rotation_matrix(axis, 0.0)
        test_vector = jnp.array([1.0, 2.0, 3.0])
        rotated = result @ test_vector
        assert jnp.allclose(rotated, test_vector)


def test_get_single_directional_rotation_matrix_invalid_axis():
    """Test that invalid rotation axis raises exception."""
    with pytest.raises(Exception):
        get_single_directional_rotation_matrix(3, jnp.pi / 2)

    with pytest.raises(Exception):
        get_single_directional_rotation_matrix(-1, jnp.pi / 2)


def test_rotate_vector_no_rotation():
    """Test vector rotation with zero angles."""
    vector = jnp.array([1.0, 2.0, 3.0])
    axes_tuple = (0, 1, 2)  # x, y, z
    result = rotate_vector(vector, 0.0, 0.0, axes_tuple)
    # With zero rotation, should get back original vector
    assert jnp.allclose(result, vector, atol=1e-6)


def test_rotate_vector_azimuth_only():
    """Test vector rotation with azimuth angle only."""
    vector = jnp.array([1.0, 0.0, 0.0])
    axes_tuple = (0, 1, 2)
    azimuth = jnp.pi / 2
    result = rotate_vector(vector, azimuth, 0.0, axes_tuple)
    # Should rotate around y-axis (vertical)
    expected = jnp.array([0.0, 0.0, -1.0])
    assert jnp.allclose(result, expected, atol=1e-6)


def test_rotate_vector_elevation_only():
    """Test vector rotation with elevation angle only."""
    vector = jnp.array([0.0, 1.0, 0.0])
    axes_tuple = (0, 1, 2)
    elevation = jnp.pi / 2
    result = rotate_vector(vector, 0.0, elevation, axes_tuple)
    # Should rotate around x-axis (horizontal)
    expected = jnp.array([0.0, 0.0, -1.0])
    assert jnp.allclose(result, expected, atol=1e-6)


def test_rotate_vector_different_axes_tuple():
    """Test vector rotation with different axes configuration."""
    vector = jnp.array([1.0, 0.0, 0.0])
    axes_tuple = (2, 0, 1)  # z, x, y
    result = rotate_vector(vector, 0.0, 0.0, axes_tuple)
    # With zero rotation, should still get back original vector
    assert jnp.allclose(result, vector, atol=1e-6)


def test_rotate_vector_combined_rotations():
    """Test vector rotation with both azimuth and elevation."""
    vector = jnp.array([1.0, 0.0, 0.0])
    axes_tuple = (0, 1, 2)
    azimuth = jnp.pi / 4  # 45 degrees
    elevation = jnp.pi / 4  # 45 degrees
    result = rotate_vector(vector, azimuth, elevation, axes_tuple)

    # Result should be normalized (assuming input was normalized)
    assert jnp.allclose(jnp.linalg.norm(result), 1.0, atol=1e-6)
