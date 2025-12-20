"""Tests for symmetry parameter transforms."""

import jax.numpy as jnp
import pytest

from fdtdx import (
    DiagonalSymmetry2D,
    DiagonalSymmetry3D,
    HorizontalSymmetry2D,
    HorizontalSymmetry3D,
    PointSymmetry2D,
    PointSymmetry3D,
    VerticalSymmetry2D,
    VerticalSymmetry3D,
)

# =============================================================================
# 2D Symmetry Tests
# =============================================================================


class TestHorizontalSymmetry2D:
    """Tests for HorizontalSymmetry2D (x-axis mirror)."""

    def test_symmetry_property(self):
        """Test that output satisfies horizontal symmetry."""
        # Create 2D array with singleton middle dimension (required by 2D transforms)
        test_arr = jnp.array(
            [
                [[1, 2, 3, 4]],
                [[5, 6, 7, 8]],
                [[9, 10, 11, 12]],
                [[13, 14, 15, 16]],
            ],
            dtype=jnp.float32,
        )

        sym = HorizontalSymmetry2D()
        result = sym({"test": test_arr})["test"]
        squeezed = result.squeeze(1)

        # Check symmetry: arr == arr[::-1, :]
        assert jnp.allclose(squeezed, squeezed[::-1, :])

    def test_point_reflection(self):
        """Test that a single point is correctly mirrored."""
        test_arr = jnp.zeros((5, 1, 5), dtype=jnp.float32)
        test_arr = test_arr.at[1, 0, 1].set(1.0)

        sym = HorizontalSymmetry2D()
        result = sym({"test": test_arr})["test"]

        # Point at [1, 0, 1] should mirror to [3, 0, 1]
        assert result[1, 0, 1] == 0.5
        assert result[3, 0, 1] == 0.5

    def test_shape_preserved(self):
        """Test that output shape matches input shape."""
        test_arr = jnp.ones((4, 1, 6), dtype=jnp.float32)
        sym = HorizontalSymmetry2D()
        result = sym({"test": test_arr})["test"]
        assert result.shape == test_arr.shape

    def test_multiple_arrays(self):
        """Test that transform works on multiple arrays in dict."""
        params = {
            "a": jnp.ones((4, 1, 4), dtype=jnp.float32),
            "b": jnp.ones((6, 1, 6), dtype=jnp.float32),
        }
        sym = HorizontalSymmetry2D()
        result = sym(params)
        assert "a" in result and "b" in result
        assert result["a"].shape == (4, 1, 4)
        assert result["b"].shape == (6, 1, 6)


class TestVerticalSymmetry2D:
    """Tests for VerticalSymmetry2D (y-axis mirror)."""

    def test_symmetry_property(self):
        """Test that output satisfies vertical symmetry."""
        test_arr = jnp.array(
            [
                [[1, 2, 3, 4]],
                [[5, 6, 7, 8]],
                [[9, 10, 11, 12]],
                [[13, 14, 15, 16]],
            ],
            dtype=jnp.float32,
        )

        sym = VerticalSymmetry2D()
        result = sym({"test": test_arr})["test"]
        squeezed = result.squeeze(1)

        # Check symmetry: arr == arr[:, ::-1]
        assert jnp.allclose(squeezed, squeezed[:, ::-1])

    def test_point_reflection(self):
        """Test that a single point is correctly mirrored."""
        test_arr = jnp.zeros((5, 1, 5), dtype=jnp.float32)
        test_arr = test_arr.at[1, 0, 1].set(1.0)

        sym = VerticalSymmetry2D()
        result = sym({"test": test_arr})["test"]

        # Point at [1, 0, 1] should mirror to [1, 0, 3]
        assert result[1, 0, 1] == 0.5
        assert result[1, 0, 3] == 0.5

    def test_shape_preserved(self):
        """Test that output shape matches input shape."""
        test_arr = jnp.ones((4, 1, 6), dtype=jnp.float32)
        sym = VerticalSymmetry2D()
        result = sym({"test": test_arr})["test"]
        assert result.shape == test_arr.shape


class TestPointSymmetry2D:
    """Tests for PointSymmetry2D (180-degree rotation)."""

    def test_symmetry_property(self):
        """Test that output satisfies 180-degree rotational symmetry."""
        test_arr = jnp.array(
            [
                [[1, 2, 3, 4]],
                [[5, 6, 7, 8]],
                [[9, 10, 11, 12]],
                [[13, 14, 15, 16]],
            ],
            dtype=jnp.float32,
        )

        sym = PointSymmetry2D()
        result = sym({"test": test_arr})["test"]
        squeezed = result.squeeze(1)

        # Check symmetry: arr == arr[::-1, ::-1]
        assert jnp.allclose(squeezed, squeezed[::-1, ::-1])

    def test_point_reflection(self):
        """Test that a single point is correctly rotated."""
        test_arr = jnp.zeros((5, 1, 5), dtype=jnp.float32)
        test_arr = test_arr.at[1, 0, 1].set(1.0)

        sym = PointSymmetry2D()
        result = sym({"test": test_arr})["test"]

        # Point at [1, 0, 1] should rotate to [3, 0, 3]
        assert result[1, 0, 1] == 0.5
        assert result[3, 0, 3] == 0.5

    def test_shape_preserved(self):
        """Test that output shape matches input shape."""
        test_arr = jnp.ones((4, 1, 6), dtype=jnp.float32)
        sym = PointSymmetry2D()
        result = sym({"test": test_arr})["test"]
        assert result.shape == test_arr.shape


class TestDiagonalSymmetry2D:
    """Tests for DiagonalSymmetry2D."""

    def test_main_diagonal_symmetry(self):
        """Test main diagonal (min_min_to_max_max=True) symmetry."""
        test_arr = jnp.zeros((4, 1, 4), dtype=jnp.float32)
        test_arr = test_arr.at[1, 0, 0].set(1.0)

        sym = DiagonalSymmetry2D(min_min_to_max_max=True)
        result = sym({"test": test_arr})["test"]
        squeezed = result.squeeze(1)

        # Check symmetry: arr == arr.T
        assert jnp.allclose(squeezed, squeezed.T)

        # Point at [1, 0] should mirror to [0, 1]
        assert squeezed[1, 0] == 0.5
        assert squeezed[0, 1] == 0.5

    def test_anti_diagonal_symmetry(self):
        """Test anti-diagonal (min_min_to_max_max=False) symmetry."""
        test_arr = jnp.zeros((4, 1, 4), dtype=jnp.float32)
        test_arr = test_arr.at[1, 0, 0].set(1.0)

        sym = DiagonalSymmetry2D(min_min_to_max_max=False)
        result = sym({"test": test_arr})["test"]
        squeezed = result.squeeze(1)

        # Check anti-diagonal symmetry
        flipped = squeezed[::-1, ::-1]
        assert jnp.allclose(squeezed, flipped.T)

        # Point at [1, 0] should mirror to [3, 2]
        assert squeezed[1, 0] == 0.5
        assert squeezed[3, 2] == 0.5

    def test_shape_preserved(self):
        """Test that output shape matches input shape."""
        test_arr = jnp.ones((4, 1, 4), dtype=jnp.float32)
        sym = DiagonalSymmetry2D(min_min_to_max_max=True)
        result = sym({"test": test_arr})["test"]
        assert result.shape == test_arr.shape


# =============================================================================
# 3D Symmetry Tests
# =============================================================================


class TestHorizontalSymmetry3D:
    """Tests for HorizontalSymmetry3D."""

    def test_x_axis_symmetry(self):
        """Test mirror symmetry along x-axis (default)."""
        test_arr = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        test_arr = test_arr.at[1, 0, 0].set(1.0)

        sym = HorizontalSymmetry3D(mirror_axis="x")
        result = sym({"test": test_arr})["test"]

        # Check symmetry: arr == jnp.flip(arr, axis=0)
        assert jnp.allclose(result, jnp.flip(result, axis=0))

        # Point at [1, 0, 0] should mirror to [2, 0, 0]
        assert result[1, 0, 0] == 0.5
        assert result[2, 0, 0] == 0.5

    def test_y_axis_symmetry(self):
        """Test mirror symmetry along y-axis."""
        test_arr = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        test_arr = test_arr.at[0, 1, 0].set(1.0)

        sym = HorizontalSymmetry3D(mirror_axis="y")
        result = sym({"test": test_arr})["test"]

        # Check symmetry: arr == jnp.flip(arr, axis=1)
        assert jnp.allclose(result, jnp.flip(result, axis=1))

        # Point at [0, 1, 0] should mirror to [0, 2, 0]
        assert result[0, 1, 0] == 0.5
        assert result[0, 2, 0] == 0.5

    def test_default_is_x_axis(self):
        """Test that default mirror_axis is 'x'."""
        sym = HorizontalSymmetry3D()
        assert sym.mirror_axis == "x"

    def test_invalid_axis_raises(self):
        """Test that invalid mirror_axis raises ValueError."""
        test_arr = jnp.ones((4, 4, 4), dtype=jnp.float32)
        sym = HorizontalSymmetry3D(mirror_axis="z")

        with pytest.raises(ValueError, match="mirror_axis must be 'x' or 'y'"):
            sym({"test": test_arr})

    def test_shape_preserved(self):
        """Test that output shape matches input shape."""
        test_arr = jnp.ones((4, 5, 6), dtype=jnp.float32)
        sym = HorizontalSymmetry3D()
        result = sym({"test": test_arr})["test"]
        assert result.shape == test_arr.shape


class TestVerticalSymmetry3D:
    """Tests for VerticalSymmetry3D (z-axis mirror)."""

    def test_z_axis_symmetry(self):
        """Test mirror symmetry along z-axis."""
        test_arr = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        test_arr = test_arr.at[0, 0, 1].set(1.0)

        sym = VerticalSymmetry3D()
        result = sym({"test": test_arr})["test"]

        # Check symmetry: arr == jnp.flip(arr, axis=2)
        assert jnp.allclose(result, jnp.flip(result, axis=2))

        # Point at [0, 0, 1] should mirror to [0, 0, 2]
        assert result[0, 0, 1] == 0.5
        assert result[0, 0, 2] == 0.5

    def test_shape_preserved(self):
        """Test that output shape matches input shape."""
        test_arr = jnp.ones((4, 5, 6), dtype=jnp.float32)
        sym = VerticalSymmetry3D()
        result = sym({"test": test_arr})["test"]
        assert result.shape == test_arr.shape


class TestPointSymmetry3D:
    """Tests for PointSymmetry3D (180-degree rotation in 3D)."""

    def test_point_symmetry(self):
        """Test 180-degree rotational symmetry."""
        test_arr = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        test_arr = test_arr.at[1, 0, 0].set(1.0)

        sym = PointSymmetry3D()
        result = sym({"test": test_arr})["test"]

        # Check symmetry: arr == arr[::-1, ::-1, ::-1]
        assert jnp.allclose(result, result[::-1, ::-1, ::-1])

        # Point at [1, 0, 0] should rotate to [2, 3, 3]
        assert result[1, 0, 0] == 0.5
        assert result[2, 3, 3] == 0.5

    def test_shape_preserved(self):
        """Test that output shape matches input shape."""
        test_arr = jnp.ones((4, 5, 6), dtype=jnp.float32)
        sym = PointSymmetry3D()
        result = sym({"test": test_arr})["test"]
        assert result.shape == test_arr.shape


class TestDiagonalSymmetry3D:
    """Tests for DiagonalSymmetry3D with all 6 diagonal planes."""

    def test_xy_main_diagonal(self):
        """Test xy-plane main diagonal symmetry."""
        test_arr = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        test_arr = test_arr.at[1, 0, 0].set(1.0)

        sym = DiagonalSymmetry3D(diagonal_plane="xy", min_min_to_max_max=True)
        result = sym({"test": test_arr})["test"]

        # Check symmetry: arr == jnp.transpose(arr, (1, 0, 2))
        assert jnp.allclose(result, jnp.transpose(result, (1, 0, 2)))

        # Point at [1, 0, 0] should mirror to [0, 1, 0]
        assert result[1, 0, 0] == 0.5
        assert result[0, 1, 0] == 0.5

    def test_xy_anti_diagonal(self):
        """Test xy-plane anti-diagonal symmetry."""
        test_arr = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        test_arr = test_arr.at[1, 0, 0].set(1.0)

        sym = DiagonalSymmetry3D(diagonal_plane="xy", min_min_to_max_max=False)
        result = sym({"test": test_arr})["test"]

        # Point at [1, 0, 0] should mirror to [3, 2, 0]
        assert result[1, 0, 0] == 0.5
        assert result[3, 2, 0] == 0.5

    def test_xz_main_diagonal(self):
        """Test xz-plane main diagonal symmetry."""
        test_arr = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        test_arr = test_arr.at[1, 0, 0].set(1.0)

        sym = DiagonalSymmetry3D(diagonal_plane="xz", min_min_to_max_max=True)
        result = sym({"test": test_arr})["test"]

        # Check symmetry: arr == jnp.transpose(arr, (2, 1, 0))
        assert jnp.allclose(result, jnp.transpose(result, (2, 1, 0)))

        # Point at [1, 0, 0] should mirror to [0, 0, 1]
        assert result[1, 0, 0] == 0.5
        assert result[0, 0, 1] == 0.5

    def test_xz_anti_diagonal(self):
        """Test xz-plane anti-diagonal symmetry."""
        test_arr = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        test_arr = test_arr.at[1, 0, 0].set(1.0)

        sym = DiagonalSymmetry3D(diagonal_plane="xz", min_min_to_max_max=False)
        result = sym({"test": test_arr})["test"]

        # Point at [1, 0, 0] should mirror to [3, 0, 2]
        assert result[1, 0, 0] == 0.5
        assert result[3, 0, 2] == 0.5

    def test_yz_main_diagonal(self):
        """Test yz-plane main diagonal symmetry."""
        test_arr = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        test_arr = test_arr.at[0, 1, 0].set(1.0)

        sym = DiagonalSymmetry3D(diagonal_plane="yz", min_min_to_max_max=True)
        result = sym({"test": test_arr})["test"]

        # Check symmetry: arr == jnp.transpose(arr, (0, 2, 1))
        assert jnp.allclose(result, jnp.transpose(result, (0, 2, 1)))

        # Point at [0, 1, 0] should mirror to [0, 0, 1]
        assert result[0, 1, 0] == 0.5
        assert result[0, 0, 1] == 0.5

    def test_yz_anti_diagonal(self):
        """Test yz-plane anti-diagonal symmetry."""
        test_arr = jnp.zeros((4, 4, 4), dtype=jnp.float32)
        test_arr = test_arr.at[1, 0, 0].set(1.0)

        sym = DiagonalSymmetry3D(diagonal_plane="yz", min_min_to_max_max=False)
        result = sym({"test": test_arr})["test"]

        # Point at [1, 0, 0] should mirror to [1, 3, 3]
        assert result[1, 0, 0] == 0.5
        assert result[1, 3, 3] == 0.5

    def test_default_values(self):
        """Test default parameter values."""
        sym = DiagonalSymmetry3D()
        assert sym.diagonal_plane == "xy"
        assert sym.min_min_to_max_max is True

    def test_invalid_plane_raises(self):
        """Test that invalid diagonal_plane raises ValueError."""
        test_arr = jnp.ones((4, 4, 4), dtype=jnp.float32)
        sym = DiagonalSymmetry3D(diagonal_plane="invalid")

        with pytest.raises(ValueError, match="diagonal_plane must be"):
            sym({"test": test_arr})

    def test_shape_preserved(self):
        """Test that output shape matches input shape."""
        test_arr = jnp.ones((4, 4, 4), dtype=jnp.float32)
        sym = DiagonalSymmetry3D()
        result = sym({"test": test_arr})["test"]
        assert result.shape == test_arr.shape


# =============================================================================
# Integration Tests
# =============================================================================


class TestSymmetryComposition:
    """Test composing multiple symmetries."""

    def test_horizontal_and_vertical_2d(self):
        """Test that combining horizontal and vertical gives point symmetry."""
        test_arr = jnp.zeros((5, 1, 5), dtype=jnp.float32)
        test_arr = test_arr.at[1, 0, 1].set(1.0)

        h_sym = HorizontalSymmetry2D()
        v_sym = VerticalSymmetry2D()

        # Apply both symmetries
        result = h_sym({"test": test_arr})
        result = v_sym(result)["test"]

        # Result should have 4 symmetric points
        squeezed = result.squeeze(1)
        assert squeezed[1, 1] == 0.25
        assert squeezed[3, 1] == 0.25
        assert squeezed[1, 3] == 0.25
        assert squeezed[3, 3] == 0.25

    def test_idempotence(self):
        """Test that applying symmetry twice gives same result."""
        test_arr = jnp.array(
            [
                [[1, 2, 3, 4]],
                [[5, 6, 7, 8]],
                [[9, 10, 11, 12]],
                [[13, 14, 15, 16]],
            ],
            dtype=jnp.float32,
        )

        sym = HorizontalSymmetry2D()
        result1 = sym({"test": test_arr})["test"]
        result2 = sym({"test": result1})["test"]

        # Applying symmetry to already-symmetric array should give same result
        assert jnp.allclose(result1, result2)
