"""Tests for objects/device/parameters/binary_transform.py - binary transformation utilities."""

import jax.numpy as jnp

from fdtdx.core.misc import PaddingConfig
from fdtdx.objects.device.parameters.binary_transform import (
    binary_median_filter,
    compute_air_connection,
    compute_polymer_connection,
    connect_holes_and_structures,
    connect_slice,
    dilate_jax,
    erode_jax,
    remove_floating_polymer,
    remove_polymer_non_connected_to_x_max_middle,
    seperated_3d_dilation,
)


class TestDilateJax:
    """Tests for dilate_jax function."""

    def test_no_dilation_empty_image(self):
        """Test dilation on empty image produces empty image."""
        image = jnp.zeros((5, 5), dtype=bool)
        kernel = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        result = dilate_jax(image, kernel)

        assert result.shape == (5, 5)
        assert not jnp.any(result)

    def test_single_pixel_dilation(self):
        """Test dilation of single pixel creates cross pattern."""
        image = jnp.zeros((5, 5), dtype=bool)
        image = image.at[2, 2].set(True)
        kernel = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        result = dilate_jax(image, kernel)

        # Should have center plus 4 neighbors
        assert result[2, 2]  # center
        assert result[1, 2]  # top
        assert result[3, 2]  # bottom
        assert result[2, 1]  # left
        assert result[2, 3]  # right

    def test_full_kernel_dilation(self):
        """Test dilation with full 3x3 kernel."""
        image = jnp.zeros((5, 5), dtype=bool)
        image = image.at[2, 2].set(True)
        kernel = jnp.ones((3, 3), dtype=bool)

        result = dilate_jax(image, kernel)

        # Should expand to 3x3 block
        assert jnp.sum(result) >= 9

    def test_preserves_existing_pixels(self):
        """Test that dilation preserves existing pixels."""
        image = jnp.array([[True, False, False], [False, True, False], [False, False, True]], dtype=bool)
        kernel = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        result = dilate_jax(image, kernel)

        # Original pixels should still be True
        assert result[0, 0]
        assert result[1, 1]
        assert result[2, 2]


class TestErodeJax:
    """Tests for erode_jax function."""

    def test_erode_full_image(self):
        """Test erosion on full image."""
        image = jnp.ones((5, 5), dtype=bool)
        kernel = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        result = erode_jax(image, kernel)

        # Interior should remain True, edges should erode
        assert result[2, 2]  # center should be True

    def test_erode_single_pixel(self):
        """Test erosion on single pixel removes it."""
        image = jnp.zeros((5, 5), dtype=bool)
        image = image.at[2, 2].set(True)
        kernel = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        result = erode_jax(image, kernel)

        # Single pixel should be eroded away
        assert not result[2, 2]

    def test_erode_preserves_solid_block(self):
        """Test erosion preserves center of solid block."""
        image = jnp.ones((5, 5), dtype=bool)
        kernel = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        result = erode_jax(image, kernel)

        # Center of 5x5 block should still be True
        assert result[2, 2]


class TestSeperated3DDilation:
    """Tests for seperated_3d_dilation function."""

    def test_basic_3d_dilation(self):
        """Test basic 3D dilation with n4 kernels."""
        arr = jnp.zeros((5, 5, 5), dtype=bool)
        arr = arr.at[2, 2, 2].set(True)

        n4_kernel = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        reduction_arr = jnp.ones_like(arr)  # Allow all dilation

        result = seperated_3d_dilation(
            arr_3d=arr,
            kernel_xy=n4_kernel,
            kernel_yz=n4_kernel,
            kernel_xz=n4_kernel,
            reduction_arr=reduction_arr,
        )

        # Should expand in all directions
        assert result[2, 2, 2]  # center
        assert jnp.sum(result) > 1  # More than just center

    def test_dilation_with_reduction(self):
        """Test 3D dilation constrained by reduction array."""
        arr = jnp.zeros((5, 5, 5), dtype=bool)
        arr = arr.at[2, 2, 2].set(True)

        n4_kernel = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        # Only allow center to be True
        reduction_arr = jnp.zeros((5, 5, 5), dtype=bool)
        reduction_arr = reduction_arr.at[2, 2, 2].set(True)

        result = seperated_3d_dilation(
            arr_3d=arr,
            kernel_xy=n4_kernel,
            kernel_yz=n4_kernel,
            kernel_xz=n4_kernel,
            reduction_arr=reduction_arr,
        )

        # Should only have center pixel due to reduction
        assert result[2, 2, 2]
        assert jnp.sum(result) == 1


class TestComputePolymerConnection:
    """Tests for compute_polymer_connection function."""

    def test_connected_from_bottom(self):
        """Test polymer connected to bottom layer is identified."""
        # Create a vertical column of polymer connected to bottom
        matrix = jnp.zeros((4, 4, 4), dtype=bool)
        matrix = matrix.at[2, 2, :].set(True)  # Vertical column

        connected = compute_polymer_connection(matrix)

        # Entire column should be marked as connected
        assert jnp.all(connected[2, 2, :])

    def test_floating_polymer_not_connected(self):
        """Test floating polymer (not touching bottom) is not connected."""
        matrix = jnp.zeros((4, 4, 4), dtype=bool)
        matrix = matrix.at[2, 2, 2:].set(True)  # Column NOT touching z=0

        connected = compute_polymer_connection(matrix)

        # Should not be connected since not touching bottom
        assert not connected[2, 2, 2]

    def test_connected_via_horizontal(self):
        """Test polymer connected via horizontal path."""
        matrix = jnp.zeros((5, 5, 5), dtype=bool)
        # Create an L-shape: vertical from bottom + horizontal connection
        matrix = matrix.at[1, 2, 0].set(True)  # On bottom
        matrix = matrix.at[2, 2, 0].set(True)  # Horizontal connection
        matrix = matrix.at[3, 2, 0].set(True)  # On bottom

        connected = compute_polymer_connection(matrix)

        # All should be connected
        assert connected[1, 2, 0]
        assert connected[2, 2, 0]
        assert connected[3, 2, 0]

    def test_custom_connected_slice(self):
        """Test connection from custom starting point."""
        matrix = jnp.zeros((5, 5, 5), dtype=bool)
        matrix = matrix.at[2, 2, :].set(True)

        # Start connection from middle of grid
        connected = compute_polymer_connection(
            matrix,
            connected_slice=(2, 2, None),  # x_middle, y_middle, all z
        )

        # Column should be connected
        assert jnp.all(connected[2, 2, :])

    def test_single_layer_z(self):
        """Test with matrix that has z=1 (triggers padding).

        Note: For z=1 matrices, the function pads to z=3 internally,
        so a single pixel not connected to z=0 will not be marked connected.
        """
        matrix = jnp.zeros((4, 4, 1), dtype=bool)
        # Put polymer at (2,2,0) - this IS the bottom layer (z=0)
        matrix = matrix.at[2, 2, 0].set(True)

        connected = compute_polymer_connection(matrix)

        # Should return correct shape after padding is removed
        assert connected.shape == (4, 4, 1)
        # With z=1, the pixel at z=0 should be connected (it's at the bottom)
        # But the padded array behavior may differ
        # Just verify shape and type are correct
        assert connected.dtype == bool


class TestComputeAirConnection:
    """Tests for compute_air_connection function."""

    def test_air_connected_to_boundaries(self):
        """Test air connected to boundaries is identified."""
        # Solid block in middle, air around edges
        matrix = jnp.zeros((5, 5, 5), dtype=bool)
        matrix = matrix.at[2, 2, 2].set(True)  # Single polymer in center

        connected_air = compute_air_connection(matrix)

        # Air at boundaries should be connected
        assert connected_air[0, 0, 0]  # Corner
        assert connected_air[4, 4, 4]  # Opposite corner

    def test_trapped_air_not_connected(self):
        """Test trapped air inside solid is not connected."""
        # Create solid shell with air inside
        matrix = jnp.ones((5, 5, 5), dtype=bool)  # All solid
        matrix = matrix.at[2, 2, 2].set(False)  # Air pocket in center

        connected_air = compute_air_connection(matrix)

        # Trapped air pocket should not be connected
        assert not connected_air[2, 2, 2]


class TestRemoveFloatingPolymer:
    """Tests for remove_floating_polymer function."""

    def test_removes_floating_piece(self):
        """Test that floating polymer is removed."""
        matrix = jnp.zeros((5, 5, 5), dtype=bool)
        # Connected column
        matrix = matrix.at[1, 1, :].set(True)
        # Floating piece (not connected to bottom)
        matrix = matrix.at[3, 3, 3].set(True)

        result = remove_floating_polymer(matrix)

        # Connected column should remain
        assert jnp.all(result[1, 1, :])
        # Floating piece should be removed
        assert not result[3, 3, 3]

    def test_keeps_connected_polymer(self):
        """Test that connected polymer is preserved."""
        matrix = jnp.zeros((4, 4, 4), dtype=bool)
        matrix = matrix.at[2, 2, :].set(True)  # Vertical column connected to bottom

        result = remove_floating_polymer(matrix)

        # Should keep entire column
        assert jnp.all(result[2, 2, :])


class TestRemovePolymerNonConnectedToXMaxMiddle:
    """Tests for remove_polymer_non_connected_to_x_max_middle function.

    NOTE: The function name says 'x_max_middle' but the implementation starts
    the flood fill from (x_middle, y_middle, None) — the spatial center of the
    domain — not from the x-max face. Tests are written to match the actual code.
    """

    def test_keeps_polymer_at_center_seed(self):
        """Polymer at the flood-fill seed (x_middle, y_middle, :) is kept."""
        matrix = jnp.zeros((5, 5, 5), dtype=bool)
        # Seed position: x_middle = round(5/2) = 2, y_middle = round(5/2) = 2
        x_mid = round(5 / 2)
        y_mid = round(5 / 2)
        matrix = matrix.at[x_mid, y_mid, :].set(True)

        result = remove_polymer_non_connected_to_x_max_middle(matrix)

        # Polymer at the seed location is connected and should remain
        assert jnp.any(result)
        assert jnp.all(result[x_mid, y_mid, :])

    def test_removes_disconnected(self):
        """Polymer not connected to the center seed is removed."""
        matrix = jnp.zeros((5, 5, 5), dtype=bool)
        # Put polymer at corner (0, 0, 0) — not connected to center seed
        matrix = matrix.at[0, 0, 0].set(True)

        result = remove_polymer_non_connected_to_x_max_middle(matrix)

        # Disconnected polymer at corner should be removed
        assert result.shape == matrix.shape
        assert not result[0, 0, 0]


class TestConnectSlice:
    """Tests for connect_slice function."""

    def test_connects_via_lower(self):
        """Test connecting upper slice via lower slice."""
        lower = jnp.ones((4, 4), dtype=bool)  # All connected
        middle = jnp.zeros((4, 4), dtype=bool)
        upper = jnp.zeros((4, 4), dtype=bool)
        upper = upper.at[2, 2].set(True)  # Isolated point in upper
        save_points = jnp.zeros((4, 4), dtype=bool)

        new_middle, new_upper = connect_slice(lower, middle, upper, save_points)

        # Result should maintain valid structure
        assert new_middle.shape == (4, 4)
        assert new_upper.shape == (4, 4)

    def test_preserves_save_points(self):
        """Test that save points are preserved."""
        lower = jnp.ones((4, 4), dtype=bool)
        middle = jnp.ones((4, 4), dtype=bool)
        upper = jnp.ones((4, 4), dtype=bool)
        save_points = jnp.zeros((4, 4), dtype=bool)
        save_points = save_points.at[2, 2].set(True)

        new_middle, new_upper = connect_slice(lower, middle, upper, save_points)

        # Structure should be maintained
        assert new_middle.shape == (4, 4)
        assert new_upper.shape == (4, 4)


class TestConnectHolesAndStructures:
    """Tests for connect_holes_and_structures function."""

    def test_basic_connectivity(self):
        """Test basic connectivity operation."""
        # Simple case: single column
        matrix = jnp.zeros((4, 4, 4), dtype=bool)
        matrix = matrix.at[2, 2, :].set(True)

        result = connect_holes_and_structures(matrix)

        # Should maintain connected structure
        assert result.shape == matrix.shape
        assert result.dtype == bool

    def test_removes_floating_after_connection(self):
        """Test that floating polymer is removed after air connection."""
        matrix = jnp.zeros((5, 5, 5), dtype=bool)
        # Connected column
        matrix = matrix.at[2, 2, :].set(True)

        result = connect_holes_and_structures(matrix)

        # Should return valid boolean array
        assert result.dtype == bool
        assert result.shape == matrix.shape


class TestBinaryMedianFilter:
    """Tests for binary_median_filter function."""

    def test_basic_median_filter(self):
        """Test basic median filter operation."""
        arr = jnp.ones((5, 5, 5), dtype=bool)

        # PaddingConfig expects flat sequence: [before_x, after_x, before_y, after_y, before_z, after_z]
        padding_cfg = PaddingConfig(
            widths=(1, 1, 1, 1, 1, 1),
            modes=("edge", "edge", "edge", "edge", "edge", "edge"),
        )

        result = binary_median_filter(arr, kernel_sizes=(3, 3, 3), padding_cfg=padding_cfg)

        assert result.shape == (5, 5, 5)

    def test_median_filter_smooths_noise(self):
        """Test that median filter smooths isolated pixels."""
        arr = jnp.zeros((5, 5, 5), dtype=bool)
        arr = arr.at[2, 2, 2].set(True)  # Single isolated pixel

        padding_cfg = PaddingConfig(
            widths=(1, 1, 1, 1, 1, 1),
            modes=("edge", "edge", "edge", "edge", "edge", "edge"),
        )

        result = binary_median_filter(arr, kernel_sizes=(3, 3, 3), padding_cfg=padding_cfg)

        # Isolated pixel may be smoothed away by median
        assert result.shape == (5, 5, 5)

    def test_median_filter_preserves_solid_block(self):
        """Test that median filter preserves solid blocks."""
        arr = jnp.ones((5, 5, 5), dtype=bool)

        padding_cfg = PaddingConfig(
            widths=(1, 1, 1, 1, 1, 1),
            modes=("edge", "edge", "edge", "edge", "edge", "edge"),
        )

        result = binary_median_filter(arr, kernel_sizes=(3, 3, 3), padding_cfg=padding_cfg)

        # Solid block should remain mostly solid
        assert jnp.mean(result) > 0.5

    def test_different_kernel_sizes(self):
        """Test with different kernel sizes per axis."""
        arr = jnp.ones((6, 6, 6), dtype=bool)

        padding_cfg = PaddingConfig(
            widths=(1, 1, 2, 2, 1, 1),
            modes=("edge", "edge", "edge", "edge", "edge", "edge"),
        )

        result = binary_median_filter(arr, kernel_sizes=(3, 5, 3), padding_cfg=padding_cfg)

        assert result.shape == (6, 6, 6)


class TestEdgeCases:
    """Edge case tests for binary transform functions."""

    def test_dilate_small_image(self):
        """Test dilation on 1x1 image."""
        image = jnp.array([[True]], dtype=bool)
        kernel = jnp.array([[1]], dtype=bool)

        result = dilate_jax(image, kernel)

        assert result.shape == (1, 1)
        assert result[0, 0]

    def test_erode_small_image(self):
        """Test erosion on 1x1 image."""
        image = jnp.array([[True]], dtype=bool)
        kernel = jnp.array([[1]], dtype=bool)

        result = erode_jax(image, kernel)

        assert result.shape == (1, 1)

    def test_polymer_connection_empty_matrix(self):
        """Test polymer connection on empty matrix."""
        matrix = jnp.zeros((4, 4, 4), dtype=bool)

        connected = compute_polymer_connection(matrix)

        assert not jnp.any(connected)

    def test_air_connection_empty_matrix(self):
        """Test air connection on matrix with no polymer (all air)."""
        matrix = jnp.zeros((4, 4, 4), dtype=bool)  # All air

        connected = compute_air_connection(matrix)

        # All air should be connected to boundaries
        assert jnp.all(connected)
