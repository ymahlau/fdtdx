"""Tests for objects/detectors/plotting/plot2d.py - 2D slice plotting."""

import matplotlib.pyplot as plt
import numpy as np

from fdtdx.objects.detectors.plotting.plot2d import plot_2d_from_slices, plot_grads


class TestPlot2dFromSlices:
    """Tests for plot_2d_from_slices function."""

    def test_basic_slices(self):
        """Test plotting with basic slice arrays."""
        xy_slice = np.random.rand(10, 10)
        xz_slice = np.random.rand(10, 8)
        yz_slice = np.random.rand(10, 8)
        resolutions = (1e-6, 1e-6, 1e-6)

        fig = plot_2d_from_slices(xy_slice, xz_slice, yz_slice, resolutions)

        assert fig is not None
        assert len(fig.axes) == 6  # 3 plots + 3 colorbars
        plt.close(fig)

    def test_with_custom_minmax(self):
        """Test plotting with custom min/max values."""
        xy_slice = np.random.rand(10, 10)
        xz_slice = np.random.rand(10, 8)
        yz_slice = np.random.rand(10, 8)
        resolutions = (1e-6, 1e-6, 1e-6)

        fig = plot_2d_from_slices(
            xy_slice,
            xz_slice,
            yz_slice,
            resolutions,
            minvals=(-1.0, -1.0, -1.0),
            maxvals=(1.0, 1.0, 1.0),
        )

        assert fig is not None
        plt.close(fig)

    def test_with_partial_minmax(self):
        """Test plotting with some None values for min/max."""
        xy_slice = np.random.rand(10, 10)
        xz_slice = np.random.rand(10, 8)
        yz_slice = np.random.rand(10, 8)
        resolutions = (1e-6, 1e-6, 1e-6)

        fig = plot_2d_from_slices(
            xy_slice,
            xz_slice,
            yz_slice,
            resolutions,
            minvals=(None, -0.5, None),
            maxvals=(0.5, None, None),
        )

        assert fig is not None
        plt.close(fig)

    def test_different_interpolations(self):
        """Test with different interpolation methods."""
        xy_slice = np.random.rand(10, 10)
        xz_slice = np.random.rand(10, 8)
        yz_slice = np.random.rand(10, 8)
        resolutions = (1e-6, 1e-6, 1e-6)

        for interp in ["nearest", "bilinear", "gaussian"]:
            fig = plot_2d_from_slices(xy_slice, xz_slice, yz_slice, resolutions, plot_interpolation=interp)
            assert fig is not None
            plt.close(fig)

    def test_with_custom_dpi(self):
        """Test plotting with custom DPI."""
        xy_slice = np.random.rand(10, 10)
        xz_slice = np.random.rand(10, 8)
        yz_slice = np.random.rand(10, 8)
        resolutions = (1e-6, 1e-6, 1e-6)

        fig = plot_2d_from_slices(xy_slice, xz_slice, yz_slice, resolutions, plot_dpi=150)

        assert fig is not None
        plt.close(fig)

    def test_different_resolutions(self):
        """Test with different spatial resolutions."""
        xy_slice = np.random.rand(10, 10)
        xz_slice = np.random.rand(10, 8)
        yz_slice = np.random.rand(10, 8)
        resolutions = (1e-6, 2e-6, 0.5e-6)  # Different resolutions

        fig = plot_2d_from_slices(xy_slice, xz_slice, yz_slice, resolutions)

        assert fig is not None
        plt.close(fig)

    def test_negative_values(self):
        """Test with negative values."""
        xy_slice = np.random.rand(10, 10) - 0.5
        xz_slice = np.random.rand(10, 8) - 0.5
        yz_slice = np.random.rand(10, 8) - 0.5
        resolutions = (1e-6, 1e-6, 1e-6)

        fig = plot_2d_from_slices(xy_slice, xz_slice, yz_slice, resolutions)

        assert fig is not None
        plt.close(fig)

    def test_uniform_values(self):
        """Test with uniform values."""
        xy_slice = np.ones((10, 10)) * 0.5
        xz_slice = np.ones((10, 8)) * 0.5
        yz_slice = np.ones((10, 8)) * 0.5
        resolutions = (1e-6, 1e-6, 1e-6)

        fig = plot_2d_from_slices(xy_slice, xz_slice, yz_slice, resolutions)

        assert fig is not None
        plt.close(fig)

    def test_small_slices(self):
        """Test with very small slice arrays."""
        xy_slice = np.random.rand(2, 2)
        xz_slice = np.random.rand(2, 2)
        yz_slice = np.random.rand(2, 2)
        resolutions = (1e-6, 1e-6, 1e-6)

        fig = plot_2d_from_slices(xy_slice, xz_slice, yz_slice, resolutions)

        assert fig is not None
        plt.close(fig)

    def test_large_slices(self):
        """Test with larger slice arrays."""
        xy_slice = np.random.rand(100, 100)
        xz_slice = np.random.rand(100, 50)
        yz_slice = np.random.rand(100, 50)
        resolutions = (1e-6, 1e-6, 1e-6)

        fig = plot_2d_from_slices(xy_slice, xz_slice, yz_slice, resolutions)

        assert fig is not None
        plt.close(fig)


class TestPlotGrads:
    """Tests for plot_grads function."""

    def test_basic_3d_gradient(self):
        """Test plotting gradients from 3D array."""
        grad_arr = np.random.rand(10, 10, 10)
        voxel_size = (1e-6, 1e-6, 1e-6)

        fig = plot_grads(grad_arr, voxel_size)

        assert fig is not None
        plt.close(fig)

    def test_non_cubic_gradient(self):
        """Test with non-cubic gradient array."""
        grad_arr = np.random.rand(20, 10, 5)
        voxel_size = (1e-6, 2e-6, 0.5e-6)

        fig = plot_grads(grad_arr, voxel_size)

        assert fig is not None
        plt.close(fig)

    def test_negative_gradients(self):
        """Test with negative gradient values."""
        grad_arr = np.random.rand(10, 10, 10) - 0.5
        voxel_size = (1e-6, 1e-6, 1e-6)

        fig = plot_grads(grad_arr, voxel_size)

        assert fig is not None
        plt.close(fig)

    def test_uniform_gradients(self):
        """Test with uniform gradient values."""
        grad_arr = np.ones((10, 10, 10)) * 0.1
        voxel_size = (1e-6, 1e-6, 1e-6)

        fig = plot_grads(grad_arr, voxel_size)

        assert fig is not None
        plt.close(fig)

    def test_zero_gradients(self):
        """Test with zero gradients."""
        grad_arr = np.zeros((10, 10, 10))
        voxel_size = (1e-6, 1e-6, 1e-6)

        fig = plot_grads(grad_arr, voxel_size)

        assert fig is not None
        plt.close(fig)

    def test_returns_figure(self):
        """Test that function returns matplotlib Figure."""
        from matplotlib.figure import Figure

        grad_arr = np.random.rand(10, 10, 10)
        voxel_size = (1e-6, 1e-6, 1e-6)

        fig = plot_grads(grad_arr, voxel_size)

        assert isinstance(fig, Figure)
        plt.close(fig)
