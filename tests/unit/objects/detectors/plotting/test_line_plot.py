"""Tests for objects/detectors/plotting/line_plot.py - line plot utilities."""

import matplotlib.pyplot as plt
import numpy as np

from fdtdx.objects.detectors.plotting.line_plot import (
    plot_line_over_time,
    plot_waterfall_over_time,
)


class TestPlotLineOverTime:
    """Tests for plot_line_over_time function."""

    def test_basic_line_plot(self):
        """Test basic line plot."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        time_steps = [0.0, 1.0, 2.0, 3.0, 4.0]

        fig = plot_line_over_time(arr, time_steps, metric_name="Energy")

        assert fig is not None
        plt.close(fig)

    def test_with_numpy_time_array(self):
        """Test with numpy array for time steps."""
        arr = np.sin(np.linspace(0, 2 * np.pi, 50))
        time_steps = np.linspace(0, 1e-12, 50)

        fig = plot_line_over_time(arr, time_steps, metric_name="Field")

        assert fig is not None
        plt.close(fig)

    def test_custom_xlabel(self):
        """Test with custom x-axis label."""
        arr = np.array([1.0, 2.0, 3.0])
        time_steps = [0.0, 1.0, 2.0]

        fig = plot_line_over_time(arr, time_steps, metric_name="Power", xlabel="Time [ps]")

        assert fig is not None
        plt.close(fig)

    def test_negative_values(self):
        """Test with negative values."""
        arr = np.array([-1.0, 0.0, 1.0, 0.0, -1.0])
        time_steps = [0.0, 0.25, 0.5, 0.75, 1.0]

        fig = plot_line_over_time(arr, time_steps, metric_name="Signal")

        assert fig is not None
        plt.close(fig)

    def test_constant_values(self):
        """Test with constant values."""
        arr = np.ones(10) * 5.0
        time_steps = list(range(10))

        fig = plot_line_over_time(arr, time_steps, metric_name="Constant")

        assert fig is not None
        plt.close(fig)

    def test_single_point(self):
        """Test with single data point."""
        arr = np.array([1.0])
        time_steps = [0.0]

        fig = plot_line_over_time(arr, time_steps, metric_name="Single")

        assert fig is not None
        plt.close(fig)

    def test_large_array(self):
        """Test with large array."""
        arr = np.random.rand(1000)
        time_steps = np.linspace(0, 1, 1000)

        fig = plot_line_over_time(arr, time_steps, metric_name="Random")

        assert fig is not None
        plt.close(fig)


class TestPlotWaterfallOverTime:
    """Tests for plot_waterfall_over_time function."""

    def test_basic_waterfall(self):
        """Test basic waterfall plot."""
        arr = np.random.rand(20, 30)  # time x space
        time_steps = np.linspace(0, 1e-12, 20)
        spatial_steps = np.linspace(0, 10e-6, 30)

        fig = plot_waterfall_over_time(arr, time_steps, spatial_steps, metric_name="Field Intensity")

        assert fig is not None
        plt.close(fig)

    def test_custom_units(self):
        """Test with custom time and spatial units."""
        arr = np.random.rand(20, 30)
        time_steps = np.linspace(0, 100, 20)
        spatial_steps = np.linspace(0, 50, 30)

        fig = plot_waterfall_over_time(
            arr,
            time_steps,
            spatial_steps,
            metric_name="Amplitude",
            time_unit="fs",
            spatial_unit="nm",
        )

        assert fig is not None
        plt.close(fig)

    def test_sinusoidal_pattern(self):
        """Test with sinusoidal pattern."""
        t = np.linspace(0, 2 * np.pi, 50)
        x = np.linspace(0, 2 * np.pi, 40)
        T, X = np.meshgrid(t, x, indexing="ij")
        arr = np.sin(T) * np.cos(X)

        time_steps = t
        spatial_steps = x

        fig = plot_waterfall_over_time(arr, time_steps, spatial_steps, metric_name="Wave")

        assert fig is not None
        plt.close(fig)

    def test_uniform_values(self):
        """Test with uniform values."""
        arr = np.ones((20, 30)) * 0.5
        time_steps = np.linspace(0, 1, 20)
        spatial_steps = np.linspace(0, 1, 30)

        fig = plot_waterfall_over_time(arr, time_steps, spatial_steps, metric_name="Uniform")

        assert fig is not None
        plt.close(fig)

    def test_negative_values(self):
        """Test with negative values."""
        arr = np.random.rand(20, 30) - 0.5
        time_steps = np.linspace(0, 1, 20)
        spatial_steps = np.linspace(0, 1, 30)

        fig = plot_waterfall_over_time(arr, time_steps, spatial_steps, metric_name="Bipolar")

        assert fig is not None
        plt.close(fig)

    def test_small_array(self):
        """Test with small array."""
        arr = np.random.rand(3, 3)
        time_steps = [0, 1, 2]
        spatial_steps = [0, 1, 2]

        fig = plot_waterfall_over_time(arr, time_steps, spatial_steps, metric_name="Small")

        assert fig is not None
        plt.close(fig)

    def test_rectangular_array(self):
        """Test with rectangular (non-square) array."""
        arr = np.random.rand(10, 50)
        time_steps = np.linspace(0, 1, 10)
        spatial_steps = np.linspace(0, 5, 50)

        fig = plot_waterfall_over_time(arr, time_steps, spatial_steps, metric_name="Rectangular")

        assert fig is not None
        plt.close(fig)

    def test_list_inputs(self):
        """Test with list inputs for steps."""
        arr = np.random.rand(5, 10)
        time_steps = [0.0, 0.1, 0.2, 0.3, 0.4]
        spatial_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        fig = plot_waterfall_over_time(arr, time_steps, spatial_steps, metric_name="List Input")

        assert fig is not None
        plt.close(fig)
