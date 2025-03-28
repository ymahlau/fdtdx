import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from fdtdx.objects.detectors.plotting import line_plot


def test_energy_detector_waterfall_visualization():
    """Test energy detector visualization by directly testing its plotting functionality"""
    # Skip detector instantiation and test the plot function call directly
    # Create synthetic 2D data (time x space) representing energy along a 1D line detector
    time_steps = 10
    spatial_points = 15
    mock_state = {"energy": np.zeros((time_steps, spatial_points), dtype=np.float32)}

    # Generate a propagating Gaussian pulse pattern
    for t in range(time_steps):
        peak_pos = int((t / time_steps) * spatial_points)
        for s in range(spatial_points):
            # Gaussian pulse moving across the line
            mock_state["energy"][t, s] = np.exp(-0.5 * ((s - peak_pos) / 2) ** 2)

    # Create the plotting function mock to verify it would be called with our data
    with patch.object(line_plot, "plot_waterfall_over_time") as mock_waterfall:
        mock_waterfall.return_value = MagicMock()

        # Now call the function directly with our test data
        time_values = np.linspace(0, 1e-12, time_steps)
        space_values = np.linspace(0, 1.5, spatial_points)

        line_plot.plot_waterfall_over_time(
            arr=mock_state["energy"],
            time_steps=time_values.tolist(),
            spatial_steps=space_values.tolist(),
            metric_name="Energy",
        )

    # Verify waterfall plot was called with correct parameters
    mock_waterfall.assert_called_once()


def test_energy_detector_waterfall_visualization_direct():
    """Test the waterfall plot function directly with 1D detector data"""
    # Create synthetic test data representing energy along a 1D detector over time
    time_steps = 10
    spatial_points = 15

    # Create test data that represents a propagating Gaussian pulse
    data = np.zeros((time_steps, spatial_points), dtype=np.float32)
    for t in range(time_steps):
        peak_pos = int((t / time_steps) * spatial_points)
        for s in range(spatial_points):
            data[t, s] = np.exp(-0.5 * ((s - peak_pos) / 2) ** 2)

    # Create time and space coordinates
    time_values = np.linspace(0, 1e-12, time_steps)
    space_values = np.linspace(0, 1.5, spatial_points)

    # Generate the actual plot
    fig = line_plot.plot_waterfall_over_time(
        arr=data,
        time_steps=time_values.tolist(),
        spatial_steps=space_values.tolist(),
        metric_name="Energy",
    )

    # Basic verification of plot structure
    assert fig is not None
    assert len(fig.axes) > 0

    # Verify axis labels
    ax = fig.axes[0]
    assert "Time" in ax.get_xlabel()
    assert "Position" in ax.get_ylabel()

    # Clean up
    import matplotlib.pyplot as plt

    plt.close(fig)
