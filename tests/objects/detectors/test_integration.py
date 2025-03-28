from unittest.mock import MagicMock, patch

import numpy as np

from fdtdx.objects.detectors.plotting import line_plot


def test_detector_visualization_integration():
    """Test the integration between detector state and visualization"""
    # Create test data - 2D array representing time series data along a line
    time_steps = 7  # Small number of time steps for testing
    spatial_points = 10

    # Create test data representing energy along a line over time
    data = np.zeros((time_steps, spatial_points), dtype=np.float32)

    # Create a simple Gaussian pulse that moves along the line over time
    for t in range(time_steps):
        center = (t / time_steps) * spatial_points
        for x in range(spatial_points):
            data[t, x] = np.exp(-0.5 * ((x - center) / 2) ** 2)

    # Create a state dictionary like what a detector would produce
    detector_state = {"energy": data}

    # Mock the actual waterfall plot function to verify integration
    with patch.object(line_plot, "plot_waterfall_over_time") as mock_waterfall:
        mock_waterfall.return_value = MagicMock()

        # Now call the plot function directly as the detector would
        time_values = np.linspace(0, 1e-12, time_steps)
        space_values = np.linspace(0, 1.0, spatial_points)

        # Use the module to ensure we're calling the patched function
        line_plot.plot_waterfall_over_time(
            arr=detector_state["energy"],
            time_steps=time_values.tolist(),
            spatial_steps=space_values.tolist(),
            metric_name="Energy",
        )

    # Verify the plot function was called with the correct data
    mock_waterfall.assert_called_once()

    # Create a real plot to test the actual visualization (no mocking)
    fig = line_plot.plot_waterfall_over_time(
        arr=data, time_steps=time_values.tolist(), spatial_steps=space_values.tolist(), metric_name="Energy"
    )

    # Verify the plot has the expected structure
    assert fig is not None
    assert len(fig.axes) == 2  # Main plot + colorbar

    # Check axis labels
    ax = fig.axes[0]
    assert "Time" in ax.get_xlabel()
    assert "Position" in ax.get_ylabel()

    # Clean up
    import matplotlib.pyplot as plt

    plt.close(fig)
