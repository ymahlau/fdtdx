from unittest.mock import MagicMock, patch

import numpy as np

from fdtdx.objects.detectors.plotting.line_plot import plot_waterfall_over_time


def test_waterfall_plot_direct():
    """Test waterfall plot functionality directly without detector dependencies"""
    # Create synthetic test data - a wave pattern moving across space over time
    time_steps = 20
    spatial_points = 10

    # Create 2D data (time × space)
    data = np.zeros((time_steps, spatial_points), dtype=np.float32)

    # Create a wave pattern that evolves over time
    for t in range(time_steps):
        for s in range(spatial_points):
            # Simple wave pattern that moves with time
            phase = 2 * np.pi * (t / time_steps + s / spatial_points)
            data[t, s] = np.sin(phase)

    # Create time and space coordinates
    time_values = np.linspace(0, 1e-13, time_steps)  # in seconds
    space_values = np.linspace(0, 1.0, spatial_points)  # in μm

    # Call the waterfall plot function directly
    with patch("matplotlib.pyplot.figure") as mock_figure:
        with patch("matplotlib.figure.Figure.colorbar"):
            mock_figure.return_value = MagicMock()
            # Call the waterfall plot function
            plot_waterfall_over_time(
                arr=data, time_steps=time_values.tolist(), spatial_steps=space_values.tolist(), metric_name="Test Field"
            )

    # Verify the plot creation was called
    mock_figure.assert_called_once()


def test_waterfall_visualization():
    """Test the waterfall plot visualization with detailed assertions"""
    # Create a small dataset for testing
    time_steps = 5
    spatial_points = 8
    data = np.zeros((time_steps, spatial_points), dtype=np.float32)

    # Create a simple gradient pattern
    for t in range(time_steps):
        for s in range(spatial_points):
            data[t, s] = t * s / (time_steps * spatial_points)

    # Create coordinate arrays
    times = np.linspace(0, 5e-14, time_steps)
    positions = np.linspace(0, 0.5, spatial_points)

    # Generate the actual figure without mocking
    fig = plot_waterfall_over_time(
        arr=data, time_steps=times.tolist(), spatial_steps=positions.tolist(), metric_name="Test Energy"
    )

    # Verify figure properties
    assert len(fig.axes) == 2  # Main plot + colorbar

    # Check main axis attributes
    ax = fig.axes[0]
    assert ax.get_xlabel().startswith("Time")
    assert ax.get_ylabel().startswith("Position")

    # Check that the data is properly displayed
    img = ax.get_images()[0]
    assert img.get_array().shape == (spatial_points, time_steps)  # Transposed in the plot

    # Clean up
    import matplotlib.pyplot as plt

    plt.close(fig)
