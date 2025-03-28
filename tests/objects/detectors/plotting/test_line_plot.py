import numpy as np
from matplotlib.figure import Figure

from fdtdx.objects.detectors.plotting.line_plot import plot_line_over_time, plot_waterfall_over_time


def test_plot_waterfall_over_time():
    """Test that waterfall plot function generates the expected figure"""
    # Create synthetic test data - a wave pattern moving across space over time
    time_steps = 10
    spatial_points = 20

    # Create data - a moving gaussian
    data = np.zeros((time_steps, spatial_points))
    for t in range(time_steps):
        peak_pos = int((t / time_steps) * spatial_points)
        for s in range(spatial_points):
            data[t, s] = np.exp(-0.5 * ((s - peak_pos) / 2) ** 2)

    # Time and space coordinates
    times = np.linspace(0, 1e-12, time_steps)  # in seconds
    positions = np.linspace(0, 1, spatial_points)  # in μm

    # Generate the plot
    fig = plot_waterfall_over_time(
        arr=data,
        time_steps=times.tolist(),
        spatial_steps=positions.tolist(),
        metric_name="Test Energy",
    )

    # Verify the plot was created with expected elements
    assert isinstance(fig, Figure)
    assert len(fig.axes) > 1  # Main axis + colorbar

    # Check axes labels
    main_ax = fig.axes[0]
    assert "Time" in main_ax.get_xlabel()
    assert "Position" in main_ax.get_ylabel()

    # Check colorbar exists and has label
    cbar_ax = fig.axes[1]
    assert "Test Energy" in cbar_ax.get_ylabel()


def test_plot_line_over_time():
    """Basic test for the existing line plot function"""
    # Create simple sine wave data
    time_steps = np.linspace(0, 1e-12, 10).tolist()
    values = np.sin(np.linspace(0, 2 * np.pi, 10))

    fig = plot_line_over_time(
        arr=values,
        time_steps=time_steps,
        metric_name="Test Metric",
    )

    assert isinstance(fig, Figure)
    assert "Time" in fig.axes[0].get_xlabel()
    assert "Test Metric" in fig.axes[0].get_ylabel()
