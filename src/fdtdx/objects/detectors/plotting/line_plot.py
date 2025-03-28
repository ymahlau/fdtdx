import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_line_over_time(
    arr: np.ndarray,
    time_steps: list[float],  # in seconds
    metric_name: str,
    xlabel: str = "Time [s]",
):
    """Creates a line plot showing the evolution of a metric over time.

    Uses seaborn's lineplot to visualize how a measured quantity changes over the simulation
    time steps. The plot includes proper axis labels with units.

    Args:
        arr: 1D numpy array containing the metric values at each time step
        time_steps: List of simulation time points in seconds
        metric_name: Name of the metric being plotted, used for y-axis label
        xlabel: Label for the x-axis, default is "Time [s]"

    Returns:
        matplotlib.figure.Figure: Figure containing the line plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.lineplot(x=time_steps, y=arr, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_name)
    return fig


def plot_waterfall_over_time(
    arr: np.ndarray,
    time_steps: list[float],
    spatial_steps: list[float],
    metric_name: str,
    time_unit: str = "s",
    spatial_unit: str = "μm",
):
    """Creates a 2D waterfall/heatmap plot showing the evolution of spatial data over time.

    Visualizes how 1D spatial data changes across multiple time steps using a 2D heatmap
    with proper axis labels and a colorbar.

    Args:
        arr: 2D numpy array containing values with shape (time_steps, spatial_points)
        time_steps: List of simulation time points
        spatial_steps: List of spatial coordinates
        metric_name: Name of the metric being plotted, used for colorbar label
        time_unit: Unit for the time axis, default is "s"
        spatial_unit: Unit for the spatial axis, default is "μm"

    Returns:
        matplotlib.figure.Figure: Figure containing the waterfall plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a meshgrid for proper extent
    t_min, t_max = min(time_steps), max(time_steps)
    s_min, s_max = min(spatial_steps), max(spatial_steps)

    # Plot as a heatmap/waterfall
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    im = ax.imshow(
        arr.T,  # Transpose to have time on x-axis, space on y-axis
        aspect="auto",
        origin="lower",
        extent=[t_min, t_max, s_min, s_max],
        interpolation="gaussian",
        cmap=cmap,
    )

    # Add labels
    ax.set_xlabel(f"Time [{time_unit}]")
    ax.set_ylabel(f"Position [{spatial_unit}]")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_name)

    fig.tight_layout()
    return fig
