import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_line_over_time(
    arr: np.ndarray,
    time_steps: list[float],  # in seconds
    metric_name: str,
):
    """Creates a line plot showing the evolution of a metric over time.

    Uses seaborn's lineplot to visualize how a measured quantity changes over the simulation
    time steps. The plot includes proper axis labels with units.

    Args:
        arr: 1D numpy array containing the metric values at each time step
        time_steps: List of simulation time points in seconds
        metric_name: Name of the metric being plotted, used for y-axis label

    Returns:
        matplotlib.figure.Figure: Figure containing the line plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.lineplot(x=time_steps, y=arr, ax=ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(metric_name)
    return fig
