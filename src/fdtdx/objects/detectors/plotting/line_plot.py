import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_line_over_time(
    arr: np.ndarray,
    time_steps: list[float] | np.ndarray,  # in seconds
    metric_name: str,
    xlabel: str = "Time [s]",
):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.lineplot(x=time_steps, y=arr, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_name)
    return fig


def plot_waterfall_over_time(
    arr: np.ndarray,
    time_steps: list[float] | np.ndarray,
    spatial_steps: list[float] | np.ndarray,
    metric_name: str,
    time_unit: str = "s",
    spatial_unit: str = "μm",
):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a meshgrid for proper extent
    t_min, t_max = min(time_steps), max(time_steps)  # type: ignore
    s_min, s_max = min(spatial_steps), max(spatial_steps)  # type: ignore

    # Plot as a heatmap/waterfall
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    im = ax.imshow(
        arr.T,  # Transpose to have time on x-axis, space on y-axis
        aspect="auto",
        origin="lower",
        extent=(t_min, t_max, s_min, s_max),
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
