from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ObjectContainer


def plot_setup_from_side(
    config: SimulationConfig,
    objects: ObjectContainer,
    resolution: float,
    ax,
    viewing_side: str,
    exclude_large_object_ratio: Optional[float] = None,
):
    """
    Plot a 2D projection of the simulation setup from a chosen axis.

    viewing_side: one of 'x', 'y', 'z' indicating which projection to plot.
    exclude_large_object_ratio: if given, objects that cover more than this
        fraction of the image area will be excluded.
    """

    volume = objects.volume

    # Map viewing_side to coordinate pairs
    if viewing_side == "x":
        plane = (1, 2)
        xlabel, ylabel = "y (µm)", "z (µm)"
        title = "YZ plane"
    elif viewing_side == "y":
        plane = (0, 2)
        xlabel, ylabel = "x (µm)", "z (µm)"
        title = "XZ plane"
    elif viewing_side == "z":
        plane = (0, 1)
        xlabel, ylabel = "x (µm)", "y (µm)"
        title = "XY plane"
    else:
        raise ValueError("viewing_side must be 'x', 'y', or 'z'")

    # Total area for filtering
    total_w = volume.grid_shape[plane[0]] * resolution
    total_h = volume.grid_shape[plane[1]] * resolution
    total_area = total_w * total_h

    for obj in objects:
        # Determine bounds in this projection
        x0 = obj.grid_bounds[plane[0]][0] * resolution
        x1 = obj.grid_bounds[plane[0]][1] * resolution
        y0 = obj.grid_bounds[plane[1]][0] * resolution
        y1 = obj.grid_bounds[plane[1]][1] * resolution

        w = x1 - x0
        h = y1 - y0
        area = w * h

        # Skip large objects
        if exclude_large_object_ratio is not None:
            if area / total_area > exclude_large_object_ratio:
                continue

        rect = Rectangle((x0, y0), w, h, facecolor=obj.color, edgecolor="black", alpha=0.6)
        ax.add_patch(rect)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim([0, total_w])
    ax.set_ylim([0, total_h])
    ax.set_aspect("equal")
    ax.grid(True)


def plot_setup(
    config: SimulationConfig,
    objects: ObjectContainer,
    resolution: float,
    fig=None,
    filename=None,
    exclude_large_object_ratio: Optional[float] = None,
):
    """
    Updated plot_setup which now calls plot_setup_from_side three times.
    """

    if fig is None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    else:
        axs = fig.subplots(1, 3)

    # XY view (looking along +z)
    plot_setup_from_side(
        config,
        objects,
        resolution,
        axs[0],
        viewing_side="z",
        exclude_large_object_ratio=exclude_large_object_ratio,
    )

    # XZ view (looking along +y)
    plot_setup_from_side(
        config,
        objects,
        resolution,
        axs[1],
        viewing_side="y",
        exclude_large_object_ratio=exclude_large_object_ratio,
    )

    # YZ view (looking along +x)
    plot_setup_from_side(
        config,
        objects,
        resolution,
        axs[2],
        viewing_side="x",
        exclude_large_object_ratio=exclude_large_object_ratio,
    )

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()

    return fig
