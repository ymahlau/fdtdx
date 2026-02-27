from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ObjectContainer
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.boundaries.periodic import PeriodicBoundary
from fdtdx.objects.object import SimulationObject


def plot_setup_from_side(
    config: SimulationConfig,
    objects: ObjectContainer,
    viewing_side: Literal["x", "y", "z"],
    exclude_object_list: list[SimulationObject] | None = None,
    filename: str | Path | None = None,
    ax: Any | None = None,
    plot_legend: bool = True,
    exclude_xy_plane_object_list: list[SimulationObject] | None = None,
    exclude_yz_plane_object_list: list[SimulationObject] | None = None,
    exclude_xz_plane_object_list: list[SimulationObject] | None = None,
    exclude_large_object_ratio: float | None = None,
) -> Figure:
    """Creates a visualization of the simulation setup from a single viewing side.

    Generates a single subplot showing a cross-section of the simulation volume and the objects
    within it from the specified viewing side. Objects are drawn as colored rectangles with
    optional legends.

    Args:
        config (SimulationConfig): Configuration object containing simulation parameters like resolution
        objects (ObjectContainer): Container holding all simulation objects to be plotted
        viewing_side (Literal['x', 'y', 'z']): Which plane to view ('x' for YZ, 'y' for XZ, 'z' for XY)
        exclude_object_list (list[SimulationObject] | None, optional): List of objects to exclude from all plots
        filename (str | Path | None, optional): If provided, saves the plot to this file instead of displaying
        ax (Any | None, optional): Optional matplotlib axis to plot on. If None, creates new figure
        plot_legend (bool, optional): Whether to add a legend showing object names/types
        exclude_xy_plane_object_list (list[SimulationObject] | None, optional): Objects to exclude from XY plane plot
        exclude_yz_plane_object_list (list[SimulationObject] | None, optional): Objects to exclude from YZ plane plot
        exclude_xz_plane_object_list (list[SimulationObject] | None, optional): Objects to exclude from XZ plane plot
        exclude_large_object_ratio (float | None, optional): If provided, excludes objects that cover more than
            this ratio of the image (e.g., 1.0 excludes objects covering 100% of the image)

    Returns:
        Figure: The generated figure object

    Note:
        The plots show object positions in micrometers, converting from simulation units.
        PML objects are automatically excluded from their respective boundary planes.
    """
    # default to empty lists
    exclude_object_list = exclude_object_list or []
    exclude_xy_plane_object_list = exclude_xy_plane_object_list or []
    exclude_yz_plane_object_list = exclude_yz_plane_object_list or []
    exclude_xz_plane_object_list = exclude_xz_plane_object_list or []

    # add boundaries to exclude lists
    for o in objects.objects:
        if not isinstance(o, (PerfectlyMatchedLayer, PeriodicBoundary)):
            continue
        if o.axis == 0:
            exclude_yz_plane_object_list.append(o)
        elif o.axis == 1:
            exclude_xz_plane_object_list.append(o)
        elif o.axis == 2:
            exclude_xy_plane_object_list.append(o)

    # add volume to exclude list
    volume = objects.volume
    exclude_object_list.append(volume)

    object_list = [o for o in objects.objects if o not in exclude_object_list]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = None

    resolution = config.resolution / 1.0e-6  # Convert to µm

    # Determine which exclude list to use based on viewing side
    if viewing_side == "z":
        plane_exclude_list = exclude_xy_plane_object_list
        axis_indices = (0, 1)  # X, Y
        axis_labels = ("x (µm)", "y (µm)")
        title = "XY plane"
        plane_size = (volume.grid_shape[0], volume.grid_shape[1])
    elif viewing_side == "y":
        plane_exclude_list = exclude_xz_plane_object_list
        axis_indices = (0, 2)  # X, Z
        axis_labels = ("x (µm)", "z (µm)")
        title = "XZ plane"
        plane_size = (volume.grid_shape[0], volume.grid_shape[2])
    elif viewing_side == "x":
        plane_exclude_list = exclude_yz_plane_object_list
        axis_indices = (1, 2)  # Y, Z
        axis_labels = ("y (µm)", "z (µm)")
        title = "YZ plane"
        plane_size = (volume.grid_shape[1], volume.grid_shape[2])
    else:
        raise ValueError(f"Invalid viewing_side: {viewing_side}. Must be 'x', 'y', or 'z'")

    # Filter objects for this plane
    colored_objects: list[SimulationObject] = [
        o for o in object_list if o.color is not None and (plane_exclude_list is None or o not in plane_exclude_list)
    ]

    # Apply exclude_large_object_ratio filter
    if exclude_large_object_ratio is not None:
        total_area = plane_size[0] * plane_size[1]
        filtered_objects = []
        for obj in colored_objects:
            slices = obj.grid_slice_tuple
            obj_area = (slices[axis_indices[0]][1] - slices[axis_indices[0]][0]) * (
                slices[axis_indices[1]][1] - slices[axis_indices[1]][0]
            )
            coverage_ratio = obj_area / total_area
            if coverage_ratio <= exclude_large_object_ratio:
                filtered_objects.append(obj)
        colored_objects = filtered_objects

    if plot_legend:
        handles = []
        used_lists = []
        for o in colored_objects:
            print_single = False
            for o2 in colored_objects:
                if o.__class__ == o2.__class__:
                    if o.color != o2.color:
                        print_single = True
                    if not o.name.startswith("Object"):
                        print_single = True
            label = o.__class__.__name__ if o.name.startswith("Object") else o.name
            color_val = str(o.color)
            patch = Patch(color=color_val, label=label)
            if print_single:
                handles.append(patch)
            else:
                if o.__class__.__name__ not in used_lists:
                    used_lists.append(o.__class__.__name__)
                    handles.append(patch)

        ax.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(1.75, 0.75),
            frameon=False,
        )

    # Plot each object
    for obj in colored_objects:
        slices = obj.grid_slice_tuple
        color = obj.color

        ax.add_patch(
            Rectangle(
                (slices[axis_indices[0]][0] * resolution, slices[axis_indices[1]][0] * resolution),
                (slices[axis_indices[0]][1] - slices[axis_indices[0]][0]) * resolution,
                (slices[axis_indices[1]][1] - slices[axis_indices[1]][0]) * resolution,
                color=color,
                alpha=0.5,
                linestyle="--" if isinstance(obj, PeriodicBoundary) else "-",
            )
        )

    # Set labels and titles
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.set_xlim((0, plane_size[0] * resolution))
    ax.set_ylim((0, plane_size[1] * resolution))
    ax.set_aspect("equal")
    ax.grid(True)

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()

    return plt.gcf() if fig is None else fig


def plot_setup(
    config: SimulationConfig,
    objects: ObjectContainer,
    exclude_object_list: list[SimulationObject] | None = None,
    filename: str | Path | None = None,
    axs: Any | None = None,
    plot_legend: bool = True,
    exclude_xy_plane_object_list: list[SimulationObject] | None = None,
    exclude_yz_plane_object_list: list[SimulationObject] | None = None,
    exclude_xz_plane_object_list: list[SimulationObject] | None = None,
    exclude_large_object_ratio: float | None = None,
) -> Figure:
    """Creates a visualization of the simulation setup showing objects in XY, XZ and YZ planes.

    Generates three subplots showing cross-sections of the simulation volume and the objects
    within it. Objects are drawn as colored rectangles with optional legends. The visualization
    helps verify the correct positioning and sizing of objects in the simulation setup.

    Args:
        config (SimulationConfig): Configuration object containing simulation parameters like resolution
        objects (ObjectContainer): Container holding all simulation objects to be plotted
        exclude_object_list (list[SimulationObject] | None, optional): List of objects to exclude from all plots
        filename (str | Path | None, optional): If provided, saves the plot to this file instead of displaying
        axs (Any | None, optional): Optional matplotlib axes to plot on. If None, creates new figure
        plot_legend (bool, optional): Whether to add a legend showing object names/types
        exclude_xy_plane_object_list (list[SimulationObject] | None, optional): Objects to exclude from XY plane plot
        exclude_yz_plane_object_list (list[SimulationObject] | None, optional): Objects to exclude from YZ plane plot
        exclude_xz_plane_object_list (list[SimulationObject] | None, optional): Objects to exclude from XZ plane plot
        exclude_large_object_ratio (float | None, optional): If provided, excludes objects that cover more than
            this ratio of the image (e.g., 1.0 excludes objects covering 100% of the image)

    Returns:
        Figure: The generated figure object

    Note:
        The plots show object positions in micrometers, converting from simulation units.
        PML objects are automatically excluded from their respective boundary planes.
    """
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig = None

    # Plot XY plane (viewing from z direction)
    plot_setup_from_side(
        config=config,
        objects=objects,
        viewing_side="z",
        exclude_object_list=exclude_object_list,
        filename=None,
        ax=axs[0],
        plot_legend=False,
        exclude_xy_plane_object_list=exclude_xy_plane_object_list,
        exclude_yz_plane_object_list=exclude_yz_plane_object_list,
        exclude_xz_plane_object_list=exclude_xz_plane_object_list,
        exclude_large_object_ratio=exclude_large_object_ratio,
    )

    # Plot XZ plane (viewing from y direction)
    plot_setup_from_side(
        config=config,
        objects=objects,
        viewing_side="y",
        exclude_object_list=exclude_object_list,
        filename=None,
        ax=axs[1],
        plot_legend=False,
        exclude_xy_plane_object_list=exclude_xy_plane_object_list,
        exclude_yz_plane_object_list=exclude_yz_plane_object_list,
        exclude_xz_plane_object_list=exclude_xz_plane_object_list,
        exclude_large_object_ratio=exclude_large_object_ratio,
    )

    # Plot YZ plane (viewing from x direction)
    plot_setup_from_side(
        config=config,
        objects=objects,
        viewing_side="x",
        exclude_object_list=exclude_object_list,
        filename=None,
        ax=axs[2],
        plot_legend=plot_legend,
        exclude_xy_plane_object_list=exclude_xy_plane_object_list,
        exclude_yz_plane_object_list=exclude_yz_plane_object_list,
        exclude_xz_plane_object_list=exclude_xz_plane_object_list,
        exclude_large_object_ratio=exclude_large_object_ratio,
    )

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()

    return plt.gcf() if fig is None else fig
