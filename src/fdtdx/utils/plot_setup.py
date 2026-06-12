from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import RectilinearGrid
from fdtdx.fdtd.container import ObjectContainer
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.boundaries.pec import PerfectElectricConductor
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.boundaries.pmc import PerfectMagneticConductor
from fdtdx.objects.object import SimulationObject


def _get_full_coverage_objects(
    objects: list[SimulationObject],
    axis_indices: tuple[int, int],
    plane_size: tuple[int, int],
    volume: SimulationObject,
) -> list[SimulationObject]:
    """Detect objects that cover 100% of the viewing plane.

    Args:
        objects: List of simulation objects to check
        axis_indices: Tuple of two axis indices defining the viewing plane
        plane_size: Tuple of (width, height) of the viewing plane in grid cells
        volume: The simulation volume object

    Returns:
        List of objects that cover 100% of the viewing plane
    """
    full_coverage_objects = []
    total_area = plane_size[0] * plane_size[1]
    # Guard against degenerate simulation volumes (zero area planes)
    if total_area <= 0:
        return []

    for obj in objects:
        if obj is volume:
            continue

        slices = obj.grid_slice_tuple
        obj_width = slices[axis_indices[0]][1] - slices[axis_indices[0]][0]
        obj_height = slices[axis_indices[1]][1] - slices[axis_indices[1]][0]
        obj_area = obj_width * obj_height

        # Check if object covers the entire plane (allowing for small floating point errors)
        if obj_area >= total_area * 0.999:  # 99.9% threshold to account for numerical issues
            full_coverage_objects.append(obj)

    return full_coverage_objects


def _axis_edges_um(
    config: SimulationConfig, axis: int, bounds: tuple[int, int], axis_length: int
) -> tuple[float, float]:
    """Return centered physical edge coordinates in micrometres for an index interval."""
    grid = getattr(config, "grid", None)
    if isinstance(grid, RectilinearGrid):
        edges = grid.edges(axis)
        domain_center = 0.5 * (float(edges[0]) + float(edges[-1]))
        return (
            (float(edges[bounds[0]]) - domain_center) / 1.0e-6,
            (float(edges[bounds[1]]) - domain_center) / 1.0e-6,
        )
    spacing = config.uniform_spacing()
    domain_center = 0.5 * axis_length
    return ((bounds[0] - domain_center) * spacing / 1.0e-6, (bounds[1] - domain_center) * spacing / 1.0e-6)


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
    auto_exclude_full_coverage: bool = True,
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
        auto_exclude_full_coverage (bool, optional): Automatically exclude objects that cover 100% of the viewing plane

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
        if not isinstance(
            o, (PerfectlyMatchedLayer, BlochBoundary, PerfectElectricConductor, PerfectMagneticConductor)
        ):
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

    # Determine which exclude list to use based on viewing side
    if viewing_side == "z":
        plane_exclude_list = list(exclude_xy_plane_object_list)  # Create a copy
        axis_indices = (0, 1)  # X, Y
        axis_labels = ("x (µm)", "y (µm)")
        title = "XY plane"
        plane_size = (volume.grid_shape[0], volume.grid_shape[1])
    elif viewing_side == "y":
        plane_exclude_list = list(exclude_xz_plane_object_list)  # Create a copy
        axis_indices = (0, 2)  # X, Z
        axis_labels = ("x (µm)", "z (µm)")
        title = "XZ plane"
        plane_size = (volume.grid_shape[0], volume.grid_shape[2])
    elif viewing_side == "x":
        plane_exclude_list = list(exclude_yz_plane_object_list)  # Create a copy
        axis_indices = (1, 2)  # Y, Z
        axis_labels = ("y (µm)", "z (µm)")
        title = "YZ plane"
        plane_size = (volume.grid_shape[1], volume.grid_shape[2])
    else:
        raise ValueError(f"Invalid viewing_side: {viewing_side}. Must be 'x', 'y', or 'z'")

    # Auto-detect and exclude objects that cover 100% of the viewing plane
    if auto_exclude_full_coverage:
        full_coverage_objects = _get_full_coverage_objects(object_list, axis_indices, plane_size, volume)
        plane_exclude_list.extend(full_coverage_objects)

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
            color_val = o.color.to_mpl() if o.color is not None else "gray"
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

        x_edges = _axis_edges_um(config, axis_indices[0], slices[axis_indices[0]], plane_size[0])
        y_edges = _axis_edges_um(config, axis_indices[1], slices[axis_indices[1]], plane_size[1])

        ax.add_patch(
            Rectangle(
                (x_edges[0], y_edges[0]),
                x_edges[1] - x_edges[0],
                y_edges[1] - y_edges[0],
                color=color.to_mpl() if color is not None else "gray",
                alpha=0.5,
                linestyle="--"
                if isinstance(obj, (BlochBoundary, PerfectElectricConductor, PerfectMagneticConductor))
                else "-",
            )
        )

    # Set labels and titles
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.set_xlim(_axis_edges_um(config, axis_indices[0], (0, plane_size[0]), plane_size[0]))
    ax.set_ylim(_axis_edges_um(config, axis_indices[1], (0, plane_size[1]), plane_size[1]))
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
    auto_exclude_full_coverage: bool = True,
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
        auto_exclude_full_coverage (bool, optional): Automatically exclude objects that cover 100% of the viewing plane

    Returns:
        Figure: The generated figure object

    Note:
        The plots show object positions in micrometers, converting from simulation units.
        PML objects are automatically excluded from their respective boundary planes.
        Objects covering 100% of a viewing plane are automatically excluded by default.
    """
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        assert axs is not None
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
        auto_exclude_full_coverage=auto_exclude_full_coverage,
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
        auto_exclude_full_coverage=auto_exclude_full_coverage,
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
        auto_exclude_full_coverage=auto_exclude_full_coverage,
    )

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()

    return plt.gcf() if fig is None else fig
