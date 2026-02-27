from pathlib import Path
from typing import Any, Literal, cast

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer

MaterialType = Literal["permittivity", "permeability"]


def plot_material_from_side(
    config: SimulationConfig,
    arrays: ArrayContainer,
    viewing_side: Literal["x", "y", "z"],
    material_axis: int = 0,
    filename: str | Path | None = None,
    ax: Any | None = None,
    plot_legend: bool = True,
    position: float = 0.0,
    type: MaterialType = "permittivity",
) -> Figure:
    """Creates a visualization of material distribution from a single viewing side.

    Generates a single subplot showing a 2D slice of the material distribution
    (permittivity or permeability) through the 3D simulation volume at a specified position.

    Args:
        config (SimulationConfig): Configuration object containing simulation parameters like resolution
        arrays (ArrayContainer): Container holding the material arrays (permittivity, permeability)
        viewing_side (Literal['x', 'y', 'z']): Which plane to view ('x' for YZ, 'y' for XZ, 'z' for XY)
        materialaxis (int): Which material axis to plot (for anisotropic materials). Can be 0, 1, 2 for x, y or z.
        filename (str | Path | None, optional): If provided, saves the plot to this file instead of displaying
        ax (Any | None, optional): Optional matplotlib axis to plot on. If None, creates new figure
        plot_legend (bool, optional): Whether to add a colorbar legend
        position (float, optional): Position of the slice in meters. Zero means at center,
            1e-6 would mean center+1µm
        type (MaterialType, optional): Type of material to plot, either "permittivity" or "permeability"

    Returns:
        Figure: The generated figure object

    Note:
        The plots show material values in a 2D cross-section, with positions in micrometers.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = None

    # Get the appropriate material array
    if type == "permittivity":
        # Invert to get actual permittivity from inverse permittivity
        material_array = 1.0 / arrays.inv_permittivities
    else:  # type == "permeability"
        # Invert to get actual permeability from inverse permeability
        # Note: inv_permeabilities can be a float or array
        if isinstance(arrays.inv_permeabilities, float):
            material_array = np.full_like(arrays.inv_permittivities, 1.0 / arrays.inv_permeabilities)
        else:
            material_array = 1.0 / arrays.inv_permeabilities

    resolution = config.resolution / 1.0e-6  # Convert to µm

    # Calculate slice index from position
    slice_offset = round(position / config.resolution)

    # Get array shape
    material_array = cast(jax.Array, material_array)
    array_shape = material_array.shape

    # Determine slice parameters based on viewing side
    if viewing_side == "z":
        # XY plane - slice through Z axis
        center_idx = array_shape[2] // 2
        slice_idx = center_idx + slice_offset
        slice_idx = max(0, min(slice_idx, array_shape[2] - 1))
        material_slice = material_array[material_axis, :, :, slice_idx]
        axis_labels = ("x (µm)", "y (µm)")
        title = f"XY plane - {type} at z={position * 1e6:.2f} µm"
        extent = [0, array_shape[0] * resolution, 0, array_shape[1] * resolution]

    elif viewing_side == "y":
        # XZ plane - slice through Y axis
        center_idx = array_shape[1] // 2
        slice_idx = center_idx + slice_offset
        slice_idx = max(0, min(slice_idx, array_shape[1] - 1))
        material_slice = material_array[material_axis, :, slice_idx, :]
        axis_labels = ("x (µm)", "z (µm)")
        title = f"XZ plane - {type} at y={position * 1e6:.2f} µm"
        extent = [0, array_shape[0] * resolution, 0, array_shape[2] * resolution]

    else:  # viewing_side == "x"
        # YZ plane - slice through X axis
        center_idx = array_shape[0] // 2
        slice_idx = center_idx + slice_offset
        slice_idx = max(0, min(slice_idx, array_shape[0] - 1))
        material_slice = material_array[material_axis, slice_idx, :, :]
        axis_labels = ("y (µm)", "z (µm)")
        title = f"YZ plane - {type} at x={position * 1e6:.2f} µm"
        extent = [0, array_shape[1] * resolution, 0, array_shape[2] * resolution]

    # Plot the material slice
    im = ax.imshow(
        material_slice.T,  # Transpose for correct orientation
        origin="lower",
        extent=cast(tuple[int | float, int | float, int | float, int | float], tuple(extent)),
        aspect="equal",
        cmap="viridis",
        interpolation="nearest",
    )

    # Set labels and titles
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add colorbar if legend is requested
    if plot_legend:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(type.capitalize())

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()

    return plt.gcf() if fig is None else fig


def plot_material(
    config: SimulationConfig,
    arrays: ArrayContainer,
    filename: str | Path | None = None,
    axs: Any | None = None,
    plot_legend: bool = True,
    positions: tuple[float, float, float] = (0.0, 0.0, 0.0),
    type: MaterialType = "permittivity",
) -> Figure:
    """Creates a visualization of material distribution showing slices in XY, XZ and YZ planes.

    Generates three subplots showing 2D slices of the material distribution
    (permittivity or permeability) through the 3D simulation volume at specified positions.

    Args:
        config (SimulationConfig): Configuration object containing simulation parameters like resolution
        arrays (ArrayContainer): Container holding the material arrays (permittivity, permeability)
        filename (str | Path | None, optional): If provided, saves the plot to this file instead of displaying
        axs (Any | None, optional): Optional matplotlib axes to plot on. If None, creates new figure
        plot_legend (bool, optional): Whether to add colorbar legends
        positions (tuple[float, float, float], optional): Positions of slices in x, y, z directions (in meters).
            Zero means at center, 1e-6 would mean center+1µm
        type (MaterialType, optional): Type of material to plot, either "permittivity" or "permeability"

    Returns:
        Figure: The generated figure object

    Note:
        The plots show material values in 2D cross-sections, with positions in micrometers.
    """
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig = None

    # Extract individual positions
    x_pos, y_pos, z_pos = positions

    # Plot XY plane (viewing from z direction)
    plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="z",
        filename=None,
        ax=axs[0],
        plot_legend=plot_legend,
        position=z_pos,
        type=type,
    )

    # Plot XZ plane (viewing from y direction)
    plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="y",
        filename=None,
        ax=axs[1],
        plot_legend=plot_legend,
        position=y_pos,
        type=type,
    )

    # Plot YZ plane (viewing from x direction)
    plot_material_from_side(
        config=config,
        arrays=arrays,
        viewing_side="x",
        filename=None,
        ax=axs[2],
        plot_legend=plot_legend,
        position=x_pos,
        type=type,
    )

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()

    return plt.gcf() if fig is None else fig
