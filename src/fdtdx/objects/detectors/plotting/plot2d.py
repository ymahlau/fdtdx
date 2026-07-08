from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_2d_from_slices(
    xy_slice: np.ndarray,
    xz_slice: np.ndarray,
    yz_slice: np.ndarray,
    resolutions: tuple[float, float, float],
    coordinate_edges_um: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    minvals: tuple[float | None, float | None, float | None] = (None, None, None),
    maxvals: tuple[float | None, float | None, float | None] = (None, None, None),
    plot_interpolation: str = "gaussian",
    plot_dpi: int | None = None,
    aspect: Literal["equal", "auto"] = "equal",
    cmap: str = "default",
    signed_data: bool = True,
) -> Figure:
    """Plot orthogonal slices using either uniform spacing or rectilinear edges.

    Args:
        xy_slice: Values on an ``(x, y)`` slice.
        xz_slice: Values on an ``(x, z)`` slice.
        yz_slice: Values on a ``(y, z)`` slice.
        resolutions: Legacy uniform ``(dx, dy, dz)`` spacing in metres.
        coordinate_edges_um: Optional rectilinear edge coordinates in micrometres
            for the plotted detector slice. When supplied, plots use
            ``pcolormesh`` so non-uniform cell widths are represented by their
            physical edge positions instead of one scalar spacing.
        minvals: Per-slice color minima.
        maxvals: Per-slice color maxima.
        plot_interpolation: Interpolation used by the legacy uniform imshow path.
        plot_dpi: Figure DPI.
        cmap: str = "default": Color map for the detector plots. "default" is inferno
            for unsigned data and bwr for signed data.
        aspect: Literal["auto", "equal"]: Size aspect of the detector plots.
            "equal" (default) uses the same scale for all axes.
            "auto" adjusts each axis's scale to fit the figure size.
        signed_data: bool: Whether the data is signed (fields, phasors, mode_overlaps...)
            or unsigned (energy, Poynting flux). Used to choose a colormap and scale its values.

    Returns:
        Matplotlib figure with XY, XZ, and YZ panels.
    """
    slices = [xy_slice, xz_slice, yz_slice]
    for a in range(3):
        if signed_data:
            if minvals[a] is None or maxvals[a] is None:
                abs_max = float(np.abs(slices[a]).max())
                min_list, max_list = list(minvals), list(maxvals)
                min_list[a] = -abs_max
                max_list[a] = abs_max
                minvals = (min_list[0], min_list[1], min_list[2])
                maxvals = (max_list[0], max_list[1], max_list[2])
        else:
            if minvals[a] is None:
                min_list = list(minvals)
                min_list[a] = 0.0
                minvals = (min_list[0], min_list[1], min_list[2])
            if maxvals[a] is None:
                max_list = list(maxvals)
                max_list[a] = slices[a].max()
                maxvals = (max_list[0], max_list[1], max_list[2])

    fig = plt.figure(figsize=(20, 10), dpi=plot_dpi)
    res_x = resolutions[0] / 1.0e-6  # Convert to µm
    res_y = resolutions[1] / 1.0e-6
    res_z = resolutions[2] / 1.0e-6
    colorbar_shrink = 0.5

    # Center rectilinear coordinates for plotting
    if coordinate_edges_um is not None:
        # Center the x, y and z coordinate edges
        center_x = (coordinate_edges_um[0][0] + coordinate_edges_um[0][-1]) / 2
        center_y = (coordinate_edges_um[1][0] + coordinate_edges_um[1][-1]) / 2
        center_z = (coordinate_edges_um[2][0] + coordinate_edges_um[2][-1]) / 2

        # Subtract the center from all edges
        x_edges_centered = coordinate_edges_um[0] - center_x
        y_edges_centered = coordinate_edges_um[1] - center_y
        z_edges_centered = coordinate_edges_um[2] - center_z
    else:
        x_edges_centered = y_edges_centered = z_edges_centered = None

    # Create XY plane plot
    ax1 = fig.add_subplot(131)
    assert minvals[0] is not None and maxvals[0] is not None
    if coordinate_edges_um is None:
        extent1 = (
            -xy_slice.shape[0] * res_x / 2,
            xy_slice.shape[0] * res_x / 2,
            -xy_slice.shape[1] * res_y / 2,
            xy_slice.shape[1] * res_y / 2,
        )
        cax1 = ax1.imshow(
            xy_slice.T,
            cmap=cmap,
            vmin=minvals[0],
            vmax=maxvals[0],
            extent=extent1,
            interpolation=plot_interpolation,
            aspect=aspect,
            origin="lower",
        )
    else:
        assert x_edges_centered is not None and y_edges_centered is not None
        cax1 = ax1.pcolormesh(
            x_edges_centered,
            y_edges_centered,
            xy_slice.T,
            cmap=cmap,
            vmin=minvals[0],
            vmax=maxvals[0],
            shading="auto",
        )
        ax1.set_aspect(aspect)
    ax1.set_xlabel("X axis (µm)")
    ax1.set_ylabel("Y axis (µm)")

    plt.colorbar(mappable=cax1, ax=ax1, shrink=colorbar_shrink)

    # Create XZ plane plot

    ax2 = fig.add_subplot(132)
    assert minvals[1] is not None and maxvals[1] is not None
    if coordinate_edges_um is None:
        extent2 = (
            -xz_slice.shape[0] * res_x / 2,
            xz_slice.shape[0] * res_x / 2,
            -xz_slice.shape[1] * res_z / 2,
            xz_slice.shape[1] * res_z / 2,
        )
        cax2 = ax2.imshow(
            xz_slice.T,
            cmap=cmap,
            vmin=minvals[1],
            vmax=maxvals[1],
            extent=extent2,
            interpolation=plot_interpolation,
            aspect=aspect,
            origin="lower",
        )
    else:
        assert x_edges_centered is not None and z_edges_centered is not None
        cax2 = ax2.pcolormesh(
            x_edges_centered,
            z_edges_centered,
            xz_slice.T,
            cmap=cmap,
            vmin=minvals[1],
            vmax=maxvals[1],
            shading="auto",
        )
        ax2.set_aspect(aspect)
    ax2.set_xlabel("X axis (µm)")
    ax2.set_ylabel("Z axis (µm)")

    plt.colorbar(mappable=cax2, ax=ax2, shrink=colorbar_shrink)

    # # Create YZ plane plot
    ax3 = fig.add_subplot(133)
    assert minvals[2] is not None and maxvals[2] is not None
    if coordinate_edges_um is None:
        extent3 = (
            -yz_slice.shape[0] * res_y / 2,
            yz_slice.shape[0] * res_y / 2,
            -yz_slice.shape[1] * res_z / 2,
            yz_slice.shape[1] * res_z / 2,
        )
        cax3 = ax3.imshow(
            yz_slice.T,
            cmap=cmap,
            vmin=minvals[2],
            vmax=maxvals[2],
            extent=extent3,
            interpolation=plot_interpolation,
            aspect=aspect,
            origin="lower",
        )
    else:
        assert y_edges_centered is not None and z_edges_centered is not None
        cax3 = ax3.pcolormesh(
            y_edges_centered,
            z_edges_centered,
            yz_slice.T,
            cmap=cmap,
            vmin=minvals[2],
            vmax=maxvals[2],
            shading="auto",
        )
        ax3.set_aspect(aspect)
    ax3.set_xlabel("Y axis (µm)")
    ax3.set_ylabel("Z axis (µm)")

    plt.colorbar(mappable=cax3, ax=ax3, shrink=colorbar_shrink)

    return fig


def plot_grads(
    grad_arr: np.ndarray,
    voxel_size: tuple[float, float, float],
) -> Figure:
    xy_slice = grad_arr.mean(axis=2)
    xz_slice = grad_arr.mean(axis=1)
    yz_slice = grad_arr.mean(axis=0)
    fig = plot_2d_from_slices(
        xy_slice=xy_slice,
        xz_slice=xz_slice,
        yz_slice=yz_slice,
        resolutions=voxel_size,
    )
    return fig
