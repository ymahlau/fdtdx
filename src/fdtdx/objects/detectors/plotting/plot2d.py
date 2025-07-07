import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


def plot_2d_from_slices(
    xy_slice: np.ndarray,
    xz_slice: np.ndarray,
    yz_slice: np.ndarray,
    resolutions: tuple[float, float, float],
    minvals: tuple[float | None, float | None, float | None] = (None, None, None),
    maxvals: tuple[float | None, float | None, float | None] = (None, None, None),
    plot_interpolation: str = "gaussian",
    plot_dpi: int | None = None,
) -> Figure:
    slices = [xy_slice, xz_slice, yz_slice]
    for a in range(3):
        if minvals[a] is None:
            min_list = list(minvals)
            min_list[a] = slices[a].min()
            minvals = min_list[0], min_list[1], min_list[2]
        if maxvals[a] is None:
            max_list = list(maxvals)
            max_list[a] = slices[a].max()
            maxvals = max_list[0], max_list[1], max_list[2]

    fig = plt.figure(figsize=(20, 10), dpi=plot_dpi)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    res_x = resolutions[0] / 1.0e-6  # Convert to µm
    res_y = resolutions[1] / 1.0e-6
    res_z = resolutions[2] / 1.0e-6
    colorbar_shrink = 0.5

    # Create XY plane plot
    ax1 = fig.add_subplot(131)
    extent1 = (
        0,
        xy_slice.shape[0] * res_x,
        0,
        xy_slice.shape[1] * res_y,
    )
    cax1 = ax1.imshow(
        xy_slice.T,
        cmap=cmap,
        vmin=minvals[0],
        vmax=maxvals[0],
        extent=extent1,
        interpolation=plot_interpolation,
        aspect="equal",
        origin="lower",
    )
    ax1.set_xlabel("X axis (µm)")
    ax1.set_ylabel("Y axis (µm)")

    plt.colorbar(mappable=cax1, ax=ax1, shrink=colorbar_shrink)

    # Create XZ plane plot

    ax2 = fig.add_subplot(132)
    extent2 = (
        0,
        xz_slice.shape[0] * res_x,
        0,
        xz_slice.shape[1] * res_z,
    )
    cax2 = ax2.imshow(
        xz_slice.T,
        cmap=cmap,
        vmin=minvals[1],
        vmax=maxvals[1],
        extent=extent2,
        interpolation=plot_interpolation,
        aspect="equal",
        origin="lower",
    )
    ax2.set_xlabel("X axis (µm)")
    ax2.set_ylabel("Z axis (µm)")

    plt.colorbar(mappable=cax2, ax=ax2, shrink=colorbar_shrink)

    # # Create YZ plane plot
    ax3 = fig.add_subplot(133)
    extent3 = (
        0,
        yz_slice.shape[0] * res_y,
        0,
        yz_slice.shape[1] * res_z,
    )
    cax3 = ax3.imshow(
        yz_slice.T,
        cmap=cmap,
        vmin=minvals[2],
        vmax=maxvals[2],
        extent=extent3,
        interpolation=plot_interpolation,
        aspect="equal",
        origin="lower",
    )
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
