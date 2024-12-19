import tempfile
from functools import partial
from multiprocessing import Pool
from typing import Callable

import matplotlib.pyplot as plt
import moviepy as mpy
import numpy as np
from rich.progress import Progress

from fdtdx.objects.detectors.plotting.plot2d import plot_2d_from_slices


def plot_from_slices(
    slice_tuple: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolutions: tuple[float, float, float],
    minvals: tuple[float, float, float],
    maxvals: tuple[float, float, float],
    plot_dpi: int | None,
    plot_interpolation: str,
):
    """Creates a figure from 2D slices at a specific timestep using shared memory arrays.

    Args:
        slice_tuple: tuple of array slices in order xy, xz, yz
        resolutions: Tuple of (dx, dy, dz) spatial resolutions in meters
        minvals: Tuple of minimum values for colormap scaling
        maxvals: Tuple of maximum values for colormap scaling
        plot_dpi: DPI resolution for the figure. None uses default.
        plot_interpolation: Interpolation method for imshow ('gaussian', 'nearest', etc)

    Returns:
        numpy.ndarray: RGB image data of the rendered figure
    """
    xy_slice, xz_slice, yz_slice = slice_tuple

    fig = plot_2d_from_slices(
        xy_slice=xy_slice,
        xz_slice=xz_slice,
        yz_slice=yz_slice,
        resolutions=resolutions,
        minvals=minvals,
        maxvals=maxvals,
        plot_dpi=plot_dpi,
        plot_interpolation=plot_interpolation,
    )
    # Convert matplotlib figure to a numpy array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def _make_animation_frame(t: float | int, precomputed_figs, fps):
    """Creates a single frame for the video animation.

    Args:
        t: Time point in seconds
        precomputed_figs: List of precomputed figure arrays
        fps: Frames per second of the video

    Returns:
        numpy.ndarray: RGB image data for the frame at time t
    """
    t = int(t * fps)
    fig = precomputed_figs[t]
    return fig


def generate_video_from_slices(
    xy_slice: np.ndarray,
    xz_slice: np.ndarray,
    yz_slice: np.ndarray,
    plt_fn: Callable,
    resolutions: tuple[float, float, float],
    num_worker: int = 8,
    fps: int = 10,
    progress: Progress | None = None,
    minvals: tuple[float | None, float | None, float | None] = (None, None, None),
    maxvals: tuple[float | None, float | None, float | None] = (None, None, None),
    plot_dpi: int | None = None,
    plot_interpolation: str = "gaussian",
):
    """Generates an MP4 video from time-series slice data using parallel processing.

    Creates a video visualization of electromagnetic field evolution over time by rendering
    2D slices through the XY, XZ and YZ planes for each timestep.

    Args:
        xy_slice: 3D array containing XY plane slice data over time
        xz_slice: 3D array containing XZ plane slice data over time
        yz_slice: 3D array containing YZ plane slice data over time
        plt_fn: Plotting function to generate each frame
        resolutions: Tuple of (dx, dy, dz) spatial resolutions in meters
        num_worker: Number of parallel worker processes
        fps: Frames per second in output video
        progress: Optional progress bar instance
        minvals: Tuple of minimum values for colormap scaling. None values are auto-scaled.
        maxvals: Tuple of maximum values for colormap scaling. None values are auto-scaled.
        plot_dpi: DPI resolution for the figures. None uses default.
        plot_interpolation: Interpolation method for imshow ('gaussian', 'nearest', etc)

    Returns:
        str: Path to the generated MP4 video file
    """
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

    _, path = tempfile.mkstemp(suffix=".mp4")
    with Pool(num_worker) as pool:
        time_steps = xy_slice.shape[0]
        if progress is None:
            progress = Progress()
        task_id = progress.add_task("Generating video", total=time_steps)
        precomputed_figs = []
        partial_fn = partial(
            plt_fn,
            resolutions=resolutions,
            minvals=minvals,
            maxvals=maxvals,
            plot_dpi=plot_dpi,
            plot_interpolation=plot_interpolation,
        )
        slice_arr_list = [(xy_slice[t], xz_slice[t], yz_slice[t]) for t in range(time_steps)]
        for fig in pool.imap(partial_fn, slice_arr_list):
            precomputed_figs.append(fig)
            progress.update(task_id, advance=1)
        animation = mpy.VideoClip(
            lambda t: _make_animation_frame(t, precomputed_figs, fps),
            duration=time_steps / fps,
        )
        animation.write_videofile(path, fps=fps, logger=None)
        progress.update(task_id, visible=False)

    return path
