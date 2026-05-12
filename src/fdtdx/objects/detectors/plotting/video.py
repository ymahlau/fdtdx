import multiprocessing as mp
import tempfile
from functools import partial
from typing import Callable

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import moviepy as mpy
import numpy as np
from rich.progress import Progress

from fdtdx.objects.detectors.plotting.plot2d import plot_2d_from_slices

mp.set_start_method("spawn", force=True)


def plot_from_slices(
    slice_tuple: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolutions: tuple[float, float, float],
    minvals: tuple[float, float, float],
    maxvals: tuple[float, float, float],
    plot_dpi: int | None,
    plot_interpolation: str,
):
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
    # Get the canvas dimensions, accounting for device pixel ratio
    width, height = fig.canvas.get_width_height()
    device_pixel_ratio = fig.canvas.device_pixel_ratio
    if device_pixel_ratio != 1:
        width = int(width * device_pixel_ratio)
        height = int(height * device_pixel_ratio)

    try:
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore
        data = data.reshape(height, width, 4)
        data = data[:, :, :3]  # remove alpha channel
    except AttributeError:
        # Fall back to tostring_argb method
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)  # type: ignore
        data = data.reshape((height, width, 4))
        data = data[:, :, 1:]  # Remove alpha channel
    plt.close(fig)
    return data


def _make_animation_frame(t: float | int, precomputed_figs, fps):
    t = int(t * fps)
    fig = precomputed_figs[t]
    return fig


def generate_video_from_slices(
    xy_slice: np.ndarray,
    xz_slice: np.ndarray,
    yz_slice: np.ndarray,
    plt_fn: Callable,
    resolutions: tuple[float, float, float],
    num_worker: int | None,
    plot_interpolation: str,
    plot_dpi: int | None,
    fps: int = 10,
    progress: Progress | None = None,
    minvals: tuple[float | None, float | None, float | None] = (None, None, None),
    maxvals: tuple[float | None, float | None, float | None] = (None, None, None),
) -> str:
    """Generates an MP4 video from time-series slice data using parallel processing.

    Creates a video visualization of electromagnetic field evolution over time by rendering
    2D slices through the XY, XZ and YZ planes for each timestep.

    Args:
        xy_slice (np.ndarray): 3D array containing XY plane slice data over time
        xz_slice (np.ndarray): 3D array containing XZ plane slice data over time
        yz_slice (np.ndarray): 3D array containing YZ plane slice data over time
        plt_fn (Callable): Plotting function to generate each frame
        resolutions (tuple[float, float, float]): Tuple of (dx, dy, dz) spatial resolutions in meters
        num_worker (int | None): Number of parallel worker processes. If None, frames are processes sequentially.
        plot_interpolation (str): Interpolation method for imshow ('gaussian', 'nearest', etc)
        plot_dpi (int | None): DPI resolution for the figures. None uses default.
        fps (int): Frames per second in output video
        progress (Progress | None, optional): Optional progress bar instance. Defaults to None.
        minvals (tuple[float | None, float | None, float | None], optional): Tuple of minimum values for colormap
            scaling. None values are auto-scaled. Defaults to (None, None, None)
        maxvals (tuple[float | None, float | None, float | None]): Tuple of maximum values for colormap scaling.
            None values are auto-scaled. Defaults to (None, None, None).

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
    if num_worker is None:
        # no multiprocessing, render video frames on after the other
        for s in slice_arr_list:
            fig = partial_fn(s)
            precomputed_figs.append(fig)
            progress.update(task_id, advance=1)
    else:
        # multiprocessing pool to render video frames in parallel
        with mp.Pool(num_worker) as pool:
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
