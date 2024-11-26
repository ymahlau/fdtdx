from functools import partial
import tempfile
from multiprocessing import Pool
from typing import Callable

import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import SharedArray as sa
from rich.progress import Progress

from fdtdx.objects.detectors.plotting.plot2d import plot_2d_from_slices


def plot_from_slices(
    t: int,
    resolutions: tuple[float, float, float],
    minvals: tuple[float, float, float],
    maxvals: tuple[float, float, float],
    plot_dpi: int | None,
    plot_interpolation: str,
):
    xy_slice = sa.attach("shm://xy")[t, :, :]
    xz_slice = sa.attach("shm://xz")[t, :, :]
    yz_slice = sa.attach("shm://yz")[t, :, :]

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
    minvals: tuple[float | None, float | None, float | None]  = (None, None, None),
    maxvals: tuple[float | None, float | None, float | None]  = (None, None, None),
    plot_dpi: int | None = None,
    plot_interpolation: str = "gaussian",
):
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
        try:
            sa.delete("xy")
            sa.delete("xz")
            sa.delete("yz")
        except Exception as e:
            pass
        
        shared_xy = sa.create("shm://xy", xy_slice.shape)
        shared_xy[:] = xy_slice[:]
        shared_xz = sa.create("shm://xz", xz_slice.shape)
        shared_xz[:] = xz_slice[:]
        shared_yz = sa.create("shm://yz", yz_slice.shape)
        shared_yz[:] = yz_slice[:]
        
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
        for fig in pool.imap(partial_fn, range(time_steps)):
            precomputed_figs.append(fig)
            progress.update(task_id, advance=1)
        animation = mpy.VideoClip(
            lambda t: _make_animation_frame(t, precomputed_figs, fps),
            duration=time_steps / fps,
        )
        animation.write_videofile(path, fps=fps, logger=None, verbose=False)
        progress.update(task_id, visible=False)
    
    try:
        sa.delete("xy")
        sa.delete("xz")
        sa.delete("yz")
    except Exception as e:
        print(e)
    return path