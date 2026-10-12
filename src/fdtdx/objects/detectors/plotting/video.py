import contextlib
import multiprocessing as mp
import subprocess
import tempfile
from functools import partial
from typing import Callable, Iterable, Literal, cast

import imageio_ffmpeg
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
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
    coordinate_edges_um: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    aspect: Literal["auto", "equal"] = "equal",
    cmap: str = "default",
    signed_data: bool = True,
):
    xy_slice, xz_slice, yz_slice = slice_tuple

    fig = plot_2d_from_slices(
        xy_slice=xy_slice,
        xz_slice=xz_slice,
        yz_slice=yz_slice,
        resolutions=resolutions,
        coordinate_edges_um=coordinate_edges_um,
        minvals=minvals,
        maxvals=maxvals,
        plot_dpi=plot_dpi,
        plot_interpolation=plot_interpolation,
        aspect=aspect,
        cmap=cmap,
        signed_data=signed_data,
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


# The yuv420p pixel format, which H.264 players widely support, requires even frame dimensions.
_FRAME_ALIGNMENT = 2


def _pad_to_even_size(frame: np.ndarray) -> np.ndarray:
    """Pads an RGB frame by one row at the bottom and/or one column at the right, repeating the border
    pixels, such that both dimensions are even."""
    height, width = frame.shape[:2]
    pad_h, pad_w = -height % _FRAME_ALIGNMENT, -width % _FRAME_ALIGNMENT
    if pad_h or pad_w:
        frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return frame


def _ffmpeg_command(path: str, width: int, height: int, fps: int) -> list[str]:
    """Builds the ffmpeg command that encodes raw RGB frames from stdin as H.264 (yuv420p) into an MP4 file."""
    # fmt: off
    return [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        # input: raw RGB frames piped through stdin
        "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{width}x{height}", "-pix_fmt", "rgb24",
        "-r", f"{fps:.02f}", "-an", "-i", "-",
        # output: H.264 with the encoder's default quality, readable by virtually all players
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", path,
    ]
    # fmt: on


def _write_mp4(path: str, frames: Iterable[np.ndarray], fps: int, on_frame: Callable[[], None]) -> None:
    """Encodes a stream of equally sized uint8 RGB frames as an H.264 MP4 file using the ffmpeg binary
    bundled with imageio-ffmpeg. on_frame is called after each frame has been written.

    ffmpeg is started directly instead of through imageio_ffmpeg.write_frames, because the latter uses a
    preexec_fn on POSIX systems, which is unsafe in multithreaded processes such as those running JAX.
    """
    proc = None
    first_shape = None
    # ffmpeg's error output goes to a file instead of a pipe, so it can never fill up and block ffmpeg
    with tempfile.TemporaryFile() as stderr_file:
        try:
            for frame in frames:
                if frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError(f"Expected RGB frames of shape (height, width, 3), got {frame.shape}")
                if first_shape is None:
                    first_shape = frame.shape
                elif frame.shape != first_shape:
                    raise ValueError(f"All video frames must have the same shape, got {frame.shape} and {first_shape}")
                frame = np.ascontiguousarray(_pad_to_even_size(frame), dtype=np.uint8)
                if proc is None:
                    proc = subprocess.Popen(
                        _ffmpeg_command(path, width=frame.shape[1], height=frame.shape[0], fps=fps),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=stderr_file,
                        # do not open a console window on Windows
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                assert proc.stdin is not None
                try:
                    proc.stdin.write(frame.data)
                except BrokenPipeError:
                    # ffmpeg exited early, its error is reported below
                    break
                on_frame()
            if proc is None:
                raise ValueError("Cannot write a video without frames")
            assert proc.stdin is not None
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            return_code = proc.wait()
        except BaseException:
            # do not leave a running ffmpeg process behind on errors or KeyboardInterrupt
            if proc is not None:
                proc.kill()
                if proc.stdin is not None:
                    with contextlib.suppress(OSError):
                        # flushing frames buffered for the killed process fails, but closes the pipe anyway
                        proc.stdin.close()
                proc.wait()
            raise
        if return_code != 0:
            stderr_file.seek(0)
            ffmpeg_error = stderr_file.read().decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed with exit code {return_code} while writing {path}:\n{ffmpeg_error}")


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
    coordinate_edges_um: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    aspect: Literal["auto", "equal"] = "equal",
    cmap: str = "default",
    signed_data: bool = True,
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
        coordinate_edges_um: Optional rectilinear edge coordinates in micrometres
            for plotting non-uniform grids without resampling.
        num_worker (int | None): Number of parallel worker processes. If None, frames are processes sequentially.
        plot_interpolation (str): Interpolation method for imshow ('gaussian', 'nearest', etc)
        plot_dpi (int | None): DPI resolution for the figures. None uses default.
        fps (int): Frames per second in output video
        progress (Progress | None, optional): Optional progress bar instance. Defaults to None.
        minvals (tuple[float | None, float | None, float | None], optional): Tuple of minimum values for colormap
            scaling. None values are auto-scaled. Defaults to (None, None, None)
        maxvals (tuple[float | None, float | None, float | None]): Tuple of maximum values for colormap scaling.
            None values are auto-scaled. Defaults to (None, None, None).
        cmap: str = "default": Color map for the detector plots. "default" is turbo
            for unsigned data and RdBu_r for signed data.
        aspect: Literal["auto", "equal"]: Size aspect of the detector plots.
            "equal" (default) uses the same scale for all axes.
            "auto" adjusts each axis's scale to fit the figure size.
        signed_data: bool: Whether the data is signed (fields, phasors, mode_overlaps...)
            or unsigned (energy, Poynting flux). Used to choose a colormap and scale its values.

    Returns:
        str: Path to the generated MP4 video file
    """
    slices = [xy_slice, xz_slice, yz_slice]
    minlist = list(minvals)  # Convert to list for mutability
    maxlist = list(maxvals)  # Convert to list for mutability
    # Reset max and min color values
    for a in range(3):
        if signed_data:
            if minvals[a] is None or maxvals[a] is None:
                if minlist[a] is None and maxlist[a] is None:
                    # Set values symmetrically with absolute maximum
                    abs_max = float(np.abs(slices[a]).max())
                    minlist[a] = -abs_max
                    maxlist[a] = abs_max
                elif minlist[a] is not None:
                    # Keep user input, set values symmetrically
                    # minvals[a] is guaranteed to be not None here
                    maxlist[a] = -cast(float, minlist[a])
                elif maxlist[a] is not None:
                    # Keep user input, set values symmetrically
                    # maxvals[a] is guaranteed to be not None here
                    minlist[a] = -cast(float, maxlist[a])
        else:  # Data is unsigned
            if minlist[a] is None:
                minlist[a] = 0.0
            if maxlist[a] is None:
                maxlist[a] = float(slices[a].max())

    _, path = tempfile.mkstemp(suffix=".mp4")

    time_steps = xy_slice.shape[0]
    if progress is None:
        progress = Progress()
    task_id = progress.add_task("Generating video", total=time_steps)
    partial_fn = partial(
        plt_fn,
        resolutions=resolutions,
        minvals=(minlist[0], minlist[1], minlist[2]),
        maxvals=(maxlist[0], maxlist[1], maxlist[2]),
        plot_dpi=plot_dpi,
        plot_interpolation=plot_interpolation,
        coordinate_edges_um=coordinate_edges_um,
        aspect=aspect,
        cmap=cmap,
        signed_data=signed_data,
    )
    slice_arr_list = [(xy_slice[t], xz_slice[t], yz_slice[t]) for t in range(time_steps)]
    # without workers, frames are rendered one after the other. Otherwise, a multiprocessing pool renders
    # them in parallel. In both cases, frames are encoded in order as soon as they are available.
    with mp.Pool(num_worker) if num_worker is not None else contextlib.nullcontext() as pool:
        frames = map(partial_fn, slice_arr_list) if pool is None else pool.imap(partial_fn, slice_arr_list)
        _write_mp4(path, frames, fps=fps, on_frame=lambda: progress.update(task_id, advance=1))
    progress.update(task_id, visible=False)

    return path
