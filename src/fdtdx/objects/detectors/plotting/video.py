import tempfile
from typing import Callable
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from rich.progress import Progress

def plot_from_slices(slice_arr_tuple, **kwargs):
    import matplotlib.pyplot as plt
    
    # Unpack the tuple of arrays
    xy_slice, xz_slice, yz_slice = slice_arr_tuple
    
    # Filter out all non-imshow kwargs
    valid_imshow_kwargs = {
        k: v for k, v in kwargs.items() 
        if k in ['cmap', 'norm', 'aspect', 'interpolation', 'alpha']
    }
    
    # Create the figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot each slice
    ax1.imshow(xy_slice, **valid_imshow_kwargs)
    ax1.set_title('XY Slice')
    plt.colorbar(ax1.images[0], ax=ax1)
    
    ax2.imshow(xz_slice, **valid_imshow_kwargs)
    ax2.set_title('XZ Slice')
    plt.colorbar(ax2.images[0], ax=ax2)
    
    ax3.imshow(yz_slice, **valid_imshow_kwargs)
    ax3.set_title('YZ Slice')
    plt.colorbar(ax3.images[0], ax=ax3)
    
    plt.tight_layout()
    
    # Get the canvas dimensions
    width, height = fig.canvas.get_width_height()
    
    # Convert the figure to image data
    try:
        # Try buffer_rgba() method first (most reliable)
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape((height, width, 4))
        data = data[:, :, :3]  # Remove alpha channel
    except AttributeError:
        # Fall back to tostring_argb method
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        data = data.reshape((height, width, 4))
        data = data[:, :, 1:]  # Remove alpha channel
    
    plt.close(fig)
    return data

def generate_video_from_slices(
    xy_slice: np.ndarray,
    xz_slice: np.ndarray,
    yz_slice: np.ndarray,
    plt_fn: Callable,
    resolutions: tuple[float, float, float],
    num_worker: int = 8,  # Kept for API compatibility but not used
    fps: int = 10,
    progress: Progress | None = None,
    minvals: tuple[float | None, float | None, float | None] = (None, None, None),
    maxvals: tuple[float | None, float | None, float | None] = (None, None, None),
    plot_dpi: int | None = None,
    plot_interpolation: str = "gaussian",
):
    """Generates an MP4 video from time-series slice data using matplotlib."""
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

    # Create output paths
    base_path = tempfile.mkdtemp()
    video_path = os.path.join(base_path, "animation.mp4")
    
    print(f"\nSaving data to:")
    print(f"  Base directory: {base_path}")
    print(f"  Video path: {video_path}\n")
    
    # Setup progress bar
    if progress is None:
        progress = Progress()
    task_id = progress.add_task("Generating video", total=xy_slice.shape[0])

    # Create figure and animation
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initialize plots
    im1 = ax1.imshow(xy_slice[0], interpolation=plot_interpolation)
    ax1.set_title('XY Slice')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(xz_slice[0], interpolation=plot_interpolation)
    ax2.set_title('XZ Slice')
    plt.colorbar(im2, ax=ax2)
    
    im3 = ax3.imshow(yz_slice[0], interpolation=plot_interpolation)
    ax3.set_title('YZ Slice')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()

    def update(frame):
        im1.set_array(xy_slice[frame])
        im2.set_array(xz_slice[frame])
        im3.set_array(yz_slice[frame])
        progress.update(task_id, advance=1)
        return [im1, im2, im3]

    print(f"Generating animation with {xy_slice.shape[0]} frames...")
    anim = animation.FuncAnimation(
        fig, update, frames=xy_slice.shape[0], 
        interval=1000/fps, blit=True
    )

    try:
        print("Saving animation...")
        anim.save(
            video_path, 
            fps=fps,
            extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'],
            dpi=plot_dpi if plot_dpi else 100
        )
        print(f"Video successfully saved to: {video_path}")
        plt.close(fig)
        return video_path
    
    except Exception as e:
        print(f"Video creation failed with error: {str(e)}")
        print("Saving first frame as fallback...")
        frame_path = os.path.join(base_path, "frame_0000.png")
        plt.savefig(frame_path)
        plt.close(fig)
        return frame_path
    
    finally:
        progress.update(task_id, visible=False)
