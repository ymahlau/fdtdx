from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import jax
import datetime
import uuid

def generate_unique_filename(prefix="file", extension=None):
    """
    Generate a unique filename using timestamp and UUID.
    
    Parameters:
    -----------
    prefix : str, optional
        Prefix for the filename
    extension : str, optional
        File extension (without dot)
        
    Returns:
    --------
    str : Unique filename
    """
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add a short UUID segment for extra uniqueness
    unique_id = str(uuid.uuid4())[:8]
    
    # Combine components
    if extension:
        return f"{prefix}_{timestamp}_{unique_id}.{extension}"
    return f"{prefix}_{timestamp}_{unique_id}"


def debug_plot_2d(
    array: np.ndarray | jax.Array,
    cmap: str = "viridis", 
    show_values: bool = False,
    tmp_dir: str | Path = "outputs/tmp/debug",
    filename: str | None = None,
):
    """
    Plot a 2D numpy array as a heatmap.
    
    Parameters:
    -----------
    array : numpy.ndarray
        2D array to visualize
    title : str, optional
        Title for the plot
    cmap : str, optional
        Colormap to use for the visualization
    show_values : bool, optional
        Whether to show numerical values in each cell
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    
    if filename is None:
        filename = generate_unique_filename("debug", "png")
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    im = plt.imshow(
        array.T, 
        cmap=cmap, 
        origin="lower",
        aspect="equal",
    )
    
    plt.colorbar(im)
    plt.xlabel('First Array axis (x)')
    plt.ylabel('Second Array axis (y)')
    
    # Show values in cells if requested
    if show_values:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                text_color = 'white' if im.norm(array[i, j]) > 0.5 else 'black'  # type: ignore
                plt.text(j, i, f'{array[i, j]:.2f}',
                        ha='center', va='center', color=text_color)
    
    if isinstance(tmp_dir, str):
        tmp_dir = Path(tmp_dir)
    
    plt.grid(True)
    
    plt.savefig(tmp_dir / filename, dpi=400, bbox_inches="tight")