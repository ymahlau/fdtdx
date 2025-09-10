import datetime
import uuid
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np


def generate_unique_filename(prefix="file", extension=None):
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add a short UUID segment for extra uniqueness
    unique_id = str(uuid.uuid4())[:8]

    # Combine components
    if extension:
        return f"{prefix}_{timestamp}_{unique_id}.{extension}"
    return f"{prefix}_{timestamp}_{unique_id}"


def debug_plot_2d(
    #: The 2D array to visualize. Can be either a numpy array or JAX array.
    array: np.ndarray | jax.Array,
    #: The matplotlib colormap to use for the visualization. Defaults to "viridis".
    cmap: str = "viridis",
    #: If True, overlays the numerical values on each cell. Defaults to False.
    show_values: bool = False,
    #: Directory where the plot will be saved. Will be created if it doesn't exist.
    #: Defaults to "outputs/tmp/debug".
    tmp_dir: str | Path = "outputs/tmp/debug",
    #: Name for the output file. If None, generates a unique name using timestamp.
    #: The .png extension will be added automatically. Defaults to None.
    filename: str | None = None,
    #: If true, center all values around x=0. Defaults to False.
    center_zero: bool = False,
) -> None:
    """Creates a debug visualization of a 2D array and saves it to disk.

    This function is useful for debugging array values during development and testing.
    It creates a heatmap visualization with optional value annotations and automatically
    saves it to a specified directory.

    The resulting plot includes:
        - A heatmap visualization of the array values
        - A colorbar showing the value scale
        - Grid lines for better readability
        - Axis labels indicating array dimensions
        - Optional numerical value annotations in each cell
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)

    if filename is None:
        filename = generate_unique_filename("debug", "png")

    plt.figure(figsize=(10, 8))

    # Set up colormap centering
    if center_zero:
        # Find the maximum absolute value to center around zero
        vmax = np.max(np.abs(array))
        vmin = -vmax
    else:
        vmin, vmax = None, None

    # Create heatmap
    im = plt.imshow(
        array.T,
        cmap=cmap,
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im)
    plt.xlabel("First Array axis (x)")
    plt.ylabel("Second Array axis (y)")

    # Show values in cells if requested
    if show_values:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                text_color = "white" if im.norm(array[i, j]) > 0.5 else "black"  # type: ignore
                plt.text(j, i, f"{array[i, j]:.2f}", ha="center", va="center", color=text_color)

    if isinstance(tmp_dir, str):
        tmp_dir = Path(tmp_dir)

    plt.grid(True)

    plt.savefig(tmp_dir / filename, dpi=400, bbox_inches="tight")


def debug_plot_lines(
    #: Dictionary mapping names to 1D arrays (numpy or JAX arrays).
    data_dict: dict[str, np.ndarray | jax.Array],
    #: Optional x-axis values for all lines.
    #: If None, indices will be used. Defaults to None.
    x_values: np.ndarray | jax.Array | None = None,
    #: Optional dictionary mapping names to colors.
    #: If None, default color cycle is used.
    colors: dict[str, str] | None = None,
    #: Optional dictionary mapping names to line styles.
    #: If None, solid lines are used.
    line_styles: dict[str, str] | None = None,
    #: Optional dictionary mapping names to markers.
    #: If None, no markers are used.
    markers: dict[str, str] | None = None,
    #: Label for the x-axis. Defaults to "X".
    x_label: str = "X",
    #: Label for the y-axis. Defaults to "Y".
    y_label: str = "Y",
    #: Title for the plot. Defaults to "Debug Line Plot".
    title: str = "Debug Line Plot",
    #: Location for the legend. Defaults to "best".
    legend_loc: str = "best",
    #: If True, adds grid lines to the plot. Defaults to True.
    grid: bool = True,
    #: Directory where the plot will be saved. Will be created if it doesn't exist.
    #: Defaults to "outputs/tmp/debug".
    tmp_dir: str | Path = "outputs/tmp/debug",
    #: Name for the output file. If None, generates a unique name using timestamp.
    #: The .png extension will be added automatically.
    filename: str | None = None,
) -> None:
    """Creates a debug visualization of multiple 1D arrays as line plots and saves it to disk.

    This function is useful for debugging array values during development and testing.
    It creates a multi-line plot for comparing multiple 1D arrays and automatically
    saves it to a specified directory.

    The resulting plot includes:
        - Multiple lines representing each 1D array in the input dictionary
        - A legend identifying each line
        - Axis labels and title
        - Optional grid lines for better readability
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 8))

    # Create plot for each array in the dictionary
    for name, array in data_dict.items():
        # Convert to numpy if needed
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)

        # Ensure array is 1D
        if array.ndim != 1:
            raise ValueError(f"Array '{name}' must be 1-dimensional, got shape {array.shape}")

        # Get plot parameters for this line
        color = colors.get(name) if colors else None
        line_style = line_styles.get(name) if line_styles else "-"
        marker = markers.get(name) if markers else None

        # Plot the line
        if x_values is not None:
            plt.plot(x_values, array, label=name, color=color, linestyle=line_style, marker=marker)
        else:
            plt.plot(array, label=name, color=color, linestyle=line_style, marker=marker)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Add legend
    if len(data_dict) > 1:
        plt.legend(loc=legend_loc)

    # Add grid if requested
    if grid:
        plt.grid(True, linestyle="--", alpha=0.7)

    # Create output directory if it doesn't exist
    if isinstance(tmp_dir, str):
        tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_lines_{timestamp}.png"
    elif not filename.endswith(".png"):
        filename = f"{filename}.png"

    # Save the plot
    tmp_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(tmp_dir / filename, dpi=400, bbox_inches="tight")
    plt.close()
