from typing import cast

import jax
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from fdtdx.core.misc import get_background_material_name
from fdtdx.materials import Material, compute_ordered_names
from fdtdx.typing import ParameterType


def index_matrix_to_str(indices: jax.Array) -> str:
    """Converts a 2D matrix of indices to a formatted string representation.

    Args:
        indices (jax.Array): A 2D JAX array containing numerical indices.

    Returns:
        str: A string representation of the matrix where each row is space-separated
        and rows are separated by newlines.
    """
    indices_str = ""
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            indices_str += str(indices[i, j].squeeze()) + " "
        indices_str += "\n"
    return indices_str


def device_matrix_index_figure(
    device_matrix_indices: jax.Array,
    material: dict[str, Material],
    parameter_type: ParameterType,
) -> Figure:
    """Creates a visualization figure of device matrix indices with permittivity configurations.

    Args:
        device_matrix_indices (jax.Array): A 3D JAX array containing the device matrix indices.
            Shape should be (height, width, channels) where channels is typically 1.
        material (dict[str, Material]): A tuple of (name, value) pairs defining the permittivity
            configurations, where name is a string identifier (e.g., "Air") and value
            is the corresponding permittivity value.
        parameter_type (ParameterType): Type of the parameters to be plotted

    Returns:
        Figure: A matplotlib Figure object containing the visualization with:
        - A heatmap of the device matrix indices
        - Color-coded regions based on permittivity configurations
        - Optional text labels showing index values (for smaller matrices)
        - A legend mapping colors to permittivity configurations
        - Proper axis labels and grid settings
    """
    assert device_matrix_indices.ndim == 3

    fig, ax = cast(tuple[Figure, Axes], plt.subplots(figsize=(12, 12)))
    image_palette = sns.color_palette("YlOrBr", as_cmap=True)
    background_name = get_background_material_name(material)
    ordered_name_list = compute_ordered_names(material)
    background_index = ordered_name_list.index(background_name)

    if parameter_type == ParameterType.CONTINUOUS:
        matrix_inverse_permittivity_indices_sorted = device_matrix_indices.mean(axis=-1)
        indices = np.arange(len(ordered_name_list))
    elif device_matrix_indices.shape[-1] == 1:
        device_matrix_indices = device_matrix_indices.astype(np.int32)
        device_matrix_indices = device_matrix_indices[..., 0]
        matrix_inverse_permittivity_indices_sorted = device_matrix_indices
        indices = np.unique(device_matrix_indices)
    else:
        device_matrix_indices = device_matrix_indices.astype(np.int32)
        device_matrix_indices_flat = np.reshape(device_matrix_indices, (-1, device_matrix_indices.shape[-1]))
        indices = np.unique(
            device_matrix_indices_flat,
            axis=0,
        )
        background_count = np.count_nonzero(indices == background_index, axis=-1)
        background_count_argsort = np.argsort(background_count)
        indices_sorted = indices[background_count_argsort]
        matrix_inverse_permittivity_indices_sorted = np.array(
            [
                np.where((indices_sorted == device_matrix_indices_flat[i]).all(axis=1))[0][0]
                for i in range(device_matrix_indices_flat.shape[0])
            ]
        ).reshape(device_matrix_indices.shape[:-1])

    cax = ax.imshow(
        matrix_inverse_permittivity_indices_sorted.T,
        cmap=image_palette,
        aspect="auto",
        origin="lower",
    )
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    height, width = (
        device_matrix_indices.shape[0],
        device_matrix_indices.shape[1],
    )
    if height * width < 1500:
        for y in range(height):
            for x in range(width):
                value = matrix_inverse_permittivity_indices_sorted[x, y]
                text_color = "w" if cax.norm(value) > 0.5 else "k"  # type: ignore
                ax.text(x, y, str(int(value)), ha="center", va="center", color=text_color)
    assert cax.cmap is not None
    if indices.ndim == 1:
        legend_elements = [
            Patch(
                facecolor=cax.cmap(cax.norm(int(i))),
                label=f"({i}) {ordered_name_list[int(i)]}",
            )
            for i in indices
        ]
    else:
        legend_elements = [
            Patch(
                facecolor=cax.cmap(cax.norm(int(i))),
                label=f"({i}) " + "|".join([ordered_name_list[int(e)] for e in indices[i]]),
            )
            for i in np.unique(matrix_inverse_permittivity_indices_sorted)
        ]

    legend_cols = max(1, int(len(legend_elements) / height))
    if len(legend_elements) < 100:
        ax.legend(
            handles=legend_elements,
            loc="center left",
            frameon=False,
            bbox_to_anchor=(1, 0.5),
            ncols=legend_cols,
        )
    ax.set_aspect("equal")
    for line in ax.get_xgridlines() + ax.get_ygridlines():
        line.set_alpha(0.0)
    return fig
