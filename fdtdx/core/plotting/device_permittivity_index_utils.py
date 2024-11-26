from typing import cast
import jax
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


def index_matrix_to_str(indices: jax.Array):
    indices_str = ""
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            indices_str += str(indices[i, j].squeeze()) + " "
        indices_str += "\n"
    return indices_str


def device_matrix_index_figure(
    device_matrix_indices: jax.Array,
    permittivity_configs: tuple[tuple[str, float], ...],
):
    assert device_matrix_indices.ndim == 3
    device_matrix_indices = device_matrix_indices.astype(np.int32)
    fig, ax = cast(tuple[Figure, Axes], plt.subplots(figsize=(12, 12)))
    image_palette = sns.color_palette("YlOrBr", as_cmap=True)
    if device_matrix_indices.shape[-1] == 1:
        device_matrix_indices = device_matrix_indices[..., 0]
        matrix_inverse_permittivity_indices_sorted = device_matrix_indices
        indices = np.unique(device_matrix_indices)
    else:
        air_index = None
        for i, cfg in enumerate(permittivity_configs):
            if cfg[0] == "Air":
                air_index = i
                break
        device_matrix_indices_flat = np.reshape(
            device_matrix_indices, (-1, device_matrix_indices.shape[-1])
        )
        indices = np.unique(
            device_matrix_indices_flat,
            axis=0,
        )
        air_count = np.count_nonzero(indices == air_index, axis=-1)
        air_count_argsort = np.argsort(air_count)
        indices_sorted = indices[air_count_argsort]
        matrix_inverse_permittivity_indices_sorted = np.array(
            [
                np.where((indices_sorted == device_matrix_indices_flat[i]).all(axis=1))[
                    0
                ][0]
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
                ax.text(
                    x, y, str(int(value)), ha="center", va="center", color=text_color
                )
    assert cax.cmap is not None
    if indices.ndim == 1:
        legend_elements = [
            Patch(
                facecolor=cax.cmap(cax.norm(int(i))),
                label=f"({i}) {permittivity_configs[int(i)][0]}",
            )
            for i in indices
        ]
    else:
        legend_elements = [
            Patch(
                facecolor=cax.cmap(cax.norm(int(i))),
                label=f"({i}) "
                + "|".join([permittivity_configs[int(e)][0] for e in indices[i]]),
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
