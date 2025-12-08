from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_field_slice_component(
        field: jnp.ndarray,
        component_name: str,
        ax: Any,
        plot_legend: bool = True,
) -> None:
    """Plots a single component of the electromagnetic field.

    Args:
        field (jnp.ndarray): 2D array of shape (w, h) containing field values
        component_name (str): Name of the component (e.g., 'Ex', 'Hy')
        ax (Any): Matplotlib axis to plot on
        plot_legend (bool, optional): Whether to add a colorbar legend

    Raises:
        ValueError: If field is not 2D or contains invalid values
    """
    # Sanity checks
    if field.ndim != 2:
        raise ValueError(f"Field must be 2D, got shape {field.shape}")

    if jnp.any(jnp.isnan(field)):
        raise ValueError(f"{component_name} contains NaN values")

    if jnp.any(jnp.isinf(field)):
        raise ValueError(f"{component_name} contains infinite values")

    # Plot the field component
    im = ax.imshow(
        field.T,  # Transpose for correct orientation
        origin="lower",
        aspect="equal",
        cmap="RdBu_r",  # Red-blue colormap, centered at zero
        interpolation="nearest",
    )

    # Set labels and title
    ax.set_xlabel("x (grid points)")
    ax.set_ylabel("y (grid points)")
    ax.set_title(component_name)
    ax.grid(True, alpha=0.3)

    # Add colorbar if legend is requested
    if plot_legend:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Field value")


def plot_field_slice(
        E: jnp.ndarray,
        H: jnp.ndarray,
        filename: str | Path | None = None,
        axs: Any | None = None,
        plot_legend: bool = True,
) -> Figure:
    """Creates a visualization of electromagnetic field components.

    Generates a 2x3 subplot showing all six components of the electromagnetic field
    (Ex, Ey, Ez, Hx, Hy, Hz) in a single figure.

    Args:
        E (jnp.ndarray): Electric field array of shape (3, nx, ny, nz) or (3, w, h)
        H (jnp.ndarray): Magnetic field array of shape (3, nx, ny, nz) or (3, w, h)
        filename (str | Path | None, optional): If provided, saves the plot to this file
        axs (Any | None, optional): Optional matplotlib axes to plot on. If None, creates new figure
        plot_legend (bool, optional): Whether to add colorbar legends

    Returns:
        Figure: The generated figure object

    Raises:
        ValueError: If arrays have incorrect shapes or contain invalid values

    Note:
        For 4D inputs (3, nx, ny, nz), exactly one of nx, ny, nz must be 1.
        The function will automatically squeeze out the singleton dimension.
    """
    # Sanity checks for E field
    if E.shape[0] != 3:
        raise ValueError(f"E field first dimension must be 3, got shape {E.shape}")

    if E.ndim not in (3, 4):
        raise ValueError(f"E field must be 3D (3, w, h) or 4D (3, nx, ny, nz), got shape {E.shape}")

    # Sanity checks for H field
    if H.shape[0] != 3:
        raise ValueError(f"H field first dimension must be 3, got shape {H.shape}")

    if H.ndim not in (3, 4):
        raise ValueError(f"H field must be 3D (3, w, h) or 4D (3, nx, ny, nz), got shape {H.shape}")

    # Check if shapes match
    if E.shape != H.shape:
        raise ValueError(f"E and H fields must have same shape, got E: {E.shape}, H: {H.shape}")

    # Handle 4D case - exactly one dimension must be 1
    if E.ndim == 4:
        spatial_dims = E.shape[1:]
        singleton_count = sum(dim == 1 for dim in spatial_dims)

        if singleton_count != 1:
            raise ValueError(
                f"For 4D input (3, nx, ny, nz), exactly one of nx, ny, nz must be 1. "
                f"Got shape {E.shape} with {singleton_count} singleton dimensions"
            )

        # Squeeze out the singleton dimension
        E = jnp.squeeze(E, axis=tuple(i + 1 for i, dim in enumerate(spatial_dims) if dim == 1))
        H = jnp.squeeze(H, axis=tuple(i + 1 for i, dim in enumerate(spatial_dims) if dim == 1))

    # Now E and H should be (3, w, h)
    if E.ndim != 3 or E.shape[0] != 3:
        raise ValueError(f"After processing, E should be (3, w, h), got {E.shape}")

    # Check for NaN and infinite values
    if jnp.any(jnp.isnan(E)):
        raise ValueError("E field contains NaN values")

    if jnp.any(jnp.isinf(E)):
        raise ValueError("E field contains infinite values")

    if jnp.any(jnp.isnan(H)):
        raise ValueError("H field contains NaN values")

    if jnp.any(jnp.isinf(H)):
        raise ValueError("H field contains infinite values")

    # Create figure if axes not provided
    if axs is None:
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    else:
        fig = None

    # Component names
    E_components = ['Ex', 'Ey', 'Ez']
    H_components = ['Hx', 'Hy', 'Hz']

    # Plot E field components (top row)
    for i, comp_name in enumerate(E_components):
        plot_field_slice_component(
            field=E[i],
            component_name=comp_name,
            ax=axs[0, i],
            plot_legend=plot_legend,
        )

    # Plot H field components (bottom row)
    for i, comp_name in enumerate(H_components):
        plot_field_slice_component(
            field=H[i],
            component_name=comp_name,
            ax=axs[1, i],
            plot_legend=plot_legend,
        )

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()

    return plt.gcf() if fig is None else fig