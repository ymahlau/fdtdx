"""Plotting helpers for far-field projector results (accessor-driven).

These take the arrays returned by ``PlanarFarFieldProjector`` / ``BoxFarFieldProjector``
accessors (e.g. :meth:`spherical`, :meth:`kspace`) and render matplotlib figures. They are not
wired through ``Detector.draw_plot`` (whose slice machinery does not fit far-field outputs);
call them directly with accessor output.
"""

import numpy as np
from matplotlib.figure import Figure


def plot_radiation_pattern(
    theta: np.ndarray,
    intensity: np.ndarray,
    *,
    normalize: bool = True,
    title: str = "Radiation pattern",
    plot_dpi: int | None = None,
) -> Figure:
    """Polar plot of a far-field intensity cut ``|E(theta)|^2`` at fixed azimuth.

    Args:
        theta: 1D polar angles (radians).
        intensity: 1D intensity ``|E_theta|^2 + |E_phi|^2`` (same length as ``theta``).
        normalize: Scale the pattern to a peak of 1.
        title: Plot title.
        plot_dpi: Optional figure DPI.

    Returns:
        A matplotlib ``Figure`` with a polar axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.projections.polar import PolarAxes

    theta = np.asarray(theta)
    intensity = np.asarray(intensity, dtype=float)
    if normalize and float(np.max(intensity)) > 0:
        intensity = intensity / float(np.max(intensity))
    fig = plt.figure(dpi=plot_dpi)
    ax = fig.add_subplot(111, projection="polar")
    assert isinstance(ax, PolarAxes)
    ax.plot(theta, intensity)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title)
    return fig


def plot_kspace(
    ux: np.ndarray,
    uy: np.ndarray,
    intensity: np.ndarray,
    *,
    normalize: bool = True,
    title: str = "Fourier plane (k-space)",
    plot_dpi: int | None = None,
) -> Figure:
    """Image of a far-field intensity on the direction-cosine ``(ux, uy)`` grid.

    Args:
        ux, uy: 2D direction-cosine grids (e.g. from ``PlanarFarFieldProjector.kspace``).
        intensity: 2D intensity ``|E_theta|^2 + |E_phi|^2`` on the same grid.
        normalize: Scale the image to a peak of 1.
        title: Plot title.
        plot_dpi: Optional figure DPI.

    Returns:
        A matplotlib ``Figure`` with the unit circle marked.
    """
    import matplotlib.pyplot as plt

    ux = np.asarray(ux)
    uy = np.asarray(uy)
    intensity = np.asarray(intensity, dtype=float)
    if normalize and float(np.max(intensity)) > 0:
        intensity = intensity / float(np.max(intensity))
    fig = plt.figure(dpi=plot_dpi)
    ax = fig.add_subplot(111)
    mesh = ax.pcolormesh(ux, uy, intensity, shading="auto")
    circle = plt.Circle((0, 0), 1.0, fill=False, color="white", linestyle="--", linewidth=0.8)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlabel("ux = kx / k")
    ax.set_ylabel("uy = ky / k")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="intensity")
    return fig
