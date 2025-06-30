"""Plotting utilities for visualizing data with error bounds and confidence intervals.

This module provides functions for creating plots with filled regions representing
standard deviations or custom confidence intervals. The utilities support:
- Plotting mean curves with standard deviation bands
- Custom upper/lower bound visualization
- Configurable styling including colors, markers, and transparency
- Optional value clipping
"""

from typing import Any

import numpy as np
from matplotlib import pyplot as plt


def plot_filled_std_curves(
    x: np.ndarray,
    mean: np.ndarray,
    color: Any,
    lighter_color: Any,
    std: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    linestyle: str = "-",
    marker: str | None = None,
    label: str | None = None,
    alpha: float = 0.2,
    min_val: float | None = None,
    max_val: float | None = None,
):
    """Plots a curve with filled standard deviation or confidence intervals.

    Creates a plot showing a mean curve with a filled region representing either
    standard deviation bounds or custom upper/lower bounds. The filled region uses
    a lighter color with transparency.

    The function supports two modes:
    1. Standard deviation mode: Provide std parameter to create bounds at mean ± std
    2. Custom bounds mode: Provide explicit upper and lower bound arrays

    The plotted curves can be optionally clipped to minimum/maximum values.

    Args:
        x (np.ndarray): Array of x-axis values.
        mean (np.ndarray): Array of y-axis values for the mean curve.
        color (Any): Color for the mean curve line.
        lighter_color (Any): Color for the filled standard deviation region.
        std (np.ndarray | None, optional): Optional standard deviation array.
            If provided, used to compute upper/lower bounds. Defaults to None.
        upper (np.ndarray | None, optional): Optional array of upper bound values. Must be provided with lower.
        lower (np.ndarray | None, optional): Optional array of lower bound values. Must be provided with upper.
        linestyle (str, optional): Style of the mean curve line. Defaults to solid line "-".
        marker (str | None, optional): Optional marker style for data points on the mean curve. Defaults to None
        label (str | None, optional): Optional label for the plot legend. Defaults to None.
        alpha (float, optional): Transparency value for the filled region. Defaults to 0.2.
        min_val (float | None, optional): Optional minimum value to clip the curves. Defaults to None.
        max_val (float | None, optional): Optional maximum value to clip the curves. Defaults to None.

    Example:
        >>> x = np.linspace(0, 10, 100)
        >>> mean = np.sin(x)
        >>> std = 0.1 * np.ones_like(x)
        >>> plot_filled_std_curves(x, mean, 'blue', 'lightblue', std=std)
    """
    if (upper is None) != (lower is None):
        raise ValueError("Need to specify both upper and lower")
    if (std is None) == (upper is None):
        raise ValueError("Need to specify either std or upper/lower")
    if std is not None:
        upper = mean + std
        lower = mean - std
    if min_val is not None and lower is not None and upper is not None:
        mean = np.maximum(mean, min_val)
        lower = np.maximum(lower, min_val)
        upper = np.maximum(upper, min_val)
    if max_val is not None and lower is not None and upper is not None:
        mean = np.minimum(mean, max_val)
        upper = np.minimum(upper, max_val)
        lower = np.minimum(lower, max_val)
    if upper is None or lower is None:
        raise Exception("This should never happen")
    plt.plot(x, upper, color=lighter_color, alpha=alpha)
    plt.plot(x, lower, color=lighter_color, alpha=alpha)
    plt.fill_between(x, lower, upper, color=lighter_color, alpha=alpha)
    plt.plot(x, mean, color=color, label=label, linestyle=linestyle, marker=marker, markersize=4)
