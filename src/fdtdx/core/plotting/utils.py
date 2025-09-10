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
    #: Array of x-axis values.
    x: np.ndarray,
    #: Array of y-axis values for the mean curve.
    mean: np.ndarray,
    #: Color for the mean curve line.
    color: Any,
    #: Color for the filled standard deviation region.
    lighter_color: Any,
    #: Optional standard deviation array.
    #: If provided, used to compute upper/lower bounds. Defaults to None.
    std: np.ndarray | None = None,
    #: Optional array of upper bound values. Must be provided with lower.
    upper: np.ndarray | None = None,
    #: Optional array of lower bound values. Must be provided with upper.
    lower: np.ndarray | None = None,
    #: Style of the mean curve line. Defaults to solid line "-".
    linestyle: str = "-",
    #: Optional marker style for data points on the mean curve. Defaults to None
    marker: str | None = None,
    #: Optional label for the plot legend. Defaults to None.
    label: str | None = None,
    #: Transparency value for the filled region. Defaults to 0.2.
    alpha: float = 0.2,
    #: Optional minimum value to clip the curves. Defaults to None.
    min_val: float | None = None,
    #: Optional maximum value to clip the curves. Defaults to None.
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
