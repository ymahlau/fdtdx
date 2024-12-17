"""Plotting utilities for visualizing data with error bounds and confidence intervals.

This module provides functions for creating plots with filled regions representing
standard deviations or custom confidence intervals. The utilities support:
- Plotting mean curves with standard deviation bands
- Custom upper/lower bound visualization
- Configurable styling including colors, markers, and transparency
- Optional value clipping
"""

from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt


def plot_filled_std_curves(
    x: np.ndarray,
    mean: np.ndarray,
    color: Any,
    lighter_color: Any,
    std: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    lower: Optional[np.ndarray] = None,
    linestyle: str = "-",
    marker: Optional[str] = None,
    label: Optional[str] = None,
    alpha: float = 0.2,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
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
        x: Array of x-axis values.
        mean: Array of y-axis values for the mean curve.
        color: Color for the mean curve line.
        lighter_color: Color for the filled standard deviation region.
        std: Optional standard deviation array. If provided, used to compute upper/lower bounds.
        upper: Optional array of upper bound values. Must be provided with lower.
        lower: Optional array of lower bound values. Must be provided with upper.
        linestyle: Style of the mean curve line. Defaults to solid line "-".
        marker: Optional marker style for data points on the mean curve.
        label: Optional label for the plot legend.
        alpha: Transparency value for the filled region. Defaults to 0.2.
        min_val: Optional minimum value to clip the curves.
        max_val: Optional maximum value to clip the curves.

    Raises:
        ValueError: If neither std nor both upper/lower bounds are provided, or if only
            one of upper/lower is provided.

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
