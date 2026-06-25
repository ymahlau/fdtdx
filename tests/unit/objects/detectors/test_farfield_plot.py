"""Unit tests for the far-field plotting helpers."""

import matplotlib
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")

from fdtdx.objects.detectors.plotting.farfield_plot import plot_kspace, plot_radiation_pattern


def test_plot_radiation_pattern_returns_figure():
    theta = np.linspace(0.0, np.pi, 50)
    intensity = np.sin(theta) ** 2
    fig = plot_radiation_pattern(theta, intensity)
    assert isinstance(fig, Figure)


def test_plot_kspace_returns_figure():
    u = np.linspace(-1.0, 1.0, 20)
    ux, uy = np.meshgrid(u, u)
    intensity = np.exp(-(ux**2 + uy**2) * 5.0)
    fig = plot_kspace(ux, uy, intensity)
    assert isinstance(fig, Figure)
