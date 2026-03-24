"""Unit tests for utils/plot_setup.py using mocked ObjectContainer/SimulationConfig."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fdtdx.utils.plot_setup import plot_setup, plot_setup_from_side


class _MockColor:
    """Wraps a matplotlib color string to satisfy the .to_mpl() interface."""

    def __init__(self, color_str: str):
        self._color = color_str

    def to_mpl(self) -> str:
        return self._color


class _MockObj:
    """Minimal simulation object for unit-testing the plot_setup functions."""

    def __init__(self, name: str, color, grid_slice):
        self.name = name
        # color can be a string (wrapped into _MockColor) or None to be excluded
        self.color = _MockColor(color) if color is not None else None
        # grid_slice_tuple: ((x0,x1),(y0,y1),(z0,z1))
        self.grid_slice_tuple = grid_slice


def _make_container(objects=None, volume_grid_shape=(50, 50, 50)):
    """Return a mock ObjectContainer.

    objects: list of _MockObj (or similar) to populate container.objects.
    The volume is a separate MagicMock whose grid_shape is volume_grid_shape.
    """
    if objects is None:
        objects = []
    container = MagicMock()
    # volume is NOT in objects list – simulates real behaviour where volume is
    # accessed separately and then excluded
    mock_volume = MagicMock()
    mock_volume.grid_shape = volume_grid_shape
    container.volume = mock_volume
    container.objects = objects
    return container


def _make_config(resolution: float = 50e-9) -> MagicMock:
    config = MagicMock()
    config.resolution = resolution
    return config


class TestPlotSetupFromSide:
    """Unit tests for plot_setup_from_side."""

    def test_viewing_side_z_returns_figure(self):
        """Viewing side z (XY plane) returns a Figure."""
        config = _make_config()
        container = _make_container()
        fig, ax = plt.subplots()
        result = plot_setup_from_side(config=config, objects=container, viewing_side="z", ax=ax)
        assert result is not None
        plt.close("all")

    def test_viewing_side_y_returns_figure(self):
        """Viewing side y (XZ plane) returns a Figure."""
        config = _make_config()
        container = _make_container()
        fig, ax = plt.subplots()
        result = plot_setup_from_side(config=config, objects=container, viewing_side="y", ax=ax)
        assert result is not None
        plt.close("all")

    def test_viewing_side_x_returns_figure(self):
        """Viewing side x (YZ plane) returns a Figure."""
        config = _make_config()
        container = _make_container()
        fig, ax = plt.subplots()
        result = plot_setup_from_side(config=config, objects=container, viewing_side="x", ax=ax)
        assert result is not None
        plt.close("all")

    def test_invalid_viewing_side_raises(self):
        """Unsupported viewing_side raises ValueError."""
        config = _make_config()
        container = _make_container()
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="Invalid viewing_side"):
            plot_setup_from_side(config=config, objects=container, viewing_side="w", ax=ax)
        plt.close("all")

    def test_axis_labels_and_title_z(self):
        """Viewing from z sets x/y labels and XY title."""
        config = _make_config()
        container = _make_container()
        fig, ax = plt.subplots()
        plot_setup_from_side(config=config, objects=container, viewing_side="z", ax=ax)
        assert "x" in ax.get_xlabel()
        assert "y" in ax.get_ylabel()
        assert "XY" in ax.get_title()
        plt.close("all")

    def test_axis_labels_and_title_y(self):
        """Viewing from y sets x/z labels and XZ title."""
        config = _make_config()
        container = _make_container()
        fig, ax = plt.subplots()
        plot_setup_from_side(config=config, objects=container, viewing_side="y", ax=ax)
        assert "x" in ax.get_xlabel()
        assert "z" in ax.get_ylabel()
        assert "XZ" in ax.get_title()
        plt.close("all")

    def test_axis_labels_and_title_x(self):
        """Viewing from x sets y/z labels and YZ title."""
        config = _make_config()
        container = _make_container()
        fig, ax = plt.subplots()
        plot_setup_from_side(config=config, objects=container, viewing_side="x", ax=ax)
        assert "y" in ax.get_xlabel()
        assert "z" in ax.get_ylabel()
        assert "YZ" in ax.get_title()
        plt.close("all")

    def test_no_ax_creates_new_figure(self):
        """When ax=None a new figure is created and returned."""
        config = _make_config()
        container = _make_container()
        result = plot_setup_from_side(config=config, objects=container, viewing_side="z")
        assert result is not None
        plt.close("all")

    def test_colored_objects_rendered_as_patches(self):
        """Objects with a non-None color are added to the axis as patches."""
        config = _make_config()
        obj = _MockObj("box", "blue", ((5, 15), (5, 15), (5, 15)))
        container = _make_container(objects=[obj])
        fig, ax = plt.subplots()
        plot_setup_from_side(config=config, objects=container, viewing_side="z", ax=ax, plot_legend=False)
        # At least one Rectangle patch should be present
        assert len(ax.patches) >= 1
        plt.close("all")

    def test_objects_without_color_excluded(self):
        """Objects with color=None are not rendered."""
        config = _make_config()
        obj_with_color = _MockObj("visible", "red", ((5, 15), (5, 15), (5, 15)))
        obj_no_color = _MockObj("invisible", None, ((20, 30), (20, 30), (20, 30)))
        container = _make_container(objects=[obj_with_color, obj_no_color])
        fig, ax = plt.subplots()
        plot_setup_from_side(config=config, objects=container, viewing_side="z", ax=ax, plot_legend=False)
        # Only 1 patch (from obj_with_color); obj_no_color is skipped
        assert len(ax.patches) == 1
        plt.close("all")

    def test_exclude_object_list(self):
        """Objects in exclude_object_list are not rendered."""
        config = _make_config()
        obj_a = _MockObj("a", "blue", ((5, 15), (5, 15), (5, 15)))
        obj_b = _MockObj("b", "green", ((20, 30), (20, 30), (20, 30)))
        container = _make_container(objects=[obj_a, obj_b])
        fig, ax = plt.subplots()
        plot_setup_from_side(
            config=config,
            objects=container,
            viewing_side="z",
            ax=ax,
            plot_legend=False,
            exclude_object_list=[obj_b],
        )
        assert len(ax.patches) == 1
        plt.close("all")

    def test_exclude_large_object_ratio_removes_large(self):
        """Objects covering more than the threshold fraction are excluded."""
        config = _make_config()
        # A large object covering most of the 50x50 plane (40x40 = 64 %)
        large = _MockObj("big", "red", ((5, 45), (5, 45), (0, 50)))
        # A small object covering 4x4=0.26 % of the plane
        small = _MockObj("tiny", "blue", ((10, 14), (10, 14), (0, 50)))
        container = _make_container(objects=[large, small], volume_grid_shape=(50, 50, 50))
        fig, ax = plt.subplots()
        # threshold = 0.5: large covers 1600/2500 = 0.64 > 0.5, small covers 16/2500 < 0.5
        plot_setup_from_side(
            config=config,
            objects=container,
            viewing_side="z",
            ax=ax,
            plot_legend=False,
            exclude_large_object_ratio=0.5,
        )
        assert len(ax.patches) == 1  # only small object
        plt.close("all")

    def test_no_legend_skips_legend_drawing(self):
        """plot_legend=False does not add a legend to the axis."""
        config = _make_config()
        obj = _MockObj("item", "red", ((5, 15), (5, 15), (5, 15)))
        container = _make_container(objects=[obj])
        fig, ax = plt.subplots()
        plot_setup_from_side(config=config, objects=container, viewing_side="z", ax=ax, plot_legend=False)
        assert ax.get_legend() is None
        plt.close("all")

    def test_legend_added_when_enabled(self):
        """plot_legend=True adds a legend to the axis."""
        config = _make_config()
        obj = _MockObj("sensor", "green", ((5, 15), (5, 15), (5, 15)))
        container = _make_container(objects=[obj])
        fig, ax = plt.subplots()
        plot_setup_from_side(config=config, objects=container, viewing_side="z", ax=ax, plot_legend=True)
        assert ax.get_legend() is not None
        plt.close("all")

    def test_saves_file_when_filename_given(self):
        """Figure is saved when filename is provided."""
        config = _make_config()
        container = _make_container()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "setup.png"
            plot_setup_from_side(config=config, objects=container, viewing_side="z", filename=path)
            assert path.exists()
        plt.close("all")


class TestPlotSetup:
    """Unit tests for plot_setup (3-panel wrapper)."""

    def test_returns_figure(self):
        """plot_setup returns a Figure."""
        config = _make_config()
        container = _make_container()
        result = plot_setup(config=config, objects=container)
        assert result is not None
        plt.close("all")

    def test_creates_three_subplots(self):
        """When no axs are supplied, exactly three axes are created."""
        config = _make_config()
        container = _make_container()
        fig = plot_setup(config=config, objects=container, plot_legend=False)
        assert len(fig.axes) == 3
        plt.close("all")

    def test_with_external_axes(self):
        """When axs is provided, that figure's axes are populated."""
        config = _make_config()
        container = _make_container()
        fig, axs = plt.subplots(1, 3)
        result = plot_setup(config=config, objects=container, axs=axs, plot_legend=False)
        assert result is not None
        for ax in axs:
            assert ax.get_title() != ""
        plt.close("all")

    def test_exclude_object_list_passed_through(self):
        """exclude_object_list is forwarded to each sub-call."""
        config = _make_config()
        obj_a = _MockObj("a", "blue", ((5, 15), (5, 15), (5, 15)))
        obj_b = _MockObj("b", "red", ((20, 30), (20, 30), (20, 30)))
        container = _make_container(objects=[obj_a, obj_b])
        fig, axs = plt.subplots(1, 3)
        plot_setup(
            config=config,
            objects=container,
            axs=axs,
            plot_legend=False,
            exclude_object_list=[obj_b],
        )
        # Each of the 3 axes should show only obj_a (1 patch each)
        for ax in axs:
            assert len(ax.patches) == 1
        plt.close("all")

    def test_saves_file_when_filename_given(self):
        """Figure is saved when filename is provided."""
        config = _make_config()
        container = _make_container()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "setup_all.png"
            plot_setup(config=config, objects=container, filename=path)
            assert path.exists()
        plt.close("all")
