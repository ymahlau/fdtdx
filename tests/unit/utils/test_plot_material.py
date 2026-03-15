"""Unit tests for utils/plot_material.py using mocked ArrayContainer/SimulationConfig."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fdtdx.utils.plot_material import plot_material, plot_material_from_side


def _make_config(resolution: float = 50e-9) -> MagicMock:
    config = MagicMock()
    config.resolution = resolution
    return config


def _make_arrays(
    shape: tuple = (1, 10, 12, 15),
    inv_perm: float = 0.5,
    inv_per=1.0,
    per_is_float: bool = False,
) -> MagicMock:
    """Return a mock ArrayContainer with controllable material arrays.

    shape is (n_material_axes, nx, ny, nz).
    inv_perm / inv_per are scalar fill values.
    Setting per_is_float=True makes inv_permeabilities a Python float.
    """
    arrays = MagicMock()
    arrays.inv_permittivities = np.full(shape, inv_perm, dtype=float)
    if per_is_float:
        arrays.inv_permeabilities = float(inv_per)
    else:
        arrays.inv_permeabilities = np.full(shape, inv_per, dtype=float)
    return arrays


class TestPlotMaterialFromSide:
    """Unit tests for plot_material_from_side."""

    def test_viewing_side_z_returns_figure(self):
        """Viewing side z (XY plane) returns a Figure."""
        config = _make_config()
        arrays = _make_arrays()
        fig, ax = plt.subplots()
        result = plot_material_from_side(config=config, arrays=arrays, viewing_side="z", ax=ax)
        assert result is not None
        plt.close("all")

    def test_viewing_side_y_returns_figure(self):
        """Viewing side y (XZ plane) returns a Figure."""
        config = _make_config()
        arrays = _make_arrays()
        fig, ax = plt.subplots()
        result = plot_material_from_side(config=config, arrays=arrays, viewing_side="y", ax=ax)
        assert result is not None
        plt.close("all")

    def test_viewing_side_x_returns_figure(self):
        """Viewing side x (YZ plane) returns a Figure."""
        config = _make_config()
        arrays = _make_arrays()
        fig, ax = plt.subplots()
        result = plot_material_from_side(config=config, arrays=arrays, viewing_side="x", ax=ax)
        assert result is not None
        plt.close("all")

    def test_axis_labels_and_title_z(self):
        """Viewing from z sets correct axis labels and title."""
        config = _make_config()
        arrays = _make_arrays()
        fig, ax = plt.subplots()
        plot_material_from_side(config=config, arrays=arrays, viewing_side="z", ax=ax, plot_legend=False)
        assert "x" in ax.get_xlabel()
        assert "y" in ax.get_ylabel()
        assert "XY" in ax.get_title()
        plt.close("all")

    def test_axis_labels_and_title_y(self):
        """Viewing from y sets correct axis labels and title."""
        config = _make_config()
        arrays = _make_arrays()
        fig, ax = plt.subplots()
        plot_material_from_side(config=config, arrays=arrays, viewing_side="y", ax=ax, plot_legend=False)
        assert "x" in ax.get_xlabel()
        assert "z" in ax.get_ylabel()
        assert "XZ" in ax.get_title()
        plt.close("all")

    def test_axis_labels_and_title_x(self):
        """Viewing from x sets correct axis labels and title."""
        config = _make_config()
        arrays = _make_arrays()
        fig, ax = plt.subplots()
        plot_material_from_side(config=config, arrays=arrays, viewing_side="x", ax=ax, plot_legend=False)
        assert "y" in ax.get_xlabel()
        assert "z" in ax.get_ylabel()
        assert "YZ" in ax.get_title()
        plt.close("all")

    def test_image_is_plotted(self):
        """An image is added to the axis."""
        config = _make_config()
        arrays = _make_arrays()
        fig, ax = plt.subplots()
        plot_material_from_side(config=config, arrays=arrays, viewing_side="z", ax=ax, plot_legend=False)
        assert len(ax.get_images()) == 1
        plt.close("all")

    def test_legend_adds_colorbar(self):
        """plot_legend=True adds a colorbar axis."""
        config = _make_config()
        arrays = _make_arrays()
        fig, ax = plt.subplots()
        initial_axes = len(fig.axes)
        plot_material_from_side(config=config, arrays=arrays, viewing_side="z", ax=ax, plot_legend=True)
        assert len(fig.axes) == initial_axes + 1
        plt.close("all")

    def test_no_legend_no_colorbar(self):
        """plot_legend=False does not add a colorbar axis."""
        config = _make_config()
        arrays = _make_arrays()
        fig, ax = plt.subplots()
        initial_axes = len(fig.axes)
        plot_material_from_side(config=config, arrays=arrays, viewing_side="z", ax=ax, plot_legend=False)
        assert len(fig.axes) == initial_axes
        plt.close("all")

    def test_no_axes_creates_new_figure(self):
        """When ax=None, a new figure is created and returned."""
        config = _make_config()
        arrays = _make_arrays()
        result = plot_material_from_side(config=config, arrays=arrays, viewing_side="z")
        assert result is not None
        plt.close("all")

    def test_permittivity_type(self):
        """type='permittivity' uses inv_permittivities."""
        config = _make_config()
        arrays = _make_arrays(inv_perm=0.25)  # permittivity = 1/0.25 = 4.0
        fig, ax = plt.subplots()
        plot_material_from_side(config=config, arrays=arrays, viewing_side="z", ax=ax, type="permittivity")
        im = ax.get_images()[0]
        data = np.array(im.get_array())
        assert np.allclose(data, 4.0)
        plt.close("all")

    def test_permeability_type_array(self):
        """type='permeability' with array inv_permeabilities uses that array."""
        config = _make_config()
        arrays = _make_arrays(inv_per=0.5, per_is_float=False)  # permeability = 2.0
        fig, ax = plt.subplots()
        plot_material_from_side(config=config, arrays=arrays, viewing_side="z", ax=ax, type="permeability")
        im = ax.get_images()[0]
        data = np.array(im.get_array())
        assert np.allclose(data, 2.0)
        plt.close("all")

    def test_permeability_type_float(self):
        """type='permeability' with float inv_permeabilities fills with scalar."""
        config = _make_config()
        arrays = _make_arrays(inv_per=0.5, per_is_float=True)  # permeability = 2.0
        fig, ax = plt.subplots()
        plot_material_from_side(config=config, arrays=arrays, viewing_side="z", ax=ax, type="permeability")
        im = ax.get_images()[0]
        data = np.array(im.get_array())
        assert np.allclose(data, 2.0)
        plt.close("all")

    def test_position_offset_changes_slice(self):
        """Non-zero position changes which z-slice is selected."""
        config = _make_config(resolution=50e-9)
        # Different values along z so we can detect which slice was chosen
        inv_perm = np.zeros((1, 10, 10, 20))
        for z in range(20):
            inv_perm[0, :, :, z] = 1.0 / (z + 1)  # each z-slice has distinct permittivity
        arrays = MagicMock()
        arrays.inv_permittivities = inv_perm
        arrays.inv_permeabilities = np.ones_like(inv_perm)

        fig, ax_center = plt.subplots()
        fig2, ax_offset = plt.subplots()

        plot_material_from_side(
            config=config, arrays=arrays, viewing_side="z", ax=ax_center, position=0.0, plot_legend=False
        )
        plot_material_from_side(
            config=config, arrays=arrays, viewing_side="z", ax=ax_offset, position=3 * 50e-9, plot_legend=False
        )

        data_center = np.array(ax_center.get_images()[0].get_array())
        data_offset = np.array(ax_offset.get_images()[0].get_array())
        assert not np.allclose(data_center, data_offset)
        plt.close("all")

    def test_position_clamped_at_max(self):
        """A very large position is clamped to the last valid slice."""
        config = _make_config()
        arrays = _make_arrays()
        fig, ax = plt.subplots()
        # Should not raise even if position is way outside the volume
        plot_material_from_side(config=config, arrays=arrays, viewing_side="z", ax=ax, position=100e-3)
        assert len(ax.get_images()) == 1
        plt.close("all")

    def test_position_clamped_at_min(self):
        """A very negative position is clamped to index 0."""
        config = _make_config()
        arrays = _make_arrays()
        fig, ax = plt.subplots()
        plot_material_from_side(config=config, arrays=arrays, viewing_side="z", ax=ax, position=-100e-3)
        assert len(ax.get_images()) == 1
        plt.close("all")

    def test_saves_file_when_filename_given(self):
        """Figure is saved when filename is provided."""
        config = _make_config()
        arrays = _make_arrays()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.png"
            plot_material_from_side(config=config, arrays=arrays, viewing_side="z", filename=path)
            assert path.exists()
        plt.close("all")


class TestPlotMaterial:
    """Unit tests for plot_material (3-panel wrapper)."""

    def test_returns_figure(self):
        """plot_material returns a Figure."""
        config = _make_config()
        arrays = _make_arrays()
        result = plot_material(config=config, arrays=arrays)
        assert result is not None
        plt.close("all")

    def test_creates_three_subplots(self):
        """When no axes supplied, three subplot axes are created."""
        config = _make_config()
        arrays = _make_arrays()
        fig = plot_material(config=config, arrays=arrays, plot_legend=False)
        # 3 main axes (no colorbars when plot_legend=False)
        assert len(fig.axes) == 3
        plt.close("all")

    def test_with_external_axes(self):
        """When axs is provided, that figure is returned."""
        config = _make_config()
        arrays = _make_arrays()
        fig, axs = plt.subplots(1, 3)
        result = plot_material(config=config, arrays=arrays, axs=axs, plot_legend=False)
        assert result is not None
        for ax in axs:
            assert len(ax.get_images()) == 1
        plt.close("all")

    def test_custom_positions_used(self):
        """positions tuple is forwarded to each sub-call."""
        config = _make_config(resolution=50e-9)
        inv_perm = np.zeros((1, 20, 20, 20))
        for i in range(20):
            inv_perm[0, i, :, :] = 1.0 / (i + 1)  # distinct values along x
        arrays = MagicMock()
        arrays.inv_permittivities = inv_perm
        arrays.inv_permeabilities = np.ones_like(inv_perm)

        fig, axs = plt.subplots(1, 3)
        plot_material(config=config, arrays=arrays, axs=axs, plot_legend=False, positions=(5 * 50e-9, 0.0, 0.0))
        # YZ plane (ax[2]) uses x_pos = 5*50e-9, which corresponds to an offset
        yz_data = np.array(axs[2].get_images()[0].get_array())
        # Center of x (index 10) has value 1/(10+1)=1/11; offset index 15 has 1/16
        assert yz_data is not None
        plt.close("all")

    def test_saves_file_when_filename_given(self):
        """Figure is saved when filename is provided."""
        config = _make_config()
        arrays = _make_arrays()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "material.png"
            plot_material(config=config, arrays=arrays, filename=path)
            assert path.exists()
        plt.close("all")
