import matplotlib

matplotlib.use("Agg")

from unittest.mock import patch

import numpy as np
import pytest
from matplotlib import pyplot as plt

from fdtdx.core.plotting.utils import plot_filled_std_curves


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


class TestPlotFilledStdCurvesValidation:
    """Tests for input validation in plot_filled_std_curves."""

    def test_only_upper_raises(self):
        x = np.arange(5, dtype=float)
        mean = np.ones(5)
        with pytest.raises(ValueError, match="both upper and lower"):
            plot_filled_std_curves(x, mean, "blue", "lightblue", upper=np.ones(5))

    def test_only_lower_raises(self):
        x = np.arange(5, dtype=float)
        mean = np.ones(5)
        with pytest.raises(ValueError, match="both upper and lower"):
            plot_filled_std_curves(x, mean, "blue", "lightblue", lower=np.ones(5))

    def test_neither_std_nor_bounds_raises(self):
        x = np.arange(5, dtype=float)
        mean = np.ones(5)
        with pytest.raises(ValueError, match="either std or upper/lower"):
            plot_filled_std_curves(x, mean, "blue", "lightblue")

    def test_both_std_and_bounds_raises(self):
        x = np.arange(5, dtype=float)
        mean = np.ones(5)
        with pytest.raises(ValueError, match="either std or upper/lower"):
            plot_filled_std_curves(
                x,
                mean,
                "blue",
                "lightblue",
                std=np.ones(5),
                upper=np.ones(5),
                lower=np.ones(5),
            )


class TestPlotFilledStdCurvesStdMode:
    """Tests for std-based plotting."""

    def test_std_mode_plots_correctly(self):
        x = np.array([0.0, 1.0, 2.0])
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.1, 0.2, 0.3])

        with patch.object(plt, "plot") as mock_plot, patch.object(plt, "fill_between") as mock_fill:
            plot_filled_std_curves(x, mean, "blue", "lightblue", std=std)

        expected_upper = mean + std
        expected_lower = mean - std

        assert mock_plot.call_count == 3
        np.testing.assert_array_equal(mock_plot.call_args_list[0][0][1], expected_upper)
        np.testing.assert_array_equal(mock_plot.call_args_list[1][0][1], expected_lower)
        np.testing.assert_array_equal(mock_plot.call_args_list[2][0][1], mean)

        assert mock_fill.call_count == 1
        np.testing.assert_array_equal(mock_fill.call_args[0][1], expected_lower)
        np.testing.assert_array_equal(mock_fill.call_args[0][2], expected_upper)


class TestPlotFilledStdCurvesBoundsMode:
    """Tests for explicit upper/lower bound plotting."""

    def test_upper_lower_mode(self):
        x = np.array([0.0, 1.0, 2.0])
        mean = np.array([1.0, 2.0, 3.0])
        upper = np.array([1.5, 2.5, 3.5])
        lower = np.array([0.5, 1.5, 2.5])

        with patch.object(plt, "plot") as mock_plot, patch.object(plt, "fill_between") as mock_fill:
            plot_filled_std_curves(x, mean, "green", "lightgreen", upper=upper, lower=lower)

        np.testing.assert_array_equal(mock_plot.call_args_list[0][0][1], upper)
        np.testing.assert_array_equal(mock_plot.call_args_list[1][0][1], lower)
        np.testing.assert_array_equal(mock_plot.call_args_list[2][0][1], mean)
        np.testing.assert_array_equal(mock_fill.call_args[0][1], lower)
        np.testing.assert_array_equal(mock_fill.call_args[0][2], upper)


class TestPlotFilledStdCurvesClipping:
    """Tests for min_val and max_val clipping."""

    def test_min_val_clips_all_curves(self):
        x = np.array([0.0, 1.0])
        mean = np.array([0.5, -0.5])
        std = np.array([1.0, 1.0])

        with patch.object(plt, "plot") as mock_plot, patch.object(plt, "fill_between"):
            plot_filled_std_curves(x, mean, "b", "lb", std=std, min_val=0.0)

        plotted_upper = mock_plot.call_args_list[0][0][1]
        plotted_lower = mock_plot.call_args_list[1][0][1]
        plotted_mean = mock_plot.call_args_list[2][0][1]

        assert np.all(plotted_mean >= 0.0)
        assert np.all(plotted_upper >= 0.0)
        assert np.all(plotted_lower >= 0.0)
        np.testing.assert_array_equal(plotted_lower, np.array([0.0, 0.0]))

    def test_max_val_clips_all_curves(self):
        x = np.array([0.0, 1.0])
        mean = np.array([5.0, 10.0])
        std = np.array([1.0, 1.0])

        with patch.object(plt, "plot") as mock_plot, patch.object(plt, "fill_between"):
            plot_filled_std_curves(x, mean, "b", "lb", std=std, max_val=8.0)

        plotted_upper = mock_plot.call_args_list[0][0][1]
        plotted_lower = mock_plot.call_args_list[1][0][1]
        plotted_mean = mock_plot.call_args_list[2][0][1]

        assert np.all(plotted_mean <= 8.0)
        assert np.all(plotted_upper <= 8.0)
        assert np.all(plotted_lower <= 8.0)
        np.testing.assert_array_equal(plotted_upper, np.array([6.0, 8.0]))


class TestPlotFilledStdCurvesStyling:
    """Tests for styling parameters passed to matplotlib."""

    def test_colors_and_alpha_passed(self):
        x = np.array([0.0, 1.0])
        mean = np.ones(2)
        std = np.ones(2) * 0.1

        with patch.object(plt, "plot") as mock_plot, patch.object(plt, "fill_between") as mock_fill:
            plot_filled_std_curves(x, mean, "red", "salmon", std=std, alpha=0.5)

        assert mock_plot.call_args_list[0][1]["color"] == "salmon"
        assert mock_plot.call_args_list[1][1]["color"] == "salmon"
        assert mock_plot.call_args_list[2][1]["color"] == "red"
        assert mock_fill.call_args[1]["color"] == "salmon"
        assert mock_plot.call_args_list[0][1]["alpha"] == 0.5
        assert mock_fill.call_args[1]["alpha"] == 0.5

    def test_label_linestyle_marker(self):
        x = np.array([0.0, 1.0])
        mean = np.ones(2)
        std = np.ones(2) * 0.1

        with patch.object(plt, "plot") as mock_plot, patch.object(plt, "fill_between"):
            plot_filled_std_curves(
                x,
                mean,
                "b",
                "lb",
                std=std,
                label="test_label",
                linestyle="--",
                marker="o",
            )

        mean_call = mock_plot.call_args_list[2]
        assert mean_call[1]["label"] == "test_label"
        assert mean_call[1]["linestyle"] == "--"
        assert mean_call[1]["marker"] == "o"
        assert mean_call[1]["markersize"] == 4
