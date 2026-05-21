"""Tests for objects/detectors/detector.py - Detector base class draw_plot method."""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from fdtdx.core.switch import OnOffSwitch
from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.detectors.field import FieldDetector
from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector


@pytest.fixture
def small_switch():
    """Switch that records exactly 3 time steps."""
    return OnOffSwitch(fixed_on_time_steps=[0, 1, 2])


@pytest.fixture
def single_switch():
    """Switch that records exactly 1 time step."""
    return OnOffSwitch(fixed_on_time_steps=[0])


class TestDrawPlotLinePlots:
    """Tests for draw_plot → 1D data branches."""

    def test_1d_multiple_timesteps_returns_figure(self, simulation_config, plane_grid_slice, random_key, small_switch):
        """PoyntingFlux (scalar/step) with T>1 → line plot over time."""
        detector = PoyntingFluxDetector(direction="+", switch=small_switch)
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = detector.init_state()
        state_np = {k: np.asarray(v) for k, v in state.items()}

        figs = detector.draw_plot(state_np)

        assert "poynting_flux" in figs
        assert isinstance(figs["poynting_flux"], Figure)
        plt.close(figs["poynting_flux"])

    def test_1d_single_timestep_x_axis_label(self, simulation_config, random_key, single_switch):
        """1D on x-line with T==1 → spatial line plot with X axis label."""
        x_line_slice = ((0, 8), (0, 1), (0, 1))
        detector = FieldDetector(components=("Ex",), switch=single_switch)
        detector = detector.place_on_grid(x_line_slice, simulation_config, random_key)
        state = detector.init_state()
        state_np = {k: np.asarray(v) for k, v in state.items()}

        figs = detector.draw_plot(state_np)

        assert "fields" in figs
        assert isinstance(figs["fields"], Figure)
        plt.close(figs["fields"])

    def test_1d_single_timestep_y_axis_label(self, simulation_config, random_key, single_switch):
        """1D on y-line with T==1 → spatial line plot with Y axis label."""
        y_line_slice = ((0, 1), (0, 8), (0, 1))
        detector = FieldDetector(components=("Ex",), switch=single_switch)
        detector = detector.place_on_grid(y_line_slice, simulation_config, random_key)
        state = detector.init_state()
        state_np = {k: np.asarray(v) for k, v in state.items()}

        figs = detector.draw_plot(state_np)

        assert isinstance(figs["fields"], Figure)
        plt.close(figs["fields"])

    def test_1d_single_timestep_z_axis_label(self, simulation_config, random_key, single_switch):
        """1D on z-line with T==1 → spatial line plot with Z axis label."""
        z_line_slice = ((0, 1), (0, 1), (0, 8))
        detector = FieldDetector(components=("Ex",), switch=single_switch)
        detector = detector.place_on_grid(z_line_slice, simulation_config, random_key)
        state = detector.init_state()
        state_np = {k: np.asarray(v) for k, v in state.items()}

        figs = detector.draw_plot(state_np)

        assert isinstance(figs["fields"], Figure)
        plt.close(figs["fields"])


class TestDrawPlotWaterfallAndSlices:
    """Tests for draw_plot → 2D data branches."""

    def test_2d_multiple_timesteps_waterfall(self, simulation_config, random_key, small_switch):
        """1D spatial + T>1 time steps → waterfall plot."""
        x_line_slice = ((0, 8), (0, 1), (0, 1))
        detector = FieldDetector(components=("Ex",), switch=small_switch)
        detector = detector.place_on_grid(x_line_slice, simulation_config, random_key)
        state = detector.init_state()
        state_np = {k: np.asarray(v) for k, v in state.items()}

        figs = detector.draw_plot(state_np)

        assert "fields" in figs
        assert isinstance(figs["fields"], Figure)
        plt.close(figs["fields"])

    def test_2d_single_timestep_slices_plot(self, simulation_config, small_grid_slice, random_key, single_switch):
        """XY/XZ/YZ slices with T==1 → plot_2d_from_slices."""
        detector = EnergyDetector(as_slices=True, switch=single_switch)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        state = detector.init_state()
        state_np = {k: np.asarray(v) for k, v in state.items()}

        figs = detector.draw_plot(state_np)

        assert "sliced_plot" in figs
        assert isinstance(figs["sliced_plot"], Figure)
        plt.close(figs["sliced_plot"])

    def test_2d_single_timestep_wrong_keys_raises(self, simulation_config, small_grid_slice, random_key, single_switch):
        """2D state with T==1 but wrong keys raises."""
        detector = EnergyDetector(as_slices=True, switch=single_switch)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        # Pass 2D arrays but with wrong key names
        state_np = {"energy": np.random.rand(8, 8)}

        with pytest.raises(Exception):
            detector.draw_plot(state_np)


class TestDrawPlot3D:
    """Tests for draw_plot → 3D data branches."""

    def test_3d_single_timestep_mean_slices(self, simulation_config, small_grid_slice, random_key, single_switch):
        """3D volume with T==1 → 2D from mean."""
        detector = EnergyDetector(switch=single_switch)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        state = detector.init_state()
        state_np = {k: np.asarray(v) for k, v in state.items()}

        figs = detector.draw_plot(state_np)

        assert "energy" in figs
        assert isinstance(figs["energy"], Figure)
        plt.close(figs["energy"])

    def test_3d_multiple_timesteps_video_from_slices(
        self, simulation_config, small_grid_slice, random_key, small_switch
    ):
        """XY/XZ/YZ slices with T>1 → generate video."""
        detector = EnergyDetector(as_slices=True, switch=small_switch)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        state = detector.init_state()
        state_np = {k: np.asarray(v) for k, v in state.items()}

        figs = detector.draw_plot(state_np)

        assert "sliced_video" in figs
        path = figs["sliced_video"]
        assert isinstance(path, str)
        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)

    def test_3d_multiple_timesteps_wrong_keys_raises(
        self, simulation_config, small_grid_slice, random_key, small_switch
    ):
        """3D state with T>1 but wrong keys raises."""
        detector = EnergyDetector(as_slices=True, switch=small_switch)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        # Pass 3D array but with wrong key name (not XY/XZ/YZ)
        state_np = {"energy": np.random.rand(3, 8, 8)}

        with pytest.raises(Exception):
            detector.draw_plot(state_np)


class TestDrawPlot4D:
    """Tests for draw_plot → 4D data branch."""

    def test_4d_multiple_timesteps_video(self, simulation_config, small_grid_slice, random_key, small_switch):
        """3D spatial volume + T>1 → video with mean slices."""
        detector = FieldDetector(components=("Ex",), switch=small_switch)
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        state = detector.init_state()
        state_np = {k: np.asarray(v) for k, v in state.items()}

        figs = detector.draw_plot(state_np)

        assert "fields" in figs
        path = figs["fields"]
        assert isinstance(path, str)
        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)


class TestDrawPlotErrors:
    """Tests for draw_plot error cases."""

    def test_empty_state_raises(self, simulation_config, plane_grid_slice, random_key, small_switch):
        """Empty state dict raises with 'empty state' message."""
        detector = PoyntingFluxDetector(direction="+", switch=small_switch)
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)

        with pytest.raises(Exception, match="empty state"):
            detector.draw_plot({})

    def test_different_ndim_arrays_raises(self, simulation_config, plane_grid_slice, random_key, small_switch):
        """State with inconsistent ndim across keys raises."""
        detector = PoyntingFluxDetector(direction="+", switch=small_switch)
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)
        # "a" squeezes to 1D, "b" squeezes to 3D
        state_np = {
            "a": np.ones((3, 1)),
            "b": np.ones((3, 4, 2)),
        }

        with pytest.raises(Exception, match="different ndim"):
            detector.draw_plot(state_np)

    def test_5d_data_raises(self, simulation_config, small_grid_slice, random_key, small_switch):
        """5D squeezed data (T>1, 6 components, 3D grid) raises."""
        detector = FieldDetector(switch=small_switch)  # 6 default components
        detector = detector.place_on_grid(small_grid_slice, simulation_config, random_key)
        state = detector.init_state()  # shape (3, 6, 8, 8, 8), squeezes to 5D
        state_np = {k: np.asarray(v) for k, v in state.items()}

        with pytest.raises(Exception, match="Cannot plot"):
            detector.draw_plot(state_np)


class TestDrawPlotInverse:
    """Tests for draw_plot with inverse detector."""

    def test_inverse_line_plot_succeeds(self, simulation_config, plane_grid_slice, random_key, small_switch):
        """inverse=True detector with T>1 still returns a figure."""
        detector = PoyntingFluxDetector(direction="+", inverse=True, switch=small_switch)
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = detector.init_state()
        # Assign distinct values so reversal is meaningful
        state["poynting_flux"] = jnp.array([[1.0], [2.0], [3.0]])
        state_np = {k: np.asarray(v) for k, v in state.items()}

        figs = detector.draw_plot(state_np)

        assert "poynting_flux" in figs
        assert isinstance(figs["poynting_flux"], Figure)
        plt.close(figs["poynting_flux"])

    def test_if_inverse_plot_backwards_false_no_reversal(
        self, simulation_config, plane_grid_slice, random_key, small_switch
    ):
        """inverse=True but if_inverse_plot_backwards=False → no reversal."""
        detector = PoyntingFluxDetector(
            direction="+",
            inverse=True,
            if_inverse_plot_backwards=False,
            switch=small_switch,
        )
        detector = detector.place_on_grid(plane_grid_slice, simulation_config, random_key)
        state = detector.init_state()
        state_np = {k: np.asarray(v) for k, v in state.items()}

        figs = detector.draw_plot(state_np)

        assert isinstance(figs["poynting_flux"], Figure)
        plt.close(figs["poynting_flux"])
