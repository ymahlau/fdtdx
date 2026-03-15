"""Tests for objects/detectors/plotting/video.py - video generation utilities."""

from pathlib import Path

import numpy as np
import pytest

from fdtdx.objects.detectors.plotting.video import (
    _make_animation_frame,
    generate_video_from_slices,
    plot_from_slices,
)


class TestPlotFromSlices:
    """Tests for plot_from_slices function."""

    def test_basic_plot(self):
        """Test basic slice plotting."""
        xy = np.random.rand(10, 10)
        xz = np.random.rand(10, 8)
        yz = np.random.rand(10, 8)
        slice_tuple = (xy, xz, yz)
        resolutions = (1e-6, 1e-6, 1e-6)
        minvals = (0.0, 0.0, 0.0)
        maxvals = (1.0, 1.0, 1.0)

        result = plot_from_slices(
            slice_tuple,
            resolutions,
            minvals,
            maxvals,
            plot_dpi=72,
            plot_interpolation="nearest",
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3  # height x width x 3 (RGB)

    def test_returns_rgb_array(self):
        """Test that output is RGB (3 channels)."""
        xy = np.random.rand(10, 10)
        xz = np.random.rand(10, 8)
        yz = np.random.rand(10, 8)
        slice_tuple = (xy, xz, yz)
        resolutions = (1e-6, 1e-6, 1e-6)
        minvals = (0.0, 0.0, 0.0)
        maxvals = (1.0, 1.0, 1.0)

        result = plot_from_slices(
            slice_tuple,
            resolutions,
            minvals,
            maxvals,
            plot_dpi=72,
            plot_interpolation="nearest",
        )

        assert result.shape[2] == 3  # RGB channels

    def test_different_interpolations(self):
        """Test with different interpolation methods."""
        xy = np.random.rand(10, 10)
        xz = np.random.rand(10, 8)
        yz = np.random.rand(10, 8)
        slice_tuple = (xy, xz, yz)
        resolutions = (1e-6, 1e-6, 1e-6)
        minvals = (0.0, 0.0, 0.0)
        maxvals = (1.0, 1.0, 1.0)

        for interp in ["nearest", "bilinear"]:
            result = plot_from_slices(
                slice_tuple,
                resolutions,
                minvals,
                maxvals,
                plot_dpi=72,
                plot_interpolation=interp,
            )
            assert result is not None


class TestMakeAnimationFrame:
    """Tests for _make_animation_frame function."""

    def test_returns_correct_frame(self):
        """Test that correct frame is returned for given time."""
        precomputed = [np.zeros((10, 10, 3)), np.ones((10, 10, 3)), np.ones((10, 10, 3)) * 0.5]
        fps = 10

        # t=0 should return frame 0
        frame = _make_animation_frame(0, precomputed, fps)
        assert np.array_equal(frame, precomputed[0])

        # t=0.1 should return frame 1
        frame = _make_animation_frame(0.1, precomputed, fps)
        assert np.array_equal(frame, precomputed[1])

        # t=0.2 should return frame 2
        frame = _make_animation_frame(0.2, precomputed, fps)
        assert np.array_equal(frame, precomputed[2])

    def test_integer_time(self):
        """Test with integer time values."""
        precomputed = [np.zeros((10, 10, 3)) + i for i in range(5)]
        fps = 1

        frame = _make_animation_frame(2, precomputed, fps)
        assert np.array_equal(frame, precomputed[2])


class TestGenerateVideoFromSlices:
    """Tests for generate_video_from_slices function."""

    @pytest.fixture
    def sample_slices(self):
        """Create sample time-series slice data."""
        time_steps = 5
        xy = np.random.rand(time_steps, 10, 10)
        xz = np.random.rand(time_steps, 10, 8)
        yz = np.random.rand(time_steps, 10, 8)
        return xy, xz, yz

    def test_generates_video_file(self, sample_slices):
        """Test that video file is generated."""
        xy, xz, yz = sample_slices

        path = generate_video_from_slices(
            xy_slice=xy,
            xz_slice=xz,
            yz_slice=yz,
            plt_fn=plot_from_slices,
            resolutions=(1e-6, 1e-6, 1e-6),
            num_worker=None,  # Sequential processing
            plot_interpolation="nearest",
            plot_dpi=72,
            fps=2,
        )

        assert path is not None
        assert Path(path).exists()
        assert path.endswith(".mp4")

        # Cleanup
        Path(path).unlink(missing_ok=True)

    def test_auto_minmax_calculation(self, sample_slices):
        """Test that min/max are auto-calculated when None."""
        xy, xz, yz = sample_slices

        path = generate_video_from_slices(
            xy_slice=xy,
            xz_slice=xz,
            yz_slice=yz,
            plt_fn=plot_from_slices,
            resolutions=(1e-6, 1e-6, 1e-6),
            num_worker=None,
            plot_interpolation="nearest",
            plot_dpi=72,
            fps=2,
            minvals=(None, None, None),
            maxvals=(None, None, None),
        )

        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)

    def test_custom_minmax(self, sample_slices):
        """Test with custom min/max values."""
        xy, xz, yz = sample_slices

        path = generate_video_from_slices(
            xy_slice=xy,
            xz_slice=xz,
            yz_slice=yz,
            plt_fn=plot_from_slices,
            resolutions=(1e-6, 1e-6, 1e-6),
            num_worker=None,
            plot_interpolation="nearest",
            plot_dpi=72,
            fps=2,
            minvals=(-1.0, -1.0, -1.0),
            maxvals=(1.0, 1.0, 1.0),
        )

        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)

    def test_partial_minmax(self, sample_slices):
        """Test with partial min/max values (some None)."""
        xy, xz, yz = sample_slices

        path = generate_video_from_slices(
            xy_slice=xy,
            xz_slice=xz,
            yz_slice=yz,
            plt_fn=plot_from_slices,
            resolutions=(1e-6, 1e-6, 1e-6),
            num_worker=None,
            plot_interpolation="nearest",
            plot_dpi=72,
            fps=2,
            minvals=(None, -0.5, None),
            maxvals=(0.5, None, 1.0),
        )

        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)

    def test_with_progress_bar(self, sample_slices):
        """Test with custom progress bar."""
        from rich.progress import Progress

        xy, xz, yz = sample_slices

        with Progress() as progress:
            path = generate_video_from_slices(
                xy_slice=xy,
                xz_slice=xz,
                yz_slice=yz,
                plt_fn=plot_from_slices,
                resolutions=(1e-6, 1e-6, 1e-6),
                num_worker=None,
                plot_interpolation="nearest",
                plot_dpi=72,
                fps=2,
                progress=progress,
            )

        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)

    def test_different_fps(self, sample_slices):
        """Test with different FPS values."""
        xy, xz, yz = sample_slices

        for fps in [1, 5, 10]:
            path = generate_video_from_slices(
                xy_slice=xy,
                xz_slice=xz,
                yz_slice=yz,
                plt_fn=plot_from_slices,
                resolutions=(1e-6, 1e-6, 1e-6),
                num_worker=None,
                plot_interpolation="nearest",
                plot_dpi=72,
                fps=fps,
            )

            assert Path(path).exists()
            Path(path).unlink(missing_ok=True)
