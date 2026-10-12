"""Tests for objects/detectors/plotting/video.py - video generation utilities."""

import os
import warnings
from pathlib import Path
from unittest import mock

import imageio_ffmpeg
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.objects.detectors.plotting.video import (
    _pad_to_even_size,
    _write_mp4,
    generate_video_from_slices,
    plot_from_slices,
)


def _read_mp4(path: str) -> tuple[dict, list[np.ndarray]]:
    """Decodes an MP4 file into its metadata and a list of RGB frames."""
    # disable the preexec_fn of imageio-ffmpeg, which is unsafe in the multithreaded JAX test process
    with mock.patch.dict(os.environ, {"IMAGEIO_FFMPEG_NO_PREVENT_SIGINT": "1"}):
        reader = imageio_ffmpeg.read_frames(path)
        meta = next(reader)
        width, height = meta["size"]
        frames = [np.frombuffer(f, dtype=np.uint8).reshape(height, width, 3) for f in reader]
    return meta, frames


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


class TestPadToEvenSize:
    """Tests for _pad_to_even_size function."""

    def test_even_frame_unchanged(self):
        frame = np.random.randint(0, 256, (30, 46, 3), dtype=np.uint8)
        assert _pad_to_even_size(frame) is frame

    def test_odd_frame_padded_with_edge_pixels(self):
        frame = np.random.randint(0, 256, (17, 31, 3), dtype=np.uint8)
        padded = _pad_to_even_size(frame)
        assert padded.shape == (18, 32, 3)
        assert np.array_equal(padded[:17, :31], frame)
        assert np.array_equal(padded[17, :31], frame[-1])
        assert np.array_equal(padded[:17, 31], frame[:, -1])


class TestWriteMp4:
    """Tests for _write_mp4 function."""

    def test_roundtrip(self, tmp_path):
        """Every frame is written exactly once, in order, at the padded size."""
        num_frames = 7
        frames = [np.full((20, 30, 3), 30 * i, dtype=np.uint8) for i in range(num_frames)]
        path = str(tmp_path / "video.mp4")
        calls = []

        _write_mp4(path, iter(frames), fps=5, on_frame=lambda: calls.append(1))

        meta, decoded = _read_mp4(path)
        assert len(calls) == num_frames
        assert len(decoded) == num_frames
        assert meta["size"] == (30, 20)
        assert meta["fps"] == pytest.approx(5)
        # lossy compression, but uniform gray levels should survive closely
        for i, frame in enumerate(decoded):
            assert np.abs(frame[:20, :30].astype(int) - 30 * i).max() <= 3

    def test_accepts_non_contiguous_frames(self, tmp_path):
        """RGB views of RGBA buffers (as produced by plot_from_slices) are encoded correctly."""
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        rgba[..., 0] = 200
        path = str(tmp_path / "video.mp4")

        _write_mp4(path, [rgba[..., :3]], fps=1, on_frame=lambda: None)

        _, decoded = _read_mp4(path)
        assert len(decoded) == 1
        assert np.abs(decoded[0].astype(int) - np.array([200, 0, 0])).max() <= 10

    def test_odd_frame_size(self, tmp_path):
        """Odd frame sizes are padded by one pixel and stored in the widely supported yuv420p format."""
        frames = [np.full((21, 33, 3), 100, dtype=np.uint8) for _ in range(3)]
        path = str(tmp_path / "video.mp4")

        _write_mp4(path, frames, fps=2, on_frame=lambda: None)

        meta, decoded = _read_mp4(path)
        assert len(decoded) == 3
        assert meta["size"] == (34, 22)
        assert meta["pix_fmt"].startswith("yuv420p")

    def test_no_frames_raises(self, tmp_path):
        with pytest.raises(ValueError, match="without frames"):
            _write_mp4(str(tmp_path / "video.mp4"), [], fps=2, on_frame=lambda: None)

    def test_inconsistent_frame_shapes_raise(self, tmp_path):
        frames = [np.zeros((16, 16, 3), dtype=np.uint8), np.zeros((16, 18, 3), dtype=np.uint8)]
        with pytest.raises(ValueError, match="same shape"):
            _write_mp4(str(tmp_path / "video.mp4"), frames, fps=2, on_frame=lambda: None)

    def test_ffmpeg_error_is_raised(self, tmp_path):
        """Failures of ffmpeg (here: unwritable output path) surface as an exception with ffmpeg's message."""
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(100)]
        path = str(tmp_path / "missing_dir" / "video.mp4")
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            _write_mp4(path, frames, fps=2, on_frame=lambda: None)

    def test_exception_in_frame_source_propagates(self, tmp_path):
        def frames():
            yield np.zeros((16, 16, 3), dtype=np.uint8)
            raise KeyError("render failed")

        with pytest.raises(KeyError, match="render failed"):
            _write_mp4(str(tmp_path / "video.mp4"), frames(), fps=2, on_frame=lambda: None)

    def test_no_unsafe_fork_with_jax(self, tmp_path):
        """ffmpeg must be started without running Python code in a forked child (e.g. a preexec_fn), which
        can deadlock in multithreaded processes. JAX warns about such forks once its backend is running."""
        jnp.zeros(1).block_until_ready()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _write_mp4(
                str(tmp_path / "video.mp4"), [np.zeros((16, 16, 3), dtype=np.uint8)], fps=1, on_frame=lambda: None
            )
        assert not [w for w in caught if "fork" in str(w.message)]

    def test_high_fps_keeps_every_frame_in_order(self, tmp_path):
        """Regression test: frame sampling by time used to drop/duplicate frames at some frame rates (e.g. 49)."""
        num_frames = 50
        frames = [np.full((16, 16, 3), 5 * i, dtype=np.uint8) for i in range(num_frames)]
        path = str(tmp_path / "video.mp4")

        _write_mp4(path, frames, fps=49, on_frame=lambda: None)

        _, decoded = _read_mp4(path)
        levels = [round(float(f.mean()) / 5) for f in decoded]
        assert levels == list(range(num_frames))


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
        _, frames = _read_mp4(path)
        assert len(frames) == xy.shape[0]

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
            _, frames = _read_mp4(path)
            assert len(frames) == xy.shape[0]
            Path(path).unlink(missing_ok=True)

    def test_multiprocessing(self, sample_slices):
        """Frames rendered by a worker pool are all encoded."""
        xy, xz, yz = sample_slices

        path = generate_video_from_slices(
            xy_slice=xy,
            xz_slice=xz,
            yz_slice=yz,
            plt_fn=plot_from_slices,
            resolutions=(1e-6, 1e-6, 1e-6),
            num_worker=2,
            plot_interpolation="nearest",
            plot_dpi=72,
            fps=2,
        )

        _, frames = _read_mp4(path)
        assert len(frames) == xy.shape[0]
        Path(path).unlink(missing_ok=True)
