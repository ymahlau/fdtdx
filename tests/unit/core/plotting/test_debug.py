import re

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.plotting.debug import (
    debug_plot_2d,
    debug_plot_lines,
    generate_unique_filename,
)

# ── generate_unique_filename ──────────────────────────────────────────


class TestGenerateUniqueFilename:
    def test_default_prefix(self):
        result = generate_unique_filename()
        assert result.startswith("file_")

    def test_format_pattern(self):
        result = generate_unique_filename(prefix="p", extension="txt")
        # p_YYYYMMDD_HHMMSS_xxxxxxxx.txt
        pattern = r"^p_\d{8}_\d{6}_[0-9a-f]{8}\.txt$"
        assert re.match(pattern, result)

    def test_uniqueness(self):
        results = {generate_unique_filename() for _ in range(10)}
        assert len(results) == 10


# ── debug_plot_2d ─────────────────────────────────────────────────────


class TestDebugPlot2d:
    def test_saves_file_with_given_filename(self, tmp_path):
        arr = np.ones((3, 4))
        debug_plot_2d(arr, tmp_dir=tmp_path, filename="test.png")
        assert (tmp_path / "test.png").exists()

    def test_saves_file_with_generated_filename(self, tmp_path):
        arr = np.zeros((2, 2))
        debug_plot_2d(arr, tmp_dir=tmp_path)
        files = list(tmp_path.glob("debug_*.png"))
        assert len(files) == 1

    def test_accepts_jax_array(self, tmp_path):
        arr = jnp.ones((3, 3))
        debug_plot_2d(arr, tmp_dir=tmp_path, filename="jax.png")
        assert (tmp_path / "jax.png").exists()

    def test_accepts_string_tmp_dir(self, tmp_path):
        arr = np.ones((2, 2))
        debug_plot_2d(arr, tmp_dir=str(tmp_path), filename="str_dir.png")
        assert (tmp_path / "str_dir.png").exists()

    def test_center_zero(self, tmp_path):
        arr = np.array([[1.0, -2.0], [3.0, -4.0]])
        debug_plot_2d(arr, tmp_dir=tmp_path, filename="center.png", center_zero=True)
        assert (tmp_path / "center.png").exists()

    def test_show_values(self, tmp_path):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        debug_plot_2d(arr, tmp_dir=tmp_path, filename="values.png", show_values=True)
        assert (tmp_path / "values.png").exists()


# ── debug_plot_lines ──────────────────────────────────────────────────


class TestDebugPlotLines:
    def test_single_line(self, tmp_path):
        data = {"line1": np.array([1.0, 2.0, 3.0])}
        debug_plot_lines(data, tmp_dir=tmp_path, filename="single.png")
        assert (tmp_path / "single.png").exists()

    def test_multiple_lines(self, tmp_path):
        data = {
            "a": np.array([1.0, 2.0]),
            "b": np.array([3.0, 4.0]),
        }
        debug_plot_lines(data, tmp_dir=tmp_path, filename="multi.png")
        assert (tmp_path / "multi.png").exists()

    def test_with_x_values(self, tmp_path):
        data = {"y": np.array([10.0, 20.0, 30.0])}
        x = np.array([0.0, 0.5, 1.0])
        debug_plot_lines(data, x_values=x, tmp_dir=tmp_path, filename="xval.png")
        assert (tmp_path / "xval.png").exists()

    def test_with_jax_arrays(self, tmp_path):
        data = {"jax_line": jnp.array([1.0, 2.0, 3.0])}
        debug_plot_lines(data, tmp_dir=tmp_path, filename="jax.png")
        assert (tmp_path / "jax.png").exists()

    def test_custom_colors_styles_markers(self, tmp_path):
        data = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
        debug_plot_lines(
            data,
            colors={"a": "red", "b": "blue"},
            line_styles={"a": "--", "b": ":"},
            markers={"a": "o", "b": "x"},
            tmp_dir=tmp_path,
            filename="styled.png",
        )
        assert (tmp_path / "styled.png").exists()

    def test_no_grid(self, tmp_path):
        data = {"y": np.array([1.0, 2.0])}
        debug_plot_lines(data, grid=False, tmp_dir=tmp_path, filename="nogrid.png")
        assert (tmp_path / "nogrid.png").exists()

    def test_generated_filename(self, tmp_path):
        data = {"y": np.array([1.0, 2.0])}
        debug_plot_lines(data, tmp_dir=tmp_path)
        files = list(tmp_path.glob("debug_lines_*.png"))
        assert len(files) == 1

    def test_filename_without_extension(self, tmp_path):
        data = {"y": np.array([1.0, 2.0])}
        debug_plot_lines(data, tmp_dir=tmp_path, filename="noext")
        assert (tmp_path / "noext.png").exists()

    def test_string_tmp_dir(self, tmp_path):
        out = tmp_path / "subdir"
        data = {"y": np.array([1.0, 2.0])}
        debug_plot_lines(data, tmp_dir=str(out), filename="str.png")
        assert (out / "str.png").exists()

    def test_rejects_non_1d_array(self, tmp_path):
        data = {"bad": np.ones((2, 3))}
        with pytest.raises(ValueError, match="1-dimensional"):
            debug_plot_lines(data, tmp_dir=tmp_path, filename="fail.png")
