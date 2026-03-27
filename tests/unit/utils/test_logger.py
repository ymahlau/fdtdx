"""Tests for utils/logger.py - experiment logging utilities."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from fdtdx.utils.logger import (
    Logger,
    _log_formatter,
    init_working_directory,
    snapshot_python_files,
)


class TestInitWorkingDirectory:
    """Tests for init_working_directory function."""

    def test_creates_directory(self):
        """Test that directory is created."""
        with tempfile.TemporaryDirectory() as base:
            with patch("fdtdx.utils.logger.Path.cwd", return_value=Path(base)):
                wd = init_working_directory("test_experiment", wd_name="test_run")
                assert wd.exists()
                assert "test_experiment" in str(wd)
                assert "test_run" in str(wd)

    def test_timestamp_name_when_none(self):
        """Test that timestamp is used when wd_name is None."""
        with tempfile.TemporaryDirectory() as base:
            with patch("fdtdx.utils.logger.Path.cwd", return_value=Path(base)):
                wd = init_working_directory("test_experiment", wd_name=None)
                assert wd.exists()
                # Should have timestamp-like name (contains digits and dashes)
                assert any(c.isdigit() for c in wd.name)


class TestLogFormatter:
    """Tests for _log_formatter function."""

    def test_formats_info_level(self):
        """Test formatting INFO level message."""
        record = {
            "level": MagicMock(name="INFO"),
            "file": MagicMock(path="test.py"),
            "line": 42,
            "message": "Test message",
        }
        record["level"].name = "INFO"

        result = _log_formatter(record)

        assert "test.py:42" in result
        assert "Test message" in result

    def test_formats_error_level(self):
        """Test formatting ERROR level message."""
        record = {
            "level": MagicMock(name="ERROR"),
            "file": MagicMock(path="error.py"),
            "line": 100,
            "message": "Error occurred",
        }
        record["level"].name = "ERROR"

        result = _log_formatter(record)

        assert "error.py:100" in result
        assert "Error occurred" in result

    def test_formats_warning_level(self):
        """Test formatting WARNING level message."""
        record = {
            "level": MagicMock(name="WARNING"),
            "file": MagicMock(path="warn.py"),
            "line": 50,
            "message": "Warning message",
        }
        record["level"].name = "WARNING"

        result = _log_formatter(record)

        assert "warn.py:50" in result
        assert "Warning message" in result

    def test_escapes_special_characters(self):
        """Test that special characters in message are escaped."""
        record = {
            "level": MagicMock(name="INFO"),
            "file": MagicMock(path="test.py"),
            "line": 1,
            "message": "Message with [brackets] and <angles>",
        }
        record["level"].name = "INFO"

        result = _log_formatter(record)

        # rich markup should be escaped
        assert "brackets" in result


class TestSnapshotPythonFiles:
    """Tests for snapshot_python_files function."""

    def test_creates_zip_file(self):
        """Test that zip file is created."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_dir = Path(tmp_dir) / "snapshot"

            # Mock sys.argv to avoid issues
            with patch("sys.argv", ["test_script.py"]):
                with patch("shutil.copy"):  # Don't actually copy files
                    snapshot_python_files(snapshot_dir, save_source=False, save_script=False)

            # Check that zip was created
            assert (Path(tmp_dir) / "code.zip").exists()


class TestLogger:
    """Tests for Logger class."""

    @pytest.fixture
    def temp_logger(self):
        """Create a temporary logger for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("fdtdx.utils.logger.Path.cwd", return_value=Path(tmp_dir)):
                with patch("sys.argv", ["test.py"]):
                    logger = Logger("test_experiment", name="test", save_source=False, save_script=False)
                    yield logger
                    # Cleanup
                    logger.csvfile.close()
                    logger.progress.stop()

    def test_logger_creation(self, temp_logger):
        """Test that logger is created successfully."""
        assert temp_logger.cwd.exists()

    def test_stl_dir_property(self, temp_logger):
        """Test stl_dir property creates directory."""
        stl_dir = temp_logger.stl_dir
        assert stl_dir.exists()
        assert "stl" in str(stl_dir)

    def test_params_dir_property(self, temp_logger):
        """Test params_dir property creates directory."""
        params_dir = temp_logger.params_dir
        assert params_dir.exists()
        assert "params" in str(params_dir)

    def test_savefig(self, temp_logger):
        """Test saving a matplotlib figure."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        temp_logger.savefig(temp_logger.cwd, "test_figure.png", fig)

        assert (temp_logger.cwd / "figures" / "test_figure.png").exists()

    def test_write_stats(self, temp_logger):
        """Test writing statistics."""
        stats = {"loss": 0.5, "accuracy": 0.95, "iteration": 1}

        temp_logger.write(stats, do_print=False)

        # Check that CSV file was written
        temp_logger.csvfile.flush()
        csv_path = temp_logger.cwd / "metrics.csv"
        assert csv_path.exists()

        # Read and verify contents
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert float(rows[0]["loss"]) == 0.5

    def test_write_multiple_stats(self, temp_logger):
        """Test writing multiple rows of statistics."""
        for i in range(3):
            stats = {"iteration": i, "loss": 1.0 / (i + 1)}
            temp_logger.write(stats, do_print=False)

        temp_logger.csvfile.flush()
        csv_path = temp_logger.cwd / "metrics.csv"

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3

    def test_write_jax_array_stats(self, temp_logger):
        """Test writing JAX array statistics."""
        stats = {"value": jnp.array(0.5), "count": 10}

        temp_logger.write(stats, do_print=False)

        temp_logger.csvfile.flush()
        csv_path = temp_logger.cwd / "metrics.csv"

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert float(rows[0]["value"]) == 0.5

    def test_write_filters_non_scalar(self, temp_logger):
        """Test that non-scalar values are filtered from stats."""
        stats = {
            "scalar": 1.0,
            "array": jnp.array([1, 2, 3]),  # Should be filtered
            "string": "text",  # Should be filtered
        }

        temp_logger.write(stats, do_print=False)

        temp_logger.csvfile.flush()
        csv_path = temp_logger.cwd / "metrics.csv"

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert "scalar" in rows[0]
            assert "array" not in rows[0]
            assert "string" not in rows[0]

    def test_savefig_custom_dpi(self, temp_logger):
        """Test saving figure with custom DPI."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])

        temp_logger.savefig(temp_logger.cwd, "high_dpi.png", fig, dpi=600)

        assert (temp_logger.cwd / "figures" / "high_dpi.png").exists()

    def test_multiple_figures_saved(self, temp_logger):
        """Test that multiple figures can be saved and each lands on disk."""
        for i in range(3):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [i, i + 1, i + 2])
            temp_logger.savefig(temp_logger.cwd, f"figure_{i}.png", fig)

        figures_dir = temp_logger.cwd / "figures"
        assert len(list(figures_dir.glob("*.png"))) == 3
