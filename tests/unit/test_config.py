"""Unit tests for fdtdx.config module."""

import math
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from fdtdx import constants
from fdtdx.config import DUMMY_SIMULATION_CONFIG, GradientConfig, SimulationConfig


class TestGradientConfigConstruction:
    """Tests for GradientConfig class construction."""

    def test_construct_reversible_with_recorder(self):
        """Test creating GradientConfig with reversible method and recorder."""
        mock_recorder = MagicMock()
        config = GradientConfig(method="reversible", recorder=mock_recorder)
        assert config.method == "reversible"
        assert config.recorder is mock_recorder
        assert config.num_checkpoints is None

    def test_construct_checkpointed_with_num_checkpoints(self):
        """Test creating GradientConfig with checkpointed method."""
        config = GradientConfig(method="checkpointed", num_checkpoints=10)
        assert config.method == "checkpointed"
        assert config.recorder is None
        assert config.num_checkpoints == 10

    def test_construct_reversible_without_recorder_raises(self):
        """Test that reversible method without recorder raises exception."""
        with pytest.raises(Exception, match="Need Recorder in gradient config"):
            GradientConfig(method="reversible")

    def test_construct_checkpointed_without_num_checkpoints_raises(self):
        """Test that checkpointed method without num_checkpoints raises exception."""
        with pytest.raises(Exception, match="Need Checkpoint Number in gradient config"):
            GradientConfig(method="checkpointed")


def _create_mock_backend(platform="cpu"):
    """Helper to create mock backend for SimulationConfig tests."""
    mock_backend = MagicMock()
    mock_backend.platform = platform
    return mock_backend


class TestSimulationConfigConstruction:
    """Tests for SimulationConfig class construction."""

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_construct_with_all_params(self, mock_get_backend, mock_devices, _):
        """Test creating SimulationConfig with all parameters."""
        mock_get_backend.return_value = _create_mock_backend("cpu")

        config = SimulationConfig(
            time=1e-12,
            resolution=1e-9,
            backend="cpu",
            dtype=jnp.float64,
            courant_factor=0.95,
        )
        assert config.time == 1e-12
        assert config.resolution == 1e-9
        assert config.backend == "cpu"
        assert config.dtype == jnp.float64
        assert config.courant_factor == 0.95
        assert config.gradient_config is None


class TestSimulationConfigBackendHandling:
    """Tests for SimulationConfig backend initialization."""

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_gpu_backend_when_available(self, mock_get_backend, mock_devices, mock_config_update):
        """Test GPU backend is used when available."""
        mock_get_backend.return_value = _create_mock_backend("gpu")
        mock_devices.return_value = [MagicMock()]

        config = SimulationConfig(time=1e-12, resolution=1e-9, backend="gpu")
        assert config.backend == "gpu"
        mock_config_update.assert_called_with("jax_platform_name", "gpu")

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_gpu_fallback_to_cpu_when_unavailable(self, mock_get_backend, mock_devices, _):
        """Test GPU falls back to CPU when unavailable."""
        mock_get_backend.return_value = _create_mock_backend("cpu")
        mock_devices.side_effect = RuntimeError("No GPU found")

        config = SimulationConfig(time=1e-12, resolution=1e-9, backend="gpu")
        assert config.backend == "cpu"

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_tpu_backend_when_available(self, mock_get_backend, mock_devices, mock_config_update):
        """Test TPU backend is used when available."""
        mock_get_backend.return_value = _create_mock_backend("tpu")
        mock_devices.return_value = [MagicMock()]

        config = SimulationConfig(time=1e-12, resolution=1e-9, backend="tpu")
        assert config.backend == "tpu"
        mock_config_update.assert_called_with("jax_platform_name", "tpu")

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_tpu_fallback_to_cpu_when_unavailable(self, mock_get_backend, mock_devices, _):
        """Test TPU falls back to CPU when unavailable."""
        mock_get_backend.return_value = _create_mock_backend("cpu")
        mock_devices.side_effect = RuntimeError("No TPU found")

        config = SimulationConfig(time=1e-12, resolution=1e-9, backend="tpu")
        assert config.backend == "cpu"

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_cpu_backend_explicit(self, mock_get_backend, _, mock_config_update):
        """Test explicit CPU backend."""
        mock_get_backend.return_value = _create_mock_backend("cpu")

        config = SimulationConfig(time=1e-12, resolution=1e-9, backend="cpu")
        assert config.backend == "cpu"
        mock_config_update.assert_called_with("jax_platform_name", "cpu")

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_metal_backend_auto_detection(self, mock_get_backend, mock_devices, mock_config_update):
        """Test METAL backend is auto-detected when platform is METAL."""
        mock_get_backend.return_value = _create_mock_backend("METAL")
        mock_devices.return_value = [MagicMock()]

        config = SimulationConfig(time=1e-12, resolution=1e-9, backend="gpu")
        assert config.backend == "METAL"
        mock_config_update.assert_called_with("jax_platform_name", "metal")

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_metal_fallback_to_cpu_when_unavailable(self, mock_get_backend, mock_devices, _):
        """Test METAL falls back to CPU when initialization fails."""
        mock_get_backend.return_value = _create_mock_backend("cpu")
        mock_devices.side_effect = RuntimeError("METAL init failed")

        config = SimulationConfig(time=1e-12, resolution=1e-9, backend="METAL")
        assert config.backend == "cpu"


class TestSimulationConfigProperties:
    """Tests for SimulationConfig computed properties."""

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_courant_number_calculation(self, mock_get_backend, *_):
        """Test courant_number calculation."""
        mock_get_backend.return_value = _create_mock_backend("cpu")

        config = SimulationConfig(time=1e-12, resolution=1e-9, backend="cpu", courant_factor=0.5)
        expected = 0.5 / math.sqrt(3)
        assert abs(config.courant_number - expected) < 1e-10

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_time_step_duration_calculation(self, mock_get_backend, *_):
        """Test time_step_duration calculation."""
        mock_get_backend.return_value = _create_mock_backend("cpu")

        resolution = 1e-9
        config = SimulationConfig(time=1e-12, resolution=resolution, backend="cpu")
        expected = config.courant_number * resolution / constants.c
        assert abs(config.time_step_duration - expected) < 1e-30

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_time_steps_total_calculation(self, mock_get_backend, *_):
        """Test time_steps_total calculation."""
        mock_get_backend.return_value = _create_mock_backend("cpu")

        config = SimulationConfig(time=1e-12, resolution=1e-9, backend="cpu")
        expected = round(config.time / config.time_step_duration)
        assert config.time_steps_total == expected

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_max_travel_distance_calculation(self, mock_get_backend, *_):
        """Test max_travel_distance calculation."""
        mock_get_backend.return_value = _create_mock_backend("cpu")

        time = 1e-12
        config = SimulationConfig(time=time, resolution=1e-9, backend="cpu")
        expected = constants.c * time
        assert abs(config.max_travel_distance - expected) < 1e-20


class TestSimulationConfigGradientProperties:
    """Tests for SimulationConfig gradient-related properties."""

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_only_forward_without_gradient_config(self, mock_get_backend, *_):
        """Test only_forward is True when gradient_config is None."""
        mock_get_backend.return_value = _create_mock_backend("cpu")

        config = SimulationConfig(time=1e-12, resolution=1e-9, backend="cpu")
        assert config.only_forward is True
        assert config.invertible_optimization is False

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_invertible_optimization_with_recorder(self, mock_get_backend, *_):
        """Test invertible_optimization is True when recorder is provided."""
        mock_get_backend.return_value = _create_mock_backend("cpu")

        mock_recorder = MagicMock()
        gradient_config = GradientConfig(method="reversible", recorder=mock_recorder)
        config = SimulationConfig(time=1e-12, resolution=1e-9, backend="cpu", gradient_config=gradient_config)
        assert config.only_forward is False
        assert config.invertible_optimization is True

    @patch("fdtdx.config.jax.config.update")
    @patch("fdtdx.config.jax.devices")
    @patch("jax.extend.backend.get_backend")
    def test_invertible_optimization_false_for_checkpointed(self, mock_get_backend, *_):
        """Test invertible_optimization is False for checkpointed method."""
        mock_get_backend.return_value = _create_mock_backend("cpu")

        gradient_config = GradientConfig(method="checkpointed", num_checkpoints=10)
        config = SimulationConfig(time=1e-12, resolution=1e-9, backend="cpu", gradient_config=gradient_config)
        assert config.only_forward is False
        assert config.invertible_optimization is False


class TestDummySimulationConfig:
    """Tests for DUMMY_SIMULATION_CONFIG constant."""

    def test_dummy_config_values(self):
        """Test DUMMY_SIMULATION_CONFIG has expected sentinel values."""
        assert isinstance(DUMMY_SIMULATION_CONFIG, SimulationConfig)
        assert DUMMY_SIMULATION_CONFIG.time == -1
        assert DUMMY_SIMULATION_CONFIG.resolution == -1
