from unittest.mock import Mock, patch

import jax
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.wrapper import run_fdtd


class TestRunFdtd:
    """Test run_fdtd function"""

    @pytest.fixture
    def setup(self):
        """Setup common test data"""
        # Create mock arrays
        mock_arrays = Mock(spec=ArrayContainer)

        # Create mock objects
        mock_objects = Mock(spec=ObjectContainer)

        # Create mock config
        mock_config = Mock(spec=SimulationConfig)
        mock_config.gradient_config = None

        # Create random key
        key = jax.random.PRNGKey(0)

        return mock_arrays, mock_objects, mock_config, key

    def test_no_gradient_config(self, setup):
        """Test run_fdtd with no gradient config (should use checkpointed_fdtd)"""
        arrays, objects, config, key = setup
        config.gradient_config = None

        # Mock the checkpointed_fdtd function
        with patch("fdtdx.fdtd.wrapper.checkpointed_fdtd") as mock_checkpointed:
            mock_result = Mock(spec=SimulationState)
            mock_checkpointed.return_value = mock_result

            result = run_fdtd(arrays, objects, config, key)

            # Verify checkpointed_fdtd was called
            mock_checkpointed.assert_called_once_with(arrays=arrays, objects=objects, config=config, key=key)

            # Verify the result is returned
            assert result == mock_result

    def test_reversible_gradient_method(self, setup):
        """Test run_fdtd with reversible gradient method"""
        arrays, objects, config, key = setup

        # Mock gradient config
        mock_gradient_config = Mock()
        mock_gradient_config.method = "reversible"
        config.gradient_config = mock_gradient_config

        # Mock the reversible_fdtd function
        with patch("fdtdx.fdtd.wrapper.reversible_fdtd") as mock_reversible:
            mock_result = Mock(spec=SimulationState)
            mock_reversible.return_value = mock_result

            result = run_fdtd(arrays, objects, config, key)

            # Verify reversible_fdtd was called
            mock_reversible.assert_called_once_with(arrays=arrays, objects=objects, config=config, key=key)

            # Verify the result is returned
            assert result == mock_result

    def test_checkpointed_gradient_method(self, setup):
        """Test run_fdtd with checkpointed gradient method"""
        arrays, objects, config, key = setup

        # Mock gradient config
        mock_gradient_config = Mock()
        mock_gradient_config.method = "checkpointed"
        config.gradient_config = mock_gradient_config

        # Mock the checkpointed_fdtd function
        with patch("fdtdx.fdtd.wrapper.checkpointed_fdtd") as mock_checkpointed:
            mock_result = Mock(spec=SimulationState)
            mock_checkpointed.return_value = mock_result

            result = run_fdtd(arrays, objects, config, key)

            # Verify checkpointed_fdtd was called
            mock_checkpointed.assert_called_once_with(arrays=arrays, objects=objects, config=config, key=key)

            # Verify the result is returned
            assert result == mock_result

    def test_unknown_gradient_method(self, setup):
        """Test run_fdtd with unknown gradient method (should raise exception)"""
        arrays, objects, config, key = setup

        # Mock gradient config with unknown method
        mock_gradient_config = Mock()
        mock_gradient_config.method = "unknown_method"
        config.gradient_config = mock_gradient_config

        # Should raise an exception
        with pytest.raises(Exception, match="Unknown gradient computation method: unknown_method"):
            run_fdtd(arrays, objects, config, key)

    def test_none_gradient_config_with_method_access(self, setup):
        """Test edge case where gradient_config is None but code tries to access method"""
        arrays, objects, config, key = setup

        # Set gradient_config to None
        config.gradient_config = None

        # This should work fine and use checkpointed_fdtd
        with patch("fdtdx.fdtd.wrapper.checkpointed_fdtd") as mock_checkpointed:
            mock_result = Mock(spec=SimulationState)
            mock_checkpointed.return_value = mock_result

            run_fdtd(arrays, objects, config, key)

            # Verify checkpointed_fdtd was called
            mock_checkpointed.assert_called_once_with(arrays=arrays, objects=objects, config=config, key=key)

    def test_gradient_config_with_none_method(self, setup):
        """Test edge case where gradient_config exists but method is None"""
        arrays, objects, config, key = setup

        # Mock gradient config with None method
        mock_gradient_config = Mock()
        mock_gradient_config.method = None
        config.gradient_config = mock_gradient_config

        # Should raise an exception when trying to access the method
        with pytest.raises(Exception, match="Unknown gradient computation method: None"):
            run_fdtd(arrays, objects, config, key)
