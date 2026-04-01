"""Unit tests for fdtdx.fdtd.wrapper"""

from unittest.mock import Mock, patch

import jax
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.wrapper import run_fdtd


class TestRunFdtd:
    """Test run_fdtd dispatch logic."""

    @pytest.fixture
    def setup(self):
        mock_arrays = Mock(spec=ArrayContainer)
        mock_objects = Mock(spec=ObjectContainer)
        mock_config = Mock(spec=SimulationConfig)
        mock_config.gradient_config = None
        key = jax.random.PRNGKey(0)
        return mock_arrays, mock_objects, mock_config, key

    def test_no_gradient_config_uses_checkpointed(self, setup):
        """gradient_config=None delegates to checkpointed_fdtd."""
        arrays, objects, config, key = setup
        config.gradient_config = None
        mock_result = Mock(spec=SimulationState)

        with patch("fdtdx.fdtd.wrapper.checkpointed_fdtd", return_value=mock_result) as mock_ckpt:
            result = run_fdtd(arrays, objects, config, key, stopping_condition=None)

        mock_ckpt.assert_called_once_with(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            stopping_condition=None,
            show_progress=True,
        )
        assert result is mock_result

    def test_reversible_gradient_method_uses_reversible_fdtd(self, setup):
        """gradient_config.method='reversible' delegates to reversible_fdtd."""
        arrays, objects, config, key = setup
        config.gradient_config = Mock()
        config.gradient_config.method = "reversible"
        mock_result = Mock(spec=SimulationState)

        with patch("fdtdx.fdtd.wrapper.reversible_fdtd", return_value=mock_result) as mock_rev:
            result = run_fdtd(arrays, objects, config, key)

        mock_rev.assert_called_once_with(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            show_progress=True,
        )
        assert result is mock_result

    def test_checkpointed_gradient_method_uses_checkpointed(self, setup):
        """gradient_config.method='checkpointed' delegates to checkpointed_fdtd."""
        arrays, objects, config, key = setup
        config.gradient_config = Mock()
        config.gradient_config.method = "checkpointed"
        mock_result = Mock(spec=SimulationState)

        with patch("fdtdx.fdtd.wrapper.checkpointed_fdtd", return_value=mock_result) as mock_ckpt:
            result = run_fdtd(arrays, objects, config, key, stopping_condition=None)

        mock_ckpt.assert_called_once_with(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            stopping_condition=None,
            show_progress=True,
        )
        assert result is mock_result

    def test_unknown_gradient_method_raises(self, setup):
        """An unrecognised gradient_config.method raises an Exception."""
        arrays, objects, config, key = setup
        config.gradient_config = Mock()
        config.gradient_config.method = "unknown_method"

        with pytest.raises(Exception, match="Unknown gradient computation method: unknown_method"):
            run_fdtd(arrays, objects, config, key)

    def test_none_method_raises(self, setup):
        """gradient_config.method=None (not a known string) also raises."""
        arrays, objects, config, key = setup
        config.gradient_config = Mock()
        config.gradient_config.method = None

        with pytest.raises(Exception, match="Unknown gradient computation method: None"):
            run_fdtd(arrays, objects, config, key)

    def test_stopping_condition_with_gradient_config_raises(self, setup):
        """Combining a custom stopping_condition with gradient_config raises NotImplementedError."""
        arrays, objects, config, key = setup
        config.gradient_config = Mock()  # not None
        stopping = Mock()  # not None

        with pytest.raises(NotImplementedError):
            run_fdtd(arrays, objects, config, key, stopping_condition=stopping)

    def test_stopping_condition_none_with_gradient_config_ok(self, setup):
        """stopping_condition=None with gradient_config set does not raise."""
        arrays, objects, config, key = setup
        config.gradient_config = Mock()
        config.gradient_config.method = "reversible"
        mock_result = Mock(spec=SimulationState)

        with patch("fdtdx.fdtd.wrapper.reversible_fdtd", return_value=mock_result):
            result = run_fdtd(arrays, objects, config, key, stopping_condition=None)
        assert result is mock_result

    def test_stopping_condition_forwarded_to_checkpointed(self, setup):
        """stopping_condition is passed through to checkpointed_fdtd when no gradient_config."""
        arrays, objects, config, key = setup
        config.gradient_config = None
        stopping = Mock()
        mock_result = Mock(spec=SimulationState)

        with patch("fdtdx.fdtd.wrapper.checkpointed_fdtd", return_value=mock_result) as mock_ckpt:
            run_fdtd(arrays, objects, config, key, stopping_condition=stopping)

        assert mock_ckpt.call_args[1]["stopping_condition"] is stopping

    # ------------------------------------------------------------------ #
    # show_progress propagation                                            #
    # ------------------------------------------------------------------ #

    def test_show_progress_false_forwarded_to_checkpointed_no_gradient(self, setup):
        """show_progress=False is passed to checkpointed_fdtd (no gradient_config)."""
        arrays, objects, config, key = setup
        config.gradient_config = None
        mock_result = Mock(spec=SimulationState)

        with patch("fdtdx.fdtd.wrapper.checkpointed_fdtd", return_value=mock_result) as mock_ckpt:
            run_fdtd(arrays, objects, config, key, show_progress=False)

        assert mock_ckpt.call_args[1]["show_progress"] is False

    def test_show_progress_false_forwarded_to_reversible(self, setup):
        """show_progress=False is passed to reversible_fdtd."""
        arrays, objects, config, key = setup
        config.gradient_config = Mock()
        config.gradient_config.method = "reversible"
        mock_result = Mock(spec=SimulationState)

        with patch("fdtdx.fdtd.wrapper.reversible_fdtd", return_value=mock_result) as mock_rev:
            run_fdtd(arrays, objects, config, key, show_progress=False)

        assert mock_rev.call_args[1]["show_progress"] is False

    def test_show_progress_false_forwarded_to_checkpointed_gradient(self, setup):
        """show_progress=False is passed to checkpointed_fdtd (checkpointed gradient)."""
        arrays, objects, config, key = setup
        config.gradient_config = Mock()
        config.gradient_config.method = "checkpointed"
        mock_result = Mock(spec=SimulationState)

        with patch("fdtdx.fdtd.wrapper.checkpointed_fdtd", return_value=mock_result) as mock_ckpt:
            run_fdtd(arrays, objects, config, key, show_progress=False)

        assert mock_ckpt.call_args[1]["show_progress"] is False

    def test_show_progress_defaults_to_true(self, setup):
        """show_progress defaults to True when not supplied."""
        arrays, objects, config, key = setup
        config.gradient_config = None
        mock_result = Mock(spec=SimulationState)

        with patch("fdtdx.fdtd.wrapper.checkpointed_fdtd", return_value=mock_result) as mock_ckpt:
            run_fdtd(arrays, objects, config, key)

        assert mock_ckpt.call_args[1]["show_progress"] is True
