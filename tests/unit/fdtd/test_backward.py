from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.backward import backward, cond_fn, full_backward
from fdtdx.fdtd.container import ObjectContainer


class TestCondFn:
    def test_returns_true_when_above_start(self):
        assert bool(cond_fn((10, None), start_time_step=5)) is True

    def test_returns_false_at_start(self):
        assert bool(cond_fn((5, None), start_time_step=5)) is False

    def test_returns_false_below_start(self):
        assert bool(cond_fn((3, None), start_time_step=5)) is False


class TestFullBackward:
    @patch("fdtdx.fdtd.backward.eqxi.while_loop")
    def test_calls_while_loop_with_correct_args(self, mock_while_loop):
        state = (5, Mock())
        mock_while_loop.return_value = (0, state[1])

        result = full_backward(
            state=state,
            objects=Mock(spec=ObjectContainer),
            config=Mock(spec=SimulationConfig),
            key=jax.random.PRNGKey(42),
            record_detectors=False,
            reset_fields=False,
            start_time_step=0,
        )

        mock_while_loop.assert_called_once()
        kwargs = mock_while_loop.call_args.kwargs
        assert kwargs["kind"] == "lax"
        assert kwargs["init_val"] == state
        assert result == (0, state[1])

    @patch("fdtdx.fdtd.backward.eqxi.while_loop")
    def test_cond_fun_uses_start_time_step(self, mock_while_loop):
        mock_while_loop.return_value = (3, Mock())

        full_backward(
            state=(10, Mock()),
            objects=Mock(spec=ObjectContainer),
            config=Mock(spec=SimulationConfig),
            key=jax.random.PRNGKey(0),
            record_detectors=False,
            reset_fields=False,
            start_time_step=3,
        )

        cond_fun = mock_while_loop.call_args.kwargs["cond_fun"]
        assert cond_fun((4, None)) is True
        assert cond_fun((3, None)) is False


class TestBackward:
    @pytest.fixture
    def mock_arrays(self):
        arrays = Mock()
        arrays.E = jnp.ones((3, 10, 10, 10))
        arrays.H = jnp.full((3, 10, 10, 10), 2.0)

        aset_log = {}

        def aset(field_name, value):
            aset_log[field_name] = value
            new = Mock()
            new.E = value if field_name == "E" else arrays.E
            new.H = value if field_name == "H" else arrays.H
            new.aset = aset
            return new

        arrays.aset = aset
        arrays._aset_log = aset_log
        return arrays

    @pytest.fixture
    def mock_objects(self):
        objects = Mock(spec=ObjectContainer)
        objects.pml_objects = []
        objects.boundary_objects = []
        return objects

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def patched_updates(self, mock_arrays):
        with (
            patch("fdtdx.fdtd.backward.add_interfaces", return_value=mock_arrays) as add_int,
            patch("fdtdx.fdtd.backward.update_H_reverse", return_value=mock_arrays) as upd_H,
            patch("fdtdx.fdtd.backward.update_E_reverse", return_value=mock_arrays) as upd_E,
            patch("fdtdx.fdtd.backward.update_detector_states", return_value=mock_arrays) as upd_det,
        ):
            yield {
                "add_interfaces": add_int,
                "update_H": upd_H,
                "update_E": upd_E,
                "update_detectors": upd_det,
            }

    def test_basic_backward_step(self, mock_arrays, mock_objects, key, patched_updates):
        result = backward(
            state=(5, mock_arrays),
            config=Mock(spec=SimulationConfig),
            objects=mock_objects,
            key=key,
            record_detectors=False,
            reset_fields=False,
        )
        assert result[0] == 4
        patched_updates["add_interfaces"].assert_called_once()
        patched_updates["update_H"].assert_called_once()
        patched_updates["update_E"].assert_called_once()
        patched_updates["update_detectors"].assert_not_called()

    def test_records_detectors_when_enabled(self, mock_arrays, mock_objects, key, patched_updates):
        backward(
            state=(5, mock_arrays),
            config=Mock(spec=SimulationConfig),
            objects=mock_objects,
            key=key,
            record_detectors=True,
            reset_fields=False,
        )
        patched_updates["update_detectors"].assert_called_once()
        call_kwargs = patched_updates["update_detectors"].call_args.kwargs
        assert call_kwargs["inverse"] is True
        assert jnp.array_equal(call_kwargs["H_prev"], mock_arrays.H)

    def test_reset_fields_calls_apply_field_reset_on_all_boundaries(
        self, mock_arrays, mock_objects, key, patched_updates
    ):
        b1 = Mock()
        b2 = Mock()
        b1.apply_field_reset.side_effect = lambda f: f
        b2.apply_field_reset.side_effect = lambda f: f
        mock_objects.boundary_objects = [b1, b2]

        backward(
            state=(5, mock_arrays),
            config=Mock(spec=SimulationConfig),
            objects=mock_objects,
            key=key,
            record_detectors=False,
            reset_fields=True,
        )

        b1.apply_field_reset.assert_called_once()
        b2.apply_field_reset.assert_called_once()

    def test_reset_fields_passes_only_requested_field_names(self, mock_arrays, mock_objects, key, patched_updates):
        b = Mock()
        b.apply_field_reset.side_effect = lambda f: f
        mock_objects.boundary_objects = [b]

        backward(
            state=(5, mock_arrays),
            config=Mock(spec=SimulationConfig),
            objects=mock_objects,
            key=key,
            record_detectors=False,
            reset_fields=True,
            fields_to_reset=("E",),
        )

        called_fields = b.apply_field_reset.call_args[0][0]
        assert set(called_fields.keys()) == {"E"}

    def test_custom_fields_to_reset(self, mock_arrays, mock_objects, key, patched_updates):
        backward(
            state=(5, mock_arrays),
            config=Mock(spec=SimulationConfig),
            objects=mock_objects,
            key=key,
            record_detectors=False,
            reset_fields=True,
            fields_to_reset=("E",),
        )

        assert "E" in mock_arrays._aset_log
        assert "H" not in mock_arrays._aset_log
