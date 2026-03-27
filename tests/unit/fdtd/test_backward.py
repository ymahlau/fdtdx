from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.backward import backward, cond_fn, full_backward
from fdtdx.fdtd.container import ObjectContainer
from fdtdx.objects.boundaries.periodic import PeriodicBoundary


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

    def test_pml_regions_zeroed(self, mock_arrays, mock_objects, key, patched_updates):
        pml = Mock()
        pml.grid_slice = (slice(0, 3), slice(0, 3), slice(0, 3))
        mock_objects.pml_objects = [pml]

        backward(
            state=(5, mock_arrays),
            config=Mock(spec=SimulationConfig),
            objects=mock_objects,
            key=key,
            record_detectors=False,
            reset_fields=True,
        )

        # E was ones, PML region should now be zeros
        e_field = mock_arrays._aset_log["E"]
        assert jnp.allclose(e_field[:, 0:3, 0:3, 0:3], 0.0)
        # Non-PML region should remain unchanged
        assert jnp.allclose(e_field[:, 3:, :, :], 1.0)

        # H was 2.0, PML region should now be zeros
        h_field = mock_arrays._aset_log["H"]
        assert jnp.allclose(h_field[:, 0:3, 0:3, 0:3], 0.0)
        assert jnp.allclose(h_field[:, 3:, :, :], 2.0)

    def test_periodic_boundary_positive_direction(self, mock_objects, key):
        # Use a non-uniform E field so we can verify the copy direction.
        arrays = Mock()
        # x=0 slice is 1.0, x=1 slice is 5.0; rest is 3.0
        e = jnp.ones((3, 10, 10, 10))
        e = e.at[:, 0, :, :].set(1.0)
        e = e.at[:, 1, :, :].set(5.0)
        arrays.E = e
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

        boundary = Mock(spec=PeriodicBoundary)
        boundary.axis = 0
        boundary.direction = "+"
        # destination: x=0:2; source end is x=1 (idx 1)
        boundary.grid_slice = (slice(0, 2), slice(0, 10), slice(0, 10))
        boundary._grid_slice_tuple = ((0, 2), (0, 10), (0, 10))
        mock_objects.boundary_objects = [boundary]

        with (
            patch("fdtdx.fdtd.backward.add_interfaces", return_value=arrays),
            patch("fdtdx.fdtd.backward.update_H_reverse", return_value=arrays),
            patch("fdtdx.fdtd.backward.update_E_reverse", return_value=arrays),
            patch("fdtdx.fdtd.backward.update_detector_states", return_value=arrays),
        ):
            backward(
                state=(5, arrays),
                config=Mock(spec=SimulationConfig),
                objects=mock_objects,
                key=key,
                record_detectors=False,
                reset_fields=True,
            )

        assert "E" in aset_log
        assert "H" in aset_log
        # The boundary region (x=0:2) should be overwritten with values from
        # the opposite edge (x=0:1 per the copy logic); both should equal 1.0.
        e_result = aset_log["E"]
        assert jnp.allclose(e_result[:, 0:2, :, :], e_result[:, 0:1, :, :])

    def test_periodic_boundary_negative_direction(self, mock_objects, key):
        # Use a non-uniform E field so we can verify the copy direction.
        arrays = Mock()
        e = jnp.ones((3, 10, 10, 10))
        e = e.at[:, :, 8, :].set(7.0)
        e = e.at[:, :, 9, :].set(3.0)
        arrays.E = e
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

        boundary = Mock(spec=PeriodicBoundary)
        boundary.axis = 1
        boundary.direction = "-"
        # destination: y=8:10; source is y=9:10 per copy logic
        boundary.grid_slice = (slice(0, 10), slice(8, 10), slice(0, 10))
        boundary._grid_slice_tuple = ((0, 10), (8, 10), (0, 10))
        mock_objects.boundary_objects = [boundary]

        with (
            patch("fdtdx.fdtd.backward.add_interfaces", return_value=arrays),
            patch("fdtdx.fdtd.backward.update_H_reverse", return_value=arrays),
            patch("fdtdx.fdtd.backward.update_E_reverse", return_value=arrays),
            patch("fdtdx.fdtd.backward.update_detector_states", return_value=arrays),
        ):
            backward(
                state=(5, arrays),
                config=Mock(spec=SimulationConfig),
                objects=mock_objects,
                key=key,
                record_detectors=False,
                reset_fields=True,
            )

        assert "E" in aset_log
        assert "H" in aset_log
        # The destination region (y=8:10) should be overwritten with values
        # from y=9:10, so both rows should equal the original y=9 values (3.0).
        e_result = aset_log["E"]
        assert jnp.allclose(e_result[:, :, 8:10, :], e_result[:, :, 9:10, :])

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
