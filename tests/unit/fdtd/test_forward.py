from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.forward import forward, forward_single_args_wrapper
from fdtdx.interfaces.state import RecordingState
from fdtdx.objects.detectors.detector import DetectorState


@pytest.fixture
def arrays():
    """Create an ArrayContainer with small test arrays."""
    return ArrayContainer(
        E=jnp.ones((3, 4, 4, 4)),
        H=jnp.ones((3, 4, 4, 4)) * 2.0,
        psi_E=jnp.zeros((6, 4, 4, 4)),
        psi_H=jnp.zeros((6, 4, 4, 4)),
        alpha=jnp.zeros((3, 4, 4, 4)),
        kappa=jnp.ones((3, 4, 4, 4)),
        sigma=jnp.zeros((3, 4, 4, 4)),
        inv_permittivities=jnp.ones((4, 4, 4)),
        inv_permeabilities=jnp.ones((4, 4, 4)),
        detector_states={"det1": Mock(spec=DetectorState)},
        recording_state=Mock(spec=RecordingState),
    )


@pytest.fixture
def updated_arrays(arrays):
    """Arrays returned by update_E/update_H (E field modified to distinguish)."""
    return arrays.aset("E", arrays.E * 3.0)


@pytest.fixture
def config():
    return Mock(spec=SimulationConfig)


@pytest.fixture
def objects():
    return Mock(spec=ObjectContainer)


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


class TestForward:
    """Tests for the forward() function."""

    def test_basic_forward_no_flags(self, arrays, updated_arrays, config, objects, key):
        """forward() with all flags False calls update_E and update_H only."""
        with (
            patch("fdtdx.fdtd.forward.update_E", return_value=updated_arrays) as mock_E,
            patch("fdtdx.fdtd.forward.update_H", return_value=updated_arrays) as mock_H,
        ):
            time_step = jnp.array(5)
            result = forward(
                state=(time_step, arrays),
                config=config,
                objects=objects,
                key=key,
                record_detectors=False,
                record_boundaries=False,
                simulate_boundaries=False,
            )

            # Returns (time_step+1, arrays)
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] == 6
            assert isinstance(result[1], ArrayContainer)

            # update_E called with correct args
            mock_E.assert_called_once_with(
                time_step=time_step,
                arrays=arrays,
                objects=objects,
                config=config,
                simulate_boundaries=False,
            )
            # update_H called with arrays returned by update_E
            mock_H.assert_called_once_with(
                time_step=time_step,
                arrays=updated_arrays,
                objects=objects,
                config=config,
                simulate_boundaries=False,
            )

    def test_simulate_boundaries_passed_through(self, arrays, updated_arrays, config, objects, key):
        """simulate_boundaries=True is forwarded to update_E and update_H."""
        with (
            patch("fdtdx.fdtd.forward.update_E", return_value=updated_arrays) as mock_E,
            patch("fdtdx.fdtd.forward.update_H", return_value=updated_arrays) as mock_H,
        ):
            forward(
                state=(jnp.array(0), arrays),
                config=config,
                objects=objects,
                key=key,
                record_detectors=False,
                record_boundaries=False,
                simulate_boundaries=True,
            )

            assert mock_E.call_args[1]["simulate_boundaries"] is True
            assert mock_H.call_args[1]["simulate_boundaries"] is True

    def test_record_detectors_calls_update_detector_states(self, arrays, updated_arrays, config, objects, key):
        """record_detectors=True triggers update_detector_states with inverse=False."""
        with (
            patch("fdtdx.fdtd.forward.update_E", return_value=updated_arrays),
            patch("fdtdx.fdtd.forward.update_H", return_value=updated_arrays),
            patch("fdtdx.fdtd.forward.update_detector_states", return_value=updated_arrays) as mock_det,
        ):
            time_step = jnp.array(3)
            result = forward(
                state=(time_step, arrays),
                config=config,
                objects=objects,
                key=key,
                record_detectors=True,
                record_boundaries=False,
                simulate_boundaries=False,
            )

            mock_det.assert_called_once()
            call_kwargs = mock_det.call_args[1]
            assert call_kwargs["time_step"] is time_step
            assert call_kwargs["arrays"] is updated_arrays
            assert call_kwargs["objects"] is objects
            assert call_kwargs["inverse"] is False
            # The returned state must contain the detector-updated arrays.
            assert result[1] is updated_arrays

    def test_h_prev_captured_before_update_e(self, arrays, config, objects, key):
        """H_prev is saved from original arrays.H before update_E modifies arrays."""
        modified_H = arrays.H * 99.0
        arrays_after_E = arrays.aset("H", modified_H)

        with (
            patch("fdtdx.fdtd.forward.update_E", return_value=arrays_after_E),
            patch("fdtdx.fdtd.forward.update_H", return_value=arrays_after_E),
            patch("fdtdx.fdtd.forward.update_detector_states", return_value=arrays_after_E) as mock_det,
        ):
            forward(
                state=(jnp.array(0), arrays),
                config=config,
                objects=objects,
                key=key,
                record_detectors=True,
                record_boundaries=False,
                simulate_boundaries=False,
            )

            # H_prev should be the original H, not the modified one
            h_prev_arg = mock_det.call_args[1]["H_prev"]
            assert jnp.array_equal(h_prev_arg, arrays.H)
            assert not jnp.array_equal(h_prev_arg, modified_H)

    def test_record_boundaries_calls_collect_interfaces_with_stop_gradient(
        self, arrays, updated_arrays, config, objects, key
    ):
        """record_boundaries=True calls collect_interfaces wrapped in stop_gradient."""
        collected = updated_arrays.aset("E", updated_arrays.E + 1.0)

        with (
            patch("fdtdx.fdtd.forward.update_E", return_value=updated_arrays),
            patch("fdtdx.fdtd.forward.update_H", return_value=updated_arrays),
            patch("fdtdx.fdtd.forward.collect_interfaces", return_value=collected) as mock_ci,
            patch("fdtdx.fdtd.forward.jax.lax.stop_gradient", return_value=collected) as mock_sg,
        ):
            time_step = jnp.array(2)
            result = forward(
                state=(time_step, arrays),
                config=config,
                objects=objects,
                key=key,
                record_detectors=False,
                record_boundaries=True,
                simulate_boundaries=False,
            )

            # collect_interfaces called with correct args
            mock_ci.assert_called_once_with(
                time_step=time_step,
                arrays=updated_arrays,
                objects=objects,
                config=config,
                key=key,
            )
            # stop_gradient wraps the collect_interfaces result
            mock_sg.assert_called_once_with(collected)
            # The stop_gradient result is used in the returned state
            assert result[1] is collected

    def test_no_detectors_when_flag_false(self, arrays, updated_arrays, config, objects, key):
        """record_detectors=False skips update_detector_states entirely."""
        with (
            patch("fdtdx.fdtd.forward.update_E", return_value=updated_arrays),
            patch("fdtdx.fdtd.forward.update_H", return_value=updated_arrays),
            patch("fdtdx.fdtd.forward.update_detector_states") as mock_det,
        ):
            forward(
                state=(jnp.array(0), arrays),
                config=config,
                objects=objects,
                key=key,
                record_detectors=False,
                record_boundaries=False,
                simulate_boundaries=False,
            )
            mock_det.assert_not_called()

    def test_no_boundaries_when_flag_false(self, arrays, updated_arrays, config, objects, key):
        """record_boundaries=False skips collect_interfaces entirely."""
        with (
            patch("fdtdx.fdtd.forward.update_E", return_value=updated_arrays),
            patch("fdtdx.fdtd.forward.update_H", return_value=updated_arrays),
            patch("fdtdx.fdtd.forward.collect_interfaces") as mock_ci,
        ):
            forward(
                state=(jnp.array(0), arrays),
                config=config,
                objects=objects,
                key=key,
                record_detectors=False,
                record_boundaries=False,
                simulate_boundaries=False,
            )
            mock_ci.assert_not_called()

    def test_both_record_flags_true(self, arrays, updated_arrays, config, objects, key):
        """Both record_detectors and record_boundaries can be active simultaneously."""
        with (
            patch("fdtdx.fdtd.forward.update_E", return_value=updated_arrays),
            patch("fdtdx.fdtd.forward.update_H", return_value=updated_arrays),
            patch("fdtdx.fdtd.forward.collect_interfaces", return_value=updated_arrays) as mock_ci,
            patch("fdtdx.fdtd.forward.jax.lax.stop_gradient", return_value=updated_arrays),
            patch("fdtdx.fdtd.forward.update_detector_states", return_value=updated_arrays) as mock_det,
        ):
            forward(
                state=(jnp.array(0), arrays),
                config=config,
                objects=objects,
                key=key,
                record_detectors=True,
                record_boundaries=True,
                simulate_boundaries=False,
            )
            mock_ci.assert_called_once()
            mock_det.assert_called_once()

    def test_boundary_recording_before_detector_recording(self, arrays, updated_arrays, config, objects, key):
        """Boundary recording happens before detector recording (order matters)."""
        call_order = []
        boundary_result = updated_arrays.aset("E", updated_arrays.E + 10.0)

        def track_ci(**kwargs):
            call_order.append("collect_interfaces")
            return boundary_result

        def track_det(**kwargs):
            call_order.append("update_detector_states")
            # Verify detector sees boundary-updated arrays
            assert kwargs["arrays"] is boundary_result
            return boundary_result

        with (
            patch("fdtdx.fdtd.forward.update_E", return_value=updated_arrays),
            patch("fdtdx.fdtd.forward.update_H", return_value=updated_arrays),
            patch("fdtdx.fdtd.forward.collect_interfaces", side_effect=track_ci),
            patch("fdtdx.fdtd.forward.jax.lax.stop_gradient", return_value=boundary_result),
            patch("fdtdx.fdtd.forward.update_detector_states", side_effect=track_det),
        ):
            forward(
                state=(jnp.array(0), arrays),
                config=config,
                objects=objects,
                key=key,
                record_detectors=True,
                record_boundaries=True,
                simulate_boundaries=False,
            )
            assert call_order == ["collect_interfaces", "update_detector_states"]


class TestForwardSingleArgsWrapper:
    """Tests for the forward_single_args_wrapper() function."""

    def test_wrapper_constructs_array_container_and_calls_forward(self, arrays, config, objects, key):
        """Wrapper creates ArrayContainer from individual args and calls forward."""
        result_arrays = arrays.aset("E", arrays.E * 5.0)
        mock_state = (jnp.array(1), result_arrays)

        with patch("fdtdx.fdtd.forward.forward", return_value=mock_state) as mock_fwd:
            forward_single_args_wrapper(
                time_step=jnp.array(0),
                E=arrays.E,
                H=arrays.H,
                psi_E=arrays.psi_E,
                psi_H=arrays.psi_H,
                alpha=arrays.alpha,
                kappa=arrays.kappa,
                sigma=arrays.sigma,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                detector_states=arrays.detector_states,
                recording_state=arrays.recording_state,
                config=config,
                objects=objects,
                key=key,
                record_detectors=True,
                record_boundaries=True,
                simulate_boundaries=True,
            )

            mock_fwd.assert_called_once()
            call_kwargs = mock_fwd.call_args[1]
            # Verify the ArrayContainer was constructed
            state_arg = call_kwargs["state"]
            assert jnp.array_equal(state_arg[0], jnp.array(0))
            assert isinstance(state_arg[1], ArrayContainer)
            assert jnp.array_equal(state_arg[1].E, arrays.E)
            assert jnp.array_equal(state_arg[1].H, arrays.H)
            # Verify flags are passed through
            assert call_kwargs["config"] is config
            assert call_kwargs["objects"] is objects
            assert call_kwargs["key"] is key
            assert call_kwargs["record_detectors"] is True
            assert call_kwargs["record_boundaries"] is True
            assert call_kwargs["simulate_boundaries"] is True

    def test_wrapper_returns_all_12_unpacked_fields(self, arrays, config, objects, key):
        """Wrapper unpacks the returned SimulationState into 12 individual values."""
        result_arrays = ArrayContainer(
            E=arrays.E * 2.0,
            H=arrays.H * 3.0,
            psi_E=arrays.psi_E + 1.0,
            psi_H=arrays.psi_H + 2.0,
            alpha=arrays.alpha + 0.1,
            kappa=arrays.kappa * 1.5,
            sigma=arrays.sigma + 0.5,
            inv_permittivities=arrays.inv_permittivities * 4.0,
            inv_permeabilities=arrays.inv_permeabilities * 5.0,
            detector_states={"det1": Mock(spec=DetectorState)},
            recording_state=Mock(spec=RecordingState),
        )
        mock_state = (jnp.array(7), result_arrays)

        with patch("fdtdx.fdtd.forward.forward", return_value=mock_state):
            result = forward_single_args_wrapper(
                time_step=jnp.array(6),
                E=arrays.E,
                H=arrays.H,
                psi_E=arrays.psi_E,
                psi_H=arrays.psi_H,
                alpha=arrays.alpha,
                kappa=arrays.kappa,
                sigma=arrays.sigma,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                detector_states=arrays.detector_states,
                recording_state=arrays.recording_state,
                config=config,
                objects=objects,
                key=key,
                record_detectors=False,
                record_boundaries=False,
                simulate_boundaries=False,
            )

            assert len(result) == 12
            assert result[0] == 7  # time_step
            assert jnp.array_equal(result[1], result_arrays.E)
            assert jnp.array_equal(result[2], result_arrays.H)
            assert jnp.array_equal(result[3], result_arrays.psi_E)
            assert jnp.array_equal(result[4], result_arrays.psi_H)
            assert jnp.array_equal(result[5], result_arrays.alpha)
            assert jnp.array_equal(result[6], result_arrays.kappa)
            assert jnp.array_equal(result[7], result_arrays.sigma)
            assert jnp.array_equal(result[8], result_arrays.inv_permittivities)
            assert jnp.array_equal(result[9], result_arrays.inv_permeabilities)
            assert result[10] is result_arrays.detector_states
            assert result[11] is result_arrays.recording_state
