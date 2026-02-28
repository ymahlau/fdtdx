from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.core.progress import (
    SimulationProgressBar,
    _auto_update_interval,
    _make_pbar,
    _wrap_body_with_progress,
)
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.fdtd import checkpointed_fdtd, custom_fdtd_forward, reversible_fdtd
from fdtdx.objects.object import SimulationObject


class DummySimulationObject(SimulationObject):
    """Minimal mock implementation of a SimulationObject."""

    def __init__(self, name="dummy"):
        self.name = name

    def update(self, *args, **kwargs):
        return args, kwargs


@pytest.fixture
def dummy_arrays():
    field_shape = (3, 2, 2, 2)  # (components, nx, ny, nz)
    auxiliary_field_shape = (6, 2, 2, 2)
    mat_shape = (3, 2, 2, 2)

    return ArrayContainer(
        E=jnp.zeros(field_shape),
        H=jnp.zeros(field_shape),
        psi_E=jnp.zeros(auxiliary_field_shape),
        psi_H=jnp.zeros(auxiliary_field_shape),
        alpha=jnp.zeros(field_shape),
        kappa=jnp.ones(field_shape),
        sigma=jnp.zeros(field_shape),
        inv_permittivities=jnp.ones(mat_shape),
        inv_permeabilities=jnp.ones(mat_shape),
        detector_states={},
        recording_state=None,
        electric_conductivity=None,
        magnetic_conductivity=None,
    )


@pytest.fixture
def dummy_objects():
    return ObjectContainer(object_list=[DummySimulationObject()], volume_idx=0)


@pytest.fixture
def empty_objects():
    return ObjectContainer(object_list=[], volume_idx=0)


@pytest.fixture
def dummy_config():
    return SimulationConfig(
        time=400e-15,
        resolution=1.0,
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )


@pytest.fixture
def dummy_config_with_checkpointing():
    return SimulationConfig(
        time=400e-15,
        resolution=1.0,
        backend="gpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=GradientConfig(
            method="checkpointed",
            num_checkpoints=10,
        ),
    )


def test_reversible_fdtd_runs(dummy_arrays, dummy_objects, dummy_config):
    key = jax.random.PRNGKey(0)
    t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, dummy_config, key)

    assert isinstance(t, jax.Array)
    assert isinstance(arrs, ArrayContainer)
    assert arrs.E.shape == dummy_arrays.E.shape


def test_checkpointed_fdtd_runs(dummy_arrays, dummy_objects, dummy_config_with_checkpointing):
    key = jax.random.PRNGKey(0)
    t, arrs = checkpointed_fdtd(dummy_arrays, dummy_objects, dummy_config_with_checkpointing, key)

    assert isinstance(t, jax.Array)
    assert isinstance(arrs, ArrayContainer)


def test_custom_fdtd_forward_runs(dummy_arrays, dummy_objects, dummy_config):
    key = jax.random.PRNGKey(0)
    t, arrs = custom_fdtd_forward(
        dummy_arrays,
        dummy_objects,
        dummy_config,
        key,
        reset_container=True,
        record_detectors=False,
        start_time=0,
        end_time=1,
    )

    assert isinstance(t, jax.Array)
    assert isinstance(arrs, ArrayContainer)


# --- Edge case tests ---


def test_zero_time(dummy_arrays, dummy_objects):
    config = SimulationConfig(
        time=0.0,
        resolution=1.0,
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )
    key = jax.random.PRNGKey(0)
    t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, config, key)
    assert int(t) == 0
    assert isinstance(arrs, ArrayContainer)


def test_empty_objects(dummy_arrays, empty_objects, dummy_config_with_checkpointing):
    key = jax.random.PRNGKey(0)
    t, arrs = checkpointed_fdtd(dummy_arrays, empty_objects, dummy_config_with_checkpointing, key)
    assert isinstance(t, jax.Array)
    assert isinstance(arrs, ArrayContainer)


def test_custom_fdtd_forward_same_start_end(dummy_arrays, dummy_objects, dummy_config):
    key = jax.random.PRNGKey(0)
    t, arrs = custom_fdtd_forward(
        dummy_arrays,
        dummy_objects,
        dummy_config,
        key,
        reset_container=False,
        record_detectors=False,
        start_time=1,
        end_time=1,
    )
    assert int(t) == 1
    assert isinstance(arrs, ArrayContainer)


# _auto_update_interval unit tests


class TestAutoUpdateInterval:
    """Tests for the _auto_update_interval helper."""

    def test_few_steps_returns_one(self):
        """When total_steps <= target_updates the interval should be 1."""
        assert _auto_update_interval(10, target_updates=20) == 1
        assert _auto_update_interval(20, target_updates=20) == 1

    def test_result_is_nice_number(self):
        """Result must be a multiple of 1, 2, or 5 times a power of ten."""
        nice_factors = {1, 2, 5}
        for total in (100, 500, 1000, 5000, 10000):
            interval = _auto_update_interval(total)
            assert interval >= 1
            # Divide out powers of ten until we reach the leading factor
            n = interval
            while n % 10 == 0:
                n //= 10
            assert n in nice_factors, f"interval={interval} is not a nice number"

    def test_updates_do_not_exceed_target(self):
        """total_steps / interval must not exceed target_updates (up to rounding)."""
        for total in (100, 1000, 5000):
            interval = _auto_update_interval(total, target_updates=20)
            assert total / interval <= 20 + 1  # allow one extra for rounding

    def test_one_step(self):
        assert _auto_update_interval(1) == 1

    def test_large_simulation(self):
        """For a very large simulation the interval should be large too."""
        interval = _auto_update_interval(1_000_000, target_updates=20)
        assert interval >= 1000


# SimulationProgressBar unit tests


class TestSimulationProgressBar:
    """Unit tests for SimulationProgressBar that do not require a running simulation."""

    def test_context_manager_opens_and_closes_bar(self):
        """Entering the context manager should create a tqdm bar; exiting should close it.

        We patch tqdm.auto.tqdm at the module level so __enter__ picks it up.
        """
        mock_bar = MagicMock()
        mock_tqdm_cls = MagicMock(return_value=mock_bar)

        pbar = SimulationProgressBar(total_steps=100, desc="Test")
        with patch("tqdm.auto.tqdm", mock_tqdm_cls):
            with pbar:
                assert pbar._bar is mock_bar

        mock_bar.close.assert_called_once()
        assert pbar._bar is None

    def test_exit_forces_bar_to_100_percent(self):
        """__exit__ must set bar.n = total_steps before closing so the bar reaches 100 %.

        The in-loop callback fires with the pre-step counter, so without this
        the bar would stop at total_steps - update_interval.
        """
        mock_bar = MagicMock()
        mock_tqdm_cls = MagicMock(return_value=mock_bar)

        pbar = SimulationProgressBar(total_steps=100, desc="Test", update_interval=10)
        with patch("tqdm.auto.tqdm", mock_tqdm_cls):
            with pbar:
                # Simulate the last in-loop callback firing at step 90 (pre-step)
                pbar._host_update(90)

        # __exit__ must have set n=100 before close
        assert mock_bar.n == 100
        mock_bar.refresh.assert_called()
        mock_bar.close.assert_called_once()

    def test_default_attributes(self):
        pbar = SimulationProgressBar(total_steps=200, desc="Fwd", update_interval=5)
        assert pbar.total_steps == 200
        assert pbar.desc == "Fwd"
        assert pbar.update_interval == 5
        assert pbar.step_offset == 0

    def test_step_offset_default_is_zero(self):
        pbar = SimulationProgressBar(total_steps=50)
        assert pbar.step_offset == 0

    def test_default_update_interval(self):
        pbar = SimulationProgressBar(total_steps=50)
        assert pbar.update_interval == 1

    def test_init_does_not_import_tqdm(self):
        """Constructing SimulationProgressBar must not touch tqdm at all.

        The import is deferred to __enter__ so that construction is always
        safe, even in environments without tqdm installed.
        """
        with patch.dict("sys.modules", {"tqdm": None, "tqdm.auto": None}):
            # Must not raise ImportError
            pbar = SimulationProgressBar(total_steps=10)
            assert pbar._bar is None

    def test_host_update_sets_bar_position(self):
        """_host_update should set bar.n to the given step and call refresh."""
        mock_bar = MagicMock()
        pbar = SimulationProgressBar(total_steps=10)
        pbar._bar = mock_bar

        pbar._host_update(7)

        assert mock_bar.n == 7
        mock_bar.refresh.assert_called_once()

    def test_host_update_applies_step_offset(self):
        """bar.n must equal time_step - step_offset so partial runs show relative progress.

        For a run from step 50 to 100, step_offset=50, so when the callback
        fires with time_step=60 the bar should show n=10, not n=60.
        """
        mock_bar = MagicMock()
        pbar = SimulationProgressBar(total_steps=50, step_offset=50)
        pbar._bar = mock_bar

        pbar._host_update(60)

        assert mock_bar.n == 10  # 60 - 50

    def test_host_update_with_zero_offset(self):
        """With step_offset=0 (default), bar.n equals the raw time_step."""
        mock_bar = MagicMock()
        pbar = SimulationProgressBar(total_steps=100, step_offset=0)
        pbar._bar = mock_bar

        pbar._host_update(42)

        assert mock_bar.n == 42

    def test_host_update_always_refreshes_when_called(self):
        """_host_update must unconditionally refresh on every call.

        The update_interval gating is done on the device (jax.lax.cond), so
        _host_update is only invoked when the condition is already satisfied —
        it must never skip a refresh itself.
        """
        mock_bar = MagicMock()
        pbar = SimulationProgressBar(total_steps=20, update_interval=5)
        pbar._bar = mock_bar

        # Simulate only the steps that pass the device-side cond
        for step in range(0, 20, 5):
            pbar._host_update(step)

        assert mock_bar.refresh.call_count == 4  # steps 0, 5, 10, 15

    def test_host_update_noop_when_bar_is_none(self):
        """_host_update should not raise when called outside of a context manager."""
        pbar = SimulationProgressBar(total_steps=10)
        pbar._host_update(3)  # must not raise

    def test_get_callback_returns_callable(self):
        pbar = SimulationProgressBar(total_steps=10)
        cb = pbar.get_callback()
        assert callable(cb)

    def test_get_callback_uses_io_callback(self):
        """The callback must invoke jax.experimental.io_callback with ordered=True."""
        pbar = SimulationProgressBar(total_steps=10)
        pbar._bar = MagicMock()

        with patch("jax.experimental.io_callback") as mock_io_cb:
            cb = pbar.get_callback()
            cb(jnp.asarray(3, dtype=jnp.int32))
            mock_io_cb.assert_called_once()
            _, kwargs = mock_io_cb.call_args
            assert kwargs.get("ordered") is True

    def test_get_callback_gates_io_callback_on_device(self):
        """io_callback must only fire on steps satisfying step % update_interval == 0.

        The modulo check runs via jax.lax.cond on the device.
        jax.experimental.io_callback is fully supported inside jax.lax.while_loop
        with ordered=True (confirmed in JAX docs: https://docs.jax.dev/en/latest/external-callbacks.html).
        We test here outside of while_loop for determinism.
        """
        pbar = SimulationProgressBar(total_steps=20, update_interval=5)
        pbar._bar = MagicMock()
        cb = pbar.get_callback()

        for step in range(20):
            cb(jnp.asarray(step, dtype=jnp.int32))
        jax.effects_barrier()

        # Only steps 0, 5, 10, 15 pass the device-side cond → exactly 4 host calls
        assert pbar._bar.refresh.call_count == 4


class TestMakePbar:
    """Unit tests for the _make_pbar factory — the single gating point for tqdm availability."""

    def test_returns_none_when_show_progress_false(self):
        assert _make_pbar(show_progress=False, total_steps=100, desc="x") is None

    def test_returns_none_when_total_steps_zero(self):
        assert _make_pbar(show_progress=True, total_steps=0, desc="x") is None

    def test_returns_none_when_total_steps_negative(self):
        assert _make_pbar(show_progress=True, total_steps=-1, desc="x") is None

    def test_returns_pbar_when_tqdm_available(self):
        """When tqdm is installed _make_pbar must return a SimulationProgressBar."""
        from fdtdx.core.progress import SimulationProgressBar

        result = _make_pbar(show_progress=True, total_steps=100, desc="Test")
        assert isinstance(result, SimulationProgressBar)

    def test_returns_none_and_warns_when_tqdm_missing(self):
        """When tqdm is absent _make_pbar must return None (not a bar) and warn.

        Returning None means _wrap_body_with_progress leaves the body function
        untouched — zero JAX overhead, no io_callbacks, no device→host syncs.
        """
        with patch.dict("sys.modules", {"tqdm": None, "tqdm.auto": None}):
            import importlib

            import fdtdx.core.progress as progress_mod

            importlib.reload(progress_mod)

            with pytest.warns(ImportWarning, match="tqdm"):
                result = progress_mod._make_pbar(show_progress=True, total_steps=100, desc="Test")

            assert result is None

    def test_step_offset_forwarded_to_pbar(self):
        """step_offset passed to _make_pbar must be stored on the returned pbar."""
        result = _make_pbar(show_progress=True, total_steps=50, desc="x", step_offset=25)
        assert result is not None
        assert result.step_offset == 25

    def test_auto_interval_applied(self):
        """_make_pbar must set update_interval via _auto_update_interval."""
        result = _make_pbar(show_progress=True, total_steps=1000, desc="x")
        assert result is not None
        expected = _auto_update_interval(1000)
        assert result.update_interval == expected


# Integration tests: show_progress flag wired into each simulation function


class TestShowProgressFlag:
    """Verify that show_progress=True/False does not alter simulation outputs,
    and that the progress bar wiring (_wrap_body_with_progress) works correctly."""

    # show_progress=True: simulation must still return correct results

    def test_reversible_fdtd_with_progress(self, dummy_arrays, dummy_objects, dummy_config):
        key = jax.random.PRNGKey(0)
        t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, dummy_config, key, show_progress=True)
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)
        assert arrs.E.shape == dummy_arrays.E.shape

    def test_checkpointed_fdtd_with_progress(self, dummy_arrays, dummy_objects, dummy_config_with_checkpointing):
        key = jax.random.PRNGKey(0)
        t, arrs = checkpointed_fdtd(
            dummy_arrays, dummy_objects, dummy_config_with_checkpointing, key, show_progress=True
        )
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_custom_fdtd_forward_with_progress(self, dummy_arrays, dummy_objects, dummy_config):
        key = jax.random.PRNGKey(0)
        t, arrs = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            dummy_config,
            key,
            reset_container=True,
            record_detectors=False,
            start_time=0,
            end_time=5,
            show_progress=True,
        )
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    # show_progress=False: baseline, no bar created

    def test_reversible_fdtd_without_progress(self, dummy_arrays, dummy_objects, dummy_config):
        key = jax.random.PRNGKey(0)
        t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, dummy_config, key, show_progress=False)
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_checkpointed_fdtd_without_progress(self, dummy_arrays, dummy_objects, dummy_config_with_checkpointing):
        key = jax.random.PRNGKey(0)
        t, arrs = checkpointed_fdtd(
            dummy_arrays,
            dummy_objects,
            dummy_config_with_checkpointing,
            key,
            show_progress=False,
        )
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_custom_fdtd_no_steps_no_bar(self, dummy_arrays, dummy_objects, dummy_config):
        """When start_time == end_time no loop iterations run and no bar is created."""
        key = jax.random.PRNGKey(0)
        t, arrs = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            dummy_config,
            key,
            reset_container=False,
            record_detectors=False,
            start_time=3,
            end_time=3,
            show_progress=True,  # n_steps=0, so pbar=None anyway
        )
        assert int(t) == 3
        assert isinstance(arrs, ArrayContainer)

    def test_custom_fdtd_jax_array_times_disables_progress(self, dummy_arrays, dummy_objects, dummy_config):
        """Passing jax.Array start_time/end_time must not cause concretization errors.

        When start_time or end_time are traced jax.Arrays the function must
        silently disable the progress bar rather than calling int() and
        triggering a ConcretizationTypeError.
        """
        key = jax.random.PRNGKey(0)
        start = jnp.asarray(0, dtype=jnp.int32)
        end = jnp.asarray(2, dtype=jnp.int32)
        # This must not raise a ConcretizationTypeError
        t, arrs = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            dummy_config,
            key,
            reset_container=True,
            record_detectors=False,
            start_time=start,
            end_time=end,
            show_progress=True,
        )
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_partial_run_bar_shows_relative_progress(self):
        """Progress bar for a partial run must display steps relative to start_time.

        With start_time=50 and total_steps=50, a callback at absolute step 60
        must set bar.n to 10, not 60.  Without step_offset the bar would
        immediately overflow its total.
        """
        mock_bar = MagicMock()
        pbar = SimulationProgressBar(
            total_steps=50,
            step_offset=50,
            update_interval=1,
        )
        pbar._bar = mock_bar

        # Simulate callbacks as they would arrive from the device
        for abs_step in range(50, 100, 10):
            pbar._host_update(abs_step)

        # Just verify the last assignment via attribute tracking
        assert mock_bar.n == 40  # last callback: abs 90 - offset 50 = 40
        assert mock_bar.n <= 50  # must never exceed total_steps

    # Callback wiring tests (outside lax.while_loop for determinism)

    def test_wrap_body_calls_callback_before_delegating(self):
        """_wrap_body_with_progress must invoke the callback before body_fun."""
        call_log = []

        def fake_body(state):
            call_log.append(("body", int(state[0])))
            return (state[0] + 1, state[1])

        pbar = SimulationProgressBar(total_steps=10)
        pbar._bar = MagicMock()

        # Bypass io_callback for synchronous testing
        def patched_get_callback():
            def _cb(time_step):
                call_log.append(("callback", int(time_step)))

            return _cb

        pbar.get_callback = patched_get_callback
        wrapped = _wrap_body_with_progress(fake_body, pbar)

        dummy_state = (jnp.asarray(3, dtype=jnp.int32), None)
        wrapped(dummy_state)

        assert call_log[0] == ("callback", 3), "callback must fire before body_fun"
        assert call_log[1] == ("body", 3), "body_fun must fire after callback"
        assert len(call_log) == 2

    def test_wrap_body_with_none_pbar_is_identity(self):
        """_wrap_body_with_progress(body, None) must return the original body unchanged."""

        def fake_body(state):
            return state

        wrapped = _wrap_body_with_progress(fake_body, None)
        assert wrapped is fake_body

    def test_get_callback_fires_host_update_outside_while_loop(self):
        """Calling the callback outside of while_loop triggers _host_update via io_callback.

        Per the JAX documentation, jax.experimental.io_callback is fully
        compatible with jax.lax.while_loop when ordered=True.
        Reference: https://docs.jax.dev/en/latest/external-callbacks.html
        """
        pbar = SimulationProgressBar(total_steps=20)
        pbar._bar = MagicMock()
        cb = pbar.get_callback()

        cb(jnp.asarray(7, dtype=jnp.int32))
        jax.effects_barrier()

        assert pbar._bar.n == 7
        assert pbar._bar.refresh.call_count == 1
