from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.fdtd import SimulationProgressBar, checkpointed_fdtd, custom_fdtd_forward, reversible_fdtd
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

    auxiliary_field_shape = (6, 2, 2, 2)  # (components, nx, ny, nz)
    mat_shape = (3, 2, 2, 2)  # scalar per voxel

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
    # Include one mock SimulationObject in the object list
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
    # No evolution should occur
    assert int(t) == 1
    assert isinstance(arrs, ArrayContainer)


# SimulationProgressBar unit tests


class TestSimulationProgressBar:
    """Unit tests for SimulationProgressBar that do not require a running simulation."""

    def test_context_manager_opens_and_closes_tqdm(self):
        """Entering the context manager should create a tqdm bar; exiting should close it."""
        mock_bar = MagicMock()
        mock_tqdm_cls = MagicMock(return_value=mock_bar)

        # _tqdm is an instance attribute, so we inject it directly after construction
        pbar = SimulationProgressBar(total_steps=100, desc="Test")
        pbar._tqdm = mock_tqdm_cls

        with pbar:
            assert pbar._bar is mock_bar

        mock_bar.close.assert_called_once()
        assert pbar._bar is None

    def test_default_attributes(self):
        """Constructor should store the supplied parameters."""
        pbar = SimulationProgressBar(total_steps=200, desc="Fwd", update_interval=5)
        assert pbar.total_steps == 200
        assert pbar.desc == "Fwd"
        assert pbar.update_interval == 5

    def test_default_update_interval(self):
        """Default update_interval should be 1."""
        pbar = SimulationProgressBar(total_steps=50)
        assert pbar.update_interval == 1

    def test_host_update_sets_bar_position(self):
        """_host_update should set bar.n to the given step and call refresh."""
        mock_bar = MagicMock()
        pbar = SimulationProgressBar(total_steps=10)
        pbar._bar = mock_bar

        pbar._host_update(7)

        assert mock_bar.n == 7
        mock_bar.refresh.assert_called_once()

    def test_host_update_respects_update_interval(self):
        """_host_update should only call refresh on multiples of update_interval."""
        mock_bar = MagicMock()
        pbar = SimulationProgressBar(total_steps=20, update_interval=5)
        pbar._bar = mock_bar

        for step in range(20):
            pbar._host_update(step)

        # Steps 0, 5, 10, 15 → 4 refreshes
        assert mock_bar.refresh.call_count == 4

    def test_host_update_noop_when_bar_is_none(self):
        """_host_update should not raise or do anything if called outside context."""
        pbar = SimulationProgressBar(total_steps=10)
        # _bar is None — should be a silent no-op
        pbar._host_update(3)  # must not raise

    def test_reset_resets_underlying_bar(self):
        """reset() should delegate to tqdm's reset method."""
        mock_bar = MagicMock()
        pbar = SimulationProgressBar(total_steps=10)
        pbar._bar = mock_bar

        pbar.reset()

        mock_bar.reset.assert_called_once()

    def test_reset_noop_when_bar_is_none(self):
        """reset() should not raise when called outside a context manager."""
        pbar = SimulationProgressBar(total_steps=10)
        pbar.reset()  # must not raise

    def test_get_callback_returns_callable(self):
        """get_callback() should return a callable even before the bar is opened."""
        pbar = SimulationProgressBar(total_steps=10)
        cb = pbar.get_callback()
        assert callable(cb)

    def test_get_callback_uses_io_callback(self):
        """The callback returned by get_callback should invoke jax.experimental.io_callback."""
        pbar = SimulationProgressBar(total_steps=10)
        pbar._bar = MagicMock()  # simulate open bar

        with patch("jax.experimental.io_callback") as mock_io_cb:
            cb = pbar.get_callback()
            cb(jnp.asarray(3, dtype=jnp.int32))
            mock_io_cb.assert_called_once()
            # Verify ordered=True is forwarded
            _, kwargs = mock_io_cb.call_args
            assert kwargs.get("ordered") is True

    def test_missing_tqdm_raises_import_error(self):
        """If tqdm is not installed, constructing SimulationProgressBar should raise ImportError."""
        with patch.dict("sys.modules", {"tqdm": None, "tqdm.auto": None}):
            with pytest.raises(ImportError, match="tqdm"):
                # Re-import to trigger the ImportError path inside __init__
                import importlib

                import fdtdx.fdtd.fdtd as fdtd_mod

                importlib.reload(fdtd_mod)
                fdtd_mod.SimulationProgressBar(total_steps=10)


# Integration tests: progress bar wired into each simulation function


class TestProgressBarIntegration:
    """Verify that passing a SimulationProgressBar to the simulation functions
    does not break their outputs, and test callback wiring independently of
    JAX's lax.while_loop effect execution.

    Note: jax.experimental.io_callback effects fired inside jax.lax.while_loop
    (used by eqxi.while_loop with kind="lax") are not guaranteed to execute on
    all JAX backends/versions. The integration tests therefore only assert that
    (a) simulation outputs are correct when a progress bar is supplied, and
    (b) the callback plumbing (_wrap_body_with_progress, get_callback) works
    correctly when tested outside of lax.while_loop.
    """

    def _make_pbar(self, total_steps: int, desc: str = "Test") -> SimulationProgressBar:
        """Return a SimulationProgressBar with a mock tqdm bar attached."""
        pbar = SimulationProgressBar(total_steps=total_steps, desc=desc)
        pbar._bar = MagicMock()
        return pbar

    # Simulation correctness: outputs must be unchanged by a progress bar

    def test_reversible_fdtd_with_progress_bar(self, dummy_arrays, dummy_objects, dummy_config):
        """Simulation completes successfully and returns correct types when a bar is supplied."""
        key = jax.random.PRNGKey(0)
        pbar = self._make_pbar(dummy_config.time_steps_total)

        t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, dummy_config, key, progress_bar=pbar)

        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)
        assert arrs.E.shape == dummy_arrays.E.shape

    def test_checkpointed_fdtd_with_progress_bar(self, dummy_arrays, dummy_objects, dummy_config_with_checkpointing):
        """Simulation completes successfully and returns correct types when a bar is supplied."""
        key = jax.random.PRNGKey(0)
        pbar = self._make_pbar(dummy_config_with_checkpointing.time_steps_total)

        t, arrs = checkpointed_fdtd(
            dummy_arrays, dummy_objects, dummy_config_with_checkpointing, key, progress_bar=pbar
        )

        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_custom_fdtd_forward_with_progress_bar(self, dummy_arrays, dummy_objects, dummy_config):
        """Simulation completes successfully and returns correct types when a bar is supplied."""
        key = jax.random.PRNGKey(0)
        end_time = 5
        pbar = self._make_pbar(end_time)

        t, arrs = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            dummy_config,
            key,
            reset_container=True,
            record_detectors=False,
            start_time=0,
            end_time=end_time,
            progress_bar=pbar,
        )

        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    # Callback wiring tests (outside lax.while_loop)

    def test_wrap_body_calls_callback_before_delegating(self):
        """_wrap_body_with_progress should invoke the callback with the current
        time step and then call the original body_fun exactly once."""
        from fdtdx.fdtd.fdtd import _wrap_body_with_progress

        call_log = []

        def fake_body(state):
            call_log.append(("body", int(state[0])))
            # Return state with incremented step so while_loop can advance
            return (state[0] + 1, state[1])

        pbar = self._make_pbar(total_steps=10)

        # Replace the JAX io_callback with a plain Python call so we can
        # observe it synchronously without going through lax.while_loop.
        pbar.get_callback()

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
        from fdtdx.fdtd.fdtd import _wrap_body_with_progress

        def fake_body(state):
            return state

        wrapped = _wrap_body_with_progress(fake_body, None)
        assert wrapped is fake_body

    def test_get_callback_fires_host_update_via_io_callback(self):
        """get_callback() must call _host_update through io_callback outside while_loop.

        We call the callback on a plain jnp scalar (not inside lax.while_loop)
        so that JAX executes the effect immediately, then flush with
        jax.effects_barrier() before asserting.
        """
        pbar = self._make_pbar(total_steps=20)
        cb = pbar.get_callback()

        cb(jnp.asarray(7, dtype=jnp.int32))
        jax.effects_barrier()

        assert pbar._bar.n == 7
        assert pbar._bar.refresh.call_count == 1

    def test_progress_bar_none_is_backward_compatible(self, dummy_arrays, dummy_objects, dummy_config):
        """Omitting progress_bar (default None) should behave identically to the original API."""
        key = jax.random.PRNGKey(0)
        t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, dummy_config, key)
        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_progress_bar_with_update_interval(self, dummy_arrays, dummy_objects, dummy_config):
        """update_interval > 1 should still yield a working simulation."""
        key = jax.random.PRNGKey(0)
        total = dummy_config.time_steps_total
        pbar = SimulationProgressBar(total_steps=total, update_interval=max(1, total // 4))
        pbar._bar = MagicMock()

        t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, dummy_config, key, progress_bar=pbar)

        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)

    def test_context_manager_usage_pattern(self, dummy_arrays, dummy_objects, dummy_config):
        """End-to-end test using SimulationProgressBar as a real context manager."""
        key = jax.random.PRNGKey(0)
        with SimulationProgressBar(total_steps=dummy_config.time_steps_total, desc="Integration") as pbar:
            t, arrs = reversible_fdtd(dummy_arrays, dummy_objects, dummy_config, key, progress_bar=pbar)

        assert isinstance(t, jax.Array)
        assert isinstance(arrs, ArrayContainer)
        # Bar should be closed after exiting the context
        assert pbar._bar is None

    def test_custom_fdtd_no_steps_no_callback(self, dummy_arrays, dummy_objects, dummy_config):
        """When start_time == end_time the body_fun never runs; bar should not be refreshed."""
        key = jax.random.PRNGKey(0)
        pbar = self._make_pbar(total_steps=10)

        t, arrs = custom_fdtd_forward(
            dummy_arrays,
            dummy_objects,
            dummy_config,
            key,
            reset_container=False,
            record_detectors=False,
            start_time=3,
            end_time=3,
            progress_bar=pbar,
        )

        assert int(t) == 3
        assert pbar._bar.refresh.call_count == 0
