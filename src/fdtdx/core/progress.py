from __future__ import annotations

import math

import jax


def _auto_update_interval(total_steps: int, target_updates: int = 20) -> int:
    """Return a visually pleasing update interval for a progress bar.

    Chooses the smallest power-of-ten multiple of 1, 2, or 5 that results in
    at most ``target_updates`` host callbacks over the full simulation.  This
    keeps the bar smooth while bounding the device→host sync overhead.

    Args:
        total_steps: Total number of simulation time steps.
        target_updates: Desired number of visible bar updates. Defaults to 20.

    Returns:
        An integer interval N such that the host callback is issued every N steps.
    """
    if total_steps <= target_updates:
        return 1
    raw = total_steps / target_updates
    # Round up to the nearest "nice" number: 1, 2, 5, 10, 20, 50, 100, …
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        candidate = factor * magnitude
        if candidate >= raw:
            return int(candidate)
    return int(magnitude * 10)


class SimulationProgressBar:
    """A tqdm progress bar for FDTD simulations driven from inside JAX loops.

    ``jax.experimental.io_callback`` (with ``ordered=True``) is used to push
    updates from compiled device code to the Python host.  The device→host
    sync is gated by ``jax.lax.cond`` so that it only fires every
    ``update_interval`` steps — steps that do not satisfy
    ``step % update_interval == 0`` incur **zero** sync overhead.

    This class is intended to be used as a context manager; it is created,
    opened, and closed automatically by the simulation functions.  Users do
    not need to instantiate it directly.

    Args:
        total_steps: Total number of simulation time steps.
        desc: Label shown next to the bar.
        update_interval: Issue a host callback only every N steps.  Use
            :func:`_auto_update_interval` to pick a sensible default.
    """

    def __init__(self, total_steps: int, desc: str = "FDTD", update_interval: int = 1):
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise ImportError("tqdm is required for the progress bar. Install it with: pip install tqdm") from exc
        self._tqdm = tqdm
        self.total_steps = total_steps
        self.desc = desc
        self.update_interval = update_interval
        self._bar = None

    # Context manager

    def __enter__(self) -> SimulationProgressBar:
        self._bar = self._tqdm(
            total=self.total_steps,
            desc=self.desc,
            unit="step",
            dynamic_ncols=True,
        )
        return self

    def __exit__(self, *_) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    # Host-side update (called by io_callback — never traced by JAX)

    def _host_update(self, time_step: int) -> None:
        """Unconditionally update the bar.

        The device-side ``lax.cond`` in :meth:`get_callback` ensures this is
        only ever called when ``step % update_interval == 0``, so no further
        filtering is needed here.
        """
        if self._bar is None:
            return
        self._bar.n = int(time_step)
        self._bar.refresh()

    # JAX-side callback factory

    def get_callback(self):
        """Return a JAX-traceable function that updates the bar on the host.

        The ``update_interval`` check runs **on the device** via
        ``jax.lax.cond``, so steps that do not satisfy the condition never
        issue an ``io_callback`` and incur no device→host sync cost.
        """
        host_fn = self._host_update
        update_interval = self.update_interval

        def _do_update(time_step: jax.Array) -> None:
            jax.experimental.io_callback(
                host_fn,
                result_shape_dtypes=(),
                time_step=time_step,
                ordered=True,
            )

        def _noop(_time_step: jax.Array) -> None:
            pass

        def _callback(time_step: jax.Array) -> None:
            jax.lax.cond(
                time_step % update_interval == 0,
                _do_update,
                _noop,
                time_step,
            )

        return _callback


# Internal helpers used by the simulation functions


def _wrap_body_with_progress(body_fun, progress_bar: SimulationProgressBar | None):
    """Wrap *body_fun* so that it fires the progress-bar callback each step.

    Returns *body_fun* unchanged when *progress_bar* is ``None``.
    """
    if progress_bar is None:
        return body_fun

    callback = progress_bar.get_callback()

    def wrapped(state):
        # Fire callback before the step so step 0 appears immediately.
        callback(state[0])
        return body_fun(state)

    return wrapped
