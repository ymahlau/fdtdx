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
    opened, and closed automatically by the simulation functions via
    :func:`_make_pbar`.  Users do not need to instantiate it directly.

    The tqdm import is deferred to :meth:`__enter__` so that constructing this
    object never raises or warns — only opening the context manager does.

    Args:
        total_steps: Total number of simulation time steps (length of this segment).
        desc: Label shown next to the bar.
        update_interval: Issue a host callback only every N steps.  Use
            :func:`_auto_update_interval` to pick a sensible default.
        step_offset: The absolute time step that corresponds to position 0 on
            the bar.  Set to ``start_time`` for partial simulations so that
            the displayed position stays in ``[0, total_steps]``.
    """

    def __init__(
        self,
        total_steps: int,
        desc: str = "FDTD",
        update_interval: int = 1,
        step_offset: int = 0,
    ):
        self.total_steps = total_steps
        self.desc = desc
        self.update_interval = update_interval
        # Absolute step value that corresponds to position 0 on the bar.
        # For full simulations this is 0; for partial runs (custom_fdtd_forward)
        # it equals start_time so that bar.n stays in [0, total_steps].
        self.step_offset = step_offset
        self._bar = None

    # Context manager

    def __enter__(self) -> SimulationProgressBar:
        # Defer the tqdm import to here so __init__ never raises.
        try:
            from tqdm.auto import tqdm

            self._bar = tqdm(
                total=self.total_steps,
                desc=self.desc,
                unit="step",
                dynamic_ncols=True,
            )
        except ImportError:
            # tqdm not installed — bar stays None, no callbacks will fire
            # because _make_pbar already returned None in this case.
            pass
        return self

    def __exit__(self, *_) -> None:
        if self._bar is not None:
            # Force the bar to 100 % before closing.  The in-loop callback
            # fires with the *pre-step* counter, so the last update lands at
            # total_steps - update_interval rather than total_steps.
            self._bar.n = self.total_steps
            self._bar.refresh()
            self._bar.close()
            self._bar = None

    # Host-side update (called by io_callback — never traced by JAX)

    def _host_update(self, time_step: int) -> None:
        """Unconditionally update the bar.

        The device-side ``lax.cond`` in :meth:`get_callback` ensures this is
        only ever called when ``step % update_interval == 0``, so no further
        filtering is needed here.

        ``step_offset`` is subtracted so the bar position stays relative to
        the start of this particular simulation segment rather than showing
        the absolute time step.  For full simulations ``step_offset`` is 0.
        """
        if self._bar is None:
            return
        self._bar.n = int(time_step) - self.step_offset
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


def _make_pbar(
    show_progress: bool,
    total_steps: int,
    desc: str,
    step_offset: int = 0,
) -> SimulationProgressBar | None:
    """Return a :class:`SimulationProgressBar` or ``None``.

    This is the single point where we decide whether to create a progress bar.
    Returning ``None`` causes :func:`_wrap_body_with_progress` to leave the
    body function completely unmodified — zero JAX overhead.

    ``None`` is returned when any of the following apply:

    * ``show_progress`` is ``False``
    * ``total_steps`` is zero or negative (nothing to show)
    * tqdm is not installed (checked here via a cheap import probe so the
      simulation functions never import tqdm themselves)

    The tqdm probe only attempts ``import tqdm`` — it does **not** import
    ``tqdm.auto`` or instantiate anything, so its cost is a single
    ``sys.modules`` lookup on repeated calls.
    """
    if not show_progress or total_steps <= 0:
        return None
    try:
        import tqdm  # noqa: F401 — availability probe only
    except ImportError:
        import warnings

        warnings.warn(
            "tqdm is not installed — progress bar disabled. Install it with: pip install tqdm",
            ImportWarning,
            stacklevel=3,
        )
        return None
    return SimulationProgressBar(
        total_steps=total_steps,
        desc=desc,
        update_interval=_auto_update_interval(total_steps),
        step_offset=step_offset,
    )


def _wrap_body_with_progress(body_fun, progress_bar: SimulationProgressBar | None):
    """Wrap *body_fun* so that it fires the progress-bar callback each step.

    Returns *body_fun* unchanged when *progress_bar* is ``None``, adding
    zero JAX overhead.
    """
    if progress_bar is None:
        return body_fun

    callback = progress_bar.get_callback()

    def wrapped(state):
        # Fire callback before the step so step 0 appears immediately.
        callback(state[0])
        return body_fun(state)

    return wrapped
