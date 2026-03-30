from __future__ import annotations

import math

import jax
import jax.experimental


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

    The bar's entire lifetime — open, update, close — is managed through
    ``io_callback`` invocations that fire at **XLA execution time**, not at
    Python trace time.  This means the bar works correctly whether the
    simulation function is wrapped in ``jax.jit``:

    * On the first ``io_callback`` (step == step_offset) the bar is created.
    * Subsequent callbacks update ``bar.n``.
    * A dedicated close callback fires after the loop body exits for the last
      time, forcing the bar to 100 % and calling ``bar.close()``.

    This class is managed automatically by the simulation functions via
    :func:`_make_pbar`.  Users do not need to instantiate it directly.

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

    # Host-side callbacks

    def _host_update(self, time_step: int) -> None:
        """Open the bar lazily on the first call, then update its position.

        Opening here rather than in ``__init__`` means the bar appears at
        XLA execution time, not at Python trace time.  This makes the bar
        work correctly even when the simulation is wrapped in ``jax.jit``.

        The device-side ``lax.cond`` in :meth:`get_callbacks` ensures this
        is only called when ``step % update_interval == 0``, so no modulo
        check is needed here.

        ``step_offset`` is subtracted so the bar position stays relative to
        the start of this particular simulation segment.
        """
        if self._bar is None:
            # Lazy import: only reached at XLA execution time.
            from tqdm.auto import tqdm

            self._bar = tqdm(
                total=self.total_steps,
                desc=self.desc,
                unit="step",
                dynamic_ncols=True,
            )
        self._bar.n = int(time_step) - self.step_offset
        self._bar.refresh()

    def _host_close(self) -> None:
        """Force the bar to 100 % and close it.

        Called via a dedicated ``io_callback`` injected *after* the
        ``while_loop`` body exits for the final time.  This guarantees
        completion regardless of whether the last step landed on an
        update-interval boundary.
        """
        if self._bar is not None:
            self._bar.n = self.total_steps
            self._bar.refresh()
            self._bar.close()
            self._bar = None

    # JAX-side callback factory

    def get_callbacks(self):
        """Return ``(step_callback, close_callback)`` for use inside JAX code.

        ``step_callback(time_step)``
            JAX-traceable function.  The ``update_interval`` check runs on
            the device via ``jax.lax.cond`` so that only matching steps issue
            an ``io_callback`` (and thus a device→host sync).

        ``close_callback()``
            JAX-traceable no-argument function.  Should be called exactly once
            after the ``while_loop`` completes.  Forces the bar to 100 % and
            closes it.

        Both callbacks use ``ordered=True`` so that updates arrive in the
        correct sequence and the close always follows the last update.
        """
        host_update = self._host_update
        host_close = self._host_close
        update_interval = self.update_interval

        def _do_update(time_step: jax.Array) -> None:
            jax.experimental.io_callback(
                host_update,
                result_shape_dtypes=(),
                time_step=time_step,
                ordered=True,
            )

        def _noop(_time_step: jax.Array) -> None:
            pass

        def step_callback(time_step: jax.Array) -> None:
            jax.lax.cond(
                time_step % update_interval == 0,
                _do_update,
                _noop,
                time_step,
            )

        def close_callback() -> None:
            jax.experimental.io_callback(
                host_close,
                result_shape_dtypes=(),
                ordered=True,
            )

        return step_callback, close_callback


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
    """Wrap *body_fun* so that it fires the progress-bar step callback each step.

    Returns *body_fun* unchanged when *progress_bar* is ``None``, adding
    zero JAX overhead.

    Also returns a ``close_fn`` that must be called once after the loop
    completes to force the bar to 100 % and close it.  When *progress_bar*
    is ``None`` the returned ``close_fn`` is a no-op.
    """
    if progress_bar is None:
        return body_fun, lambda: None

    step_callback, close_callback = progress_bar.get_callbacks()

    def wrapped(state):
        # Fire callback before the step so step 0 appears immediately.
        step_callback(state[0])
        return body_fun(state)

    return wrapped, close_callback
