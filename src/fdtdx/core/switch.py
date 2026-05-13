import math

import numpy as np

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field


def _resolve_window_params(
    is_always_off: bool,
    start_time: float | None,
    start_after_periods: float | None,
    end_time: float | None,
    end_after_periods: float | None,
    on_for_time: float | None,
    on_for_periods: float | None,
    period: float | None,
) -> tuple[float, float]:
    """Resolve switch parameters to an absolute (start_time, end_time) window in seconds.

    This is the single source of truth for parameter validation and period-based
    conversions. Both ``OnOffSwitch._resolve_window`` and the standalone
    ``is_on_at_time_step`` function delegate here so that the logic cannot drift
    between the two call paths.

    Returns a ``(start, end)`` pair where ``end < start`` signals "always off"
    (only when ``is_always_off=True``).
    """
    if is_always_off:
        return (0.0, -1.0)

    need_period = any(x is not None for x in [start_after_periods, end_after_periods, on_for_periods])
    if need_period and period is None:
        raise Exception("Need to specify period!")

    num_start_specs = sum(
        [
            start_time is not None,
            start_after_periods is not None,
            on_for_time is not None and end_time is not None,
            on_for_periods is not None and end_time is not None,
            on_for_time is not None and end_after_periods is not None,
            on_for_periods is not None and end_after_periods is not None,
        ]
    )
    if num_start_specs > 1:
        raise Exception("Invalid start time specification!")
    if num_start_specs == 0:
        start_time = 0.0

    num_end_specs = sum(
        [
            end_time is not None,
            end_after_periods is not None,
            on_for_time is not None and start_time is not None,
            on_for_periods is not None and start_time is not None,
            on_for_time is not None and start_after_periods is not None,
            on_for_periods is not None and start_after_periods is not None,
        ]
    )
    if num_end_specs > 1:
        raise Exception("Invalid end time specification!")
    if num_end_specs == 0:
        end_time = math.inf

    if start_after_periods is not None:
        if period is None:
            raise Exception("This should never happen")
        start_time = start_after_periods * period
    if end_after_periods is not None:
        if period is None:
            raise Exception("This should never happen")
        end_time = end_after_periods * period
    if on_for_periods is not None:
        if period is None:
            raise Exception("This should never happen")
        on_for_time = on_for_periods * period

    if start_time is None and on_for_time is not None:
        if end_time is None:
            raise Exception("This should never happen")
        start_time = end_time - on_for_time
    if end_time is None and on_for_time is not None:
        if start_time is None:
            raise Exception("This should never happen")
        end_time = start_time + on_for_time

    if start_time is None or end_time is None:
        raise Exception("This should never happen")
    return (float(start_time), float(end_time))


@autoinit
class OnOffSwitch(TreeClass):
    #: start time of the switch
    start_time: float | None = frozen_field(default=None)

    #: start time after the period
    start_after_periods: float | None = frozen_field(default=None)

    #: end time of the switch
    end_time: float | None = frozen_field(default=None)

    #: end time after the period
    end_after_periods: float | None = frozen_field(default=None)

    #: time when the switch is active
    on_for_time: float | None = frozen_field(default=None)

    #: period when the switch is active
    on_for_periods: float | None = frozen_field(default=None)

    #:  period of the switch
    period: float | None = frozen_field(default=None)

    #: list of fixed time steps
    fixed_on_time_steps: list[int] | None = frozen_field(default=None)

    #: whether switch is always off
    is_always_off: bool = frozen_field(default=False)

    #: interval of the switch
    interval: int = frozen_field(default=1)

    def _resolve_window(self) -> tuple[float, float]:
        """Return the active time window as an absolute ``(start, end)`` pair in seconds.

        Delegates to ``_resolve_window_params`` — the single source of truth for
        parameter validation and period conversions shared with the standalone
        ``is_on_at_time_step`` function.
        """
        return _resolve_window_params(
            is_always_off=self.is_always_off,
            start_time=self.start_time,
            start_after_periods=self.start_after_periods,
            end_time=self.end_time,
            end_after_periods=self.end_after_periods,
            on_for_time=self.on_for_time,
            on_for_periods=self.on_for_periods,
            period=self.period,
        )

    def calculate_on_list(
        self,
        num_total_time_steps: int,
        time_step_duration: float,
    ) -> list[bool]:
        if self.fixed_on_time_steps is not None:
            fixed = set(self.fixed_on_time_steps)
            return [t in fixed for t in range(num_total_time_steps)]

        if self.is_always_off:
            return [False] * num_total_time_steps

        start_time, end_time = self._resolve_window()
        t = np.arange(num_total_time_steps, dtype=np.float64) * time_step_duration
        on = (start_time <= t) & (t <= end_time)
        if self.interval != 1:
            on = on & (np.arange(num_total_time_steps) % self.interval == 0)
        return on.tolist()

    def is_on_at_time_step(
        self,
        time_step: int,
        time_step_duration: float,
    ) -> bool:
        if self.fixed_on_time_steps is not None:
            return time_step in self.fixed_on_time_steps
        start_time, end_time = self._resolve_window()
        if end_time < start_time:
            return False
        if self.interval != 1 and time_step % self.interval != 0:
            return False
        time_passed = time_step * time_step_duration
        return bool(start_time <= time_passed <= end_time)

    def calculate_time_step_to_on_arr_idx(
        self,
        num_total_time_steps: int,
        time_step_duration: float,
    ) -> list[int]:
        on = np.asarray(
            self.calculate_on_list(
                num_total_time_steps=num_total_time_steps,
                time_step_duration=time_step_duration,
            )
        )
        result = np.full(num_total_time_steps, -1, dtype=np.intp)
        on_indices = np.where(on)[0]
        result[on_indices] = np.arange(len(on_indices), dtype=np.intp)
        return result.tolist()


def is_on_at_time_step(
    is_always_off: bool,
    start_time: float | None,
    start_after_periods: float | None,
    end_time: float | None,
    end_after_periods: float | None,
    on_for_time: float | None,
    on_for_periods: float | None,
    time_step: int,
    time_step_duration: float,
    period: float | None,
) -> bool:  # scalar bool
    """Determine whether a component is active at a given time step.

    Delegates parameter resolution to ``_resolve_window_params`` — the same
    helper used by ``OnOffSwitch`` — so both call paths are guaranteed to
    agree on validation rules and period conversions.

    Args:
        is_always_off (bool): Base on/off state
        start_time (float | None): Absolute start time
        start_after_periods (float | None): Start time in terms of periods
        end_time (float | None): Absolute end time
        end_after_periods (float | None): End time in terms of periods
        on_for_time (float | None): Duration to stay on in absolute time
        on_for_periods (float | None): Duration to stay on in terms of periods
        time_step (int): Current simulation time step
        time_step_duration (float): Duration of each time step
        period (float | None): Period length for period-based timing

    Returns:
        bool: True if the component should be active at the given time step
    """
    resolved_start, resolved_end = _resolve_window_params(
        is_always_off=is_always_off,
        start_time=start_time,
        start_after_periods=start_after_periods,
        end_time=end_time,
        end_after_periods=end_after_periods,
        on_for_time=on_for_time,
        on_for_periods=on_for_periods,
        period=period,
    )
    if resolved_end < resolved_start:
        return False
    time_passed = time_step * time_step_duration
    return bool(resolved_start <= time_passed <= resolved_end)


def is_on_at_time_step_from_switch(time_step: int, time_step_duration: float, switch: OnOffSwitch) -> bool:
    """Convenience wrapper — delegates to ``OnOffSwitch.is_on_at_time_step``."""
    return switch.is_on_at_time_step(time_step=time_step, time_step_duration=time_step_duration)
