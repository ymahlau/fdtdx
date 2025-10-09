import math

import jax.numpy as jnp

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field
from fdtdx.units import s
from fdtdx.units.unitful import Unitful


@autoinit
class OnOffSwitch(TreeClass):
    start_time: Unitful | None = frozen_field(default=None)
    start_after_periods: Unitful | None = frozen_field(default=None)
    end_time: Unitful | None = frozen_field(default=None)
    end_after_periods: Unitful | None = frozen_field(default=None)
    on_for_time: Unitful | None = frozen_field(default=None)
    on_for_periods: Unitful | None = frozen_field(default=None)
    period: Unitful | None = frozen_field(default=None)
    fixed_on_time_steps: list[int] | None = frozen_field(default=None)
    is_always_off: bool = frozen_field(default=False)
    interval: int = frozen_field(default=1)

    def calculate_on_list(
        self,
        num_total_time_steps: int,
        time_step_duration: Unitful,
    ) -> list[bool]:
        # case 1: list with fixed time steps is provided
        if self.fixed_on_time_steps is not None:
            on_list = [False for _ in range(num_total_time_steps)]
            for t_idx in self.fixed_on_time_steps:
                on_list[t_idx] = True
            return on_list
        # case 2: calculate on list from other parameters
        on_list = []
        for t in range(num_total_time_steps):
            cur_on = self.is_on_at_time_step(
                time_step=t,
                time_step_duration=time_step_duration,
            )
            cur_on = cur_on and t % self.interval == 0
            on_list.append(cur_on)
        return on_list

    def is_on_at_time_step(
        self,
        time_step: int,
        time_step_duration: Unitful,
    ) -> bool:
        return is_on_at_time_step(
            is_always_off=self.is_always_off,
            start_time=self.start_time,
            start_after_periods=self.start_after_periods,
            end_time=self.end_time,
            end_after_periods=self.end_after_periods,
            on_for_time=self.on_for_time,
            on_for_periods=self.on_for_periods,
            time_step=time_step,
            time_step_duration=time_step_duration,
            period=self.period,
        )

    def calculate_time_step_to_on_arr_idx(
        self,
        num_total_time_steps: int,
        time_step_duration: Unitful,
    ) -> list[int]:
        on_list = self.calculate_on_list(
            num_total_time_steps=num_total_time_steps,
            time_step_duration=time_step_duration,
        )
        counter = 0
        time_to_arr_idx_list = [-1 for _ in range(num_total_time_steps)]
        for t in range(num_total_time_steps):
            if on_list[t]:
                time_to_arr_idx_list[t] = counter
                counter += 1
        return time_to_arr_idx_list


def is_on_at_time_step(
    is_always_off: bool,
    start_time: Unitful | None,
    start_after_periods: Unitful | None,
    end_time: Unitful | None,
    end_after_periods: Unitful | None,
    on_for_time: Unitful | None,
    on_for_periods: Unitful | None,
    time_step: int,
    time_step_duration: Unitful,
    period: Unitful | None,
) -> bool:  # scalar bool
    """Determines if a time-dependent component should be active at a given time step.

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
    if is_always_off:
        return False

    # validate start/end/on time
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
        start_time = 0 * s
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
        end_time = math.inf * s

    # period to actual time
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

    # determine start/end time
    if start_time is None and on_for_time is not None:
        if end_time is None:
            raise Exception("This should never happen")
        start_time = end_time - on_for_time

    if end_time is None and on_for_time is not None:
        if start_time is None:
            raise Exception("This should never happen")
        end_time = start_time + on_for_time

    # check if on
    if start_time is None or end_time is None:
        raise Exception("This should never happen")
    time_passed = time_step * time_step_duration

    on = True
    on = on and jnp.array(start_time <= time_passed).item()
    on = on and jnp.array(time_passed <= end_time).item()
    assert isinstance(on, bool)
    return on
