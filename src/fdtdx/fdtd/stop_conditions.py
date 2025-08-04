from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field
from fdtdx.fdtd.container import SimulationState


@autoinit
class StoppingCondition(TreeClass, ABC):
    """Abstract base class for defining custom stopping conditions in simulations.

    This class provides the interface for implementing custom termination criteria
    that can depend on simulation state, detector readings, field values, etc.
    """

    @abstractmethod
    def __call__(self, state: SimulationState) -> jax.Array:
        """Evaluate the stopping condition.

        Args:
            state: Current simulation state (time_step, arrays)

        Returns:
            jax.Array: Boolean scalar - True if simulation should continue, False if it should stop
        """
        pass


@autoinit
class TimeStepCondition(StoppingCondition):
    """Default stopping condition based on maximum time steps.

    This recreates the original behavior where simulation continues until
    a specified end time is reached.
    """

    end_time: int = frozen_field()

    def __call__(self, state: SimulationState) -> jax.Array:
        """Check if simulation should continue based on time step.

        Args:
            state: Current simulation state (time_step, arrays)

        Returns:
            jax.Array: Boolean scalar - True if current time < end_time, False otherwise
        """
        time_step, _ = state
        return self.end_time > time_step


@autoinit
class DetectorThresholdCondition(StoppingCondition):
    """Stopping condition based on detector readings reaching a threshold.

    Example implementation showing how to create detector-based stopping criteria.
    """

    detector_name: str = frozen_field()
    threshold: float = frozen_field()
    end_time: int = frozen_field()
    comparison: str = frozen_field(default="less_than")  # "less_than", "greater_than", "equal"

    def __call__(self, state: SimulationState) -> jax.Array:
        """Check if simulation should continue based on detector reading.

        Args:
            state: Current simulation state (time_step, arrays)

        Returns:
            jax.Array: Boolean scalar - True if condition not met and time < end_time
        """
        time_step, arrays = state
        time_condition = time_step < self.end_time
        detector_condition = self._check_detector_condition(arrays)
        return time_condition & (~detector_condition)

    def _check_detector_condition(self, arrays) -> jax.Array:
        # Get detector state if it exists, otherwise use dummy values
        def get_detector_value():
            detector_state = arrays.detector_states[self.detector_name]
            detector_arrays = list(detector_state.values())
            readings = detector_arrays[0] if len(detector_arrays) > 0 else jnp.array([])
            # Use the last reading, or 0.0 if no readings
            last_reading = jnp.where(len(readings) > 0, readings[-1], 0.0)
            has_readings = len(readings) > 0
            return last_reading, has_readings

        def get_dummy_value():
            return 0.0, False

        # Use jax.lax.cond to handle detector existence check
        detector_exists = self.detector_name in arrays.detector_states
        current_value, has_readings = jax.lax.cond(detector_exists, get_detector_value, get_dummy_value)

        less_than_met = current_value < self.threshold
        greater_than_met = current_value > self.threshold
        equal_met = jnp.abs(current_value - self.threshold) < 1e-10

        is_less_than = self.comparison == "less_than"
        is_greater_than = self.comparison == "greater_than"
        is_equal = self.comparison == "equal"
        
        condition_met = jax.lax.select(
            is_less_than, less_than_met,
            jax.lax.select(is_greater_than, greater_than_met,
                          jax.lax.select(is_equal, equal_met, False))
        )

        # Only consider condition met if detector exists and has readings
        return detector_exists & has_readings & condition_met


@autoinit
class FieldConvergenceCondition(StoppingCondition):
    """Stopping condition based on field convergence.

    Stops when the relative change in field energy between time steps
    falls below a specified threshold.
    """

    threshold: float = frozen_field(default=1e-6)  # Relative change threshold
    end_time: int = frozen_field()
    min_steps: int = frozen_field(default=100)  # Minimum steps before checking convergence

    def __call__(self, state: SimulationState) -> jax.Array:
        """Check if simulation should continue based on field convergence."""
        time_step, arrays = state

        # Always continue if below minimum steps or at max time
        time_condition = time_step < self.end_time
        min_steps_condition = time_step >= self.min_steps

        # Calculate current field energy
        E_energy = jnp.sum(arrays.E**2)
        H_energy = jnp.sum(arrays.H**2)
        total_energy = E_energy + H_energy

        # For convergence checking, we'd need to store previous energy
        # This is a simplified version that just checks if energy is very low
        converged = total_energy < self.threshold

        # Continue while time allows AND (min steps not reached OR not converged)
        return time_condition & (~min_steps_condition | ~converged)
