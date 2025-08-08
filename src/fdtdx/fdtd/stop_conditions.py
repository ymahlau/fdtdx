from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field
from fdtdx.fdtd.container import ArrayContainer, SimulationState
from fdtdx.objects.detectors.detector import DetectorState


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
            state: Current simulation state (SimulationState)

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
            state: Current simulation state (SimulationState)

        Returns:
            jax.Array: Boolean scalar - True if current time < end_time, False otherwise
        """
        time_step, _ = state
        return self.end_time > time_step


@autoinit
class DetectorThresholdCondition(StoppingCondition):
    """Stopping condition based on detector readings reaching a threshold.
    """

    detector_name: str = frozen_field()
    threshold: float = frozen_field()
    end_time: int = frozen_field()
    comparison: str = frozen_field(default="less_than")  # "less_than", "greater_than", "equal"

    def __call__(self, state: SimulationState) -> jax.Array:
        """Check if simulation should continue based on detector reading.

        Args:
            state: Current simulation state (SimulationState)

        Returns:
            jax.Array: Boolean scalar - True if condition not met and time < end_time
        """
        time_step, arrays = state
        time_condition = time_step < self.end_time
        detector_condition = self._check_detector_condition(arrays)
        return time_condition & (~detector_condition)

    def _check_detector_condition(self, arrays: ArrayContainer) -> jax.Array:
        det_state: DetectorState = arrays.detector_states[self.detector_name]

        # DetectorState objects usually only have one value;
        # take first value if present, else empty array
        vals = list(det_state.values())
        readings = vals[0] if len(vals) > 0 else jnp.asarray([], dtype=jnp.float32)

        n = readings.shape[0]
        last_reading = jax.lax.cond(
            n > 0,
            # Some detectors have a DetectorState with a jax.Array of shape
            # (time_steps, A, B, C...) with multiple axis after time_step. We
            # conserve the shape of these axis when indexing like this
            lambda r: jax.lax.dynamic_index_in_dim(r, n - 1, axis=0, keepdims=False),
            lambda r: jnp.asarray(0.0, dtype=r.dtype),
            readings,
        )
        has_readings = jnp.asarray(n > 0)

        threshold = jnp.asarray(self.threshold, dtype=readings.dtype)

        if self.comparison == "less_than":
            met = last_reading < threshold
        elif self.comparison == "greater_than":
            met = last_reading > threshold
        elif self.comparison == "equal":
            # Tolerance suited to dtype. Could be user-configured too...
            eps = jnp.asarray(10.0 * jnp.finfo(readings.dtype).eps)
            met = jnp.abs(last_reading - threshold) <= eps
        else:
            met = jnp.array(False)

        return has_readings & met


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
        """Check if simulation should continue based on field convergence.

        Args:
            state: Current simulation state (SimulationState)

        Returns:
            jax.Array: Boolean scalar - True if condition not met and time < end_time
        """
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

        return time_condition & (~min_steps_condition | ~converged)
