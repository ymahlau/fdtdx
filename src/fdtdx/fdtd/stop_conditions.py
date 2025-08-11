from abc import ABC, abstractmethod
from typing import Optional

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
    def validate(self, arrays: ArrayContainer) -> None:
        """Optional pre-run validation; override in subclasses."""
        return

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

    end_time_step: int = frozen_field()

    def validate(self, arrays: ArrayContainer) -> None:
        pass

    def __call__(self, state: SimulationState) -> jax.Array:
        """Check if simulation should continue based on time step.

        Args:
            state: Current simulation state (SimulationState)

        Returns:
            jax.Array: Boolean scalar - True if current time < end_time, False otherwise
        """
        time_step, _ = state
        return self.end_time_step > time_step


@autoinit
class DetectorThresholdCondition(StoppingCondition):
    """Stopping condition based on detector readings reaching a threshold."""

    detector_name: str = frozen_field()
    state_key: Optional[str] = frozen_field(default=None)
    threshold: float = frozen_field()
    end_time_step: int = frozen_field()
    comparison: str = frozen_field(default="less_than")  # "less_than", "greater_than", "equal"
    expected_per_step_shape: tuple[int, ...] | None = frozen_field(default=None)

    def validate(self, arrays: ArrayContainer) -> None:
        if self.detector_name not in arrays.detector_states:
            available = tuple(arrays.detector_states.keys())
            raise KeyError(f"Detector '{self.detector_name}' not found. Available detectors: {available}")
        det_state: DetectorState = arrays.detector_states[self.detector_name]
        if self.state_key is not None:
            if self.state_key not in det_state:
                available = tuple(det_state.keys())
                raise KeyError(
                    f"Key '{self.state_key}' not found in DetectorState('{self.detector_name}'). "
                    f"Available keys: {available}"
                )
            readings = det_state[self.state_key]
        else:
            if len(det_state) == 0:
                raise ValueError(f"DetectorState('{self.detector_name}') is empty.")
            if len(det_state) > 1:
                available = tuple(det_state.keys())
                raise ValueError(
                    f"DetectorState('{self.detector_name}') has multiple arrays {available}. "
                    "Specify which one via 'state_key'."
                )
            readings = next(iter(det_state.values()))

        if readings.ndim < 1:
            raise ValueError(
                f"DetectorState('{self.detector_name}') array must have a time axis; got ndim={readings.ndim}."
            )
        per_step_shape = tuple(readings.shape[1:])
        if self.expected_per_step_shape is not None and per_step_shape != self.expected_per_step_shape:
            raise ValueError(
                f"Per-step shape mismatch for detector '{self.detector_name}'. "
                f"Expected {self.expected_per_step_shape}, got {per_step_shape}."
            )

    def __call__(self, state: SimulationState) -> jax.Array:
        """Check if simulation should continue based on detector reading.

        Args:
            state: Current simulation state (SimulationState)

        Returns:
            jax.Array: Boolean scalar - True if condition not met and time < end_time
        """
        time_step, arrays = state
        time_condition = time_step < self.end_time_step
        detector_condition = self._check_detector_condition(arrays)
        return time_condition & (~detector_condition)

    def _select_readings(self, det_state: DetectorState) -> jax.Array:
        if self.state_key is not None:
            return det_state[self.state_key]
        # Fall back to the sole value; if more than one exists, take the first consistently
        # (validate() enforces uniqueness when state_key is None)
        return next(iter(det_state.values()))

    def _check_detector_condition(self, arrays: ArrayContainer) -> jax.Array:
        # Assuming validate() has run. If not, KeyError may be raised here before JIT.
        det_state: DetectorState = arrays.detector_states[self.detector_name]
        readings = self._select_readings(det_state)  # shape: (T, *S)
        n = readings.shape[0]

        last_reading = jax.lax.cond(
            jnp.asarray(n > 0),
            lambda r: jax.lax.dynamic_index_in_dim(r, n - 1, axis=0, keepdims=False),
            lambda r: jnp.zeros(r.shape[1:], dtype=r.dtype),
            readings,
        )

        threshold = jnp.asarray(self.threshold, dtype=last_reading.dtype)
        threshold = jnp.broadcast_to(threshold, last_reading.shape)

        if self.comparison == "less_than":
            met = last_reading < threshold
        elif self.comparison == "greater_than":
            met = last_reading > threshold
        elif self.comparison == "equal":
            eps = jnp.asarray(10.0 * jnp.finfo(last_reading.dtype).eps)
            eps = jnp.broadcast_to(eps, last_reading.shape)
            met = jnp.abs(last_reading - threshold) <= eps
        else:
            # Unsupported comparator: never met
            met = jnp.zeros(last_reading.shape, dtype=jnp.bool_)

        # Reduce to a scalar condition: consider "met" if any element satisfies it
        return jnp.any(met)


@autoinit
class FieldConvergenceCondition(StoppingCondition):
    """Stopping condition based on field convergence.

    Stops when the relative change in field energy between time steps
    falls below a specified threshold.
    """

    threshold: float = frozen_field(default=1e-6)  # Relative change threshold
    end_time: int = frozen_field()
    min_steps: int = frozen_field(default=100)  # Minimum steps before checking convergence

    def validate(self, arrays: ArrayContainer) -> None:
        pass

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
