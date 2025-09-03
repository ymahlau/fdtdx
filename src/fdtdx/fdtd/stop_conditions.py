from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field
from fdtdx.fdtd.container import ArrayContainer, DetectorState, SimulationState, ObjectContainer


@autoinit
class StoppingCondition(TreeClass, ABC):
    """Abstract base class for defining custom stopping conditions in simulations.

    This class provides the interface for implementing custom termination criteria
    that can depend on simulation state, detector readings, field values, etc.
    """

    @abstractmethod
    def validate(self, arrays: ArrayContainer, objects: ObjectContainer) -> None:
        """Optional pre-run validation; override in subclasses."""
        return

    @abstractmethod
    def __call__(self, state: SimulationState, objects: ObjectContainer) -> jax.Array:
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

    Attributes:
        end_time_step (int): Time step at which simulation ends.
            Can be found at SimulationConfig.time_step_duration
    """

    end_time_step: int = frozen_field()

    def validate(self, arrays: ArrayContainer, objects: ObjectContainer) -> None:
        pass

    def __call__(self, state: SimulationState, objects: ObjectContainer) -> jax.Array:
        """Check if simulation should continue based on time step.

        Args:
            state: Current simulation state (SimulationState)

        Returns:
            jax.Array: Boolean scalar - True if current time < end_time, False otherwise
        """
        time_step, _ = state
        return self.end_time_step > time_step


@autoinit
class EnergyConvergenceCondition(StoppingCondition):
    """Stopping condition based on convergence of energy values read off by an
    EnergyDetector with `reduce_volume=True`.

    This condition stops a simulation when the relative change in field
    energy between time steps falls below a specified threshold. A minimum
    number of steps can be enforced before convergence checks are applied,
    and a hard cutoff is imposed at ``end_time``.

    Attributes:
        threshold (float): Relative change threshold for determining
            convergence. Defaults to ``1e-6``.
        end_time (int): The maximum number of time steps before the
            simulation is stopped, regardless of convergence.
        min_steps (int): The minimum number of time steps that must be
            completed before convergence checking begins. Defaults to
            ``100``.
        detector_name (str): The name of the EnergyDetector from which
            energy values will be obtained in order to determine the stopping condition.
    """

    threshold: float = frozen_field(default=1e-6)  # Relative change threshold
    end_time: int = frozen_field()
    min_steps: int = frozen_field(default=100)  # Minimum steps before checking convergence
    detector_name: str = frozen_field()

    def validate(self, arrays: ArrayContainer, objects: ObjectContainer) -> None:
        if self.detector_name not in arrays.detector_states:
            available = tuple(arrays.detector_states.keys())
            raise KeyError(f"Detector '{self.detector_name}' not found. Available detectors: {available}")
        det_state: DetectorState = arrays.detector_states[self.detector_name]
        if "energy_detector" not in det_state:
            available = tuple(det_state.keys())
            raise KeyError(
                f"Chosen detector does not seem to be an EnergyDetector, \"energy_detector\" key not found in DetectorState('{self.detector_name}'). "
                f"Available keys: {available}"
            )
        readings = det_state["energy"]

        if readings.ndim < 1:
            raise ValueError(
                f"DetectorState('{self.detector_name}') array must have a time axis; got ndim={readings.ndim}."
            )
        per_step_shape = tuple(readings.shape[1:])
        if per_step_shape != (1,):
            raise ValueError(
                f"Per-step shape mismatch for detector '{self.detector_name}'. "
                f"Expected {(1,)}, got {per_step_shape}."
            )

    def __call__(self, state: SimulationState, objects: ObjectContainer) -> jax.Array:
        """Check if simulation should continue based on field convergence.

        Args:
            state: Current simulation state (SimulationState)

        Returns:
            jax.Array: Boolean scalar - True if condition not met and time < end_time
        """
        time_step, arrays = state
        converged: jnp.ndarray = jnp.array(False, dtype=bool)

        # Always continue if below minimum steps or at max time
        time_condition = time_step < self.end_time
        min_steps_condition = time_step >= self.min_steps

        # Calculate current field energy
        E_energy = jnp.sum(arrays.E**2)
        H_energy = jnp.sum(arrays.H**2)
        curr_total_energy = E_energy + H_energy

        # Mask
        past_boundary = time_step > (self.min_steps - 1)

        # Compute with the *old* prev_total_energy
        converged = jnp.where(
            past_boundary,
            (curr_total_energy - prev_total_energy) < self.threshold,
            converged,
        )
        
        # Update prev_total_energy
        prev_total_energy = jnp.where(
            time_step >= (self.min_steps - 1),
            curr_total_energy,
            prev_total_energy,
        )

        return time_condition & (~min_steps_condition | ~converged)
