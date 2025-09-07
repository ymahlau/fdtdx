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
    def validate(self, state: SimulationState, objects: ObjectContainer) -> None:
        """Optional pre-run validation; override in subclasses."""
        return

    @abstractmethod
    def __call__(self, state: SimulationState, objects: ObjectContainer) -> jax.Array:
        """Evaluate the stopping condition.

        Args:
            state: Current simulation state (SimulationState)
            objects: Objects in simulation (ObjectContainer)

        Returns:
            jax.Array: Boolean scalar - True if simulation should continue, False if it should stop

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()


@autoinit
class TimeStepCondition(StoppingCondition):
    """Default stopping condition based on maximum time steps.

    This recreates the original behavior where simulation continues until
    a specified end time is reached.

    Attributes:
        end_step (int): Time step at which simulation ends.
            Can be found at SimulationConfig.time_step_duration
    """

    end_step: int = frozen_field()

    def validate(self, state: SimulationState, objects: ObjectContainer) -> None:
        pass

    def __call__(self, state: SimulationState, objects: ObjectContainer) -> jax.Array:
        """Check if simulation should continue based on time step.

        Args:
            state: Current simulation state (SimulationState)
            objects: Objects in simulation (ObjectContainer)

        Returns:
            jax.Array: Boolean scalar - True if current time < end_time, False otherwise
        """
        time_step, _ = state
        return self.end_step > time_step


@autoinit
class EnergyThresholdCondition(StoppingCondition):
    """This condition stops a simulation when the total energy in the simulation volume
    falls below a specified threshold. A minimum number of steps can be enforced
    before convergence checks are applied, and a hard cutoff is imposed at
    ``end_time``. This condition is intended to be used with pulsed sources, where
    the total amount of energy input in the volume is finite, therefore total energy
    in the system is expected to converge towards zero eventually.

    Attributes:
        threshold (float): Lower energy threshold for determining
            simulation termination. Defaults to ``1e-6``.
        end_step (int): The maximum number of time steps before the
            simulation is stopped, regardless of the threshold being reached.
        min_steps (int): The minimum number of time steps that must be
            completed before convergence checking begins. Defaults to
            ``100``.
    """

    threshold: float = frozen_field(default=1e-6)
    end_step: int = frozen_field()
    min_steps: int = frozen_field(default=100)

    def validate(self, state: SimulationState, objects: ObjectContainer) -> None:
        pass

    def __call__(self, state: SimulationState, objects: ObjectContainer) -> jax.Array:
        """Check if simulation should continue based on energy reaching a threshold.
        Energy checks are only able to terminate the simulation after the time step is larger than ``min_steps``.

        Args:
            state: Current simulation state (SimulationState)
            objects: Objects in simulation (ObjectContainer)

        Returns:
            jax.Array: Boolean scalar - True if condition not met and time < end_time
        """
        curr_time_step, arrays = state
        time_condition = curr_time_step < self.end_step
        total_energy = jnp.sum(arrays.E**2) + jnp.sum(arrays.H**2)
        converged = total_energy < self.threshold

        return time_condition & (~(curr_time_step < self.min_steps) | ~converged)


@autoinit
class DetectorConvergenceCondition(StoppingCondition):
    """Stopping condition based on convergence of values read off by a
    Detector with `reduce_volume=True`.

    This condition stops a simulation when the relative change in field
    energy between time steps falls below a specified threshold. A minimum
    number of steps can be enforced before convergence checks are applied,
    and a hard cutoff is imposed at ``end_time``.

    Attributes:
        threshold (float): Relative change threshold for determining
            convergence. Defaults to ``1e-6``.
        end_step (int): The maximum number of time steps before the
            simulation is stopped, regardless of convergence.
        min_steps (int): The minimum number of time steps that must be
            completed before convergence checking begins. Defaults to
            ``100``.
        detector_name (str): The name of the EnergyDetector from which
            energy values will be obtained in order to determine the stopping condition.
    """

    threshold: float = frozen_field(default=1e-6)
    end_step: int = frozen_field()
    min_steps: int = frozen_field(default=100)
    detector_name: str = frozen_field()

    def validate(self, state: SimulationState, objects: ObjectContainer) -> None:
        _, arrays = state
        if self.detector_name not in arrays.detector_states:
            available = tuple(arrays.detector_states.keys())
            raise KeyError(f"Detector '{self.detector_name}' not found. Available detectors: {available}")
        det_state: DetectorState = arrays.detector_states[self.detector_name]
        if ("energy" not in det_state) or ("poynting_flux" not in det_state) or ("fields" not in det_state):
            available = tuple(det_state.keys())
            raise KeyError(
                f"Chosen detector does not seem to be an EnergyDetector, PoyntingFluxDetector, FieldDetector"
                f"with reduce_volume=True, \"energy_detector\" key not found in DetectorState('{self.detector_name}').\n "
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
                f"Per-timestep shape mismatch for detector '{self.detector_name}'. "
                f"Expected {(1,)}, got {per_step_shape}."
            )

    def __call__(self, state: SimulationState, objects: ObjectContainer) -> jax.Array:
        """Check if simulation should continue based on energy convergence.

        Args:
            state: Current simulation state (SimulationState)
            objects: Objects in simulation (ObjectContainer)

        Returns:
            jax.Array: Boolean scalar - True if condition not met and time < end_time
        """
        curr_time_step, arrays = state
        converged: jnp.ndarray = jnp.array(False, dtype=bool)
        energy_readings: jax.Array = arrays.detector_states[self.detector_name]["energy"]

        # Always continue if below minimum steps or below max time
        time_condition = curr_time_step < self.end_step
        min_steps_condition = curr_time_step >= self.min_steps

        # Getting index of detector reading array from time step
        # Detector reading does not necessarily have same number of readings as time steps
        # due to an OnOffSwitch possibly being ingested
        mapping = objects[self.detector_name]._time_step_to_arr_idx
        time_steps = jnp.arange(self.end_step)
        valid_mask = (time_steps <= curr_time_step) & (mapping != -1)
        last_valid_pos = jnp.max(jnp.where(valid_mask, time_steps, -1))
        # Clamp to 0 if no valid index found
        last_valid_pos = jnp.maximum(last_valid_pos, 0)
        idx = jnp.take(mapping, last_valid_pos)

        prev_idx = jnp.maximum(idx - 1, 0)
        curr_total_energy = jax.lax.dynamic_index_in_dim(energy_readings, idx, axis=0, keepdims=False)
        prev_total_energy = jax.lax.dynamic_index_in_dim(energy_readings, prev_idx, axis=0, keepdims=False)

        # Mask -- we do not check convergence until past_boundary is true,
        # ergo we do not check it until after self.min_steps time steps have
        # passed
        past_boundary = curr_time_step > (self.min_steps - 1)

        converged = jnp.where(
            past_boundary,
            (curr_total_energy - prev_total_energy) < self.threshold,
            converged,
        )
        
        return time_condition & (~min_steps_condition | ~converged)
