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
    Detector with `reduce_volume=True` and number of readings equal to
    number of time steps.

    This stopping condition starts off estimating convergence of values read off by a
    user-specified Detector by considering two subsets of its readings:
      (i) the k previous full periods [t - (k+1)P, t - P), and
      (ii) the last full period [t - P, t),
    where t is the current time step, P is the number of samples of the period
    over which the reading changes (it's usually half the period of the CW
    source in the case of PoyntingFluxDetector and EnergyDetector).
    Once it has them, it computes the mean over all periods of (i) such that the
    result has the same shape as (ii). After this, their Fourier transform is
    computed, and then the L2 norm of these two transforms, and stops the
    simulation once this norm falls below a specified threshold.
    A minimum number of steps can be enforced before convergence
    checks are applied, and a hard cutoff is imposed when the time step is equal
    to ``end_step``.

    Attributes:
        k (int): Number of *previous* full periods used as a reference (>=1).
        spp (int): Number of samples per period. The period depends on the detector used.
            That is, if using a PoyntingFluxDetector or EnergyDetector, this period
            is half that of the CW source used. So, to obtain spp, do
            (cw_source_period / 2) / time_step_duration. The latter can be found
            at SimulationConfig.time_step_duration.
        threshold (float): Relative change threshold for determining
            convergence. Defaults to ``1e-6``.
        end_step (int): The maximum number of time steps before the
            simulation is stopped, regardless of convergence.
            Can be obtained from SimulationConfig.time_steps_total.
        min_steps (int): The minimum number of time steps that must be
            completed before convergence checking begins. A good value would be
            (k_periods + 1) * spp, where spp is samples per period. Defaults to
            ``100``.
        detector_name (str): The name of the Detector from which
            detector readings will be obtained in order to determine when to
            halt the simulation.
    """

    k: int = frozen_field(default=4)
    spp: int = frozen_field()
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

        if any(k not in det_state for k in ("energy", "poynting_flux", "fields")):
            available = tuple(det_state.keys())
            raise KeyError(
                f"Chosen detector does not seem to be an EnergyDetector, PoyntingFluxDetector, FieldDetector.\n "
                f"Available keys: {available}"
            )
        readings = next(iter(det_state.values()))

        if readings.ndim != 2:
            raise ValueError(
                f"The selected detector must have reduce_volume=True. Therefore, "
                f"DetectorState('{self.detector_name}') array must have two "
                f"dimensions; got ndim={readings.ndim}.\n"
            )
        
        if self.spp <= 0:
            raise ValueError(f"The number of samples per period must be larger than 0; got spp='{self.spp}'.")

        if readings.shape[0] != self.end_step:
            raise ValueError("The number of detector readings must be exactly the same as the number of time steps in the simulation.")

    def __call__(self, state: SimulationState, objects: ObjectContainer) -> jax.Array:
        """Check if simulation should continue based on L2 norm between Fourier
        transforms of last period and k previous periods.

        Args:
            state: Current simulation state (SimulationState)
            objects: Objects in simulation (ObjectContainer)

        Returns:
            jax.Array: Boolean scalar - True if condition not met and time < end_time
        """
        curr_time_step, arrays = state
        converged: jnp.ndarray = jnp.array(False, dtype=bool)
        readings: jax.Array = next(iter(arrays.detector_states[self.detector_name].values()))

        # Always continue if below minimum steps or below max time
        time_condition = curr_time_step < self.end_step
        min_steps_condition = curr_time_step >= self.min_steps

        # Wrapping this in a func so we don't compute it until min_steps_condition == True
        def _compute_converged(_):
            start_ref = curr_time_step - (self.k + 1) * self.spp
            start_last = curr_time_step - self.spp

            # Clamp to valid bounds to avoid OOB under JIT
            start_ref = jnp.clip(start_ref, 0, self.end_step - self.k * self.spp)
            start_last = jnp.clip(start_last, 0, self.end_step - self.spp)
        
            # Slicing up detector readings
            ref_2d = jax.lax.dynamic_slice(readings, (start_ref, 0), (self.k * self.spp, 1))
            last_2d = jax.lax.dynamic_slice(readings, (start_last, 0), (self.spp, 1))

            # Dropping singleton column
            readings_ref = jnp.squeeze(ref_2d, axis=1)   # (k*spp,)
            readings_last = jnp.squeeze(last_2d, axis=1)  # (spp,)
        
            # Take the reference readings and average all k periods together (k*spp,) -> (spp,)
            ref_periods = readings_ref.reshape(self.k, self.spp)  # (k, spp)
            ref_mean = jnp.mean(ref_periods, axis=0)  # (spp,)
        
            # FFTs of mean over k periods in reference, and last period
            fft_ref = jnp.fft.rfft(ref_mean, n=self.spp)  # (spp//2 + 1,)
            fft_last = jnp.fft.rfft(readings_last, n=self.spp)  # (spp//2 + 1,)
        
            # L2 norm
            spectra_distance = jnp.linalg.norm(jnp.abs(fft_ref) - jnp.abs(fft_last))
            return spectra_distance < self.threshold

        converged = jax.lax.cond(
            min_steps_condition,
            _compute_converged,
            lambda _: jnp.array(False, dtype=bool),
            operand=None,
        )

        return time_condition & (~min_steps_condition | ~converged)
