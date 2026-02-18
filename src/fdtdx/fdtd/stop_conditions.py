from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field, frozen_private_field
from fdtdx.core.physics.metrics import compute_energy
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.fdtd.container import DetectorState, ObjectContainer, SimulationState


@autoinit
class StoppingCondition(TreeClass, ABC):
    """Abstract base class for defining custom stopping conditions in simulations.

    This class provides the interface for implementing custom termination criteria
    that can depend on simulation state, detector readings, field values, etc.
    """

    @abstractmethod
    def setup(self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer) -> Self:
        """Preparing condition for its use in the simulation."""
        return self

    @abstractmethod
    def _validate(self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer) -> None:
        """Pre-run validation; override in subclasses."""
        pass

    @abstractmethod
    def __call__(self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer) -> jax.Array:
        """Evaluate the stopping condition.

        Args:
            state (SimulationState): Current simulation state
            config (SimulationConfig): Configuration of the simulation
            objects (ObjectContainer): Objects in simulation

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
    """

    def setup(self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer) -> Self:
        self._validate(state, config, objects)
        return self

    def _validate(self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer) -> None:
        pass

    def __call__(self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer) -> jax.Array:
        """Check if simulation should continue based on time step.

        Args:
            state (SimulationState): Current simulation state.
            config (SimulationConfig): Configuration of the simulation.
            objects (ObjectContainer): Objects in simulation.

        Returns:
            jax.Array: Boolean scalar - True if current time < end_time, False otherwise
        """
        curr_time_step, _ = state
        return config.time_steps_total > curr_time_step


@autoinit
class EnergyThresholdCondition(StoppingCondition):
    """This condition stops a simulation when the total energy in the simulation volume
    falls below a specified threshold. A minimum number of steps can be enforced
    before convergence checks are applied, and a hard cutoff is imposed at
    ``SimulationConfig.time_steps_total``. This condition is intended to be used with
    pulsed sources, where the total amount of energy input in the volume is
    finite, therefore total energy in the system is expected to converge towards zero eventually.

    Attributes:
        threshold (float, optional): Lower energy threshold for determining
            simulation termination. Defaults to ``1e-6``.
        min_steps (int, optional): The minimum number of time steps that must be
            completed before convergence checking begins. Defaults to
            ``0.1 * SimulationConfig.time_steps_total``.
        max_steps (int, optional): The maximum number of time steps in a
            simulation, regardless of the condition being fulfilled. Defaults to
            ``SimulationConfig.time_steps_total``.
    """

    threshold: float = frozen_field(default=1e-6)
    min_steps: int | None = frozen_field(default=None)
    max_steps: int | None = frozen_field(default=None)

    def setup(self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer) -> Self:
        self = self.aset(
            "max_steps", config.time_steps_total if self.max_steps is None else self.max_steps, create_new_ok=True
        )
        self = self.aset(
            "min_steps",
            int(round(config.time_steps_total * 0.1)) if self.min_steps is None else self.min_steps,
            create_new_ok=True,
        )
        self._validate(state, config, objects)

        return self

    def _validate(self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer) -> None:
        if self.threshold <= 0:
            raise ValueError(f"Energy threshold must be positive, got {self.threshold}.")

        if self.min_steps is not None and self.min_steps < 0:
            raise ValueError(f"Minimum steps must be non-negative, got {self.min_steps}.")

    def __call__(self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer) -> jax.Array:
        """Check if simulation should continue based on energy reaching a lower threshold.
        Energy checks are only able to terminate the simulation after the time step is larger than ``min_steps``.

        Args:
            state (SimulationState): Current simulation state.
            config (SimulationConfig): Configuration of the simulation.
            objects (ObjectContainer): Objects in simulation.

        Returns:
            jax.Array: Boolean scalar - True if condition not met and curr_time_step < max_steps.
        """
        if self.max_steps is None or self.min_steps is None:
            raise RuntimeError("EnergyThresholdCondition.setup() must be calle before use. ")
        curr_time_step, arrays = state
        time_condition = curr_time_step < self.max_steps
        min_steps_condition = curr_time_step < self.min_steps
        total_energy = jnp.sum(compute_energy(arrays.E, arrays.H, arrays.inv_permittivities, arrays.inv_permeabilities))
        converged = total_energy < self.threshold

        return time_condition & (min_steps_condition | ~converged)


@autoinit
class DetectorConvergenceCondition(StoppingCondition):
    """Stopping condition based on convergence of values read off by a
    Detector with ``reduce_volume=True`` and number of readings equal to
    number of time steps (with ``switch=OnOffSwitch()``).

    This stopping condition starts off estimating convergence of values read off by a
    user-specified Detector by considering two subsets of its readings:
      (i) a number of previous periods [t - (prev_periods + 1)*spp, t - spp), and
      (ii) the last full period [t - spp, t),
    where t is the current time step, spp is the number of samples of the period
    over which the reading changes (it's usually half the period of the CW
    source in the case of PoyntingFluxDetector and EnergyDetector).
    Once it has them, it computes the mean over all periods of (i) such that the
    result has the same shape as (ii). After this, their Fourier transform is
    computed, and then the L2 norm of these two transforms, and stops the
    simulation once this norm falls below a specified threshold.
    A minimum number of steps are enforced before convergence checks are
    applied, and a hard cutoff of the simulation is imposed when the time step
    is equal to ``max_steps``.

    Attributes:
        detector_name (str): The name of the Detector from which
            readings will be obtained in order to determine when to
            halt the simulation.
        wave_character (WaveCharacter): The WaveCharacter fed to the source in the simulation.
            This is used to calculate the period over which the detector reading changes.
            For PoyntingFluxDetector and EnergyDetector, this is typically half
            the period of the CW source specified in this WaveCharacter.
        prev_periods (int, optional): Number of previous full periods used as a reference (>=1).
        threshold (float, optional): Relative change threshold for determining
            convergence. Defaults to ``1e-6``.
        min_steps (int, optional): The minimum number of time steps that the
            simulation must go on for before convergence checking begins. Defaults
            to ``(prev_periods + 1) * spp``, where ``spp`` is samples per period of
            the source. This ensures that there are enough readings to compute the
            Fourier transforms.
        max_steps (int, optional): The maximum number of time steps before the
            simulation is stopped, regardless of convergence.
            Defaults to ``SimulationConfig.time_steps_total``.
    """

    detector_name: str = frozen_field()
    wave_character: WaveCharacter = frozen_field()
    prev_periods: int = frozen_field(default=4)
    threshold: float = frozen_field(default=1e-6)
    min_steps: int | None = frozen_field(default=None)
    max_steps: int | None = frozen_field(default=None)
    _spp: int | None = frozen_private_field(default=None)  # type: ignore

    def setup(self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer) -> Self:
        """Setting up internal attributes and validating inputs."""
        spp = int(round(self.wave_character.get_period() / config.time_step_duration))
        self = self.aset("_spp", spp, create_new_ok=True)
        self = self.aset(
            "max_steps", config.time_steps_total if self.max_steps is None else self.max_steps, create_new_ok=True
        )
        self = self.aset(
            "min_steps",
            int(round((self.prev_periods + 1) * spp)) if self.min_steps is None else self.min_steps,
            create_new_ok=True,
        )
        self._validate(state, config, objects)

        return self

    def _validate(self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer) -> None:
        if self._spp is None:
            raise RuntimeError("DetectorConvergenceCondition: _spp was not initialized. Run setup() first.")
        _, arrays = state

        if (self.prev_periods + 1) * self._spp > config.time_steps_total:
            raise ValueError(
                "Number of samples over which DetectorConvergenceCondition computes is "
                "greater than the number of time steps in the simulation. "
                "Increase the time over which the simulation runs in SimulationConfig, "
                "decrease prev_periods, or use a source with a shorter period."
            )

        if self.detector_name not in arrays.detector_states:
            available = tuple(arrays.detector_states.keys())
            raise KeyError(f"Detector '{self.detector_name}' not found. Available detectors: {available}")
        det_state: DetectorState = arrays.detector_states[self.detector_name]

        if all(k not in det_state for k in ("energy", "poynting_flux", "fields")):
            available = tuple(det_state.keys())
            raise KeyError(
                f"Chosen detector does not seem to be an EnergyDetector, PoyntingFluxDetector, FieldDetector.\n "
                f"Available keys: {available}"
            )
        readings = next(iter(det_state.values()))

        if readings.ndim != 2:
            raise ValueError(
                f"The selected detector must have reduce_volume=True. Therefore, "
                f"the DetectorState('{self.detector_name}') array must have two "
                f"dimensions; got ndim={readings.ndim}.\n"
            )

        if readings.shape[0] != config.time_steps_total:
            raise ValueError(
                f"The number of detector readings must be exactly the same as the number of time steps in the simulation. "
                f"Number of detector readings: {readings.shape[0]}, time steps: {config.time_steps_total}.\n"
            )

        if self.prev_periods < 1:
            raise ValueError(f"prev_periods must be >= 1; got {self.prev_periods}.")

        if self.threshold < 0:
            raise ValueError(f"Detector convergence threshold must be non-negative, got {self.threshold}.")
        if self.min_steps is None:
            raise RuntimeError("DetectorConvergenceCondition: min_steps was not initialized.")
        if self.min_steps is not None and self.min_steps < (self.prev_periods + 1) * self._spp:
            raise ValueError(
                "min_steps must be larger than the number of steps used to compute convergence, "
                f"got {self.min_steps}, need more than {(self.prev_periods + 1) * self._spp}. "
                "You can also decrease prev_periods to match min_steps, or you can leave min_steps unset, "
                "as a suitable default will be used."
            )

    def __call__(self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer) -> jax.Array:
        """Check if simulation should continue based on the L2 norm between Fourier
        transforms of the last period and an average over a number of previous
        periods ``prev_periods``.

        Args:
            state (SimulationState): Current simulation state.
            config (SimulationConfig): Configuration of the simulation.
            objects (ObjectContainer): Objects in simulation.

        Returns:
            jax.Array: Boolean scalar - True if condition not met and curr_time_step < max_steps.
        """
        if self._spp is None or self.min_steps is None or self.max_steps is None:
            raise RuntimeError("DetectorConvergenceCondition.setup() must be called before use.")

        # Assigning to local Variables
        spp = self._spp
        min_steps = self.min_steps

        threshold = self.threshold
        prev_periods = self.prev_periods

        curr_time_step, arrays = state
        converged: jnp.ndarray = jnp.array(False, dtype=bool)
        readings: jax.Array = next(iter(arrays.detector_states[self.detector_name].values()))

        # Always continue if below minimum steps, always stop if at end_step
        time_condition = curr_time_step < config.time_steps_total
        min_steps_condition = curr_time_step >= min_steps

        # Wrapping this in a func so we don't compute it until min_steps_condition == True
        def _compute_converged(_):
            start_ref = curr_time_step - (prev_periods + 1) * spp
            start_last = curr_time_step - spp

            # Clamp to valid bounds to avoid OOB under JIT (it doesn't like that)
            start_ref = jnp.clip(start_ref, 0, config.time_steps_total - prev_periods * spp)
            start_last = jnp.clip(start_last, 0, config.time_steps_total - spp)

            ref_2d = jax.lax.dynamic_slice(readings, (start_ref, 0), (prev_periods * spp, 1))
            last_2d = jax.lax.dynamic_slice(readings, (start_last, 0), (self._spp, 1))

            readings_ref = jnp.squeeze(ref_2d, axis=1)  # (k*spp,)
            readings_last = jnp.squeeze(last_2d, axis=1)  # (spp,)

            # Take the reference readings and average all prev_periods together (prev_periods*spp,) -> (spp,)
            ref_periods = readings_ref.reshape(self.prev_periods, self._spp)  # (prev_periods, spp)
            ref_mean = jnp.mean(ref_periods, axis=0)  # (spp,)

            fft_ref = jnp.fft.rfft(ref_mean, n=self._spp)  # (spp//2 + 1,)
            fft_last = jnp.fft.rfft(readings_last, n=self._spp)  # (spp//2 + 1,)

            spectra_distance = jnp.linalg.norm(jnp.abs(fft_ref) - jnp.abs(fft_last))
            return spectra_distance < threshold

        converged = jax.lax.cond(
            min_steps_condition,
            _compute_converged,
            lambda _: jnp.array(False, dtype=bool),
            operand=None,
        )

        return (~min_steps_condition) | (time_condition & (~converged))
