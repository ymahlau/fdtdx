import math
from typing import Literal

import jax
import jax.numpy as jnp
from loguru import logger

from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_field
from fdtdx.core.physics import constants
from fdtdx.interfaces.recorder import Recorder


class GradientConfig(ExtendedTreeClass):
    """Configuration for gradient computation in simulations.

    This class handles settings for automatic differentiation, supporting either
    invertible differentiation with a recorder or checkpointing-based differentiation.

    Args:
        recorder: Optional recorder for invertible differentiation. If provided,
            invertible differentiation will be used.
        num_checkpoints: Optional number of checkpoints for checkpointing-based
            differentiation. If provided, checkpointing will be used.

    Raises:
        Exception: If both recorder and num_checkpoints are provided, or if neither
            is provided.
    """

    def __init__(
        self,
        recorder: Recorder | None = None,  # if not none, use invertible diff
        num_checkpoints: int | None = None,
    ):
        self.recorder = recorder
        self.num_checkpoints = num_checkpoints
        if self.recorder is not None and self.num_checkpoints is not None:
            raise Exception("Cannot use both invertible and checkpointing autodiff!")
        if self.recorder is None and self.num_checkpoints is None:
            raise Exception("Need either recorder or checkpoints to define autograd!")


@extended_autoinit
class SimulationConfig(ExtendedTreeClass):
    """Configuration settings for FDTD simulations.

    This class contains all the parameters needed to configure and run an FDTD
    simulation, including spatial and temporal discretization, hardware backend,
    and gradient computation settings.

    Attributes:
        time: Total simulation time in seconds.
        resolution: Spatial resolution of the simulation grid in meters.
        backend: Computation backend ('gpu', 'tpu', or 'cpu').
        dtype: Data type for numerical computations.
        courant_factor: Safety factor for the Courant condition (default: 0.99).
        gradient_config: Optional configuration for gradient computation.
    """

    time: float
    resolution: float
    backend: Literal["gpu", "tpu", "cpu"] = frozen_field(default="gpu")
    dtype: jnp.dtype = frozen_field(default=jnp.float32)
    courant_factor: float = 0.99
    gradient_config: GradientConfig | None = None

    def __post_init__(self):
        if self.backend in ["gpu", "tpu"]:
            # Try to initialize GPU
            try:
                jax.devices(self.backend)
                logger.info(f"{str.upper(self.backend)} found and will be used for computations")
                jax.config.update("jax_platform_name", self.backend)
            except RuntimeError:
                logger.warning(f"{str.upper(self.backend)} not found, falling back to CPU!")
                self.backend = "cpu"
                jax.config.update("jax_platform_name", "cpu")
        else:
            jax.config.update("jax_platform_name", "cpu")

    @property
    def courant_number(self) -> float:
        """Calculate the Courant number for the simulation.

        The Courant number is a dimensionless quantity that determines stability
        of the FDTD simulation. It represents the ratio of the physical propagation
        speed to the numerical propagation speed.

        Returns:
            float: The Courant number, scaled by the courant_factor and normalized
                for 3D simulations.
        """
        return self.courant_factor / math.sqrt(3)

    @property
    def time_step_duration(self) -> float:
        """Calculate the duration of a single time step.

        The time step duration is determined by the Courant condition to ensure
        numerical stability. It depends on the spatial resolution and the speed
        of light.

        Returns:
            float: Time step duration in seconds, calculated using the Courant
                condition and spatial resolution.
        """
        return self.courant_number * self.resolution / constants.c

    @property
    def time_steps_total(self) -> int:
        """Calculate the total number of time steps for the simulation.

        Determines how many discrete time steps are needed to simulate the
        specified total simulation time, based on the time step duration.

        Returns:
            int: Total number of time steps needed to reach the specified
                simulation time.
        """
        return round(self.time / self.time_step_duration)

    @property
    def max_travel_distance(self) -> float:
        """Calculate the maximum distance light can travel during the simulation.

        This represents the theoretical maximum distance that light could travel
        through the simulation volume, useful for determining if the simulation
        time is sufficient for light to traverse the entire domain.

        Returns:
            float: Maximum travel distance in meters, based on the speed of light
                and total simulation time.
        """
        return constants.c * self.time

    @property
    def only_forward(self) -> bool:
        """Check if the simulation is forward-only (no gradient computation).

        Forward-only simulations don't compute gradients and are used when only
        the forward propagation of electromagnetic fields is needed, without
        optimization.

        Returns:
            bool: True if no gradient configuration is specified, False otherwise.
        """
        return self.gradient_config is None

    @property
    def invertible_optimization(self) -> bool:
        """Check if invertible optimization is enabled.

        Invertible optimization uses time-reversibility of Maxwell's equations
        to compute gradients with reduced memory requirements compared to
        checkpointing-based methods.

        Returns:
            bool: True if gradient computation uses invertible differentiation
                (recorder is specified), False otherwise.
        """
        if self.gradient_config is None:
            return False
        return self.gradient_config.recorder is not None


DUMMY_SIMULATION_CONFIG = SimulationConfig(
    time=-1,
    resolution=-1,
)
