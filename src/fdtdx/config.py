import math
from typing import Literal

import jax
import jax.numpy as jnp
from loguru import logger

from fdtdx import constants
from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field
from fdtdx.interfaces.recorder import Recorder
from fdtdx.typing import BackendOption


@autoinit
class GradientConfig(TreeClass):
    """Configuration for gradient computation in simulations.

    This class handles settings for automatic differentiation, supporting either
    invertible differentiation with a recorder or checkpointing-based differentiation.

    Attributes:
        method (Literal["reversible", "checkpointed"], optional): Method for gradient computation.
            Can be either "reversible" when using the time reversible autodiff, or "checkpointed" for the exact
            checkpointing algorithm.
        recorder (Recorder | None, optional): Optional recorder for invertible differentiation. Needs to be provided for
            reversible autodiff. Defaults to None
        num_checkpoints (int | None, optional): Optional number of checkpoints for checkpointing-based
            differentiation. Needs to be provided for checkpointing gradient computation. Defaults to None.

    """

    method: Literal["reversible", "checkpointed"] = frozen_field(default="reversible")
    recorder: Recorder | None = field(default=None)
    num_checkpoints: int | None = frozen_field(default=None)

    def __post_init__(self):
        if self.method == "reversible" and self.recorder is None:
            raise Exception("Need Recorder in gradient config to compute reversible gradients")
        if self.method == "checkpointed" and self.num_checkpoints is None:
            raise Exception("Need Checkpoint Number in gradient config to compute checkpointed gradients")


@autoinit
class SimulationConfig(TreeClass):
    """Configuration settings for FDTD simulations.

    This class contains all the parameters needed to configure and run an FDTD
    simulation, including spatial and temporal discretization, hardware backend,
    and gradient computation settings.

    Attributes:
        time (float): Total simulation time in seconds.
        resolution (float): Spatial resolution of the simulation grid in meters.
        backend (BackendOption, optional): Computation backend ('gpu', 'tpu', 'cpu' or 'METAL'). Defaults to "gpu".
        dtype (jnp.dtype, optional): Data type for numerical computations. Defaults to jnp.float32.
        courant_factor (float, optional): Safety factor for the Courant condition (default: 0.99).
        gradient_config (GradientConfig | None, optional): Optional configuration for gradient computation.
    """

    time: float = frozen_field()
    resolution: float = frozen_field()
    backend: BackendOption = frozen_field(default="gpu")
    dtype: jnp.dtype = frozen_field(default=jnp.float32)
    courant_factor: float = frozen_field(default=0.99)
    gradient_config: GradientConfig | None = field(default=None)

    def __post_init__(self):
        from jax import extend

        current_platform = extend.backend.get_backend().platform

        if current_platform == "METAL" and self.backend == "gpu":
            self.backend = "METAL"

        if self.backend == "METAL":
            try:
                jax.devices()
                if __name__ == "__main__":
                    logger.info("METAL device found and will be used for computations")
                jax.config.update("jax_platform_name", "metal")
            except RuntimeError:
                if __name__ == "__main__":
                    logger.warning("METAL initialization failed, falling back to CPU!")
                self.backend = "cpu"
        elif self.backend in ["gpu", "tpu"]:
            try:
                jax.devices(self.backend)
                if __name__ == "__main__":
                    logger.info(f"{str.upper(self.backend)} found and will be used for computations")
                jax.config.update("jax_platform_name", self.backend)
            except RuntimeError:
                if __name__ == "__main__":
                    logger.warning(f"{str.upper(self.backend)} not found, falling back to CPU!")
                self.backend = "cpu"

        if self.backend == "cpu":
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
