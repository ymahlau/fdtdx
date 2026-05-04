import math
from typing import Literal

import jax
import jax.numpy as jnp
from loguru import logger

from fdtdx import constants
from fdtdx.core.grid import RectilinearGrid
from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field
from fdtdx.interfaces.recorder import Recorder
from fdtdx.typing import BackendOption


@autoinit
class GradientConfig(TreeClass):
    """Configuration for gradient computation in simulations.

    This class handles settings for automatic differentiation, supporting either
    invertible differentiation with a recorder or checkpointing-based differentiation.

    """

    #: Method for gradient computation.
    #: Can be either "reversible" when using the time reversible autodiff, or "checkpointed" for the exact checkpointing algorithm.
    method: Literal["reversible", "checkpointed"] = frozen_field(default="reversible")

    #: Optional recorder for invertible differentiation. Needs to be provided for reversible autodiff. Defaults to None
    recorder: Recorder | None = field(default=None)

    #: Optional number of checkpoints for checkpointing-based differentiation.
    #: Needs to be provided for checkpointing gradient computation. Defaults to None.
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

    """

    #: Total simulation time in seconds.
    time: float = frozen_field()

    #: Spatial resolution of a legacy uniform simulation grid in meters.
    #:
    #: New internals should prefer ``grid``.  This field is kept as a public
    #: compatibility constructor while the codebase migrates to one canonical
    #: ``RectilinearGrid`` path.  When ``grid`` is present and non-uniform, operations
    #: that still rely on this scalar should fail loudly via ``require_uniform_grid``.
    resolution: float = frozen_field()

    #: Realized rectilinear grid metric used by compiled simulations.
    #:
    #: The grid is optional during initial configuration because the simulation
    #: volume can still be specified in physical units and resolved later during
    #: object placement.  Once objects are placed, ``place_objects`` attaches a
    #: concrete ``RectilinearGrid`` so downstream code has a single metric source.
    grid: RectilinearGrid | None = frozen_field(default=None)

    #: Computation backend ('gpu', 'tpu', 'cpu' or 'METAL'). Defaults to "gpu".
    backend: BackendOption = frozen_field(default="gpu")

    #:  Data type for numerical computations. Defaults to jnp.float32.
    dtype: jnp.dtype = frozen_field(default=jnp.float32)

    #: Whether to use complex-valued field arrays.
    #: None (default): auto-detect based on boundary conditions (e.g. Bloch).
    #: True: force complex fields (complex64 if dtype=float32, complex128 if dtype=float64).
    #: False: force real fields (raises error if Bloch boundaries are present).
    use_complex_fields: bool | None = frozen_field(default=None)

    #: Safety factor for the Courant condition (default: 0.99).
    courant_factor: float = frozen_field(default=0.99)

    #: Optional configuration for gradient computation.
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

    def require_grid(self, shape: tuple[int, int, int] | None = None) -> RectilinearGrid:
        """Return the configured grid, creating a uniform compatibility grid if needed.

        Args:
            shape: Required when the config was constructed with only
                ``resolution``.  The shape is normally the placed simulation
                volume shape.

        Returns:
            A concrete ``RectilinearGrid``.
        """
        if self.grid is not None:
            return self.grid
        if shape is None:
            raise ValueError("A grid shape is required to build a RectilinearGrid from scalar resolution.")
        return RectilinearGrid.uniform(shape=shape, spacing=self.resolution)

    def require_uniform_grid(self) -> float:
        """Return uniform spacing for legacy code that has not been metricized yet.

        This method marks scalar-resolution dependencies explicitly.  It preserves
        old behavior for uniform grids and raises for non-uniform grids, which is
        safer than silently applying a global spacing where local metrics are
        required.
        """
        if self.grid is None:
            return self.resolution
        return self.grid.uniform_spacing

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
        if self.grid is not None:
            return self.grid.cfl_time_step(self.courant_factor)
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
