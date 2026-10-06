import math
from typing import Literal

import jax
import jax.numpy as jnp
from loguru import logger

from fdtdx import constants
from fdtdx.core.grid import QuasiUniformGrid, RectilinearGrid, UniformGrid
from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field
from fdtdx.interfaces.recorder import Recorder
from fdtdx.interfaces.time_filter import TimeStepFilter
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

    #: Number of interior full-field checkpoints for the ``"reversible"`` method.
    #: The reversible backward pass reconstructs the field state by running the simulation in
    #: reverse; for lossy materials this reverse reconstruction can accumulate numerical
    #: error over the full trajectory. Setting this to ``k - 1`` partitions the run into ``k`` slices
    #: and stores a full-field checkpoint at each interior slice boundary during the forward pass. The
    #: backward pass then resets the reverse reconstruction to the exact checkpoint at every boundary,
    #: bounding the reconstruction drift to a single slice (``~time_steps_total / k`` steps) at the
    #: cost of O(k) field memory. The default ``0`` reproduces the classic single full reverse pass
    #: (no interior checkpoints; only the final field, which is available for free, is used). Ignored
    #: by the ``"checkpointed"`` method. Must not exceed ``time_steps_total - 1``.
    #:
    #: ``"auto"`` (``recording_mode="segmented"`` only): :func:`~fdtdx.place_objects` replaces it with the
    #: count that minimises the memory of the segmented backward pass (see
    #: :meth:`resolve_num_checkpoints_reversible`) and returns a config holding that number.
    num_checkpoints_reversible: int | Literal["auto"] = frozen_field(default=0)

    #: How the ``"reversible"`` method stores the PML interface values that its reverse pass replays.
    #:
    #: ``"full"`` (default): the forward pass records the interface values of every time step, so the
    #: recorder buffer holds ``time_steps_total`` entries.
    #:
    #: ``"segmented"``: the forward pass records nothing and only keeps the full-field checkpoints at the
    #: ``num_checkpoints_reversible`` interior slice boundaries. The backward pass re-simulates each slice
    #: from the checkpoint at its start to regenerate the interface record of that slice alone, then
    #: reverses it. The recorder buffer therefore holds a single slice
    #: (``ceil(time_steps_total / (num_checkpoints_reversible + 1))`` entries) at the cost of one extra
    #: forward pass, so the interface memory of the exact (uncompressed) gradient no longer grows with the
    #: length of the run. Requires ``num_checkpoints_reversible >= 1`` (or ``"auto"``) and a recorder without a
    #: :class:`~fdtdx.LinearReconstructEveryK` (or other time-step filter), whose sample times are
    #: defined over the whole run, and without compression modules that carry state between steps, since
    #: the slices are recorded out of order.
    recording_mode: Literal["full", "segmented"] = frozen_field(default="full")

    def __post_init__(self):
        if self.method == "reversible" and self.recorder is None:
            raise Exception("Need Recorder in gradient config to compute reversible gradients")
        if self.method == "checkpointed" and self.num_checkpoints is None:
            raise Exception("Need Checkpoint Number in gradient config to compute checkpointed gradients")
        if self.recording_mode not in ("full", "segmented"):
            raise Exception(f"recording_mode must be 'full' or 'segmented', got {self.recording_mode!r}")
        num_ckpt = self.num_checkpoints_reversible
        if isinstance(num_ckpt, str):
            if num_ckpt != "auto":
                raise Exception(f"num_checkpoints_reversible must be an integer or 'auto', got {num_ckpt!r}")
            if self.recording_mode != "segmented":
                raise Exception("num_checkpoints_reversible='auto' requires recording_mode='segmented'")
        elif num_ckpt < 0:
            raise Exception("num_checkpoints_reversible must be >= 0")
        if self.recording_mode == "segmented":
            if self.method != "reversible":
                raise Exception("recording_mode='segmented' requires method='reversible'")
            if isinstance(num_ckpt, int) and num_ckpt < 1:
                raise Exception("recording_mode='segmented' requires num_checkpoints_reversible >= 1")
            if self.recorder is not None and any(isinstance(m, TimeStepFilter) for m in self.recorder.modules):
                raise Exception("recording_mode='segmented' does not support time-step filters in the recorder")

    def recorder_time_steps(self, time_steps_total: int) -> int:
        """Number of time steps the recorder buffer has to hold for a run of ``time_steps_total`` steps.

        Args:
            time_steps_total (int): Total number of forward time steps of the simulation.

        Returns:
            int: ``time_steps_total`` in ``"full"`` recording mode, the length of the longest slice
            (``ceil(time_steps_total / (num_checkpoints_reversible + 1))``) in ``"segmented"`` mode.
        """
        if self.recording_mode == "segmented":
            num_ckpt = self.num_checkpoints_reversible
            if not isinstance(num_ckpt, int):
                raise Exception(
                    "num_checkpoints_reversible='auto' is resolved by place_objects; use the config it returns"
                )
            return -(-time_steps_total // (num_ckpt + 1))
        return time_steps_total

    def resolve_num_checkpoints_reversible(
        self,
        time_steps_total: int,
        record_bytes_per_step: int,
        field_state_bytes: int,
    ) -> "GradientConfig":
        """Replace ``num_checkpoints_reversible="auto"`` by the count that minimises segmented-mode memory.

        With ``n`` checkpoints the segmented backward pass holds ``n`` field states and two record buffers of
        ``ceil(time_steps_total / (n + 1))`` steps (the one created by :func:`~fdtdx.place_objects` and the one
        the backward pass writes). Their total is smallest at ``n + 1 = sqrt(2 * time_steps_total * I / F)``,
        with ``I`` the recorded bytes per step and ``F`` the bytes of one field state.

        Args:
            time_steps_total (int): Total number of forward time steps of the simulation.
            record_bytes_per_step (int): Bytes the recorder stores per time step (``I``).
            field_state_bytes (int): Bytes of one full field state, i.e. of one checkpoint (``F``).

        Returns:
            GradientConfig: A copy with the resolved count if it was ``"auto"``, otherwise this config.
        """
        if self.num_checkpoints_reversible != "auto":
            return self
        num_ckpt = round(math.sqrt(2 * time_steps_total * record_bytes_per_step / max(field_state_bytes, 1))) - 1
        return self.aset("num_checkpoints_reversible", max(1, min(num_ckpt, time_steps_total - 1)))


@autoinit
class SimulationConfig(TreeClass):
    """Configuration settings for FDTD simulations.

    This class contains all the parameters needed to configure and run an FDTD
    simulation, including spatial and temporal discretization, hardware backend,
    and gradient computation settings.

    """

    #: Total simulation time in seconds.
    time: float = frozen_field()

    #: Spatial grid configuration.
    #:
    #: ``UniformGrid`` is an unresolved policy used while the final volume shape
    #: is still being inferred.  ``RectilinearGrid`` is the realized solver grid
    #: with explicit physical edge coordinates.  Placement resolves policies to
    #: ``RectilinearGrid`` so compiled FDTD code has exactly one metric source.
    grid: UniformGrid | QuasiUniformGrid | RectilinearGrid = field()

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

    #: Per-axis mirror symmetry of the simulation, in the order (x, y, z).
    #: Each entry is one of ``{-1, 0, +1}``:
    #: ``0`` = no symmetry on this axis (default),
    #: ``-1`` = PEC (electric-wall) mirror on the axis center plane,
    #: ``+1`` = PMC (magnetic-wall) mirror on the axis center plane.
    #: When any entry is nonzero, :func:`fdtdx.place_objects` automatically reduces the
    #: domain to the symmetric half/quarter/octant (keeping the upper half along each
    #: symmetric axis) and clips every object onto that reduced grid. An electric plane
    #: lands on the reduced domain's min edge and gets a PEC wall there; a magnetic plane
    #: sits half a cell below it (sources and materials are rasterized per cell), where the
    #: zero field halo already is the exact mirror, so it gets no wall object. Mode sources
    #: and mode-overlap detectors solve on the mirrored full cross-section and restrict,
    #: rather than using the mode solver's own symmetric solve. The FDTD then runs on the
    #: reduced domain; call
    #: :func:`fdtdx.unfold_fields` / :func:`fdtdx.unfold_detector_states` afterwards to
    #: reconstruct the full-domain arrays. This is additive and independent of manually
    #: specifying PEC/PMC as ordinary boundaries via :class:`fdtdx.BoundaryConfig`.
    #: Each symmetric axis must resolve to an **even** number of grid cells (so the domain
    #: splits exactly down the middle and the unfolded result matches the full domain
    #: cell-for-cell); otherwise :func:`fdtdx.place_objects` raises a ``ValueError``.
    symmetry: tuple[int, int, int] = frozen_field(default=(0, 0, 0))

    #: Optional configuration for gradient computation.
    gradient_config: GradientConfig | None = field(default=None)

    def __post_init__(self):
        from jax import extend

        if len(self.symmetry) != 3 or any(s not in (-1, 0, 1) for s in self.symmetry):
            raise ValueError(
                f"config.symmetry must be a length-3 tuple with each entry in {{-1, 0, +1}} "
                f"(0=none, -1=PEC, +1=PMC), got {self.symmetry!r}"
            )

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
    def has_symmetry(self) -> bool:
        """Whether any axis requests mirror symmetry.

        Returns:
            bool: True if at least one entry of :attr:`symmetry` is nonzero, meaning the
                domain will be reduced and a PEC/PMC wall placed on the symmetry plane(s).
        """
        return any(s != 0 for s in self.symmetry)

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

    def resolve_grid(self, shape: tuple[int, int, int] | None = None) -> RectilinearGrid:
        """Return a concrete solver grid.

        Args:
            shape: Required when ``grid`` is an unresolved ``UniformGrid``.

        Returns:
            A concrete ``RectilinearGrid``.
        """
        if isinstance(self.grid, RectilinearGrid):
            return self.grid
        if shape is None:
            raise ValueError("A grid shape is required to resolve UniformGrid.")
        return self.grid.resolve(shape)

    @property
    def resolved_grid(self) -> RectilinearGrid | None:
        """Return the concrete solver grid, or ``None`` if not yet resolved.

        ``UniformGrid`` has no edge arrays until the simulation shape is known.
        Callers that need coordinates, areas, or volumes should use this
        property and fall back to ``uniform_spacing`` when it returns ``None``.
        """
        if isinstance(self.grid, RectilinearGrid):
            return self.grid
        return None

    @property
    def has_nonuniform_grid(self) -> bool:
        """Whether the realized solver grid is non-uniform."""
        grid = self.resolved_grid
        return grid is not None and not grid.is_uniform

    def uniform_spacing(self) -> float:
        """Return the uniform grid spacing.

        ``UniformGrid`` can answer this before placement.  ``RectilinearGrid``
        answers only when all spacings are equal and raises for non-uniform
        meshes, making unsupported scalar assumptions explicit.
        """
        if isinstance(self.grid, UniformGrid):
            return self.grid.spacing
        if isinstance(self.grid, QuasiUniformGrid):
            if self.grid.is_uniform:
                return self.grid.dx
            else:
                raise ValueError(
                    "QuasiUniformGrid has no single uniform spacing:"
                    f" ({self.grid.dx}, {self.grid.dy}, {self.grid.dz} differ). "
                )
        return self.grid.uniform_spacing  # RectilinearGrid — raises internally if non-uniform

    @property
    def time_step_duration(self) -> float:
        """Calculate the duration of a single time step.

        The time step duration is determined by the Courant condition to ensure
        numerical stability. Realized rectilinear grids use their smallest
        per-axis spacings. Unresolved uniform grids use their configured scalar
        spacing; unresolved quasi-uniform grids use their smallest per-axis
        spacing as a conservative CFL bound.

        Returns:
            float: Time step duration in seconds, calculated using the Courant
                condition and spatial resolution.
        """
        if isinstance(self.grid, RectilinearGrid):
            return self.grid.cfl_time_step(self.courant_factor)
        if isinstance(self.grid, UniformGrid):
            return self.courant_number * self.grid.spacing / constants.c
        if isinstance(self.grid, QuasiUniformGrid):
            return self.courant_number * self.grid.min_spacing / constants.c
        raise NotImplementedError(f"time_step_duration is not implemented for grid type {type(self.grid).__name__}.")

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
    grid=UniformGrid(spacing=1),
)
