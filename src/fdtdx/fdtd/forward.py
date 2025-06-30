import jax

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.update import collect_interfaces, update_detector_states, update_E, update_H
from fdtdx.interfaces.state import RecordingState
from fdtdx.objects.boundaries.boundary import BaseBoundaryState
from fdtdx.objects.detectors.detector import DetectorState


def forward_single_args_wrapper(
    time_step: jax.Array,
    E: jax.Array,
    H: jax.Array,
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array,
    boundary_states: dict[str, BaseBoundaryState],
    detector_states: dict[str, DetectorState],
    recording_state: RecordingState | None,
    config: SimulationConfig,
    objects: ObjectContainer,
    key: jax.Array,
    record_detectors: bool,
    record_boundaries: bool,
    simulate_boundaries: bool,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | float,
    dict[str, BaseBoundaryState],
    dict[str, DetectorState],
    RecordingState | None,
]:
    # Wrapper function that unpacks ArrayContainer into individual arrays for JAX transformations.
    arr = ArrayContainer(
        E=E,
        H=H,
        inv_permittivities=inv_permittivities,
        inv_permeabilities=inv_permeabilities,
        boundary_states=boundary_states,
        detector_states=detector_states,
        recording_state=recording_state,
    )
    state = forward(
        state=(time_step, arr),
        config=config,
        objects=objects,
        key=key,
        record_detectors=record_detectors,
        record_boundaries=record_boundaries,
        simulate_boundaries=simulate_boundaries,
    )
    return (
        state[0],
        state[1].E,
        state[1].H,
        state[1].inv_permittivities,
        state[1].inv_permeabilities,
        state[1].boundary_states,
        state[1].detector_states,
        state[1].recording_state,
    )


def forward(
    state: SimulationState,
    config: SimulationConfig,
    objects: ObjectContainer,
    key: jax.Array,
    record_detectors: bool,
    record_boundaries: bool,
    simulate_boundaries: bool,
) -> SimulationState:
    """Performs one forward time step of the FDTD simulation.

    Implements the core FDTD update scheme based on Maxwell's equations discretized on the Yee grid.
    Updates include:
    1. Electric field update using curl of H field
    2. Magnetic field update using curl of E field
    3. Optional PML boundary conditions
    4. Optional detector state updates
    5. Optional recording of boundary values for gradient computation

    The implementation leverages JAX for automatic compilation and GPU acceleration.
    Field updates follow the standard staggered time stepping of the Yee scheme.

    Args:
        state (SimulationState): Current simulation state (time step and field values)
        config (SimulationConfig): Simulation configuration parameters
        objects (ObjectContainer): Container with sources, PML and other simulation objects
        key (jax.Array): Random key for compression
        record_detectors (bool): Whether to record detector values
        record_boundaries (bool): Whether to record boundary values for gradients
        simulate_boundaries (bool): Whether to apply PML boundary conditions

    Returns:
        SimulationState: Updated simulation state for the next time step
    """
    time_step, arrays = state
    H_prev = arrays.H
    arrays = update_E(
        time_step=time_step,
        arrays=arrays,
        objects=objects,
        config=config,
        simulate_boundaries=simulate_boundaries,
    )
    arrays = update_H(
        time_step=time_step,
        arrays=arrays,
        objects=objects,
        config=config,
        simulate_boundaries=simulate_boundaries,
    )

    if record_boundaries:
        arrays = jax.lax.stop_gradient(
            collect_interfaces(
                time_step=time_step,
                arrays=arrays,
                objects=objects,
                config=config,
                key=key,
            )
        )

    if record_detectors:
        arrays = update_detector_states(
            time_step=time_step,
            arrays=arrays,
            objects=objects,
            H_prev=H_prev,
            inverse=False,
        )

    next_state = (time_step + 1, arrays)
    return next_state
