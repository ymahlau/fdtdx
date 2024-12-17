import jax

from fdtdx.core.config import SimulationConfig
from fdtdx.fdtd.update import collect_interfaces, update_detector_states, update_E, update_H
from fdtdx.interfaces.state import RecordingState
from fdtdx.objects.boundaries.perfectly_matched_layer import BoundaryState
from fdtdx.objects.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.objects.detectors.detector import DetectorState


def forward_single_args_wrapper(
    time_step: jax.Array,
    E: jax.Array,
    H: jax.Array,
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array,
    boundary_states: dict[str, BoundaryState],
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
    jax.Array,
    dict[str, BoundaryState],
    dict[str, DetectorState],
    RecordingState | None,
]:
    """Wrapper function that unpacks ArrayContainer into individual arrays for JAX transformations.

    This function provides a JAX-compatible interface by handling individual arrays instead of
    container objects. It converts between the array-based interface required by JAX and the
    object-oriented ArrayContainer interface used by the rest of the FDTD implementation.

    Args:
        time_step: Current simulation time step
        E: Electric field array
        H: Magnetic field array
        inv_permittivities: Inverse permittivity values
        inv_permeabilities: Inverse permeability values
        boundary_states: PML boundary conditions state
        detector_states: States of field detectors
        recording_state: Optional state for recording field values
        config: Simulation configuration parameters
        objects: Container with sources and other simulation objects
        key: Random key for compression
        record_detectors: Whether to record detector values
        record_boundaries: Whether to record boundary values
        simulate_boundaries: Whether to apply PML boundary conditions

    Returns:
        Tuple containing:
            - Updated time step
            - Updated E field array
            - Updated H field array
            - Updated inverse permittivities
            - Updated inverse permeabilities
            - Updated boundary states
            - Updated detector states
            - Updated recording state
    """
    """Wrapper function that unpacks ArrayContainer into individual arrays for JAX transformations.
    
    This function provides a JAX-compatible interface by handling individual arrays instead of
    container objects. It converts between the array-based interface required by JAX and the
    object-oriented ArrayContainer interface used by the rest of the FDTD implementation.

    Args:
        time_step: Current simulation time step
        E: Electric field array
        H: Magnetic field array 
        inv_permittivities: Inverse permittivity values
        inv_permeabilities: Inverse permeability values
        boundary_states: PML boundary conditions state
        detector_states: States of field detectors
        recording_state: Optional state for recording field values
        config: Simulation configuration parameters
        objects: Container with sources and other simulation objects
        key: Random key for compression
        record_detectors: Whether to record detector values
        record_boundaries: Whether to record boundary values
        simulate_boundaries: Whether to apply PML boundary conditions

    Returns:
        Tuple containing the updated time step and field arrays
    """
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
        state: Current simulation state (time step and field values)
        config: Simulation configuration parameters
        objects: Container with sources, PML and other simulation objects
        key: Random key for compression
        record_detectors: Whether to record detector values
        record_boundaries: Whether to record boundary values for gradients
        simulate_boundaries: Whether to apply PML boundary conditions

    Returns:
        Updated simulation state for the next time step
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
