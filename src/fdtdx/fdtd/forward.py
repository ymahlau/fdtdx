import jax

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.update import collect_interfaces, update_detector_states, update_E, update_H
from fdtdx.interfaces.state import RecordingState
from fdtdx.objects.detectors.detector import DetectorState


def forward_single_args_wrapper(
    time_step: jax.Array,
    E: jax.Array,
    H: jax.Array,
    psi_E: jax.Array,
    psi_H: jax.Array,
    alpha: jax.Array,
    kappa: jax.Array,
    sigma: jax.Array,
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array,
    dispersive_P_curr: jax.Array | None,
    dispersive_P_prev: jax.Array | None,
    detector_states: dict[str, DetectorState],
    recording_state: RecordingState | None,
    dispersive_c1: jax.Array | None = None,
    dispersive_c2: jax.Array | None = None,
    dispersive_c3: jax.Array | None = None,
    *,
    config: SimulationConfig,
    objects: ObjectContainer,
    key: jax.Array,
    record_detectors: bool,
    record_boundaries: bool,
    simulate_boundaries: bool,
    arrays_template: ArrayContainer | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | float,
    jax.Array | None,
    jax.Array | None,
    dict[str, DetectorState],
    RecordingState | None,
]:
    # ``arrays_template`` carries non-differentiated fields (electric/magnetic
    # conductivity) that are closure-captured from the outer simulation state.
    # The dispersive coefficient arrays (c1, c2, c3) may be passed either as
    # primal positional arguments (so the outer ``jax.vjp`` treats them as
    # differentiable primals — this is the code path enabled by
    # ``GradientConfig.differentiate_dispersion=True``) or left as ``None`` to
    # fall back to the template (non-differentiated closure capture).
    if arrays_template is None:
        electric_conductivity = None
        magnetic_conductivity = None
        dispersive_inv_c2 = None
    else:
        electric_conductivity = arrays_template.electric_conductivity
        magnetic_conductivity = arrays_template.magnetic_conductivity
        # ``dispersive_inv_c2`` is always pulled from the template — it's a cached
        # reciprocal of c2, never a primal VJP arg.
        dispersive_inv_c2 = arrays_template.dispersive_inv_c2

    if dispersive_c1 is None and arrays_template is not None:
        dispersive_c1 = arrays_template.dispersive_c1
        dispersive_c2 = arrays_template.dispersive_c2
        dispersive_c3 = arrays_template.dispersive_c3
    arr = ArrayContainer(
        E=E,
        H=H,
        psi_E=psi_E,
        psi_H=psi_H,
        alpha=alpha,
        kappa=kappa,
        sigma=sigma,
        inv_permittivities=inv_permittivities,
        inv_permeabilities=inv_permeabilities,
        detector_states=detector_states,
        recording_state=recording_state,
        electric_conductivity=electric_conductivity,
        magnetic_conductivity=magnetic_conductivity,
        dispersive_P_curr=dispersive_P_curr,
        dispersive_P_prev=dispersive_P_prev,
        dispersive_c1=dispersive_c1,
        dispersive_c2=dispersive_c2,
        dispersive_c3=dispersive_c3,
        dispersive_inv_c2=dispersive_inv_c2,
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
        state[1].psi_E,
        state[1].psi_H,
        state[1].alpha,
        state[1].kappa,
        state[1].sigma,
        state[1].inv_permittivities,
        state[1].inv_permeabilities,
        state[1].dispersive_P_curr,
        state[1].dispersive_P_prev,
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
            config=config,
            H_prev=H_prev,
            inverse=False,
        )

    next_state = (time_step + 1, arrays)
    return next_state
