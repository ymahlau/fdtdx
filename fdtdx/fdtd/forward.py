import jax

from fdtdx.core.config import SimulationConfig
from fdtdx.fdtd.update import update_E, update_H, update_detector_states, collect_interfaces
from fdtdx.objects.boundaries.perfectly_matched_layer import BoundaryState
from fdtdx.objects.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.objects.detectors.detector import DetectorState
from fdtdx.interfaces.state import RecordingState

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
):
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
        state[1].recording_state
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
        arrays = jax.lax.stop_gradient(collect_interfaces(
            time_step=time_step,
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
        ))
    
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
