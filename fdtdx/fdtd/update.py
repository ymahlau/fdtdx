from functools import partial
import jax

from fdtdx.core.config import SimulationConfig
from fdtdx.fdtd.curl import curl_E, curl_H, interpolate_fields
from fdtdx.objects.container import ArrayContainer, ObjectContainer
from fdtdx.objects.detectors.detector import Detector
from fdtdx.shared.misc import add_boundary_interfaces, collect_boundary_interfaces
import pytreeclass as tc

def update_E(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    simulate_boundaries: bool,
) -> ArrayContainer:
    
    boundary_states = {}
    if simulate_boundaries:
        for pml in objects.pml_objects:
            boundary_states[pml.name] = pml.update_E_boundary_state(
                boundary_state=arrays.boundary_states[pml.name],
                H=arrays.H,
            )
    
    E = (
        arrays.E + config.courant_number 
        * curl_H(arrays.H) * arrays.inv_permittivities
    )
    
    for source in objects.sources:
        def _update():
            return source.update_E(
                E=E,
                inv_permittivities=arrays.inv_permittivities,
                time_step=time_step,
                inverse=False,
            )
        E = jax.lax.cond(
            source._is_on_at_time_step_arr[time_step],
            _update,
            lambda: E,
        )
    
    if simulate_boundaries:
        for pml in objects.pml_objects:
            E = pml.update_E(
                E=E,
                boundary_state=boundary_states[pml.name],
                inverse_permittivity=arrays.inv_permittivities,
            )
    
    arrays = arrays.at["E"].set(E)
    if simulate_boundaries:
        arrays = arrays.aset("boundary_states", boundary_states)

    return arrays


def update_E_reverse(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
) -> ArrayContainer:
    
    E = arrays.E
    for source in objects.sources:
        def _update():
            return source.update_E(
                E,
                inv_permittivities=arrays.inv_permittivities,
                time_step=time_step,
                inverse=True,
            )
        E = jax.lax.cond(
            source._is_on_at_time_step_arr[time_step],
            _update,
            lambda: E,
        )
    
    E = (
        E - config.courant_number 
        * curl_H(arrays.H) * arrays.inv_permittivities
    )
    
    arrays = arrays.at["E"].set(E)
    
    return arrays


def update_H(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    simulate_boundaries: bool,
) -> ArrayContainer:
    
    boundary_states = {}
    if simulate_boundaries:
        for pml in objects.pml_objects:
            boundary_states[pml.name] = pml.update_H_boundary_state(
                boundary_state=arrays.boundary_states[pml.name],
                E=arrays.E,
            )
            
    H = (
        arrays.H - config.courant_number 
        * curl_E(arrays.E) * arrays.inv_permeabilities
    )
    
    for source in objects.sources:
        def _update():
            return source.update_H(
                H=H,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=time_step + 0.5,
                inverse=False,
            )
        H = jax.lax.cond(
            source._is_on_at_time_step_arr[time_step],
            _update,
            lambda: H,
        )

    if simulate_boundaries:
        for pml in objects.pml_objects:
            H = pml.update_H(
                H=H,
                boundary_state=boundary_states[pml.name],
                inverse_permeability=arrays.inv_permeabilities,
            )

    arrays = arrays.at["H"].set(H)
    if simulate_boundaries:
        arrays = arrays.aset("boundary_states", boundary_states)
    
    return arrays


def update_H_reverse(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
) -> ArrayContainer:
    
    H = arrays.H
    for source in objects.sources:
        def _update():
            return source.update_H(
                H,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=time_step + 0.5,
                inverse=True,
            )
        H = jax.lax.cond(
            source._is_on_at_time_step_arr[time_step],
            _update,
            lambda: H,
        )
        
    H = (
        H + config.courant_number 
        * curl_E(arrays.E) * arrays.inv_permeabilities
    )
    
    arrays = arrays.at["H"].set(H)
    return arrays


def update_detector_states(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    H_prev: jax.Array,
    inverse: bool,
) -> ArrayContainer:
    interpolated_E, interpolated_H = interpolate_fields(
        E_field=arrays.E,
        H_field=(H_prev + arrays.H) / 2,
    )
    
    def helper_fn(E_input, H_input, detector: Detector):
        detector = tc.unfreeze(detector)
        return detector.update(
            time_step=time_step,
            E=E_input,
            H=H_input,
            state=arrays.detector_states[detector.name],
            inv_permittivity=arrays.inv_permittivities,
            inv_permeability=arrays.inv_permeabilities,
        )
    
    state = arrays.detector_states
    to_update = objects.backward_detectors if inverse else objects.forward_detectors
    for d in to_update:
        state[d.name] = jax.lax.cond(
            d._is_on_at_time_step_arr[time_step],
            helper_fn,
            lambda e, h, _: state[d.name],
            interpolated_E if d.exact_interpolation else arrays.E,
            interpolated_H if d.exact_interpolation else arrays.H,
            tc.freeze(d),
        )
    arrays = arrays.aset("detector_states", state)
    return arrays


def collect_interfaces(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
) -> ArrayContainer:
    if config.gradient_config is None or config.gradient_config.recorder is None:
        raise Exception(f"Need recorder to record boundaries")
    if arrays.recording_state is None:
        raise Exception(f"Need recording state to record boundaries")
    values = collect_boundary_interfaces(
        arrays=arrays,
        pml_objects=objects.pml_objects,
    )
    recording_state = config.gradient_config.recorder.compress(
        values=values,
        state=arrays.recording_state,
        time_step=time_step,
        key=key,
    )
    arrays = arrays.aset("recording_state", recording_state)
    return arrays


def add_interfaces(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
) -> ArrayContainer:
    if config.gradient_config is None or config.gradient_config.recorder is None:
        raise Exception(f"Need recorder to record boundaries")
    if arrays.recording_state is None:
        raise Exception(f"Need recording state to record boundaries")
    
    values, state = config.gradient_config.recorder.decompress(
        state=arrays.recording_state,
        time_step=time_step,
        key=key,
    )
    arrays = arrays.aset("recording_state", state)
    
    container = add_boundary_interfaces(
        arrays=arrays,
        values=values,
        pml_objects=objects.pml_objects,
    )
    
    return container

