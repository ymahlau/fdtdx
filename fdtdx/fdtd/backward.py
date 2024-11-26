from functools import partial
from typing import Sequence

import jax
import equinox.internal as eqxi

from fdtdx.core.config import SimulationConfig
from fdtdx.objects.container import ObjectContainer, SimulationState
from fdtdx.fdtd.update import (
    add_interfaces,
    update_E_reverse,
    update_H_reverse,
    update_detector_states
)



def cond_fn(state, start_time_step: int):
    time_step = state[0]
    return time_step > start_time_step

def full_backward(
    state: SimulationState,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    record_detectors: bool,
    reset_fields: bool,
    start_time_step: int = 0,
):
    s0 = eqxi.while_loop(
        cond_fun=partial(cond_fn, start_time_step=start_time_step),
        body_fun=partial(
            backward,
            config=config,
            objects=objects,
            key=key,
            record_detectors=record_detectors,
            reset_fields=reset_fields,
        ),
        init_val=state,
        kind="lax",
    )
    return s0



def backward(
    state: SimulationState,
    config: SimulationConfig,
    objects: ObjectContainer,
    key: jax.Array,
    record_detectors: bool,
    reset_fields: bool,
    fields_to_reset: Sequence[str] = ("E", "H"),
):
    time_step, arrays = state
    time_step = time_step - 1
    
    arrays = add_interfaces(
        time_step=time_step,
        arrays=arrays,
        objects=objects,
        config=config,
        key=key,
    )

    H = arrays.H
    
    arrays = update_H_reverse(
        time_step=time_step,
        arrays=arrays,
        config=config,
        objects=objects,
    )
    
    arrays = update_E_reverse(
        time_step=time_step,
        arrays=arrays,
        config=config,
        objects=objects,
    )
    
    if reset_fields:
        new_fields = {
            f: getattr(arrays, f)
            for f in fields_to_reset
        }
        for pml in objects.pml_objects:
            for name in fields_to_reset:
                new_fields[name] = (
                    new_fields[name]
                    .at[:, *pml.grid_slice]
                    .set(0)
                )
        for name, f in new_fields.items():
            arrays = arrays.aset(name, f)
    
    if record_detectors:
        arrays = update_detector_states(
            time_step=time_step,
            arrays=arrays,
            objects=objects,
            H_prev=H,
            inverse=True,
        )
    
    next_state = (time_step, arrays)
    return next_state
