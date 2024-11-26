from functools import partial

import equinox.internal as eqxi
import jax
import jax.numpy as jnp

from fdtdx.core.config import SimulationConfig
from fdtdx.fdtd.backward import backward
from fdtdx.fdtd.forward import forward, forward_single_args_wrapper
from fdtdx.interfaces.state import RecordingState
from fdtdx.objects.boundaries.perfectly_matched_layer import BoundaryState
from fdtdx.objects.container import ArrayContainer, ObjectContainer, SimulationState, reset_array_container
from fdtdx.objects.detectors.detector import DetectorState


def reversible_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
) -> SimulationState:
    arrays = reset_array_container(
        arrays,
        objects,
    )
    
    def reversible_fdtd_base(
        arr: ArrayContainer,  
    ) -> SimulationState:
        state = (jnp.asarray(0, dtype=jnp.int32), arr)
        state = eqxi.while_loop(
            max_steps=config.time_steps_total,
            cond_fun=lambda s: config.time_steps_total > s[0],
            body_fun=partial(
                forward,
                config=config,
                objects=objects,
                key=key,
                record_detectors=True,
                record_boundaries=config.invertible_optimization,
                simulate_boundaries=True,
            ),
            init_val=state,
            kind="lax",
        )
        return (state[0], state[1])
    
    
    @jax.custom_vjp
    def reversible_fdtd_primal(
        E: jax.Array,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
        boundary_states: dict[str, BoundaryState],
        detector_states: dict[str, DetectorState],
        recording_state: RecordingState | None,
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
        state = reversible_fdtd_base(arr)
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
    
    
    def body_fn(
        sr_tuple,
    ):
        state, cot = sr_tuple
        state = backward(
            state=state,
            config=config,
            objects=objects,
            key=key,
            record_detectors=False,
            reset_fields=False,
        )
        _, update_vjp = jax.vjp(
            partial(
                forward_single_args_wrapper,
                config=config,
                objects=objects,
                key=key,
                record_detectors=True,
                record_boundaries=False,
                simulate_boundaries=True,
            ),
            state[0],
            state[1].E,
            state[1].H,
            state[1].inv_permittivities,
            state[1].inv_permeabilities,
            state[1].boundary_states,
            state[1].detector_states,
            state[1].recording_state
        )
        
        cot = update_vjp(cot)
        return state, cot


    def cond_fun(
        sr_tuple,
        start_time_step: int,
    ):
        s_k, r_k = sr_tuple
        del r_k
        time_step = s_k[0]
        return time_step >= start_time_step


    def fdtd_bwd(
        residual,
        cot,
    ):
        (
            res_time_step,
            res_E,
            res_H,
            res_inv_permittivities,
            res_inv_permeabilities,
            res_boundary_states,
            res_detector_states,
            res_recording_state,
        ) = residual

        s_k = ArrayContainer(
            E=res_E,
            H=res_H,
            inv_permittivities=res_inv_permittivities,
            inv_permeabilities=res_inv_permeabilities,
            boundary_states=res_boundary_states,
            detector_states=res_detector_states,
            recording_state=res_recording_state,
        )
        
        _, cot = eqxi.while_loop(
            cond_fun=partial(cond_fun, start_time_step=0),
            body_fun=body_fn,
            init_val=((res_time_step, s_k), cot),
            kind="lax",
        )
        return (
            None, # cot[1],
            None, # cot[2],
            cot[3],
            cot[4],
            None,  # cot[5]
            None, # cot[6],
            None, # cot[7],
        )


    def fdtd_fwd(
        E: jax.Array,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
        boundary_states: dict[str, BoundaryState],
        detector_states: dict[str, DetectorState],
        recording_state: RecordingState | None,
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
        s_k = reversible_fdtd_base(arr)
    
        primal_out = (
            s_k[0],
            s_k[1].E,
            s_k[1].H,
            s_k[1].inv_permittivities,
            s_k[1].inv_permeabilities,
            s_k[1].boundary_states,
            s_k[1].detector_states,
            s_k[1].recording_state,  # None
        )
        residual = (
            s_k[0],
            s_k[1].E,
            s_k[1].H,
            s_k[1].inv_permittivities,
            s_k[1].inv_permeabilities,
            s_k[1].boundary_states,
            s_k[1].detector_states,
            s_k[1].recording_state,
        )
        return primal_out, residual


    reversible_fdtd_primal.defvjp(fdtd_fwd, fdtd_bwd)
    
    (
        time_step,
        E,
        H,
        inv_permittivities,
        inv_permeabilities,
        boundary_states,
        detector_states,
        recording_state,
    ) = reversible_fdtd_primal(
        E=arrays.E,
        H=arrays.H,
        inv_permittivities=arrays.inv_permittivities,
        inv_permeabilities=arrays.inv_permeabilities,
        boundary_states=arrays.boundary_states,
        detector_states=arrays.detector_states,
        recording_state=arrays.recording_state,
    )
    out_arrs = ArrayContainer(
        E=E,
        H=H,
        inv_permittivities=inv_permittivities,
        inv_permeabilities=inv_permeabilities,
        boundary_states=boundary_states,
        detector_states=detector_states,
        recording_state=recording_state,
    )
    return time_step, out_arrs
    

def checkpointed_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
) -> SimulationState:
    arrays = reset_array_container(arrays, objects)
    state = (jnp.asarray(0, dtype=jnp.int32), arrays)
    state = eqxi.while_loop(
        max_steps=config.time_steps_total,
        cond_fun=lambda s: config.time_steps_total > s[0],
        body_fun=partial(
            forward,
            config=config,
            objects=objects,
            key=key,
            record_detectors=True,
            record_boundaries=config.invertible_optimization,
            simulate_boundaries=True,
        ),
        init_val=state,
        kind="lax" if config.only_forward is None else "checkpointed",
        checkpoints=(
            None if config.gradient_config is None 
            else config.gradient_config.num_checkpoints
        ),
    )

    return state



def custom_fdtd_forward(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    reset_container: bool,
    record_detectors: bool,
    start_time: int | jax.Array,
    end_time: int | jax.Array,
) -> SimulationState:
    
    if reset_container:
        arrays = reset_array_container(arrays, objects)
    state = (jnp.asarray(start_time, dtype=jnp.int32), arrays)
    state = eqxi.while_loop(
        max_steps=config.time_steps_total,
        cond_fun=lambda s: end_time > s[0],
        body_fun=partial(
            forward,
            config=config,
            objects=objects,
            key=key,
            record_detectors=record_detectors,
            record_boundaries=False,
            simulate_boundaries=True,
        ),
        init_val=state,
        kind="lax",
        checkpoints=None
    )

    return state
    
    


