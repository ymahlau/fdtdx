from functools import partial
from typing import Sequence

import equinox.internal as eqxi
import jax

from fdtdx.core.config import SimulationConfig
from fdtdx.fdtd.update import add_interfaces, update_detector_states, update_E_reverse, update_H_reverse
from fdtdx.objects.container import ObjectContainer, SimulationState


def cond_fn(state, start_time_step: int) -> bool:
    """Check if current time step is greater than start time step.

    Args:
        state: Tuple containing (time_step, arrays)
        start_time_step: Starting time step to compare against

    Returns:
        bool: True if current time step > start_time_step
    """
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
) -> SimulationState:
    """Perform full backward FDTD propagation from current state to start time.

    Uses a while loop to repeatedly call backward() until reaching start_time_step.
    Leverages time-reversibility of Maxwell's equations.

    Args:
        state: Current simulation state tuple (time_step, arrays)
        objects: Container with simulation objects (sources, detectors, etc)
        config: Simulation configuration parameters
        key: JAX PRNG key for random operations
        record_detectors: Whether to record detector states
        reset_fields: Whether to reset fields after each step
        start_time_step: Time step to propagate back to (default: 0)

    Returns:
        SimulationState: Final state after backward propagation
    """
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
) -> SimulationState:
    """Perform one step of backward FDTD propagation.

    Updates fields from time step t to t-1 using time-reversed Maxwell's equations.
    Handles interfaces, field updates, optional field resetting, and detector recording.

    Args:
        state: Current simulation state tuple (time_step, arrays)
        config: Simulation configuration parameters
        objects: Container with simulation objects (sources, detectors, etc)
        key: JAX PRNG key for random operations
        record_detectors: Whether to record detector states
        reset_fields: Whether to reset fields after updates
        fields_to_reset: Which fields to reset if reset_fields is True

    Returns:
        SimulationState: Updated state after one backward step
    """
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
        new_fields = {f: getattr(arrays, f) for f in fields_to_reset}
        for pml in objects.pml_objects:
            for name in fields_to_reset:
                new_fields[name] = new_fields[name].at[:, *pml.grid_slice].set(0)
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
