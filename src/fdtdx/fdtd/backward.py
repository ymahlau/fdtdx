from functools import partial
from typing import Sequence

import equinox.internal as eqxi
import jax

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ObjectContainer, SimulationState
from fdtdx.fdtd.update import add_interfaces, update_detector_states, update_E_reverse, update_H_reverse
from fdtdx.objects.boundaries.periodic import PeriodicBoundary


def cond_fn(state, start_time_step: int) -> bool:
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
        state (SimulationState): Current simulation state tuple (time_step, arrays)
        objects (ObjectContainer): Container with simulation objects (sources, detectors, etc)
        config (SimulationConfig): Simulation configuration parameters
        key (jax.Array): JAX PRNG key for random operations
        record_detectors (bool): Whether to record detector states
        reset_fields (bool): Whether to reset fields after each step
        start_time_step (int, optional): Time step to propagate back to (default: 0)

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
        state (SimulationState): Current simulation state tuple (time_step, arrays)
        config (SimulationConfig): Simulation configuration parameters
        objects (ObjectContainer): Container with simulation objects (sources, detectors, etc)
        key (jax.Array): JAX PRNG key for random operations
        record_detectors (bool): Whether to record detector states
        reset_fields (bool): Whether to reset fields after updates
        fields_to_reset (Sequence[str], optional): Which fields to reset if reset_fields is True. Defaults to ("E", "H").

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
        # Reset PML boundaries
        for pml in objects.pml_objects:
            for name in fields_to_reset:
                new_fields[name] = new_fields[name].at[:, *pml.grid_slice].set(0)
        # Handle periodic boundaries by copying values from opposite sides
        for boundary in objects.boundary_objects:
            if isinstance(boundary, PeriodicBoundary):
                for name in fields_to_reset:
                    field = new_fields[name]
                    # Get field values from opposite boundary
                    opposite_slice = list(boundary.grid_slice)
                    if boundary.direction == "+":
                        opposite_slice[boundary.axis] = slice(
                            boundary._grid_slice_tuple[boundary.axis][1] - 1,
                            boundary._grid_slice_tuple[boundary.axis][1],
                        )
                        boundary_slice = slice(
                            boundary._grid_slice_tuple[boundary.axis][0],
                            boundary._grid_slice_tuple[boundary.axis][0] + 1,
                        )
                    else:
                        opposite_slice[boundary.axis] = slice(
                            boundary._grid_slice_tuple[boundary.axis][0],
                            boundary._grid_slice_tuple[boundary.axis][0] + 1,
                        )
                        boundary_slice = slice(
                            boundary._grid_slice_tuple[boundary.axis][1] - 1,
                            boundary._grid_slice_tuple[boundary.axis][1],
                        )
                    opposite_slice[boundary.axis] = boundary_slice
                    field_values = field[..., opposite_slice[0], opposite_slice[1], opposite_slice[2]]
                    new_fields[name] = field.at[..., *boundary.grid_slice].set(field_values)

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
