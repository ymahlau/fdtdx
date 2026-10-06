from __future__ import annotations

from collections.abc import Callable
from functools import partial

import equinox.internal as eqxi
import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.progress import _make_pbar, _wrap_body_with_progress
from fdtdx.fdtd.backward import backward
from fdtdx.fdtd.container import ArrayContainer, FieldState, ObjectContainer, PmlAuxField, SimulationState
from fdtdx.fdtd.forward import forward, forward_single_args_wrapper
from fdtdx.fdtd.stop_conditions import StoppingCondition, TimeStepCondition
from fdtdx.interfaces.state import RecordingState
from fdtdx.objects.detectors.detector import DetectorState


def _reversible_slice_boundaries(time_steps_total: int, num_slices: int) -> list[int]:
    """Compute the time-step boundaries partitioning a run into ``num_slices`` slices.

    Returns ``[s_0, s_1, ..., s_k]`` with ``k = num_slices``, ``s_0 = 0`` and
    ``s_k = time_steps_total``. The boundaries are strictly increasing and every slice
    ``[s_i, s_{i+1}]`` has length ``>= 1`` provided ``1 <= num_slices <= time_steps_total``.
    The interior boundaries ``s_1 .. s_{k-1}`` are the times at which a full-field checkpoint
    is taken for the sliced reversible backward pass.

    Args:
        time_steps_total (int): Total number of forward time steps ``T``.
        num_slices (int): Number of slices ``k`` (``= num_checkpoints_reversible + 1``).

    Returns:
        list[int]: The ``num_slices + 1`` boundary time steps.
    """
    return [round(i * time_steps_total / num_slices) for i in range(num_slices + 1)]


def reversible_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    show_progress: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimulationState:
    """Run a memory-efficient differentiable FDTD simulation leveraging time-reversal symmetry.

    This implementation exploits the time-reversal symmetry of Maxwell's equations to perform
    backpropagation without storing the electromagnetic fields at each time step. During the
    backward pass, the fields are reconstructed by running the simulation in reverse, only
    requiring O(1) memory storage instead of O(T) where T is the number of time steps.

    The only exception is boundary conditions which break time-reversal symmetry - these are
    recorded during the forward pass and replayed during backpropagation. With
    ``config.gradient_config.recording_mode == "segmented"`` they are instead regenerated slice by
    slice during backpropagation, by re-simulating each slice from the full-field checkpoint at its
    start, so that only a single slice of them is stored at any time.

    Args:
        arrays (ArrayContainer): Initial state of the simulation containing:
            - E, H: Electric and magnetic field arrays
            - inv_permittivities, inv_permeabilities: Material properties
            - detector_states: Dictionary of field detectors
            - recording_state: Optional state for recording field evolution
        objects (ObjectContainer): Collection of physical objects in the simulation
            (sources, detectors, boundaries, etc.)
        config (SimulationConfig): Simulation parameters including:
            - time_steps_total: Total number of steps to simulate
            - invertible_optimization: Whether to record boundaries for backprop
        key (jax.Array): JAX PRNGKey for any stochastic operations
        show_progress (bool): Display a tqdm progress bar while the simulation runs.
            Set to False for a minor speed improvement; see the module-level
            benchmark note for overhead estimates. Defaults to True.
            The bar is driven entirely by ``io_callback`` at XLA execution
            time, so it works correctly whether the simulation is
            wrapped in ``jax.jit``.

    Returns:
        SimulationState: Tuple containing:
            - Final time step (int)
            - ArrayContainer with the final state of all fields and components

    Notes:
        The implementation uses custom vector-Jacobian products (VJPs) to enable
        efficient backpropagation through the entire simulation while maintaining
        numerical stability. This makes it suitable for gradient-based optimization
        of electromagnetic designs.

    Raises:
        NotImplementedError: If the simulation contains dispersive materials. Reversing
            the ADE polarization recurrence is not supported; use the ``"checkpointed"``
            gradient method for dispersive simulations.
    """
    # if arrays.magnetic_conductivity is not None or arrays.electric_conductivity is not None:
    #     raise Exception(f"Reversible FDTD does not work with Conductive Materials")

    # Checked here in addition to initialization time, since the gradient config can be
    # swapped after ``place_objects`` and this function can be called directly, bypassing
    # ``run_fdtd``.
    if arrays.dispersive_c1 is not None or arrays.fields.dispersive_P_curr is not None:
        raise NotImplementedError(
            "Dispersive time-reversible gradient computation under active development. "
            "Use GradientConfig(method='checkpointed') instead."
        )

    arrays = arrays.reset()

    # Sliced reversible backward pass: partition the run into ``num_slices`` slices and store a
    # full-field checkpoint at each interior boundary during the forward pass, so the reverse
    # reconstruction can be reset to the exact field at every boundary (bounding reconstruction
    # drift for lossy materials). ``num_checkpoints_reversible == 0`` (the default)
    # gives a single slice and reproduces the classic full reverse pass exactly. In segmented
    # recording mode the checkpoints are also where each slice's interface record is regenerated from.
    grad_cfg = config.gradient_config
    num_ckpt = 0 if grad_cfg is None else grad_cfg.num_checkpoints_reversible
    if not isinstance(num_ckpt, int):
        raise Exception("num_checkpoints_reversible='auto' is resolved by place_objects; use the config it returns")
    segmented = grad_cfg is not None and grad_cfg.recording_mode == "segmented"
    num_slices = num_ckpt + 1
    # Only the interior checkpoints (num_ckpt >= 1) impose the slice-length constraint; the default
    # single slice (num_ckpt == 0) is always valid, including the ``time_steps_total == 0`` edge case.
    if num_ckpt > 0 and num_slices > config.time_steps_total:
        raise Exception(
            "num_checkpoints_reversible must be <= time_steps_total - 1 "
            f"(got num_checkpoints_reversible={num_ckpt}, time_steps_total={config.time_steps_total})"
        )
    # The recorder buffer is sized when the arrays are initialized, and the gradient config can be swapped
    # afterwards. An index past its end would be clamped silently, so check it here.
    if grad_cfg is not None and grad_cfg.recorder is not None:
        required_steps = grad_cfg.recorder_time_steps(config.time_steps_total)
        recorder_steps = getattr(grad_cfg.recorder, "_max_time_steps", None)
        if recorder_steps is not None and recorder_steps < required_steps:
            raise Exception(
                f"The recorder holds {recorder_steps} time steps, but recording_mode='{grad_cfg.recording_mode}' "
                f"needs {required_steps}. Initialize the arrays with the gradient config that is used here."
            )
    # The slices are recorded during the backward pass, in reverse order, so a recorder module that carries
    # state from one compressed step to the next would see a different state sequence than in full mode.
    if segmented and arrays.recording_state is not None and arrays.recording_state.state:
        raise Exception(
            "recording_mode='segmented' does not support recorder modules with internal state "
            f"({sorted(arrays.recording_state.state)}), because the slices are recorded out of order."
        )
    slice_boundaries = jnp.asarray(
        _reversible_slice_boundaries(config.time_steps_total, num_slices),
        dtype=jnp.int32,
    )

    pbar = _make_pbar(
        show_progress=show_progress,
        total_steps=config.time_steps_total,
        desc="FDTD (reversible)",
        progress_callback=progress_callback,
    )

    # Build the (optionally instrumented) forward body function once so both
    # reversible_fdtd_primal and fdtd_fwd share the same wrapping logic. In segmented recording
    # mode nothing is recorded here: the backward pass regenerates the record slice by slice.
    _forward_body = partial(
        forward,
        config=config,
        objects=objects,
        key=key,
        record_detectors=True,
        record_boundaries=config.invertible_optimization and not segmented,
        simulate_boundaries=True,
    )
    _forward_body_with_progress, _close_pbar = _wrap_body_with_progress(_forward_body, pbar)

    def run_until(state: SimulationState, end_time_step: int | jax.Array, body_fun: Callable) -> SimulationState:
        """Advance ``state`` with ``body_fun`` until its time step reaches ``end_time_step``."""
        return eqxi.while_loop(
            cond_fun=lambda s: end_time_step > s[0],
            body_fun=body_fun,
            init_val=state,
            kind="lax",
        )

    def segmented_forward(
        arr: ArrayContainer,
    ) -> tuple[SimulationState, FieldState]:
        """Run the forward pass slice by slice, capturing the field state at every interior boundary.

        The checkpoints are stacked along a leading axis (``checkpoints.E[i]`` is the field at
        ``slice_boundaries[i + 1]``) and the slices run in a ``fori_loop``, so the size of the compiled
        program does not depend on the number of checkpoints. The boundary at the end of the last slice is
        the primal output and is not stored (``mode="drop"``).
        """
        state = (jnp.asarray(0, dtype=jnp.int32), arr)
        checkpoints = jax.tree.map(lambda x: jnp.zeros((num_ckpt, *x.shape), x.dtype), arr.fields)
        if num_slices == 1:
            # Without interior checkpoints the forward pass is a single loop, exactly as without slicing.
            return run_until(state, config.time_steps_total, _forward_body_with_progress), checkpoints

        def slice_body(i, carry):
            state, checkpoints = carry
            state = run_until(state, slice_boundaries[i + 1], _forward_body_with_progress)
            checkpoints = jax.tree.map(lambda buf, x: buf.at[i].set(x, mode="drop"), checkpoints, state[1].fields)
            return state, checkpoints

        return jax.lax.fori_loop(0, num_slices, slice_body, (state, checkpoints))

    @jax.custom_vjp
    def reversible_fdtd_primal(
        E: jax.Array,
        H: jax.Array,
        psi_E: PmlAuxField,
        psi_H: PmlAuxField,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
        detector_states: dict[str, DetectorState],
        recording_state: RecordingState | None,
    ):
        arr = ArrayContainer(
            fields=FieldState(
                E=E,
                H=H,
                psi_E=psi_E,
                psi_H=psi_H,
            ),
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            detector_states=detector_states,
            recording_state=recording_state,
            electric_conductivity=arrays.electric_conductivity,
            magnetic_conductivity=arrays.magnetic_conductivity,
            initial_inv_permittivities=arrays.initial_inv_permittivities,
        )
        # The non-gradient primal path needs only the final state, so it takes no checkpoints.
        state = run_until((jnp.asarray(0, dtype=jnp.int32), arr), config.time_steps_total, _forward_body_with_progress)
        return (
            state[0],
            state[1].fields.E,
            state[1].fields.H,
            state[1].fields.psi_E,
            state[1].fields.psi_H,
            state[1].inv_permittivities,
            state[1].inv_permeabilities,
            state[1].detector_states,
            state[1].recording_state,
        )

    def body_fn(
        sr_tuple,
        record_time_offset: int | jax.Array,
    ):
        state, cot = sr_tuple
        state = backward(
            state=state,
            config=config,
            objects=objects,
            key=key,
            record_detectors=False,
            reset_fields=False,
            record_time_offset=record_time_offset,
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
                electric_conductivity=arrays.electric_conductivity,
                magnetic_conductivity=arrays.magnetic_conductivity,
            ),
            state[0],
            state[1].fields.E,
            state[1].fields.H,
            state[1].fields.psi_E,
            state[1].fields.psi_H,
            state[1].inv_permittivities,
            state[1].inv_permeabilities,
            state[1].detector_states,
            state[1].recording_state,
        )

        cot = update_vjp(cot)
        return state, cot

    def cond_fun(
        sr_tuple,
        start_time_step: int | jax.Array,
    ):
        """Whether another reverse step is due.

        ``body_fn`` steps the reconstruction *back* before taking the VJP, so entering the body at
        ``time_step = k`` back-propagates the forward step that produced state ``k``, i.e. forward
        step ``k - 1``. The forward pass runs steps ``0 .. time_steps_total - 1``, so the last one
        needing a VJP is entered at ``time_step = start_time_step + 1`` and the loop must stop on
        reaching ``start_time_step`` itself. A ``>=`` here runs one extra body call, which
        reconstructs ``start_time_step - 1`` and pulls back a forward step that never happened.
        """
        s_k, r_k = sr_tuple
        del r_k
        time_step = s_k[0]
        return time_step > start_time_step

    def fdtd_bwd(
        residual,
        cot,
    ):
        primal_out, checkpoints = residual
        (
            res_time_step,
            res_E,
            res_H,
            res_psi_E,
            res_psi_H,
            res_inv_permittivities,
            res_inv_permeabilities,
            res_detector_states,
            res_recording_state,
        ) = primal_out
        del res_time_step
        final_fields = FieldState(E=res_E, H=res_H, psi_E=res_psi_E, psi_H=res_psi_H)

        def make_container(fields: FieldState, recording_state: RecordingState | None) -> ArrayContainer:
            return ArrayContainer(
                fields=fields,
                inv_permittivities=res_inv_permittivities,
                inv_permeabilities=res_inv_permeabilities,
                detector_states=res_detector_states,
                recording_state=recording_state,
                electric_conductivity=arrays.electric_conductivity,
                magnetic_conductivity=arrays.magnetic_conductivity,
                initial_inv_permittivities=arrays.initial_inv_permittivities,
            )

        def fields_at_boundary(i: jax.Array) -> FieldState:
            """Exact field state at ``slice_boundaries[i]``: zero at the start of the run (the fields are
            reset on entry), an interior checkpoint, or the primal output at the end of the run."""
            branches = [lambda: jax.tree.map(jnp.zeros_like, final_fields)]
            if num_ckpt > 0:
                branches.append(lambda: jax.tree.map(lambda buf: buf[jnp.clip(i - 1, 0, num_ckpt - 1)], checkpoints))
            branches.append(lambda: final_fields)
            branch = jnp.where(i == 0, 0, jnp.where(i == num_slices, len(branches) - 1, 1))
            return jax.lax.switch(branch, branches)

        def reverse_slice(j, carry):
            """Reverse slice ``num_slices - 1 - j``, starting from the exact field state at its end."""
            recording_state, running_cot = carry
            i = num_slices - 1 - j
            slice_start, slice_end = slice_boundaries[i], slice_boundaries[i + 1]
            record_time_offset: int | jax.Array = 0
            if segmented:
                # Regenerate the interface record of this slice (stored from buffer index 0) by
                # re-simulating it from the checkpoint at its start.
                record_time_offset = slice_start
                re_forward_body = partial(
                    forward,
                    config=config,
                    objects=objects,
                    key=key,
                    record_detectors=False,
                    record_boundaries=True,
                    simulate_boundaries=True,
                    record_time_offset=slice_start,
                )
                _, re_arrays = run_until(
                    (slice_start, make_container(fields_at_boundary(i), recording_state)),
                    slice_end,
                    re_forward_body,
                )
                recording_state = re_arrays.recording_state
            fields_end = final_fields if num_slices == 1 else fields_at_boundary(i + 1)
            (_, arrays_start), running_cot = eqxi.while_loop(
                cond_fun=partial(cond_fun, start_time_step=slice_start),
                body_fun=partial(body_fn, record_time_offset=record_time_offset),
                init_val=((slice_end, make_container(fields_end, recording_state)), running_cot),
                kind="lax",
            )
            return arrays_start.recording_state, running_cot

        if num_slices == 1:
            # The default single slice is reversed outside of a loop, which compiles to exactly the classic
            # reverse pass.
            _, cot = reverse_slice(0, (res_recording_state, cot))
        else:
            _, cot = jax.lax.fori_loop(0, num_slices, reverse_slice, (res_recording_state, cot))
        return (
            None,  # cot[1],   E
            None,  # cot[2],   H
            None,  # cot[3],   psi_E
            None,  # cot[4],   psi_H
            cot[5],  #         inv_permittivities
            cot[6],  #         inv_permeabilities
            None,  # cot[7],   detector_states
            None,  # cot[8],   recording_state
        )

    def fdtd_fwd(
        E: jax.Array,
        H: jax.Array,
        psi_E: PmlAuxField,
        psi_H: PmlAuxField,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
        detector_states: dict[str, DetectorState],
        recording_state: RecordingState | None,
    ):
        arr = ArrayContainer(
            fields=FieldState(
                E=E,
                H=H,
                psi_E=psi_E,
                psi_H=psi_H,
            ),
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            detector_states=detector_states,
            recording_state=recording_state,
            electric_conductivity=arrays.electric_conductivity,
            magnetic_conductivity=arrays.magnetic_conductivity,
            initial_inv_permittivities=arrays.initial_inv_permittivities,
        )
        s_k, checkpoints = segmented_forward(arr)

        primal_out = (
            s_k[0],
            s_k[1].fields.E,
            s_k[1].fields.H,
            s_k[1].fields.psi_E,
            s_k[1].fields.psi_H,
            s_k[1].inv_permittivities,
            s_k[1].inv_permeabilities,
            s_k[1].detector_states,
            s_k[1].recording_state,  # None
        )
        # ``checkpoints`` holds the stacked interior full-field snapshots (empty for a single slice); they
        # are threaded to ``fdtd_bwd`` via the residual so the reverse reconstruction can be reset to the
        # exact field at each boundary (and, in segmented recording mode, each slice can be re-simulated).
        residual = (primal_out, checkpoints)
        return primal_out, residual

    reversible_fdtd_primal.defvjp(fdtd_fwd, fdtd_bwd)

    (
        time_step,
        E,
        H,
        psi_E,
        psi_H,
        inv_permittivities,
        inv_permeabilities,
        detector_states,
        recording_state,
    ) = reversible_fdtd_primal(
        E=arrays.fields.E,
        H=arrays.fields.H,
        psi_E=arrays.fields.psi_E,
        psi_H=arrays.fields.psi_H,
        inv_permittivities=arrays.inv_permittivities,
        inv_permeabilities=arrays.inv_permeabilities,
        detector_states=arrays.detector_states,
        recording_state=arrays.recording_state,
    )
    _close_pbar()

    out_arrs = ArrayContainer(
        fields=FieldState(
            E=E,
            H=H,
            psi_E=psi_E,
            psi_H=psi_H,
        ),
        inv_permittivities=inv_permittivities,
        inv_permeabilities=inv_permeabilities,
        detector_states=detector_states,
        recording_state=recording_state,
        electric_conductivity=arrays.electric_conductivity,
        magnetic_conductivity=arrays.magnetic_conductivity,
        initial_inv_permittivities=arrays.initial_inv_permittivities,
    )
    return time_step, out_arrs


def checkpointed_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    stopping_condition: StoppingCondition | None = None,
    show_progress: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimulationState:
    """Run an FDTD simulation with gradient checkpointing for memory efficiency.

    This implementation uses checkpointing to reduce memory usage during backpropagation
    by only storing the field state at certain intervals and recomputing intermediate
    states as needed.

    Args:
        arrays (ArrayContainer): Initial state of the simulation containing fields and materials
        objects (ObjectContainer): Collection of physical objects in the simulation
        config (SimulationConfig): Simulation parameters including checkpointing settings
        key (jax.Array): JAX PRNGKey for any stochastic operations
        stopping_condition (StoppingCondition, optional): Custom stopping condition on which simulation is halted.
            If none is provided, we default to TimeStepCondition (simulation progresses until max time is reached)
        show_progress (bool): Display a tqdm progress bar while the simulation runs.
            Set to False for a minor speed improvement; see the module-level
            benchmark note for overhead estimates. Defaults to True.
            The bar is driven entirely by ``io_callback`` at XLA execution
            time, so it works correctly whether the simulation is
            wrapped in ``jax.jit``.

    Returns:
        SimulationState: Tuple containing final time step and ArrayContainer with final state

    Notes:
        The number of checkpoints can be configured through config.gradient_config.num_checkpoints.
        More checkpoints reduce recomputation but increase memory usage.
    """
    arrays = arrays.reset()
    state = (jnp.asarray(0, dtype=jnp.int32), arrays)
    if stopping_condition is not None:
        stopping_condition = stopping_condition.setup(state, config, objects)
    else:
        stopping_condition = TimeStepCondition().setup(state, config, objects)

    pbar = _make_pbar(
        show_progress=show_progress,
        total_steps=config.time_steps_total,
        desc="FDTD (checkpointed)",
        progress_callback=progress_callback,
    )

    _forward_body = partial(
        forward,
        config=config,
        objects=objects,
        key=key,
        record_detectors=True,
        record_boundaries=config.invertible_optimization,
        simulate_boundaries=True,
    )
    _forward_body_with_progress, _close_pbar = _wrap_body_with_progress(_forward_body, pbar)

    state = eqxi.while_loop(
        max_steps=config.time_steps_total,
        cond_fun=partial(
            stopping_condition,
            config=config,
            objects=objects,
        ),
        body_fun=_forward_body_with_progress,
        init_val=state,
        kind="lax" if config.only_forward is None else "checkpointed",
        checkpoints=(None if config.gradient_config is None else config.gradient_config.num_checkpoints),
    )
    _close_pbar()

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
    show_progress: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimulationState:
    """Run a customizable forward FDTD simulation between specified time steps.

    This function provides fine-grained control over the simulation execution,
    allowing partial time evolution and customization of recording behavior.

    Args:
        arrays (ArrayContainer): Initial state of the simulation
        objects (ObjectContainer): Collection of physical objects
        config (SimulationConfig): Simulation parameters
        key (jax.Array): JAX PRNGKey for stochastic operations
        reset_container (bool): Whether to reset the array container before starting
        record_detectors (bool): Whether to record detector readings
        start_time (int | jax.Array): Time step to start from
        end_time (int | jax.Array): Time step to end at
        show_progress (bool): Display a tqdm progress bar while the simulation runs.
            Set to False for a minor speed improvement; see the module-level
            benchmark note for overhead estimates. Defaults to True.
            The bar is driven entirely by ``io_callback`` at XLA execution
            time, so it works correctly whether the simulation is
            wrapped in ``jax.jit``.

    Returns:
        SimulationState: Tuple containing final time step and ArrayContainer with final state

    Notes:
        This function is useful for implementing custom simulation strategies or
        running partial simulations for analysis purposes.
    """
    if reset_container:
        arrays = arrays.reset()
    state = (jnp.asarray(start_time, dtype=jnp.int32), arrays)

    # start_time and end_time must be statically known Python ints here so that
    # we can compute n_steps for the progress bar without triggering JAX
    # concretization.  They are always statically known at call sites of this
    # function (they control the loop bound, not an array value).
    if isinstance(start_time, jax.Array) or isinstance(end_time, jax.Array):
        # Traced arrays: skip the progress bar entirely to avoid concretization.
        show_progress = False
        progress_callback = None
        n_steps = 0
    else:
        n_steps = int(end_time) - int(start_time)

    pbar = _make_pbar(
        show_progress=show_progress,
        total_steps=n_steps,
        desc="FDTD (forward)",
        step_offset=0 if not show_progress and progress_callback is None else int(start_time),
        progress_callback=progress_callback,
    )

    _forward_body = partial(
        forward,
        config=config,
        objects=objects,
        key=key,
        record_detectors=record_detectors,
        record_boundaries=False,
        simulate_boundaries=True,
    )
    _forward_body_with_progress, _close_pbar = _wrap_body_with_progress(_forward_body, pbar)

    state = eqxi.while_loop(
        max_steps=config.time_steps_total,
        cond_fun=lambda s: end_time > s[0],
        body_fun=_forward_body_with_progress,
        init_val=state,
        kind="lax",
        checkpoints=None,
    )
    _close_pbar()

    return state
