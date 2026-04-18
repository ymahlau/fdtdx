from __future__ import annotations

from functools import partial

import equinox.internal as eqxi
import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.progress import _make_pbar, _wrap_body_with_progress
from fdtdx.fdtd.backward import backward
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState, reset_array_container
from fdtdx.fdtd.forward import forward, forward_single_args_wrapper
from fdtdx.fdtd.stop_conditions import StoppingCondition, TimeStepCondition
from fdtdx.interfaces.state import RecordingState
from fdtdx.objects.detectors.detector import DetectorState


def _state_to_primal_tuple(state: SimulationState) -> tuple:
    """Flatten a ``SimulationState`` into the 14-tuple order used by the
    reversible custom VJP — ``time_step`` plus the 13 differentiable / state
    leaves of ``ArrayContainer`` (conductivities and dispersion coefficients
    are handled separately as closure state or extra primals)."""
    _, arr = state
    return (
        state[0],
        arr.E,
        arr.H,
        arr.psi_E,
        arr.psi_H,
        arr.alpha,
        arr.kappa,
        arr.sigma,
        arr.inv_permittivities,
        arr.inv_permeabilities,
        arr.dispersive_P_curr,
        arr.dispersive_P_prev,
        arr.detector_states,
        arr.recording_state,
    )


def _arr_from_primals(
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
    *,
    electric_conductivity: jax.Array | None,
    magnetic_conductivity: jax.Array | None,
    dispersive_c1: jax.Array | None,
    dispersive_c2: jax.Array | None,
    dispersive_c3: jax.Array | None,
    dispersive_inv_c2: jax.Array | None,
) -> ArrayContainer:
    """Build an ``ArrayContainer`` from the 13 state-like primal leaves plus
    the keyword-only non-primal fields. The two reversible-VJP branches pass
    ``dispersive_c*`` either from closure (flag off) or from primal args
    (flag on). ``dispersive_inv_c2`` is always closure-captured — it's a
    cache of ``1/c2`` and would double-count if added as an independent primal."""
    return ArrayContainer(
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


def reversible_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    show_progress: bool = True,
) -> SimulationState:
    """Run a memory-efficient differentiable FDTD simulation leveraging time-reversal symmetry.

    This implementation exploits the time-reversal symmetry of Maxwell's equations to perform
    backpropagation without storing the electromagnetic fields at each time step. During the
    backward pass, the fields are reconstructed by running the simulation in reverse, only
    requiring O(1) memory storage instead of O(T) where T is the number of time steps.

    The only exception is boundary conditions which break time-reversal symmetry - these are
    recorded during the forward pass and replayed during backpropagation.

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
    """
    # if arrays.magnetic_conductivity is not None or arrays.electric_conductivity is not None:
    #     raise Exception(f"Reversible FDTD does not work with Conductive Materials")
    arrays = reset_array_container(arrays, objects)

    differentiate_dispersion = config.differentiate_dispersion and arrays.dispersive_c1 is not None

    # When the flag is off, the coefficient arrays get closure-captured by
    # ``reversible_fdtd_primal``. A tracer passed in via ``arrays`` would leak
    # out of ``@jax.custom_vjp``. Cutting the gradient on entry matches the
    # checkpointed path and makes closure capture safe.
    if arrays.dispersive_c1 is not None and not differentiate_dispersion:
        arrays = arrays.at["dispersive_c1"].set(jax.lax.stop_gradient(arrays.dispersive_c1))
        arrays = arrays.at["dispersive_c2"].set(jax.lax.stop_gradient(arrays.dispersive_c2))
        arrays = arrays.at["dispersive_c3"].set(jax.lax.stop_gradient(arrays.dispersive_c3))
    # ``dispersive_inv_c2`` is a derived cache of ``1/c2``; stop_gradient it
    # unconditionally (even when differentiate_dispersion=True) so gradients
    # flow through ``dispersive_c2`` only and don't double-count.
    if arrays.dispersive_inv_c2 is not None:
        arrays = arrays.at["dispersive_inv_c2"].set(jax.lax.stop_gradient(arrays.dispersive_inv_c2))

    pbar = _make_pbar(
        show_progress=show_progress,
        total_steps=config.time_steps_total,
        desc="FDTD (reversible)",
    )

    # Build the (optionally instrumented) forward body function once so both
    # reversible_fdtd_base and fdtd_bwd share the same wrapping logic.
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

    def reversible_fdtd_base(
        arr: ArrayContainer,
    ) -> SimulationState:
        state = (jnp.asarray(0, dtype=jnp.int32), arr)
        state = eqxi.while_loop(
            max_steps=config.time_steps_total,
            cond_fun=lambda s: config.time_steps_total > s[0],
            body_fun=_forward_body_with_progress,
            init_val=state,
            kind="lax",
        )
        return (state[0], state[1])

    def cond_fun(
        sr_tuple,
        start_time_step: int,
    ):
        s_k, r_k = sr_tuple
        del r_k
        time_step = s_k[0]
        return time_step >= start_time_step

    # Shared closure fields — constant across both branches. ``dispersive_inv_c2``
    # is always closure-captured (never a primal arg), regardless of the
    # ``differentiate_dispersion`` flag, since it's a cached reciprocal of c2.
    closure_kwargs = {
        "electric_conductivity": arrays.electric_conductivity,
        "magnetic_conductivity": arrays.magnetic_conductivity,
        "dispersive_c1": arrays.dispersive_c1,
        "dispersive_c2": arrays.dispersive_c2,
        "dispersive_c3": arrays.dispersive_c3,
        "dispersive_inv_c2": arrays.dispersive_inv_c2,
    }
    forward_wrapper = partial(
        forward_single_args_wrapper,
        config=config,
        objects=objects,
        key=key,
        record_detectors=True,
        record_boundaries=False,
        simulate_boundaries=True,
        arrays_template=arrays,
    )

    # The two branches share the shape of fdtd_fwd / fdtd_bwd / body_fn; the
    # only real difference is whether dispersive_c1/c2/c3 are threaded through
    # as primal VJP inputs (flag on) or closure-captured (flag off). Keeping
    # two ``jax.vjp`` call shapes is what preserves the perf savings — if the
    # coefficients were always passed as primals, JAX would compute their
    # transposes even when the flag is off.

    if not differentiate_dispersion:

        @jax.custom_vjp
        def reversible_fdtd_primal(*state_primals):
            arr = _arr_from_primals(*state_primals[1:], **closure_kwargs)
            return _state_to_primal_tuple(reversible_fdtd_base(arr))

        def body_fn(sr_tuple):
            state, cot = sr_tuple
            state = backward(
                state=state,
                config=config,
                objects=objects,
                key=key,
                record_detectors=False,
                reset_fields=False,
            )
            _, update_vjp = jax.vjp(forward_wrapper, *_state_to_primal_tuple(state))
            return state, update_vjp(cot)

        def fdtd_fwd(*state_primals):
            arr = _arr_from_primals(*state_primals[1:], **closure_kwargs)
            primal_out = _state_to_primal_tuple(reversible_fdtd_base(arr))
            return primal_out, primal_out

        def fdtd_bwd(residual, cot):
            s_k = _arr_from_primals(*residual[1:], **closure_kwargs)
            _, final_cot = eqxi.while_loop(
                cond_fun=partial(cond_fun, start_time_step=0),
                body_fun=body_fn,
                init_val=((residual[0], s_k), cot),
                kind="lax",
            )
            # Only inv_permittivities (idx 8) and inv_permeabilities (idx 9)
            # propagate gradients; all other state leaves are non-trainable.
            # Indices refer to the 14-tuple primal-argument order: time_step,
            # E, H, psi_E, psi_H, alpha, kappa, sigma, inv_permittivities,
            # inv_permeabilities, P_curr, P_prev, detector_states, recording_state.
            return (
                None,  # time_step
                None,  # E
                None,  # H
                None,  # psi_E
                None,  # psi_H
                None,  # alpha
                None,  # kappa
                None,  # sigma
                final_cot[8],  # inv_permittivities
                final_cot[9],  # inv_permeabilities
                None,  # dispersive_P_curr
                None,  # dispersive_P_prev
                None,  # detector_states
                None,  # recording_state
            )

        reversible_fdtd_primal.defvjp(fdtd_fwd, fdtd_bwd)
        primal_out = reversible_fdtd_primal(*_state_to_primal_tuple((jnp.asarray(0, dtype=jnp.int32), arrays)))
    else:
        # Flag on: c1/c2/c3 become primal inputs, so the VJP machinery
        # accumulates cotangents for them naturally through ``update_vjp``'s
        # output. But because they are inputs-only (not outputs of
        # ``forward_single_args_wrapper``), their per-step cotangent is a
        # local contribution; we carry a 17-tuple cot so the three extra
        # entries accumulate over the reverse loop.
        coef_kwargs_closure = {k: v for k, v in closure_kwargs.items() if not k.startswith("dispersive_c")}

        @jax.custom_vjp
        def reversible_fdtd_primal(*primals):
            state_primals, c1, c2, c3 = primals[:14], primals[14], primals[15], primals[16]
            arr = _arr_from_primals(
                *state_primals[1:],
                **coef_kwargs_closure,
                dispersive_c1=c1,
                dispersive_c2=c2,
                dispersive_c3=c3,
            )
            return _state_to_primal_tuple(reversible_fdtd_base(arr))

        def body_fn(sr_tuple):
            state, cot17 = sr_tuple
            state = backward(
                state=state,
                config=config,
                objects=objects,
                key=key,
                record_detectors=False,
                reset_fields=False,
            )
            _, update_vjp = jax.vjp(
                forward_wrapper,
                *_state_to_primal_tuple(state),
                state[1].dispersive_c1,
                state[1].dispersive_c2,
                state[1].dispersive_c3,
            )
            # update_vjp output cot is 14-tuple (wrapper's outputs); input cot is 17-tuple.
            input_cot = update_vjp(cot17[:14])
            return state, (
                *input_cot[:14],
                cot17[14] + input_cot[14],
                cot17[15] + input_cot[15],
                cot17[16] + input_cot[16],
            )

        def fdtd_fwd(*primals):
            state_primals, c1, c2, c3 = primals[:14], primals[14], primals[15], primals[16]
            arr = _arr_from_primals(
                *state_primals[1:],
                **coef_kwargs_closure,
                dispersive_c1=c1,
                dispersive_c2=c2,
                dispersive_c3=c3,
            )
            s_k = reversible_fdtd_base(arr)
            primal_out = _state_to_primal_tuple(s_k)
            residual = (*primal_out, s_k[1].dispersive_c1, s_k[1].dispersive_c2, s_k[1].dispersive_c3)
            return primal_out, residual

        def fdtd_bwd(residual, cot):
            state_residual = residual[:14]
            c1_res, c2_res, c3_res = residual[14], residual[15], residual[16]
            s_k = _arr_from_primals(
                *state_residual[1:],
                **coef_kwargs_closure,
                dispersive_c1=c1_res,
                dispersive_c2=c2_res,
                dispersive_c3=c3_res,
            )
            # Seed the extra 3 cotangents as zero — they accumulate per step.
            cot17 = (*cot, jnp.zeros_like(c1_res), jnp.zeros_like(c2_res), jnp.zeros_like(c3_res))
            _, final_cot17 = eqxi.while_loop(
                cond_fun=partial(cond_fun, start_time_step=0),
                body_fun=body_fn,
                init_val=((state_residual[0], s_k), cot17),
                kind="lax",
            )
            return (
                None,  # time_step
                None,  # E
                None,  # H
                None,  # psi_E
                None,  # psi_H
                None,  # alpha
                None,  # kappa
                None,  # sigma
                final_cot17[8],  # inv_permittivities
                final_cot17[9],  # inv_permeabilities
                None,  # dispersive_P_curr
                None,  # dispersive_P_prev
                None,  # detector_states
                None,  # recording_state
                final_cot17[14],  # dispersive_c1
                final_cot17[15],  # dispersive_c2
                final_cot17[16],  # dispersive_c3
            )

        reversible_fdtd_primal.defvjp(fdtd_fwd, fdtd_bwd)
        primal_out = reversible_fdtd_primal(
            *_state_to_primal_tuple((jnp.asarray(0, dtype=jnp.int32), arrays)),
            arrays.dispersive_c1,
            arrays.dispersive_c2,
            arrays.dispersive_c3,
        )

    (
        time_step,
        E,
        H,
        psi_E,
        psi_H,
        alpha,
        kappa,
        sigma,
        inv_permittivities,
        inv_permeabilities,
        dispersive_P_curr,
        dispersive_P_prev,
        detector_states,
        recording_state,
    ) = primal_out
    _close_pbar()

    out_arrs = ArrayContainer(
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
        electric_conductivity=arrays.electric_conductivity,
        magnetic_conductivity=arrays.magnetic_conductivity,
        dispersive_P_curr=dispersive_P_curr,
        dispersive_P_prev=dispersive_P_prev,
        dispersive_c1=arrays.dispersive_c1,
        dispersive_c2=arrays.dispersive_c2,
        dispersive_c3=arrays.dispersive_c3,
        dispersive_inv_c2=arrays.dispersive_inv_c2,
    )
    return time_step, out_arrs


def checkpointed_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    stopping_condition: StoppingCondition | None = None,
    show_progress: bool = True,
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
    arrays = reset_array_container(arrays, objects)

    # The checkpointed path uses standard autodiff via eqxi.while_loop, so
    # dispersion coefficients carried inside ``arrays`` naturally accumulate
    # gradient contributions across time steps. To match the reversible path's
    # default behavior (no gradient through coefficients), wrap them in
    # stop_gradient unless the user opts in via GradientConfig.differentiate_dispersion.
    if arrays.dispersive_c1 is not None and not config.differentiate_dispersion:
        arrays = arrays.at["dispersive_c1"].set(jax.lax.stop_gradient(arrays.dispersive_c1))
        arrays = arrays.at["dispersive_c2"].set(jax.lax.stop_gradient(arrays.dispersive_c2))
        arrays = arrays.at["dispersive_c3"].set(jax.lax.stop_gradient(arrays.dispersive_c3))
    # ``dispersive_inv_c2`` is a derived cache — always stop_gradient here (gradients
    # flow through ``dispersive_c2`` instead, preventing double-counting).
    if arrays.dispersive_inv_c2 is not None:
        arrays = arrays.at["dispersive_inv_c2"].set(jax.lax.stop_gradient(arrays.dispersive_inv_c2))

    state = (jnp.asarray(0, dtype=jnp.int32), arrays)
    if stopping_condition is not None:
        stopping_condition = stopping_condition.setup(state, config, objects)
    else:
        stopping_condition = TimeStepCondition().setup(state, config, objects)

    pbar = _make_pbar(
        show_progress=show_progress,
        total_steps=config.time_steps_total,
        desc="FDTD (checkpointed)",
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
        arrays = reset_array_container(arrays, objects)
    state = (jnp.asarray(start_time, dtype=jnp.int32), arrays)

    # start_time and end_time must be statically known Python ints here so that
    # we can compute n_steps for the progress bar without triggering JAX
    # concretization.  They are always statically known at call sites of this
    # function (they control the loop bound, not an array value).
    if isinstance(start_time, jax.Array) or isinstance(end_time, jax.Array):
        # Traced arrays: skip the progress bar entirely to avoid concretization.
        show_progress = False
        n_steps = 0
    else:
        n_steps = int(end_time) - int(start_time)

    pbar = _make_pbar(
        show_progress=show_progress,
        total_steps=n_steps,
        desc="FDTD (forward)",
        step_offset=0 if not show_progress else int(start_time),
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
