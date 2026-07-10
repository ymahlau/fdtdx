from __future__ import annotations

from collections.abc import Callable

import jax

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.default_key import default_key
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.fdtd import checkpointed_fdtd, reversible_fdtd
from fdtdx.fdtd.stop_conditions import StoppingCondition


def run_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array | None = None,
    stopping_condition: StoppingCondition | None = None,
    show_progress: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimulationState:
    key = default_key(key)
    if stopping_condition is not None:
        if config.gradient_config is not None:
            raise NotImplementedError(
                "Custom stopping conditions are not yet compatible with gradient computation. "
                "Set config.gradient_config to None or use default time-based stopping by "
                "setting stopping_condition=None."
            )

    if (
        config.gradient_config is not None
        and config.gradient_config.method == "reversible"
        and arrays.dispersive_c1 is not None
    ):
        # The fully anisotropic update path has no closed-form time reversal for
        # dispersion. Checked here (shape-based) in addition to initialization
        # time, since the gradient config can be swapped after place_objects.
        tensor_path = (
            arrays.inv_permittivities.shape[0] == 9
            or (arrays.electric_conductivity is not None and arrays.electric_conductivity.shape[0] == 9)
            or (arrays.dispersive_c3 is not None and arrays.dispersive_c3.shape[1] == 9)
        )
        if tensor_path:
            raise NotImplementedError(
                "Dispersion combined with oriented poles or off-diagonal material tensors supports "
                "only the 'checkpointed' gradient method, not 'reversible'."
            )

    if config.gradient_config is None:
        # only forward simulation, use standard while loop of checkpointed fdtd
        return checkpointed_fdtd(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            stopping_condition=stopping_condition,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )
    if config.gradient_config.method == "reversible":
        return reversible_fdtd(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )
    elif config.gradient_config.method == "checkpointed":
        return checkpointed_fdtd(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            stopping_condition=stopping_condition,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )
    else:
        raise Exception(f"Unknown gradient computation method: {config.gradient_config.method}")
