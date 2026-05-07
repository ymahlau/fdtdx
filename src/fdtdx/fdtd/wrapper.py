from __future__ import annotations

import jax

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.fdtd import checkpointed_fdtd, reversible_fdtd
from fdtdx.fdtd.stop_conditions import StoppingCondition

_DEFAULT_KEY_SEED = 0


def _default_key(key: jax.Array | None) -> jax.Array:
    """Return *key* unchanged, or create a deterministic fallback key from a fixed seed."""
    if key is None:
        return jax.random.PRNGKey(_DEFAULT_KEY_SEED)
    return key


def run_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array | None = None,
    stopping_condition: StoppingCondition | None = None,
    show_progress: bool = True,
    progress_callback=None,
) -> SimulationState:
    key = _default_key(key)
    if stopping_condition is not None:
        if config.gradient_config is not None:
            raise NotImplementedError(
                "Custom stopping conditions are not yet compatible with gradient computation. "
                "Set config.gradient_config to None or use default time-based stopping by "
                "setting stopping_condition=None."
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
