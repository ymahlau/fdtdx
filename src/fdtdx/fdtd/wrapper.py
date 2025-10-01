import jax

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.fdtd import checkpointed_fdtd, reversible_fdtd
from fdtdx.fdtd.stop_conditions import StoppingCondition


def run_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    stopping_condition: StoppingCondition | None = None,
) -> SimulationState:
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
        )
    if config.gradient_config.method == "reversible":
        return reversible_fdtd(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
        )
    elif config.gradient_config.method == "checkpointed":
        return checkpointed_fdtd(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            stopping_condition=stopping_condition,
        )
    else:
        raise Exception(f"Unknown gradient computation method: {config.gradient_config.method}")
