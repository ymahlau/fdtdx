import jax

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.fdtd import checkpointed_fdtd, reversible_fdtd


def run_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
) -> SimulationState:
    if config.gradient_config is None:
        # only forward simulation, use standard while loop of checkpointed fdtd
        return checkpointed_fdtd(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
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
        )
    else:
        raise Exception(f"Unknown gradient computation method: {config.gradient_config.method}")
