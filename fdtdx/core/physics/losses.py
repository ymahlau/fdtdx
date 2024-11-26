import jax
import jax.numpy as jnp
from typing import Any, Sequence


def metric_efficiency(
    detector_states: dict[str, dict[str, jax.Array]],
    in_names: Sequence[str],
    out_names: Sequence[str],
    metric_name: str,
) -> tuple[jax.Array, dict[str, Any]]:
    efficiencies, info = [], {}
    for in_name in in_names:
        in_value = jax.lax.stop_gradient(
            detector_states[in_name][metric_name].mean()
        )
        info[f"{in_name}_{metric_name}"] = in_value
        for out_name in out_names:
            out_value = detector_states[out_name][metric_name].mean()
            eff = jnp.where(in_value == 0, 0, out_value / in_value)
            efficiencies.append(eff)
            info[f"{out_name}_{metric_name}"] = out_value
            info[f"{out_name}_by_{in_name}_efficiency"] = eff
    objective = jnp.mean(jnp.asarray(efficiencies))
    return objective, info