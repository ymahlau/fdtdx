from typing import Any, Sequence

import jax
import jax.numpy as jnp


def metric_efficiency(
    detector_states: dict[str, dict[str, jax.Array]],
    in_names: Sequence[str],
    out_names: Sequence[str],
    metric_name: str,
) -> tuple[jax.Array, dict[str, Any]]:
    """Calculate efficiency metrics between input and output detectors.

    Computes efficiency ratios between input and output detectors by comparing their
    metric values (e.g. energy, power). For each input-output detector pair, calculates
    the ratio of output/input metric values.

    Args:
        detector_states (dict[str, dict[str, jax.Array]]): Dictionary mapping detector names to their state dictionaries,
            which contain metric values as JAX arrays
        in_names (Sequence[str]): Names of input detectors to use as reference
        out_names (Sequence[str]): Names of output detectors to compare against inputs
        metric_name (str): Name of the metric to compare between detectors (e.g. "energy")

    Returns:
        tuple[jax.Array, dict[str, Any]]: tuple containing:
            - jax.Array: Mean efficiency across all input-output pairs
            - dict: Additional info including individual metric values and efficiencies
              with keys like:
                "{detector}_{metric}" for raw metric values
                "{out}_{by}_{in}_efficiency" for individual efficiency ratios
    """
    efficiencies, info = [], {}
    for in_name in in_names:
        in_value = jax.lax.stop_gradient(detector_states[in_name][metric_name].mean())
        info[f"{in_name}_{metric_name}"] = in_value
        for out_name in out_names:
            out_value = detector_states[out_name][metric_name].mean()
            eff = jnp.where(in_value == 0, 0, out_value / in_value)
            efficiencies.append(eff)
            info[f"{out_name}_{metric_name}"] = out_value
            info[f"{out_name}_by_{in_name}_efficiency"] = eff
    objective = jnp.mean(jnp.asarray(efficiencies))
    return objective, info
