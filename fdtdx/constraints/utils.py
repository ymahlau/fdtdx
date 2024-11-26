import itertools
from functools import partial
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import tqdm



def compute_allowed_indices(
    num_layers: int,
    indices: Sequence[int],
    fill_holes_with_index: Sequence[int],
    single_polymer_columns: bool = False,
):
    if single_polymer_columns:
        return compute_allowed_indices_without_holes_single_polymer_columns(
            num_layers, indices, fill_holes_with_index
        )
    else:
        return compute_allowed_indices_without_holes(
            num_layers, indices, fill_holes_with_index
        )


def compute_allowed_indices_without_holes_single_polymer_columns(
    num_layers: int,
    indices: Sequence[int],
    fill_holes_with_index: Sequence[int],
):
    if not fill_holes_with_index:
        all_permutations = list(itertools.product(indices, repeat=num_layers))
        return jnp.array(all_permutations)
    valid_indices = [idx for idx in indices if idx not in fill_holes_with_index]
    all_permutations = list(itertools.product(valid_indices, repeat=num_layers))
    all_valid_permutations = []
    for perm in all_permutations:
        for fill_index in fill_holes_with_index:
            for i in range(num_layers + 1):
                filled_perm = perm[: num_layers - i] + (fill_index,) * i
                unique_elements = set(filled_perm)
                if (
                    len(unique_elements) == 1
                    or len(unique_elements - set(fill_holes_with_index)) <= 1
                ):
                    all_valid_permutations.append(filled_perm)
    return jnp.unique(jnp.array(all_valid_permutations), axis=0)


def compute_allowed_indices_without_holes(
    num_layers: int,
    indices: Sequence[int],
    fill_holes_with_index: Sequence[int],
):
    valid_indices = [idx for idx in indices if idx not in fill_holes_with_index]
    unique_permutations = set()
    pbar = tqdm.tqdm(total=len(valid_indices) ** num_layers)

    for perm in itertools.product(valid_indices, repeat=num_layers):
        for fill_index in fill_holes_with_index:
            for i in range(num_layers + 1):
                filled_perm = perm[: num_layers - i] + (fill_index,) * i
                unique_permutations.add(filled_perm)
        pbar.update(1)

    # Convert the set of unique permutations to a JAX array
    final_permutations = jnp.array(list(unique_permutations))
    return final_permutations


def nearest_index(
    values: jax.Array,
    allowed_values: jax.Array,
    axis: int | None = None,
    allowed_indices: jax.Array | None = None,
    return_distances: bool = False,
    distance_metric: Literal[
        "euclidean",
        "permittivity_differences_plus_average_permittivity",
    ] = "permittivity_differences_plus_average_permittivity",
):
    if allowed_indices is None:
        distances = jnp.square(values[..., None] - allowed_values)
    else:
        if axis is None:
            raise Exception(f"Need axis when using allowed indices option")
        if values.ndim != 3:
            raise Exception(f"Invalid array shape: {values.shape}")
        
        def _index_helper(values, idx):
            return values[idx]
        vmap_idx_fn = jax.vmap(_index_helper, in_axes=(None, 0))
        allowed_values_per_index = vmap_idx_fn(allowed_values, allowed_indices)
        
        if axis == 0:
            allowed_values_per_index = allowed_values_per_index[:, :, None, None]
        elif axis == 1:
            allowed_values_per_index = allowed_values_per_index[:, None, :, None]
        elif axis == 2:
            allowed_values_per_index = allowed_values_per_index[:, None, None, :]
        else:
            raise Exception(f"Invalid axis: {axis}")
        
        if distance_metric == "euclidean" or values.shape[axis] == 1:
            distances = jnp.linalg.norm(
                values[None, ...] - allowed_values_per_index, 
                axis=axis+1,
            )
        elif distance_metric == "permittivity_differences_plus_average_permittivity":
            distances = jnp.mean(
                jnp.abs(
                    jnp.diff(values[None, ...], axis=axis+1)
                    - jnp.diff(allowed_values_per_index, axis=axis+1)
                ),
                axis=axis+1,
            ) + jnp.abs(
                values[None, ...].mean(axis=axis+1)
                - allowed_values_per_index.mean(axis=axis+1)
            )
        else:
            raise ValueError(f"Unknown distance metric {distance_metric}")

    indices = jnp.argmin(distances, axis=0)
    if allowed_indices is None:
        indices = jnp.reshape(indices, values.shape)
    if return_distances:
        return indices, distances
    return indices