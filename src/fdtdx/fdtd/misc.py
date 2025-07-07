from typing import Sequence

import jax

from fdtdx.fdtd.container import ArrayContainer
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer


def collect_boundary_interfaces(
    arrays: ArrayContainer,
    pml_objects: Sequence[PerfectlyMatchedLayer],
    fields_to_collect: Sequence[str] = ("E", "H"),
) -> dict[str, jax.Array]:
    """Collects field values at PML boundary interfaces.

    Extracts field values at the interfaces between PML regions and the main simulation
    volume. This is used to enable time-reversible automatic differentiation by saving
    boundary values that would otherwise be lost due to PML absorption.

    Args:
        arrays (ArrayContainer): Container holding the field arrays (E, H fields)
        pml_objects (Sequence[PerfectlyMatchedLayer]): Sequence of PML objects defining boundary regions
        fields_to_collect (Sequence[str], optional): Which fields to collect values for (default: E and H fields)

    Returns:
        dict[str, jax.Array]: Dictionary mapping "{pml_name}_{field_str}" to array of interface field values
    """
    res = {}
    for field_str in fields_to_collect:
        arr: jax.Array = getattr(arrays, field_str)
        for pml in pml_objects:
            cur_slice = arr[:, *pml.boundary_interface_slice()]
            res[f"{pml.name}_{field_str}"] = cur_slice
    return res


def add_boundary_interfaces(
    arrays: ArrayContainer,
    values: dict[str, jax.Array],
    pml_objects: Sequence[PerfectlyMatchedLayer],
    fields_to_add: Sequence[str] = ("E", "H"),
) -> ArrayContainer:
    """Adds saved field values back to PML boundary interfaces.

    Restores previously collected field values at the interfaces between PML regions
    and the main simulation volume. This is the inverse operation to collect_boundary_interfaces()
    and is used during time-reversed automatic differentiation.

    Args:
        arrays (ArrayContainer): Container holding the field arrays to update
        values (dict[str, jax.Array]): Dictionary of saved interface values from collect_boundary_interfaces()
        pml_objects (Sequence[PerfectlyMatchedLayer]): Sequence of PML objects defining boundary regions
        fields_to_add (Sequence[str], optional): Which fields to restore values for (default: E and H fields)

    Returns:
        ArrayContainer: Updated ArrayContainer with restored interface field values
    """
    updated_dict = {}
    for field_str in fields_to_add:
        arr: jax.Array = getattr(arrays, field_str)
        for pml in pml_objects:
            val = values[f"{pml.name}_{field_str}"]
            grid_slice = pml.boundary_interface_slice()
            arr = arr.at[:, *grid_slice].set(val)
        updated_dict[field_str] = arr

    for k, v in updated_dict.items():
        arrays = arrays.at[k].set(v)

    return arrays
