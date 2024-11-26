import jax

from typing import Sequence
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.container import ArrayContainer


def collect_boundary_interfaces(
    arrays: ArrayContainer,
    pml_objects: Sequence[PerfectlyMatchedLayer],
    fields_to_collect: Sequence[str] = ("E", "H"),
) -> dict[str, jax.Array]:
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
    


