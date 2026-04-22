from __future__ import annotations

import warnings
from typing import cast

import numpy as np

from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.objects.static_material.static import UniformMaterialObject


def _compute_default_inv_values(raw_prop: tuple, C: int, prop_name: str) -> np.ndarray:
    """Compute expected default array values for a given material property and component count.

    Args:
        raw_prop: 9-tuple material property from Material (xx, xy, xz, yx, yy, yz, zx, zy, zz).
        C: Number of components (1 isotropic, 3 diagonally anisotropic, 9 fully anisotropic).
        prop_name: Name of the property; conductivity properties have default 0.0.

    Returns:
        1-D numpy array of length C with default inverse (or zero) values.
    """
    if prop_name in ("electric_conductivity", "magnetic_conductivity"):
        return np.zeros(C)  # background has zero conductivity
    if C == 1:
        return np.array([1.0 / raw_prop[0]])
    elif C == 3:
        return np.array([1.0 / raw_prop[0], 1.0 / raw_prop[4], 1.0 / raw_prop[8]])
    else:  # C == 9
        mat = np.array(raw_prop, dtype=float).reshape(3, 3)
        return np.linalg.inv(mat).flatten()


def _check_and_extend(
    arrays: ArrayContainer,
    field_name: str,
    pml_idx: tuple,
    interior_idx: tuple,
    volume_material,
    prop_name: str,
    axis: int,
    direction: str,
) -> ArrayContainer:
    arr = getattr(arrays, field_name)
    C = arr.shape[0]

    raw_prop = getattr(volume_material, prop_name)
    default_vals = _compute_default_inv_values(raw_prop, C, prop_name)
    default_broadcast = default_vals.reshape(C, *([1] * (arr.ndim - 1)))

    pml_region = np.asarray(arr[pml_idx])
    if not np.allclose(pml_region, default_broadcast, rtol=1e-5, atol=1e-8):
        warnings.warn(
            f"PML region at axis={axis} direction={direction!r} has non-default "
            f"{field_name} values that will be overwritten.",
            UserWarning,
            stacklevel=3,
        )

    interior_vals = arr[interior_idx]  # shape (C, 1, cross...) — broadcasts across PML depth
    new_arr = arr.at[pml_idx].set(interior_vals)
    return arrays.aset(field_name, new_arr)


def extend_material_to_pml(
    objects: ObjectContainer,
    arrays: ArrayContainer,
) -> ArrayContainer:
    """Extend interior-edge material values into each PML region.

    For each PML boundary in ``objects.pml_objects``, the material values at the
    last non-PML grid cell (the "interior edge") are broadcast across the entire
    PML depth.  This is applied to ``inv_permittivities``, ``inv_permeabilities``
    (when it is an array rather than a scalar float), ``electric_conductivity``,
    and ``magnetic_conductivity`` (when they are not ``None``).

    Args:
        objects: ObjectContainer returned by :func:`place_objects`.
        arrays: ArrayContainer returned by :func:`place_objects`.

    Returns:
        Updated ArrayContainer with PML regions filled from interior-edge values.

    Warns:
        UserWarning: If a PML region already contains non-default (non-background)
            material values that would be overwritten.
    """
    volume = cast(UniformMaterialObject, objects.volume)
    for pml in objects.pml_objects:
        axis = pml.axis
        direction = pml.direction
        T = pml.thickness  # grid cells deep in the PML
        N = objects.volume.grid_shape[axis]  # full domain size along axis

        gs = pml.grid_slice  # (s0, s1, s2) — PML covers axis slice, full cross-section

        # Index of the last interior cell adjacent to this PML boundary
        edge = T if direction == "-" else N - T - 1
        edge_slice = slice(edge, edge + 1)

        interior_gs = list(gs)
        interior_gs[axis] = edge_slice
        interior_gs = tuple(interior_gs)

        pml_idx = (slice(None), *gs)  # (C, PML depth, cross...)
        interior_idx = (slice(None), *interior_gs)  # (C, 1, cross...)

        arrays = _check_and_extend(
            arrays,
            "inv_permittivities",
            pml_idx,
            interior_idx,
            volume.material,
            "permittivity",
            axis,
            direction,
        )

        if not isinstance(arrays.inv_permeabilities, float):
            arrays = _check_and_extend(
                arrays,
                "inv_permeabilities",
                pml_idx,
                interior_idx,
                volume.material,
                "permeability",
                axis,
                direction,
            )

        if arrays.electric_conductivity is not None:
            arrays = _check_and_extend(
                arrays,
                "electric_conductivity",
                pml_idx,
                interior_idx,
                volume.material,
                "electric_conductivity",
                axis,
                direction,
            )

        if arrays.magnetic_conductivity is not None:
            arrays = _check_and_extend(
                arrays,
                "magnetic_conductivity",
                pml_idx,
                interior_idx,
                volume.material,
                "magnetic_conductivity",
                axis,
                direction,
            )

    return arrays
