from typing import Sequence

import jax
import jax.numpy as jnp

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
            cur_slice = arr[:, *pml.interface_slice()]
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
            grid_slice = pml.interface_slice()
            arr = arr.at[:, *grid_slice].set(val)
        updated_dict[field_str] = arr

    for k, v in updated_dict.items():
        arrays = arrays.at[k].set(v)

    return arrays


def compute_anisotropic_update_matrices(
    inv_material_prop: jax.Array,
    sigma: jax.Array | None,
    c: float,
    eta_factor: float,
) -> tuple[jax.Array, jax.Array]:
    """Computes the A and B matrices for anisotropic FDTD updates.

    Args:
        inv_material_prop (jax.Array): Inverse material property tensor (3, 3, Nx, Ny, Nz)
        sigma (jax.Array | None): Conductivity tensor (3, 3, Nx, Ny, Nz) or None
        c (float): Courant number
        eta_factor (float): eta0 for electric, 1/eta0 for magnetic

    Returns:
        tuple[jax.Array, jax.Array]: A and B matrices
    """

    M1 = jnp.eye(3)[:, :, None, None, None]
    M2 = jnp.eye(3)[:, :, None, None, None]
    if sigma is not None:
        factor = c * eta_factor / 2 * jnp.einsum("ijxyz,jkxyz->ikxyz", inv_material_prop, sigma)
        M1 += factor
        M2 -= factor
    perm = (2, 3, 4, 0, 1)  # (3, 3, Nx, Ny, Nz) -> (Nx, Ny, Nz, 3, 3)
    inv_perm = (3, 4, 0, 1, 2)  # (Nx, Ny, Nz, 3, 3) -> (3, 3, Nx, Ny, Nz)
    A = jnp.linalg.solve(M1.transpose(perm), M2.transpose(perm)).transpose(inv_perm)
    B = c * jnp.linalg.solve(M1.transpose(perm), inv_material_prop.transpose(perm)).transpose(inv_perm)

    return A, B


def compute_anisotropic_update_matrices_reverse(
    inv_material_prop: jax.Array,
    sigma: jax.Array | None,
    c: float,
    eta_factor: float,
) -> tuple[jax.Array, jax.Array]:
    """Computes the A and B matrices for reverse anisotropic FDTD updates.

    Args:
        inv_material_prop (jax.Array): Inverse material property tensor (3, 3, Nx, Ny, Nz)
        sigma (jax.Array | None): Conductivity tensor (3, 3, Nx, Ny, Nz) or None
        c (float): Courant number
        eta_factor (float): eta0 for electric, 1/eta0 for magnetic

    Returns:
        tuple[jax.Array, jax.Array]: A and B matrices
    """
    M1 = jnp.eye(3)[:, :, None, None, None]
    M2 = jnp.eye(3)[:, :, None, None, None]
    if sigma is not None:
        factor = c * eta_factor / 2 * jnp.einsum("ijxyz,jkxyz->ikxyz", inv_material_prop, sigma)
        M1 += factor
        M2 -= factor
    perm = (2, 3, 4, 0, 1)  # (3, 3, Nx, Ny, Nz) -> (Nx, Ny, Nz, 3, 3)
    inv_perm = (3, 4, 0, 1, 2)  # (Nx, Ny, Nz, 3, 3) -> (3, 3, Nx, Ny, Nz)
    A = jnp.linalg.solve(M2.transpose(perm), M1.transpose(perm)).transpose(inv_perm)
    B = c * jnp.linalg.solve(M2.transpose(perm), inv_material_prop.transpose(perm)).transpose(inv_perm)

    return A, B


def avg_anisotropic_E_component(
    field: jax.Array,
    component: int,
    location: int,
) -> jax.Array:
    """Averages an E field component at a given location.

    Args:
        field (jax.Array): E field to average (3, Nx, Ny, Nz)
        component (int): Component to average, 0 for Ex, 1 for Ey, 2 for Ez
        location (int): Location to calculate average, 0 for Ex, 1 for Ey, 2 for Ez

    Returns:
        jax.Array: Averaged E field component
    """

    return (
        (
            field[component]
            + jnp.roll(field[component], -1, axis=location)
            + jnp.roll(field[component], 1, axis=component)
            + jnp.roll(field[component], (-1, 1), axis=(location, component))
        )
        / 4
    )[1:-1, 1:-1, 1:-1]


def avg_anisotropic_H_component(
    field: jax.Array,
    component: int,
    location: int,
) -> jax.Array:
    """Averages an H field component at a given location.

    Args:
        field (jax.Array): H field to average (3, Nx, Ny, Nz)
        component (int): Component to average, 0 for Hx, 1 for Hy, 2 for Hz
        location (int): Location to calculate average, 0 for Hx, 1 for Hy, 2 for Hz

    Returns:
        jax.Array: Averaged H field component
    """

    return (
        (
            field[component]
            + jnp.roll(field[component], 1, axis=location)
            + jnp.roll(field[component], -1, axis=component)
            + jnp.roll(field[component], (1, -1), axis=(location, component))
        )
        / 4
    )[1:-1, 1:-1, 1:-1]
