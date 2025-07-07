from typing import Literal

import jax
import jax.numpy as jnp


def compute_extent(
    kind: Literal["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"],
    thickness: int,
) -> tuple[slice, slice, slice]:
    """Computes slice objects for indexing the PML boundary region.

    Args:
        kind (Literal["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]): Which boundary to compute extent for
            ('min_x', 'max_x', etc).
        thickness (int): Number of grid cells for PML thickness

    Returns:
        tuple[slice, slice, slice]: Three slice objects for indexing the x, y, z dimensions
            of the PML boundary region

    Raises:
        ValueError: If kind is not one of the valid boundary names
    """
    if kind == "min_x":
        return (slice(None, thickness), slice(None), slice(None))
    elif kind == "max_x":
        return (slice(-thickness, None), slice(None), slice(None))
    elif kind == "min_y":
        return (slice(None), slice(None, thickness), slice(None))
    elif kind == "max_y":
        return (slice(None), slice(-thickness, None), slice(None))
    elif kind == "min_z":
        return (slice(None), slice(None), slice(None, thickness))
    elif kind == "max_z":
        return (slice(None), slice(None), slice(-thickness, None))
    else:
        raise ValueError(f"Invalid kind: {kind}")


def compute_extent_boundary(
    kind: Literal["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"],
    thickness: int,
) -> tuple[slice, slice, slice]:
    """Computes slice objects for indexing the interface between PML and main simulation region.

    Args:
        kind (Literal["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]): Which boundary interface to compute
            ('minx', 'maxx', etc).
        thickness (int): Number of grid cells for PML thickness

    Returns:
        tuple[slice, slice, slice]: Three slice objects for indexing the x, y, z dimensions
            of the boundary interface region

    Raises:
        ValueError: If kind is not one of the valid boundary names
    """
    if kind == "min_x":
        return (
            slice(thickness + 1, thickness + 2),
            slice(thickness + 1, -thickness - 1),
            slice(thickness + 1, -thickness - 1),
        )
    elif kind == "max_x":
        return (
            slice(-thickness - 2, -thickness - 1),
            slice(thickness + 1, -thickness - 1),
            slice(thickness + 1, -thickness - 1),
        )
    elif kind == "min_y":
        return (
            slice(thickness + 1, -thickness - 1),
            slice(thickness + 1, thickness + 2),
            slice(thickness + 1, -thickness - 1),
        )
    elif kind == "max_y":
        return (
            slice(thickness + 1, -thickness - 1),
            slice(-thickness - 2, -thickness - 1),
            slice(thickness + 1, -thickness - 1),
        )
    elif kind == "min_z":
        return (
            slice(thickness + 1, -thickness - 1),
            slice(thickness + 1, -thickness - 1),
            slice(thickness + 1, thickness + 2),
        )
    elif kind == "max_z":
        return (
            slice(thickness + 1, -thickness - 1),
            slice(thickness + 1, -thickness - 1),
            slice(-thickness - 2, -thickness - 1),
        )
    else:
        raise ValueError(f"Invalid kind: {kind}")


def sigma_fn(x: jax.Array, thickness: int) -> jax.Array:
    """Creates a cubically increasing conductivity profile for PML boundaries.

    Implements σ(x) = 40x³/(thickness+1)⁴ profile for PML conductivity.

    Args:
        x (jax.Array): Position values to compute conductivity for
        thickness (int): Total thickness of PML region in grid cells

    Returns:
        jax.Array: Conductivity values following cubic profile
    """
    return 40 * x**3 / (thickness + 1) ** 4


def standard_min_sigma_E_fn(thickness: int, dtype: jnp.dtype) -> jax.Array:
    """Computes standard conductivity profile for E-field at minimum boundary.

    Args:
        thickness (int): Number of grid cells for PML thickness
        dtype (jnp.dtype): Data type for the returned array

    Returns:
        jax.Array: Conductivity values for E-field, decreasing from boundary inward
    """
    return sigma_fn(
        x=jnp.arange(thickness - 0.5, -0.5, -1.0, dtype=dtype),
        thickness=thickness,
    )


def standard_min_sigma_H_fn(thickness: int, dtype: jnp.dtype) -> jax.Array:
    """Computes standard conductivity profile for H-field at minimum boundary.

    Args:
        thickness (int): Number of grid cells for PML thickness
        dtype (jnp.dtype): Data type for the returned array

    Returns:
        jax.Array: Conductivity values for H-field, decreasing from boundary inward
    """
    return sigma_fn(
        x=jnp.arange(thickness - 1.0, 0, -1.0, dtype=dtype),
        thickness=thickness,
    )


def standard_max_sigma_E_fn(thickness: int, dtype: jnp.dtype) -> jax.Array:
    """Computes standard conductivity profile for E-field at maximum boundary.

    Args:
        thickness (int): Number of grid cells for PML thickness
        dtype (jnp.dtype): Data type for the returned array

    Returns:
        jax.Array: Conductivity values for E-field, increasing from boundary inward
    """
    return sigma_fn(
        x=jnp.arange(0.5, thickness + 0.5, 1.0, dtype=dtype),
        thickness=thickness,
    )


def standard_max_sigma_H_fn(thickness: int, dtype: jnp.dtype) -> jax.Array:
    """Computes standard conductivity profile for H-field at maximum boundary.

    Args:
        thickness (int): Number of grid cells for PML thickness
        dtype (jnp.dtype): Data type for the returned array

    Returns:
        jax.Array: Conductivity values for H-field, increasing from boundary inward
    """
    return sigma_fn(
        x=jnp.arange(1.0, thickness, 1.0, dtype=dtype),
        thickness=thickness,
    )


def standard_sigma_from_direction_axis(
    thickness: int,
    direction: Literal["+", "-"],
    axis: int,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    """Computes standard conductivity profiles for E and H fields based on direction and axis.

    Args:
        thickness (int): Number of grid cells for PML thickness
        direction (Literal["+", "-"]): Direction of PML boundary ("+" for max, "-" for min)
        axis (int): Principal axis for PML (0=x, 1=y, 2=z)
        dtype (jnp.dtype): Data type for the returned arrays

    Returns:
        tuple[jax.Array, jax.Array]: tuple containing:
            - jax.Array: Conductivity values for E-field
            - jax.Array: Conductivity values for H-field

    """
    # compute sigma values based on direction
    if direction == "-":
        sigma_E_vals = standard_min_sigma_E_fn(thickness, dtype=dtype)
        sigma_H_vals = standard_min_sigma_H_fn(thickness, dtype=dtype)
    elif direction == "+":
        sigma_E_vals = standard_max_sigma_E_fn(thickness, dtype=dtype)
        sigma_H_vals = standard_max_sigma_H_fn(thickness, dtype=dtype)
    else:
        raise Exception(f"Invalid Direction: {direction}")
    # stack in corresponding component
    zeros_E = jnp.zeros_like(sigma_E_vals)
    zeros_H = jnp.zeros_like(sigma_H_vals)
    if axis == 0:
        sigma_E = jnp.stack([sigma_E_vals, zeros_E, zeros_E], axis=0)[:, :, None, None]
        sigma_H = jnp.stack([sigma_H_vals, zeros_H, zeros_H], axis=0)[:, :, None, None]
    elif axis == 1:
        sigma_E = jnp.stack([zeros_E, sigma_E_vals, zeros_E], axis=0)[:, None, :, None]
        sigma_H = jnp.stack([zeros_H, sigma_H_vals, zeros_H], axis=0)[:, None, :, None]
    elif axis == 2:
        sigma_E = jnp.stack([zeros_E, zeros_E, sigma_E_vals], axis=0)[:, None, None, :]
        sigma_H = jnp.stack([zeros_H, zeros_H, sigma_H_vals], axis=0)[:, None, None, :]
    else:
        raise Exception(f"Invalid axis: {axis}")

    zeros_H_at_axis = jnp.zeros(shape=(3, 1, 1, 1), dtype=sigma_H.dtype)
    sigma_H = jnp.concatenate([sigma_H, zeros_H_at_axis], axis=axis + 1)

    sigma_E, sigma_H = sigma_E.astype(dtype), sigma_H.astype(dtype)

    return sigma_E, sigma_H


def kappa_from_direction_axis(
    kappa_start: float,
    kappa_end: float,
    thickness: int,
    direction: Literal["+", "-"],
    axis: int,
    dtype: jnp.dtype,
) -> jax.Array:
    """Computes kappa profile for PML boundary based on direction and axis.

    Args:
        kappa_start (float): Initial kappa value at boundary interface
        kappa_end (float): Final kappa value at outer boundary
        thickness (int): Number of grid cells for PML thickness
        direction (Literal["+", "-"]): Direction of PML boundary ("+" for max, "-" for min)
        axis (int): Principal axis for PML (0=x, 1=y, 2=z)
        dtype (jnp.dtype): Data type for the returned array

    Returns:
        jax.Array: Kappa values linearly varying from start to end value

    Raises:
        Exception: If direction is not "+" or "-"
        Exception: If axis is not 0, 1, or 2
    """
    if direction == "-":
        kappa_vals = jnp.linspace(kappa_end, kappa_start, thickness, dtype=dtype)
    elif direction == "+":
        kappa_vals = jnp.linspace(kappa_start, kappa_end, thickness, dtype=dtype)
    else:
        raise Exception(f"Invalid direction: {direction}")

    if axis == 0:
        kappa = kappa_vals[None, :, None, None]
    elif axis == 1:
        kappa = kappa_vals[None, None, :, None]
    elif axis == 2:
        kappa = kappa_vals[None, None, None, :]
    else:
        raise Exception(f"Invalid axis: {axis}")
    return kappa


def axis_direction_from_kind(kind: str) -> tuple[int, Literal["+", "-"]]:
    """Extracts axis index and direction from boundary kind string.

    Args:
        kind (str): Boundary identifier string (e.g. "min_x", "max_y", etc)

    Returns:
        tuple[int, Literal["+", "-"]]: tuple containing:
            - int: Axis index (0=x, 1=y, 2=z)
            - str: Direction ("+" for max, "-" for min)
    """
    axis = -1
    if "_x" in kind:
        axis = 0
    elif "_y" in kind:
        axis = 1
    elif "_z" in kind:
        axis = 2
    else:
        raise Exception(f"Invalid kind: {kind}")

    if "max" in kind:
        direction = "+"
    elif "min" in kind:
        direction = "-"
    else:
        raise Exception(f"Invalid kind: {kind}")

    return axis, direction
