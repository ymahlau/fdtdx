from typing import Literal, Tuple

import jax
import jax.numpy as jnp


def compute_extent(
    kind: Literal["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"],
    thickness: int,
) -> Tuple[slice, slice, slice]:
    """Compute the extent of the PML boundary."""
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
    kind: Literal["minx", "maxx", "miny", "maxy", "minz", "maxz"],
    thickness: int,
) -> Tuple[slice, slice, slice]:
    """Compute the extent of the PML boundary."""
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


def sigma_fn(x: jax.Array, thickness: int):
    """create a cubicly increasing profile for the conductivity"""
    return 40 * x**3 / (thickness + 1) ** 4

def standard_min_sigma_E_fn(thickness: int, dtype: jnp.dtype):
    return sigma_fn(
        x=jnp.arange(thickness - 0.5, -0.5, -1.0, dtype=dtype),
        thickness=thickness,
    )

def standard_min_sigma_H_fn(thickness: int, dtype: jnp.dtype):
    return sigma_fn(
        x=jnp.arange(thickness - 1.0, 0, -1.0, dtype=dtype),
        thickness=thickness,
    )

def standard_max_sigma_E_fn(thickness: int, dtype: jnp.dtype):
    return sigma_fn(
        x=jnp.arange(0.5, thickness + 0.5, 1.0, dtype=dtype),
        thickness=thickness,
    )

def standard_max_sigma_H_fn(thickness: int, dtype: jnp.dtype):
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
    sigma_H = jnp.concatenate([sigma_H, zeros_H_at_axis], axis=axis+1)
    
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
