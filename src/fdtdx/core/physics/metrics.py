"""Electromagnetic field metrics and normalization utilities.

This module provides functions for computing various electromagnetic field metrics
like energy density and Poynting flux, as well as field normalization operations.
All functions support JAX's automatic differentiation and work with the standard
FDTD field array shapes.
"""

import jax
import jax.numpy as jnp


def compute_energy(
    E: jax.Array,
    H: jax.Array,
    inv_permittivity: jax.Array | float,
    inv_permeability: jax.Array | float,
    axis: int = 0,
) -> jax.Array:
    """Computes the total electromagnetic energy density of the field.

    Args:
        E (jax.Array): Electric field array with shape (3, nx, ny, nz)
        H (jax.Array): Magnetic field array with shape (3, nx, ny, nz)
        inv_permittivity (jax.Array | float): Inverse permittivity. Shape (3, nx, ny, nz) for anisotropic or scalar
        inv_permeability (jax.Array | float): Inverse permeability. Shape (3, nx, ny, nz) for anisotropic or scalar
        axis (int, optional): Axis index of the X,Y,Z component for the E and H field. Defaults to 0.
    Returns:
        jax.Array: Total energy density array with shape (nx, ny, nz)
    """
    # For anisotropic materials: energy = 0.5 * sum_i(ε_i * |E_i|² + μ_i * |H_i|²)
    # Component-wise calculation then sum
    E_squared = jnp.square(jnp.abs(E))  # shape: (3, nx, ny, nz)
    energy_E = 0.5 * (1 / inv_permittivity) * E_squared  # component-wise multiplication
    energy_E = jnp.sum(energy_E, axis=axis)  # sum over components

    H_squared = jnp.square(jnp.abs(H))  # shape: (3, nx, ny, nz)
    energy_H = 0.5 * (1 / inv_permeability) * H_squared  # component-wise multiplication
    energy_H = jnp.sum(energy_H, axis=axis)  # sum over components

    total_energy = energy_E + energy_H
    return total_energy


def normalize_by_energy(
    E: jax.Array,
    H: jax.Array,
    inv_permittivity: jax.Array | float,
    inv_permeability: jax.Array | float,
) -> tuple[jax.Array, jax.Array]:
    """Normalizes electromagnetic fields by their total energy.

    Args:
        E (jax.Array): Electric field array with shape (3, nx, ny, nz)
        H (jax.Array): Magnetic field array with shape (3, nx, ny, nz)
        inv_permittivity (jax.Array | float): Inverse of the electric permittivity array
        inv_permeability (jax.Array | float): Inverse of the magnetic permeability array

    Returns:
        tuple[jax.Array, jax.Array]: Tuple of (normalized E field, normalized H field)
    """
    total_energy = compute_energy(
        E=E,
        H=H,
        inv_permittivity=inv_permittivity,
        inv_permeability=inv_permeability,
    )
    energy_root = jnp.sqrt(jnp.sum(total_energy))
    norm_E = E / energy_root
    norm_H = H / energy_root
    return norm_E, norm_H


def compute_poynting_flux(E: jax.Array, H: jax.Array, axis: int = 0) -> jax.Array:
    """Calculates the Poynting vector (energy flux) from E and H fields.

    Args:
        E (jax.Array): Electric field array with shape (3, nx, ny, nz)
        H (jax.Array): Magnetic field array with shape (3, nx, ny, nz)
        axis (int, optional): Axis for computing the poynting flux. Defaults to 0.

    Returns:
        jax.Array: Poynting vector array with shape (3, nx, ny, nz) representing
        energy flux in each direction
    """
    return jnp.cross(
        E,
        jnp.conj(H),
        axisa=axis,
        axisb=axis,
        axisc=axis,
    )


def normalize_by_poynting_flux(E: jax.Array, H: jax.Array, axis: int) -> tuple[jax.Array, jax.Array]:
    """Normalize fields so that Poynting flux along given axis = 1."""
    # Compute Poynting vector components
    S_complex = jnp.cross(jnp.conj(E), H, axisa=0, axisb=0, axisc=0)
    S_real = 0.5 * jnp.real(S_complex[axis])  # power flow in desired direction

    # Integrate over transverse plane (axis orthogonal to `axis`)
    power = jnp.abs(jnp.sum(S_real))

    # Normalize
    norm_factor = jnp.sqrt(power)
    E_norm = E / norm_factor
    H_norm = H / norm_factor
    return E_norm, H_norm
