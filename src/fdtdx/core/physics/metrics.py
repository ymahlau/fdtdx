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
    inv_permittivity: jax.Array,
    inv_permeability: jax.Array,
) -> jax.Array:
    """Computes the total electromagnetic energy density of the field.

    Args:
        E: Electric field array with shape (3, nx, ny, nz)
        H: Magnetic field array with shape (3, nx, ny, nz)
        inv_permittivity: Inverse of the electric permittivity array
        inv_permeability: Inverse of the magnetic permeability array

    Returns:
        Total energy density array with shape (nx, ny, nz)
    """
    abs_E = jnp.sum(jnp.square(E), axis=0)
    energy_E = 0.5 * (1 / inv_permittivity) * abs_E

    abs_H = jnp.sum(jnp.square(H), axis=0)
    energy_H = 0.5 * (1 / inv_permeability) * abs_H

    total_energy = energy_E + energy_H
    return total_energy


def normalize_by_energy(
    E: jax.Array,
    H: jax.Array,
    inv_permittivity: jax.Array,
    inv_permeability: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Normalizes electromagnetic fields by their total energy.

    Args:
        E: Electric field array with shape (3, nx, ny, nz)
        H: Magnetic field array with shape (3, nx, ny, nz)
        inv_permittivity: Inverse of the electric permittivity array
        inv_permeability: Inverse of the magnetic permeability array

    Returns:
        Tuple of (normalized E field, normalized H field)
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


def poynting_flux(E: jax.Array, H: jax.Array) -> jax.Array:
    """Calculates the Poynting vector (energy flux) from E and H fields.

    Args:
        E: Electric field array with shape (3, nx, ny, nz)
        H: Magnetic field array with shape (3, nx, ny, nz)

    Returns:
        Poynting vector array with shape (3, nx, ny, nz) representing
        energy flux in each direction
    """
    return jnp.cross(
        jnp.conj(E),
        H,
        axisa=0,
        axisb=0,
        axisc=0,
    )
