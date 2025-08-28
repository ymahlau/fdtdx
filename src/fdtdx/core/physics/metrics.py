"""Electromagnetic field metrics and normalization utilities.

This module provides functions for computing various electromagnetic field metrics
like energy density and Poynting flux, as well as field normalization operations.
All functions support JAX's automatic differentiation and work with the standard
FDTD field array shapes.
"""

import jax
import jax.numpy as jnp

from fdtdx.units.composite import J, V_per_m_unit, A_per_m_unit
from fdtdx.units.typing import SI
from fdtdx.units.unitful import Unitful
import fdtdx.functional as ff
from fdtdx.constants import eps0, mu0

def compute_energy(
    E: Unitful,
    H: Unitful,
    inv_permittivity: jax.Array | float,
    inv_permeability: jax.Array | float,
    axis: int = 0,
) -> Unitful:
    """Computes the total electromagnetic energy density of the field.

    Args:
        E (Unitful): Electric field array with shape (3, nx, ny, nz)
        H (Unitful): Magnetic field array with shape (3, nx, ny, nz)
        inv_permittivity (jax.Array | float): Inverse of the electric permittivity array
        inv_permeability (jax.Array | float): Inverse of the magnetic permeability array
        axis (int, optional): Axis index of the X,Y,Z component for the E and H field. Defaults to 0.
    Returns:
        Unitful: Total energy density array with shape (nx, ny, nz)
    """
    assert E.unit.dim == V_per_m_unit.dim
    assert H.unit.dim == A_per_m_unit.dim
    
    patched_E = ff.square(ff.abs(E))
    sum_fields_E = ff.sum(patched_E, axis=axis)
    energy_E = 0.5 * eps0 * (1 / inv_permittivity) * sum_fields_E
    assert isinstance(energy_E, Unitful)

    patched_H = ff.square(ff.abs(H))
    sum_fields_H = ff.sum(patched_H, axis=axis)
    energy_H = 0.5 * mu0 * (1 / inv_permeability) * sum_fields_H
    assert isinstance(energy_H, Unitful)

    total_energy = energy_E + energy_H
    return total_energy


def normalize_by_energy(
    E: Unitful,
    H: Unitful,
    inv_permittivity: jax.Array | float,
    inv_permeability: jax.Array | float,
    resolution: Unitful,
    normalization_target: Unitful = 1*J,
) -> tuple[Unitful, Unitful]:
    """Normalizes electromagnetic fields by their total energy.

    Args:
        E (Unitful): Electric field array with shape (3, nx, ny, nz)
        H (Unitful): Magnetic field array with shape (3, nx, ny, nz)
        inv_permittivity (jax.Array | float): Inverse of the electric permittivity array
        inv_permeability (jax.Array | float): Inverse of the magnetic permeability array
        resolution (Unitful): Spatial resolution of grid points in metre.

    Returns:
        tuple[Unitful, Unitful]: Tuple of (normalized E field, normalized H field)
    """
    total_energy = compute_energy(
        E=E,
        H=H,
        inv_permittivity=inv_permittivity,
        inv_permeability=inv_permeability,
    )
    energy_sum = ff.sum(total_energy) * resolution**3
    norm_factor = (energy_sum / normalization_target).materialise()
    norm_E = E / jnp.sqrt(norm_factor)
    norm_H = H / jnp.sqrt(norm_factor)
    assert norm_E.unit.dim == E.unit.dim
    assert norm_H.unit.dim == H.unit.dim
    return norm_E, norm_H


def compute_poynting_flux(E: Unitful, H: Unitful, axis: int = 0) -> Unitful:
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
