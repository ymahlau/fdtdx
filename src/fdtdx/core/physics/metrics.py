"""Electromagnetic field metrics and normalization utilities.

This module provides functions for computing various electromagnetic field metrics
like energy density and Poynting flux, as well as field normalization operations.
All functions support JAX's automatic differentiation and work with the standard
FDTD field array shapes.
"""

import jax
import jax.numpy as jnp

import fdtdx.functional as ff
from fdtdx.constants import eps0, mu0
from fdtdx.units import A_per_m_unit, J, V_per_m_unit, W
from fdtdx.units.unitful import Unitful


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
    normalization_target: Unitful = 1 * J,
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
    norm_factor = energy_sum / normalization_target
    norm_E = E / ff.sqrt(norm_factor)
    norm_H = H / ff.sqrt(norm_factor)
    assert norm_E.unit.dim == E.unit.dim
    assert norm_H.unit.dim == H.unit.dim
    return norm_E, norm_H


def compute_poynting_vector(E: Unitful, B: Unitful, axis: int = 0) -> Unitful:
    """Calculates the instantaneous Poynting vector (energy flux) from E and B fields according to
    $ 1 / \\mu_0 E \\times B $. This is the instantaneous poynting flux of real E and B fields, for the time averaged
    flux of oscillating fields see ```compute_time_averaged_poynting_flux```

    Args:
        E (Unitful): Electric field array with E.shape[axis] == 3
        B (Unitful): Magnetic flux density array with H.shape[axis] == 3
        axis (int, optional): Axis for computing the poynting flux. Defaults to 0.

    Returns:
        Unitful: Poynting vector array with same shape as E/H representing
        energy flux in each direction.
    """
    pv = (1 / mu0) * ff.cross(
        E,
        B,
        axisa=axis,
        axisb=axis,
        axisc=axis,
    )
    return pv


def flux_from_poynting_vector(
    S: Unitful,
    resolution: Unitful,
    normal_vector: tuple[float, float, float] | jax.Array,
    axis: int = 0,
) -> Unitful:
    # normalize vector, if it is not already
    if isinstance(normal_vector, tuple):
        normal_vector = jnp.asarray(normal_vector)
    assert normal_vector.ndim == 1, f"Invalid normal vector shape: {normal_vector.shape}"
    assert normal_vector.shape[0] == 3, f"Invalid normal vector shape: {normal_vector.shape}"
    # scale to unit length
    normal_length = jnp.sqrt(jnp.sum(jnp.square(normal_vector)))
    normal_vector = normal_vector / normal_length
    # bring to correct shape
    normal_vector_shape = [-1 if idx == axis else 1 for idx, _ in enumerate(S.shape)]
    normal_vector = normal_vector.reshape(*normal_vector_shape)
    # both S and normal need to have the axis moved to the last dimension
    transpose_axes = [a for a in range(normal_vector.ndim) if a != axis] + [axis]
    normal_vector = jnp.transpose(normal_vector, axes=transpose_axes)[..., None]
    S = ff.transpose(S, axes=transpose_axes)
    # compute flux by integration
    product = ff.dot(S, normal_vector)
    flux = ff.sum(product) * ff.square(resolution)
    return flux


def compute_poynting_flux(
    E: Unitful,
    B: Unitful,
    resolution: Unitful,
    normal_vector: tuple[float, float, float] | jax.Array,
    axis: int = 0,
) -> Unitful:
    """Calculates the Poynting flux from E and H fields. In contrast to the poynting vector, the result is a scalar,
    not a vector field. The poynting vector is integrated over the surface of E and H to form the poynting flux. Therefore,
    an inherent assumption is that the E and H field represent a fields on a surface.

    Args:
        E (Unitful): Electric field array with E.shape[axis] == 3
        B (Unitful): Magnetic field array with H.shape[axis] == 3
        resolution (Unitful): Spatial resolution of the grid points of E and H field
        normal_vector (tuple[float, float, float] | jax.Array): Normal vector of the surface to integrate. If the normal
            vector is not normalized already, it is normalized within this function.
        axis (int, optional): Axis for computing the poynting flux. Defaults to 0.

    Returns:
        Unitful: scalar Poynting flux array.
    """
    assert E.shape == B.shape
    S = compute_poynting_vector(E=E, B=B, axis=axis)
    flux = flux_from_poynting_vector(
        S=S,
        resolution=resolution,
        normal_vector=normal_vector,
        axis=axis,
    )
    return flux


def normalize_by_poynting_flux(
    E: Unitful,
    B: Unitful,
    resolution: Unitful,
    normal_vector: tuple[float, float, float] | jax.Array,
    axis: int = 0,
    normalization_target: Unitful = 1 * W,
) -> tuple[Unitful, Unitful]:
    """Normalize fields so that Poynting flux along given axis = 1."""
    # Compute Poynting vector components
    flux = compute_poynting_flux(
        E=E,
        B=B,
        resolution=resolution,
        normal_vector=normal_vector,
        axis=axis,
    )
    norm_factor = flux / normalization_target
    factor = 1.0 / ff.sqrt(ff.abs(norm_factor))

    norm_E = E * factor
    norm_B = B * factor
    return norm_E, norm_B


def compute_averaged_poynting_vector(
    E: Unitful,
    H: Unitful,
) -> Unitful:
    S_m = 0.5 * ff.cross(E, ff.conj(H), axis=0)
    return S_m


def compute_averaged_flux(
    E: Unitful,
    H: Unitful,
    resolution: Unitful,
    normal_vector: tuple[float, float, float] | jax.Array,
    axis: int = 0,
):
    S_m = compute_averaged_poynting_vector(E=E, H=H)
    flux = flux_from_poynting_vector(
        S=ff.real(S_m),
        resolution=resolution,
        normal_vector=normal_vector,
        axis=axis,
    )
    return flux


def normalize_by_averaged_flux(
    E: Unitful,
    H: Unitful,
    resolution: Unitful,
    normal_vector: tuple[float, float, float] | jax.Array,
    axis: int = 0,
    normalization_target: Unitful = 1 * W,
) -> tuple[Unitful, Unitful]:
    avg_flux = compute_averaged_flux(
        E=E,
        H=H,
        resolution=resolution,
        normal_vector=normal_vector,
        axis=axis,
    )
    norm_factor = avg_flux / normalization_target
    factor = ff.sqrt(ff.abs(norm_factor))

    norm_E = E * factor
    norm_H = H * factor
    return norm_E, norm_H
