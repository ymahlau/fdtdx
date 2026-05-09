"""Electromagnetic field metrics and normalization utilities.

This module provides functions for computing various electromagnetic field metrics
like energy density and Poynting flux, as well as field normalization operations.
All functions support JAX's automatic differentiation and work with the standard
FDTD field array shapes.
"""

import jax
import jax.numpy as jnp

from fdtdx.core.misc import expand_to_3x3


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

    inv_eps_shape = getattr(inv_permittivity, "shape", ())
    inv_mu_shape = getattr(inv_permeability, "shape", ())

    if (inv_eps_shape and inv_eps_shape[0] == 9) or (inv_mu_shape and inv_mu_shape[0] == 9):
        inv_permittivity = expand_to_3x3(inv_permittivity)
        inv_permeability = expand_to_3x3(inv_permeability)

        # Invert the 3x3 matrices to get eps and mu
        perm = (2, 3, 4, 0, 1)  # (3, 3, nx, ny, nz) -> (nx, ny, nz, 3, 3)
        inv_perm = (3, 4, 0, 1, 2)  # (nx, ny, nz, 3, 3) -> (3, 3, nx, ny, nz)
        eps = jnp.linalg.inv(inv_permittivity.transpose(perm)).transpose(inv_perm)
        mu = jnp.linalg.inv(inv_permeability.transpose(perm)).transpose(inv_perm)

        # For fully anisotropic materials: 0.5 * sum_ij E_i* ε_ij E_j
        # E/H has shape (3, nx, ny, nz), eps/mu has shape (3, 3, nx, ny, nz)
        energy_E = 0.5 * jnp.real(jnp.einsum("ixyz,ijxyz,jxyz->xyz", jnp.conj(E), eps, E))
        energy_H = 0.5 * jnp.real(jnp.einsum("ixyz,ijxyz,jxyz->xyz", jnp.conj(H), mu, H))

        total_energy = energy_E + energy_H
        return total_energy

    else:
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


def normalize_by_poynting_flux(
    E: jax.Array,
    H: jax.Array,
    axis: int,
    area_weights: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Normalize fields so the integrated Poynting flux along ``axis`` is one.

    Args:
        E: Electric field array with component axis first.
        H: Magnetic field array with component axis first.
        axis: Physical propagation axis whose Poynting component is integrated.
        area_weights: Optional detector-plane area weights broadcastable to
            ``E[axis]``.  Uniform-grid callers may omit this for the historical
            raw-sum normalization; non-uniform callers should provide weights so
            refinement alone does not change the normalization.
    """
    # Compute Poynting vector components
    S_complex = jnp.cross(jnp.conj(E), H, axisa=0, axisb=0, axisc=0)
    S_real = 0.5 * jnp.real(S_complex[axis])  # power flow in desired direction
    if area_weights is not None:
        S_real = S_real * area_weights

    # Integrate over transverse plane (axis orthogonal to `axis`)
    power = jnp.abs(jnp.sum(S_real))

    # Normalize
    norm_factor = jnp.sqrt(power)
    E_norm = E / norm_factor
    H_norm = H / norm_factor
    return E_norm, H_norm


def resample_to_uniform_2d(
    field: jax.Array,
    x_centers: jax.Array,
    y_centers: jax.Array,
) -> tuple[jax.Array, jax.Array | float, jax.Array | float]:
    """Resample a ``(component, x, y)`` field onto a uniform transverse grid.

    Interpolates each component from the given non-uniform physical center
    coordinates onto a uniform grid with the same extent and number of points.
    Returns the resampled field and the uniform grid spacings.
    """
    nx, ny = field.shape[1], field.shape[2]
    target_x = jnp.linspace(x_centers[0], x_centers[-1], nx)
    target_y = jnp.linspace(y_centers[0], y_centers[-1], ny)

    def interp_x(component):
        return jax.vmap(lambda column: jnp.interp(target_x, x_centers, column), in_axes=1, out_axes=1)(component)

    def interp_y(component):
        return jax.vmap(lambda row: jnp.interp(target_y, y_centers, row))(component)

    field = jax.vmap(interp_x)(field)
    field = jax.vmap(interp_y)(field)

    dx = (target_x[1] - target_x[0]) if nx > 1 else 1.0
    dy = (target_y[1] - target_y[0]) if ny > 1 else 1.0
    return field, dx, dy
