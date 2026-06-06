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
    area_EuHv: jax.Array | None = None,
    area_EvHu: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Normalize fields so the integrated Poynting flux along ``axis`` is one.

    Args:
        E: Electric field array with component axis first.
        H: Magnetic field array with component axis first.
        axis: Physical propagation axis whose Poynting component is integrated.
        area_weights: Single area-weight array for both Poynting cross-terms.
            Used on uniform grids where both Yee positions share the same area.
            Mutually exclusive with ``area_EuHv``.
        area_EuHv: Yee-staggered area for the Eu x Hv cross-term.  When
            provided, enables the Yee-split power computation that is consistent
            with :func:`bidirectional_mode_overlap`.  Mutually exclusive with
            ``area_weights``.
        area_EvHu: Yee-staggered area for the Ev x Hu cross-term.  Defaults to
            ``area_EuHv`` when ``None``.
    """
    if area_EuHv is not None and area_weights is not None:
        raise ValueError("area_weights and area_EuHv are mutually exclusive")
    if area_EuHv is not None:
        # Yee-split path: use separate area weights for each cross-term so that
        # the power estimate is consistent with bidirectional_mode_overlap.
        u, v = [(1, 2), (2, 0), (0, 1)][axis]
        _area_EvHu = area_EvHu if area_EvHu is not None else area_EuHv
        S_EuHv = 0.5 * jnp.real(jnp.conj(E[u]) * H[v]) * area_EuHv
        S_EvHu = 0.5 * jnp.real(jnp.conj(E[v]) * H[u]) * _area_EvHu
        power = jnp.abs(jnp.sum(S_EuHv - S_EvHu))
    else:
        S_complex = jnp.cross(jnp.conj(E), H, axisa=0, axisb=0, axisc=0)
        S_real = 0.5 * jnp.real(S_complex[axis])
        if area_weights is not None:
            S_real = S_real * area_weights
        power = jnp.abs(jnp.sum(S_real))

    norm_factor = jnp.sqrt(power)
    E_norm = E / norm_factor
    H_norm = H / norm_factor
    return E_norm, H_norm


def bidirectional_mode_overlap(
    mode_E: jax.Array,
    mode_H: jax.Array,
    sim_E: jax.Array,
    sim_H: jax.Array,
    propagation_axis: int,
    area_EuHv: jax.Array | None = None,
    area_EvHu: jax.Array | None = None,
) -> jax.Array:
    """Compute the bidirectional mode-overlap integral.

    Returns ``0.25 * sum((mode_E* x sim_H + sim_E x mode_H*) · ẑ * dA)``, where ẑ is
    the unit vector along ``propagation_axis``.  The mode fields are conjugated,
    matching tidy3d's ``_dot_numpy(conjugate=True)`` convention exactly.

    On a Yee lattice the cross product decomposes into two independent terms
    that sit at different spatial locations and therefore require separate area
    weights on non-uniform grids:

    * ``(conj(mode_Eu) * sim_Hv + sim_Eu * conj(mode_Hv)) * area_EuHv``
    * ``-(conj(mode_Ev) * sim_Hu + sim_Ev * conj(mode_Hu)) * area_EvHu``

    where ``(u, v)`` are the two transverse axes.  On a uniform grid
    ``area_EuHv == area_EvHu`` and a single weight is sufficient.

    Args:
        mode_E: Mode electric field, shape ``(3, *spatial)``.
        mode_H: Mode magnetic field, shape ``(3, *spatial)``.
        sim_E: Simulation phasor electric field, shape ``(3, *spatial)``.
        sim_H: Simulation phasor magnetic field, shape ``(3, *spatial)``.
        propagation_axis: Physical axis (0, 1, or 2) of mode propagation.
        area_EuHv: Per-cell area for the Eu/Hv cross term.  ``None`` treats
            every cell as having unit area (uniform-grid raw sum).
        area_EvHu: Per-cell area for the Ev/Hu cross term.  Defaults to
            ``area_EuHv`` when ``None``.  Must not be set without ``area_EuHv``.

    Returns:
        Complex scalar overlap coefficient.  Equals 1 for a self-overlap of a
        unit-power-normalised mode, matching the tidy3d ``_dot_numpy`` convention.
    """
    if area_EvHu is not None and area_EuHv is None:
        raise ValueError("area_EvHu requires area_EuHv to also be provided")

    # cyclic transverse axes for the given propagation axis
    u, v = [(1, 2), (2, 0), (0, 1)][propagation_axis]

    EuHv = jnp.conj(mode_E[u]) * sim_H[v] + sim_E[u] * jnp.conj(mode_H[v])
    EvHu = jnp.conj(mode_E[v]) * sim_H[u] + sim_E[v] * jnp.conj(mode_H[u])

    if area_EuHv is not None:
        EuHv = EuHv * area_EuHv
    if area_EvHu is not None:
        EvHu = EvHu * area_EvHu
    elif area_EuHv is not None:
        EvHu = EvHu * area_EuHv

    return 0.25 * jnp.sum(EuHv - EvHu)


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
