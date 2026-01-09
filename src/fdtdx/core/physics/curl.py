import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.constants import eps0


def interpolate_fields(
    E_field: jax.Array,
    H_field: jax.Array,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jax.Array, jax.Array]:
    """Interpolates E and H fields onto E_z in a FDTD grid with PEC/periodic boundary conditions.

    Performs spatial interpolation of the electric and magnetic fields to align them
    onto the same grid points as E_z. This is necessary because E and H fields are
    naturally staggered in the Yee grid.

    Args:
        E_field (jax.Array): 4D tensor representing the electric field.
                Dimensions are (width, depth, height, direction).
        H_field (jax.Array): 4D tensor representing the magnetic field.
                Dimensions are (width, depth, height, direction).
        periodic_axes (tuple[bool, bool, bool], optional): Tuple of booleans indicating which axes use periodic
            boundaries (periodic_x, periodic_y, periodic_z). Defaults to (False, False, False).

    Returns:
        tuple[jax.Array, jax.Array]: A tuple (E_interp, H_interp) containing:
            - E_interp: Interpolated electric field as 4D tensor
            - H_interp: Interpolated magnetic field as 4D tensor

    Note:
        Uses PEC (Perfect Electric Conductor) boundary conditions where fields
        at boundaries are zero, unless periodic boundaries are specified.
    """
    # Apply boundary conditions: PEC (zero) or periodic for each axis separately
    for i, periodic in enumerate(periodic_axes):
        pad_mode = "wrap" if periodic else "constant"
        # Create padding tuple for current axis
        if i == 0:
            pad_width = ((0, 0), (1, 1), (0, 0), (0, 0))
        elif i == 1:
            pad_width = ((0, 0), (0, 0), (1, 1), (0, 0))
        else:  # i == 2
            pad_width = ((0, 0), (0, 0), (0, 0), (1, 1))
        E_field = jnp.pad(E_field, pad_width, mode=pad_mode)
        H_field = jnp.pad(H_field, pad_width, mode=pad_mode)

    E_x, E_y, E_z = E_field[0], E_field[1], E_field[2]
    H_x, H_y, H_z = H_field[0], H_field[1], H_field[2]

    E_x = (E_x[1:-1, 1:-1, 1:-1] + E_x[1:-1, 1:-1, :-2] + E_x[2:, 1:-1, 1:-1] + E_x[2:, 1:-1, :-2]) / 4.0
    E_y = (E_y[1:-1, 1:-1, 1:-1] + E_y[1:-1, :-2, 1:-1] + E_y[2:, 1:-1, 1:-1] + E_y[2:, :-2, 1:-1]) / 4.0
    E_z = E_z[1:-1, 1:-1, 1:-1]  # leave as is since we project onto the E_z

    H_x = (H_x[1:-1, 2:, 1:-1] + H_x[1:-1, :-2, 1:-1]) / 2.0
    H_y = (H_y[1:-1, 1:-1, 2:] + H_y[1:-1, 1:-1, :-2]) / 2.0
    H_z = (
        H_z[:-2, 2:, 2:]
        + H_z[:-2, 2:, :-2]
        + H_z[:-2, :-2, 2:]
        + H_z[:-2, :-2, :-2]
        + H_z[2:, 2:, 2:]
        + H_z[2:, 2:, :-2]
        + H_z[2:, :-2, 2:]
        + H_z[2:, :-2, :-2]
    ) / 8.0

    # Constructing the interpolated fields
    E_interp = jnp.stack([E_x, E_y, E_z], axis=0)
    H_interp = jnp.stack([H_x, H_y, H_z], axis=0)

    return E_interp, H_interp


def curl_E(
    config: SimulationConfig,
    E: jax.Array,
    psi_H: jax.Array,
    alpha: jax.Array,
    kappa: jax.Array,
    sigma: jax.Array,
    simulate_boundaries: bool,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jax.Array, jax.Array]:
    """Transforms an E-type field into an H-type field by performing a curl operation.

    Computes the discrete curl of the electric field to obtain the corresponding
    magnetic field components. The input E-field is defined on the edges of the Yee grid
    cells (integer grid points), while the output H-field is defined on the faces
    (half-integer grid points).

    Args:
        config (SimulationConfig): Simulation configuration parameters.
        E (jax.Array): Electric field to take the curl of. A 4D tensor representing the E-type field
            located on the edges of the grid cell (integer gridpoints).
            Shape is (3, nx, ny, nz) for the 3 field components.
        psi_H (jax.Array): Auxiliary field for the magnetic field.
            Shape is (6, nx, ny, nz) for the 6 auxiliary fields.
        alpha (jax.Array): Alpha parameter for the PML.
            Shape is (6, nx, ny, nz).
        kappa (jax.Array): Kappa parameter for the PML.
            Shape is (6, nx, ny, nz).
        sigma (jax.Array): Sigma parameter for the PML.
            Shape is (6, nx, ny, nz).
        simulate_boundaries (bool): Whether to simulate boundaries.
        periodic_axes (tuple[bool, bool, bool], optional): Tuple of booleans indicating which axes use periodic
            boundaries (periodic_x, periodic_y, periodic_z). Defaults to (False, False, False).

    Returns:
        jax.Array: The curl of E - an H-type field located on the faces of the grid
                  (half-integer grid points). Has same shape as input (3, nx, ny, nz).
    """
    # Pad each axis separately based on boundary conditions
    E_pad = E
    for i, periodic in enumerate(periodic_axes):
        pad_mode = "wrap" if periodic else "constant"
        # Create padding tuple for current axis
        if i == 0:
            pad_width = ((0, 0), (1, 1), (0, 0), (0, 0))
        elif i == 1:
            pad_width = ((0, 0), (0, 0), (1, 1), (0, 0))
        else:  # i == 2
            pad_width = ((0, 0), (0, 0), (0, 0), (1, 1))
        E_pad = jnp.pad(E_pad, pad_width, mode=pad_mode)

    dyEz = (jnp.roll(E_pad[2], -1, axis=1) - E_pad[2])[1:-1, 1:-1, 1:-1]
    dzEy = (jnp.roll(E_pad[1], -1, axis=2) - E_pad[1])[1:-1, 1:-1, 1:-1]
    dzEx = (jnp.roll(E_pad[0], -1, axis=2) - E_pad[0])[1:-1, 1:-1, 1:-1]
    dxEz = (jnp.roll(E_pad[2], -1, axis=0) - E_pad[2])[1:-1, 1:-1, 1:-1]
    dxEy = (jnp.roll(E_pad[1], -1, axis=0) - E_pad[1])[1:-1, 1:-1, 1:-1]
    dyEx = (jnp.roll(E_pad[0], -1, axis=1) - E_pad[0])[1:-1, 1:-1, 1:-1]

    # Auxiliary fields
    psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy = psi_H

    if simulate_boundaries:
        # Get H-field PML coefficients
        b = jnp.expm1(-config.courant_number * config.resolution / c0 / eps0 * (sigma / kappa + alpha)) + 1
        a = jnp.nan_to_num((b - 1.0) * sigma / (sigma + alpha * kappa) / kappa, nan=0.0, posinf=0.0, neginf=0.0)

        # Update auxiliary fields
        psi_Hxy = b[4, :, :, :] * psi_Hxy + a[4, :, :, :] * dyEz
        psi_Hxz = b[5, :, :, :] * psi_Hxz + a[5, :, :, :] * dzEy
        psi_Hyz = b[5, :, :, :] * psi_Hyz + a[5, :, :, :] * dzEx
        psi_Hyx = b[3, :, :, :] * psi_Hyx + a[3, :, :, :] * dxEz
        psi_Hzx = b[3, :, :, :] * psi_Hzx + a[3, :, :, :] * dxEy
        psi_Hzy = b[4, :, :, :] * psi_Hzy + a[4, :, :, :] * dyEx

    psi_H_updated = jnp.stack((psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy), axis=0)

    curl_x = (1.0 / kappa[1, :, :, :] * dyEz + psi_Hxy) - (1.0 / kappa[2, :, :, :] * dzEy + psi_Hxz)
    curl_y = (1.0 / kappa[2, :, :, :] * dzEx + psi_Hyz) - (1.0 / kappa[0, :, :, :] * dxEz + psi_Hyx)
    curl_z = (1.0 / kappa[0, :, :, :] * dxEy + psi_Hzx) - (1.0 / kappa[1, :, :, :] * dyEx + psi_Hzy)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)

    return curl, psi_H_updated


def curl_H(
    config: SimulationConfig,
    H: jax.Array,
    psi_E: jax.Array,
    alpha: jax.Array,
    kappa: jax.Array,
    sigma: jax.Array,
    simulate_boundaries: bool,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jax.Array, jax.Array]:
    """Transforms an H-type field into an E-type field by performing a curl operation.

    Computes the discrete curl of the magnetic field to obtain the corresponding
    electric field components. The input H-field is defined on the faces of the Yee grid
    cells (half-integer grid points), while the output E-field is defined on the edges
    (integer grid points).

    Args:
        config (SimulationConfig): Simulation configuration parameters.
        H (jax.Array): Magnetic field to take the curl of. A 4D tensor representing the H-type field
            located on the faces of the grid (half-integer grid points).
            Shape is (3, nx, ny, nz) for the 3 field components.
        psi_E (jax.Array): Auxiliary field for the electric field.
            Shape is (6, nx, ny, nz) for the 6 auxiliary fields.
        alpha (jax.Array): Alpha parameter for the PML.
            Shape is (6, nx, ny, nz).
        kappa (jax.Array): Kappa parameter for the PML.
            Shape is (6, nx, ny, nz).
        sigma (jax.Array): Sigma parameter for the PML.
            Shape is (6, nx, ny, nz).
        simulate_boundaries (bool): Whether to simulate boundaries.
        periodic_axes (tuple[bool, bool, bool], optional): Tuple of booleans indicating which axes use periodic
            boundaries (periodic_x, periodic_y, periodic_z). Defaults to (False, False, False).

    Returns:
        jax.Array: The curl of H - an E-type field located on the edges of the grid
                  (integer grid points). Has same shape as input (3, nx, ny, nz).
    """
    # Pad each axis separately based on boundary conditions
    H_pad = H
    for i, periodic in enumerate(periodic_axes):
        pad_mode = "wrap" if periodic else "constant"
        # Create padding tuple for current axis
        if i == 0:
            pad_width = ((0, 0), (1, 1), (0, 0), (0, 0))
        elif i == 1:
            pad_width = ((0, 0), (0, 0), (1, 1), (0, 0))
        else:  # i == 2
            pad_width = ((0, 0), (0, 0), (0, 0), (1, 1))
        H_pad = jnp.pad(H_pad, pad_width, mode=pad_mode)

    dyHz = (H_pad[2] - jnp.roll(H_pad[2], 1, axis=1))[1:-1, 1:-1, 1:-1]
    dzHy = (H_pad[1] - jnp.roll(H_pad[1], 1, axis=2))[1:-1, 1:-1, 1:-1]
    dzHx = (H_pad[0] - jnp.roll(H_pad[0], 1, axis=2))[1:-1, 1:-1, 1:-1]
    dxHz = (H_pad[2] - jnp.roll(H_pad[2], 1, axis=0))[1:-1, 1:-1, 1:-1]
    dxHy = (H_pad[1] - jnp.roll(H_pad[1], 1, axis=0))[1:-1, 1:-1, 1:-1]
    dyHx = (H_pad[0] - jnp.roll(H_pad[0], 1, axis=1))[1:-1, 1:-1, 1:-1]

    # Auxiliary fields
    psi_Exy, psi_Exz, psi_Eyz, psi_Eyx, psi_Ezx, psi_Ezy = psi_E

    if simulate_boundaries:
        # Get E-field PML coefficients
        b = jnp.expm1(-config.courant_number * config.resolution / c0 / eps0 * (sigma / kappa + alpha)) + 1
        a = jnp.nan_to_num((b - 1.0) * sigma / (sigma + alpha * kappa) / kappa, nan=0.0, posinf=0.0, neginf=0.0)

        # Update auxiliary fields
        psi_Exy = b[1, :, :, :] * psi_Exy + a[1, :, :, :] * dyHz
        psi_Exz = b[2, :, :, :] * psi_Exz + a[2, :, :, :] * dzHy
        psi_Eyz = b[2, :, :, :] * psi_Eyz + a[2, :, :, :] * dzHx
        psi_Eyx = b[0, :, :, :] * psi_Eyx + a[0, :, :, :] * dxHz
        psi_Ezx = b[0, :, :, :] * psi_Ezx + a[0, :, :, :] * dxHy
        psi_Ezy = b[1, :, :, :] * psi_Ezy + a[1, :, :, :] * dyHx

    psi_E_updated = jnp.stack((psi_Exy, psi_Exz, psi_Eyz, psi_Eyx, psi_Ezx, psi_Ezy), axis=0)

    curl_x = (1.0 / kappa[1, :, :, :] * dyHz + psi_Exy) - (1.0 / kappa[2, :, :, :] * dzHy + psi_Exz)
    curl_y = (1.0 / kappa[2, :, :, :] * dzHx + psi_Eyz) - (1.0 / kappa[0, :, :, :] * dxHz + psi_Eyx)
    curl_z = (1.0 / kappa[0, :, :, :] * dxHy + psi_Ezx) - (1.0 / kappa[1, :, :, :] * dyHx + psi_Ezy)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)

    return curl, psi_E_updated
