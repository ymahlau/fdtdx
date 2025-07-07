import jax
import jax.numpy as jnp


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


def curl_E(E: jax.Array, periodic_axes: tuple[bool, bool, bool] = (False, False, False)) -> jax.Array:
    """Transforms an E-type field into an H-type field by performing a curl operation.

    Computes the discrete curl of the electric field to obtain the corresponding
    magnetic field components. The input E-field is defined on the edges of the Yee grid
    cells (integer grid points), while the output H-field is defined on the faces
    (half-integer grid points).

    Args:
        E (jax.Array): Electric field to take the curl of. A 4D tensor representing the E-type field
            located on the edges of the grid cell (integer gridpoints).
            Shape is (3, nx, ny, nz) for the 3 field components.
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

    curl_x = jnp.roll(E_pad[2], -1, axis=1) - E_pad[2] + E_pad[1] - jnp.roll(E_pad[1], -1, axis=2)
    curl_y = jnp.roll(E_pad[0], -1, axis=2) - E_pad[0] + E_pad[2] - jnp.roll(E_pad[2], -1, axis=0)
    curl_z = jnp.roll(E_pad[1], -1, axis=0) - E_pad[1] + E_pad[0] - jnp.roll(E_pad[0], -1, axis=1)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)[:, 1:-1, 1:-1, 1:-1]

    return curl


def curl_H(H: jax.Array, periodic_axes: tuple[bool, bool, bool] = (False, False, False)) -> jax.Array:
    """Transforms an H-type field into an E-type field by performing a curl operation.

    Computes the discrete curl of the magnetic field to obtain the corresponding
    electric field components. The input H-field is defined on the faces of the Yee grid
    cells (half-integer grid points), while the output E-field is defined on the edges
    (integer grid points).

    Args:
        H (jax.Array): Magnetic field to take the curl of. A 4D tensor representing the H-type field
            located on the faces of the grid (half-integer grid points).
            Shape is (3, nx, ny, nz) for the 3 field components.
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

    curl_x = H_pad[2] - jnp.roll(H_pad[2], 1, axis=1) - H_pad[1] + jnp.roll(H_pad[1], 1, axis=2)
    curl_y = H_pad[0] - jnp.roll(H_pad[0], 1, axis=2) - H_pad[2] + jnp.roll(H_pad[2], 1, axis=0)
    curl_z = H_pad[1] - jnp.roll(H_pad[1], 1, axis=0) - H_pad[0] + jnp.roll(H_pad[0], 1, axis=1)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)[:, 1:-1, 1:-1, 1:-1]

    return curl
