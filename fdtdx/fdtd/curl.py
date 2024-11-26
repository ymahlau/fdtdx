import jax
import jax.numpy as jnp


def interpolate_fields(
    E_field: jax.Array, 
    H_field: jax.Array,
):
    """
    Interpolates E and H fields onto E_z in a FDTD grid with PEC boundary conditions.

    Parameters:
    E_field, H_field: 
    4D tensors representing the electric and magnetic fields.
    Dimensions are (width, depth, height, direction).

    Returns:
    Interpolated E and H fields as 4D tensors.
    """
    # Apply PEC boundary conditions: fields at boundaries are zero, TODO: wrapped boundaries
    E_field = jnp.pad(E_field, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="constant")
    H_field = jnp.pad(H_field, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="constant")
    
    E_x, E_y, E_z = E_field[0], E_field[1], E_field[2]
    H_x, H_y, H_z = H_field[0], H_field[1], H_field[2]
    
    E_x = (
        E_x[1:-1, 1:-1, 1:-1] + E_x[1:-1, 1:-1, :-2] + E_x[2:, 1:-1, 1:-1] + E_x[2:, 1:-1, :-2]
    ) / 4.0
    E_y = (
        E_y[1:-1, 1:-1, 1:-1] + E_y[1:-1, :-2, 1:-1] + E_y[2:, 1:-1, 1:-1] + E_y[2:, :-2, 1:-1]
    ) / 4.0
    E_z = E_z[1:-1, 1:-1, 1:-1]  # leave as is since we project onto the E_z

    H_x = (H_x[1:-1, 2:, 1:-1] + H_x[1:-1, :-2, 1:-1]) / 2.0
    H_y = (H_y[1:-1, 1:-1, 2:] + H_y[1:-1, 1:-1, :-2]) / 2.0
    H_z = (H_z[:-2, 2:, 2:]
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


def curl_E(E: jax.Array) -> jax.Array:
    """Transforms an E-type field into an H-type field by performing a curl
    operation

    Args:
        E: Electric field to take the curl of (E-type field located on the
        edges of the grid cell [integer gridpoints])

    Returns:
        The curl of E (H-type field located on the faces of the grid [half-integer grid points])
    """
    
    E_pad = jnp.pad(E, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="edge")
    
    curl_x = jnp.roll(E_pad[2], -1, axis=1) - E_pad[2] + E_pad[1] - jnp.roll(E_pad[1], -1, axis=2)
    curl_y = jnp.roll(E_pad[0], -1, axis=2) - E_pad[0] + E_pad[2] - jnp.roll(E_pad[2], -1, axis=0)
    curl_z = jnp.roll(E_pad[1], -1, axis=0) - E_pad[1] + E_pad[0] - jnp.roll(E_pad[0], -1, axis=1)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)[:, 1:-1, 1:-1, 1:-1]
    
    return curl


def curl_H(H: jax.Array) -> jax.Array:
    """Transforms an H-type field into an E-type field by performing a curl
    operation

    Args:
        H: Magnetic field to take the curl of (H-type field located on half-integer grid points)

    Returns:
        The curl of H (E-type field located on the edges of the grid [integer grid points])

    """
    H_pad = jnp.pad(H, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="edge")
    
    curl_x = H_pad[2] - jnp.roll(H_pad[2], 1, axis=1) - H_pad[1] + jnp.roll(H_pad[1], 1, axis=2)
    curl_y = H_pad[0] - jnp.roll(H_pad[0], 1, axis=2) - H_pad[2] + jnp.roll(H_pad[2], 1, axis=0)
    curl_z = H_pad[1] - jnp.roll(H_pad[1], 1, axis=0) - H_pad[0] + jnp.roll(H_pad[0], 1, axis=1)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)[:, 1:-1, 1:-1, 1:-1]
    
    return curl
