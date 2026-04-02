import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.constants import eps0


def interpolate_fields(
    E_pad: jax.Array,
    H_pad: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Interpolates E and H fields onto the E_z Yee grid point (i, j, k+½).

    All six field components are co-located at (i·Δx, j·Δy, (k+½)·Δz) using
    half-step averages. Expects pre-padded fields. Slices [1:-1]/[:-2] produce
    a backward half-step (e.g. i+½ → i) and [1:-1]/[2:] a forward half-step
    (k → k+½).

    Natural positions (Taflove convention, axis 0=x, 1=y, 2=z):
        E_x: (i+½, j,   k  )  →  shift x: −½, z: +½
        E_y: (i,   j+½, k  )  →  shift y: −½, z: +½
        E_z: (i,   j,   k+½)  →  already at target
        H_x: (i,   j+½, k+½)  →  shift y: −½
        H_y: (i+½, j,   k+½)  →  shift x: −½
        H_z: (i+½, j+½, k  )  →  shift x: −½, y: −½, z: +½

    Args:
        E_pad: Pre-padded electric field array of shape (3, Nx+2, Ny+2, Nz+2)
        H_pad: Pre-padded magnetic field array of shape (3, Nx+2, Ny+2, Nz+2)

    Returns:
        Tuple of (E_interp, H_interp), each of shape (3, Nx, Ny, Nz)
    """
    E_x, E_y, E_z = E_pad[0], E_pad[1], E_pad[2]
    H_x, H_y, H_z = H_pad[0], H_pad[1], H_pad[2]

    # E_x: (i+½, j, k) → (i, j, k+½): x backward, z forward
    E_x = (E_x[1:-1, 1:-1, 1:-1] + E_x[:-2, 1:-1, 1:-1] + E_x[1:-1, 1:-1, 2:] + E_x[:-2, 1:-1, 2:]) / 4.0

    # E_y: (i, j+½, k) → (i, j, k+½): y backward, z forward
    E_y = (E_y[1:-1, 1:-1, 1:-1] + E_y[1:-1, :-2, 1:-1] + E_y[1:-1, 1:-1, 2:] + E_y[1:-1, :-2, 2:]) / 4.0

    # E_z: (i, j, k+½) → already at target
    E_z = E_z[1:-1, 1:-1, 1:-1]

    # H_x: (i, j+½, k+½) → (i, j, k+½): y backward only
    H_x = (H_x[1:-1, 1:-1, 1:-1] + H_x[1:-1, :-2, 1:-1]) / 2.0

    # H_y: (i+½, j, k+½) → (i, j, k+½): x backward only
    H_y = (H_y[1:-1, 1:-1, 1:-1] + H_y[:-2, 1:-1, 1:-1]) / 2.0

    # H_z: (i+½, j+½, k) → (i, j, k+½): x backward, y backward, z forward
    H_z = (
        H_z[1:-1, 1:-1, 1:-1]
        + H_z[:-2, 1:-1, 1:-1]
        + H_z[1:-1, :-2, 1:-1]
        + H_z[:-2, :-2, 1:-1]
        + H_z[1:-1, 1:-1, 2:]
        + H_z[:-2, 1:-1, 2:]
        + H_z[1:-1, :-2, 2:]
        + H_z[:-2, :-2, 2:]
    ) / 8.0

    E_interp = jnp.stack([E_x, E_y, E_z], axis=0)
    H_interp = jnp.stack([H_x, H_y, H_z], axis=0)

    return E_interp, H_interp


def curl_E(
    config: SimulationConfig,
    E_pad: jax.Array,
    psi_H: jax.Array,
    alpha: jax.Array,
    kappa: jax.Array,
    sigma: jax.Array,
    simulate_boundaries: bool,
) -> tuple[jax.Array, jax.Array]:
    """Transforms an E-type field into an H-type field by performing a curl operation.

    Computes the discrete curl of the electric field to obtain the corresponding
    magnetic field components. The input E-field is defined on the edges of the Yee grid
    cells (integer grid points), while the output H-field is defined on the faces
    (half-integer grid points).

    Args:
        config (SimulationConfig): Simulation configuration parameters.
        E_pad (jax.Array): Pre-padded electric field of shape (3, nx+2, ny+2, nz+2).
        psi_H (jax.Array): Auxiliary field for the magnetic field.
            Shape is (6, nx, ny, nz) for the 6 auxiliary fields.
        alpha (jax.Array): Alpha parameter for the PML.
            Shape is (6, nx, ny, nz).
        kappa (jax.Array): Kappa parameter for the PML.
            Shape is (6, nx, ny, nz).
        sigma (jax.Array): Sigma parameter for the PML.
            Shape is (6, nx, ny, nz).
        simulate_boundaries (bool): Whether to simulate boundaries.

    Returns:
        jax.Array: The curl of E - an H-type field located on the faces of the grid
                  (half-integer grid points). Has same shape as input (3, nx, ny, nz).
    """

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
        psi_Hxy = b[4] * psi_Hxy + a[4] * dyEz
        psi_Hxz = b[5] * psi_Hxz + a[5] * dzEy
        psi_Hyz = b[5] * psi_Hyz + a[5] * dzEx
        psi_Hyx = b[3] * psi_Hyx + a[3] * dxEz
        psi_Hzx = b[3] * psi_Hzx + a[3] * dxEy
        psi_Hzy = b[4] * psi_Hzy + a[4] * dyEx

    psi_H_updated = jnp.stack((psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy), axis=0)

    curl_x = (1.0 / kappa[1] * dyEz + psi_Hxy) - (1.0 / kappa[2] * dzEy + psi_Hxz)
    curl_y = (1.0 / kappa[2] * dzEx + psi_Hyz) - (1.0 / kappa[0] * dxEz + psi_Hyx)
    curl_z = (1.0 / kappa[0] * dxEy + psi_Hzx) - (1.0 / kappa[1] * dyEx + psi_Hzy)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)

    return curl, psi_H_updated


def curl_H(
    config: SimulationConfig,
    H_pad: jax.Array,
    psi_E: jax.Array,
    alpha: jax.Array,
    kappa: jax.Array,
    sigma: jax.Array,
    simulate_boundaries: bool,
) -> tuple[jax.Array, jax.Array]:
    """Transforms an H-type field into an E-type field by performing a curl operation.

    Computes the discrete curl of the magnetic field to obtain the corresponding
    electric field components. The input H-field is defined on the faces of the Yee grid
    cells (half-integer grid points), while the output E-field is defined on the edges
    (integer grid points).

    Args:
        config (SimulationConfig): Simulation configuration parameters.
        H_pad (jax.Array): Pre-padded magnetic field of shape (3, nx+2, ny+2, nz+2).
        psi_E (jax.Array): Auxiliary field for the electric field.
            Shape is (6, nx, ny, nz) for the 6 auxiliary fields.
        alpha (jax.Array): Alpha parameter for the PML.
            Shape is (6, nx, ny, nz).
        kappa (jax.Array): Kappa parameter for the PML.
            Shape is (6, nx, ny, nz).
        sigma (jax.Array): Sigma parameter for the PML.
            Shape is (6, nx, ny, nz).
        simulate_boundaries (bool): Whether to simulate boundaries.

    Returns:
        jax.Array: The curl of H - an E-type field located on the edges of the grid
                  (integer grid points). Has same shape as input (3, nx, ny, nz).
    """

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
        psi_Exy = b[1] * psi_Exy + a[1] * dyHz
        psi_Exz = b[2] * psi_Exz + a[2] * dzHy
        psi_Eyz = b[2] * psi_Eyz + a[2] * dzHx
        psi_Eyx = b[0] * psi_Eyx + a[0] * dxHz
        psi_Ezx = b[0] * psi_Ezx + a[0] * dxHy
        psi_Ezy = b[1] * psi_Ezy + a[1] * dyHx

    psi_E_updated = jnp.stack((psi_Exy, psi_Exz, psi_Eyz, psi_Eyx, psi_Ezx, psi_Ezy), axis=0)

    curl_x = (1.0 / kappa[1] * dyHz + psi_Exy) - (1.0 / kappa[2] * dzHy + psi_Exz)
    curl_y = (1.0 / kappa[2] * dzHx + psi_Eyz) - (1.0 / kappa[0] * dxHz + psi_Eyx)
    curl_z = (1.0 / kappa[0] * dxHy + psi_Ezx) - (1.0 / kappa[1] * dyHx + psi_Ezy)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)

    return curl, psi_E_updated
