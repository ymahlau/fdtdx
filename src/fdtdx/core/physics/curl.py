import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.constants import eps0


def _metric_scale(
    config: SimulationConfig,
    axis: int,
    shape: tuple[int, int, int],
    stencil: str,
) -> jax.Array | float:
    """Return the local derivative scale for a rectilinear Yee curl term.

    fdtdx historically stores curl terms as raw finite differences and applies a
    scalar Courant number in the update equations.  On a non-uniform grid the
    equivalent term is ``(c * dt / courant_number) * diff / d_axis[i]``.  The
    prefactor equals the uniform spacing on legacy grids, so uniform behavior is
    unchanged while stretched grids get local metric factors.
    """
    if not config.has_nonuniform_grid:
        return 1.0

    grid = config.resolved_grid
    assert grid is not None
    widths = grid.cell_widths(axis)
    if stencil == "backward":
        prev_widths = jnp.concatenate([widths[:1], widths[:-1]])
        widths = 0.5 * (widths + prev_widths)
    elif stencil != "forward":
        raise ValueError(f"Unknown derivative stencil: {stencil}")
    reference_spacing = c0 * config.time_step_duration / config.courant_number
    scale = reference_spacing / widths
    broadcast_shape = [1, 1, 1]
    broadcast_shape[axis] = shape[axis]
    return scale.reshape(tuple(broadcast_shape))


def _backward_edge_average(
    current: jax.Array,
    previous: jax.Array,
    config: SimulationConfig | None,
    axis: int,
) -> jax.Array:
    """Interpolate center-staggered samples back to an edge on a rectilinear grid.

    Uniform grids use the historical arithmetic mean.  On stretched grids the
    target edge is not halfway between neighboring cell centers, so the average
    is weighted by the half-widths of the cells on each side of the edge.
    """
    if config is None or not config.has_nonuniform_grid:
        return 0.5 * (current + previous)

    grid = config.resolved_grid
    assert grid is not None
    widths = grid.cell_widths(axis)
    previous_widths = jnp.concatenate([widths[:1], widths[:-1]])
    current_half_width = 0.5 * widths
    previous_half_width = 0.5 * previous_widths
    broadcast_shape = [1, 1, 1]
    broadcast_shape[axis] = current.shape[axis]
    current_half_width = current_half_width.reshape(tuple(broadcast_shape))
    previous_half_width = previous_half_width.reshape(tuple(broadcast_shape))
    return (current * previous_half_width + previous * current_half_width) / (current_half_width + previous_half_width)


def interpolate_fields(
    E_pad: jax.Array,
    H_pad: jax.Array,
    config: SimulationConfig | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Interpolates E and H fields onto the E_z Yee grid point (i, j, k+½).

    All six field components are co-located at (i·Δx, j·Δy, (k+½)·Δz) using
    half-step averages. Expects pre-padded fields. Slices [1:-1]/[:-2] produce
    a backward half-step (e.g. i+½ → i) and [1:-1]/[2:] a forward half-step
    (k → k+½).

    Natural positions (Taflove convention, axis 0=x, 1=y, 2=z):
        E_x: (i+½, j,   k  )  →  shift x: -½, z: +½
        E_y: (i,   j+½, k  )  →  shift y: -½, z: +½
        E_z: (i,   j,   k+½)  →  already at target
        H_x: (i,   j+½, k+½)  →  shift y: -½
        H_y: (i+½, j,   k+½)  →  shift x: -½
        H_z: (i+½, j+½, k  )  →  shift x: -½, y: -½, z: +½

    Args:
        E_pad: Pre-padded electric field array of shape (3, Nx+2, Ny+2, Nz+2)
        H_pad: Pre-padded magnetic field array of shape (3, Nx+2, Ny+2, Nz+2)
        config: Optional simulation configuration.  When it carries a
            non-uniform ``RectilinearGrid``, center-to-edge interpolations use local
            physical distances instead of equal weights.

    Returns:
        Tuple of (E_interp, H_interp), each of shape (3, Nx, Ny, Nz)
    """
    E_x, E_y, E_z = E_pad[0], E_pad[1], E_pad[2]
    H_x, H_y, H_z = H_pad[0], H_pad[1], H_pad[2]

    # E_x: (i+½, j, k) → (i, j, k+½): x backward, z forward
    E_x_lower_z = _backward_edge_average(
        current=E_x[1:-1, 1:-1, 1:-1],
        previous=E_x[:-2, 1:-1, 1:-1],
        config=config,
        axis=0,
    )
    E_x_upper_z = _backward_edge_average(
        current=E_x[1:-1, 1:-1, 2:],
        previous=E_x[:-2, 1:-1, 2:],
        config=config,
        axis=0,
    )
    E_x = (E_x_lower_z + E_x_upper_z) / 2.0

    # E_y: (i, j+½, k) → (i, j, k+½): y backward, z forward
    E_y_lower_z = _backward_edge_average(
        current=E_y[1:-1, 1:-1, 1:-1],
        previous=E_y[1:-1, :-2, 1:-1],
        config=config,
        axis=1,
    )
    E_y_upper_z = _backward_edge_average(
        current=E_y[1:-1, 1:-1, 2:],
        previous=E_y[1:-1, :-2, 2:],
        config=config,
        axis=1,
    )
    E_y = (E_y_lower_z + E_y_upper_z) / 2.0

    # E_z: (i, j, k+½) → already at target
    E_z = E_z[1:-1, 1:-1, 1:-1]

    # H_x: (i, j+½, k+½) → (i, j, k+½): y backward only
    H_x = _backward_edge_average(
        current=H_x[1:-1, 1:-1, 1:-1],
        previous=H_x[1:-1, :-2, 1:-1],
        config=config,
        axis=1,
    )

    # H_y: (i+½, j, k+½) → (i, j, k+½): x backward only
    H_y = _backward_edge_average(
        current=H_y[1:-1, 1:-1, 1:-1],
        previous=H_y[:-2, 1:-1, 1:-1],
        config=config,
        axis=0,
    )

    # H_z: (i+½, j+½, k) → (i, j, k+½): x backward, y backward, z forward
    H_z_lower_z_x = _backward_edge_average(
        current=H_z[1:-1, 1:-1, 1:-1],
        previous=H_z[:-2, 1:-1, 1:-1],
        config=config,
        axis=0,
    )
    H_z_lower_z_xy = _backward_edge_average(
        current=H_z_lower_z_x,
        previous=_backward_edge_average(
            current=H_z[1:-1, :-2, 1:-1],
            previous=H_z[:-2, :-2, 1:-1],
            config=config,
            axis=0,
        ),
        config=config,
        axis=1,
    )
    H_z_upper_z_x = _backward_edge_average(
        current=H_z[1:-1, 1:-1, 2:],
        previous=H_z[:-2, 1:-1, 2:],
        config=config,
        axis=0,
    )
    H_z_upper_z_xy = _backward_edge_average(
        current=H_z_upper_z_x,
        previous=_backward_edge_average(
            current=H_z[1:-1, :-2, 2:],
            previous=H_z[:-2, :-2, 2:],
            config=config,
            axis=0,
        ),
        config=config,
        axis=1,
    )
    H_z = (H_z_lower_z_xy + H_z_upper_z_xy) / 2.0

    E_interp = jnp.stack([E_x, E_y, E_z], axis=0)
    H_interp = jnp.stack([H_x, H_y, H_z], axis=0)

    return E_interp, H_interp


def compute_pml_coefficients(
    alpha: jax.Array,
    kappa: jax.Array,
    sigma: jax.Array,
    time_step_duration: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Precompute the time-invariant CPML recurrence coefficients.

    The auxiliary-field recurrence coefficients ``a`` and ``b`` and the reciprocal
    stretching factor ``1 / kappa`` depend only on the PML profiles and the time step,
    so they can be computed once at setup instead of every FDTD step. Computing them in
    the same dtype and with the same expressions as the historical per-step code keeps
    the result bit-identical.

    Args:
        alpha (jax.Array): Alpha (complex frequency shift) profile, shape (6, nx, ny, nz).
        kappa (jax.Array): Kappa (coordinate stretching) profile, shape (6, nx, ny, nz).
        sigma (jax.Array): Sigma (conductivity) profile, shape (6, nx, ny, nz).
        time_step_duration (float): FDTD time step duration in seconds.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: ``(pml_a, pml_b, pml_inv_kappa)``, each
            of shape (6, nx, ny, nz).
    """
    b = jnp.expm1(-time_step_duration / eps0 * (sigma / kappa + alpha)) + 1
    a = jnp.nan_to_num((b - 1.0) * sigma / (sigma + alpha * kappa) / kappa, nan=0.0, posinf=0.0, neginf=0.0)
    inv_kappa = 1.0 / kappa
    return a, b, inv_kappa


def curl_E(
    config: SimulationConfig,
    E_pad: jax.Array,
    psi_H: jax.Array,
    pml_a: jax.Array,
    pml_b: jax.Array,
    pml_inv_kappa: jax.Array,
    pml_indices: jax.Array,
    simulate_boundaries: bool,
) -> tuple[jax.Array, jax.Array]:
    """Transforms an E-type field into an H-type field by performing a curl operation.

    Computes the discrete curl of the electric field to obtain the corresponding
    magnetic field components. The input E-field is defined on the edges of the Yee grid
    cells (integer grid points), while the output H-field is defined on the faces
    (half-integer grid points).

    The CPML correction is applied sparsely: the full-volume curl is computed first
    (kappa=1, psi=0), then the per-cell PML correction is gathered/updated/scattered
    over the ``M`` shell cells given by ``pml_indices``.

    Args:
        config (SimulationConfig): Simulation configuration parameters.
        E_pad (jax.Array): Pre-padded electric field of shape (3, nx+2, ny+2, nz+2).
        psi_H (jax.Array): Auxiliary field for the magnetic field, stored sparsely.
            Shape is (6, M) for the 6 auxiliary fields over M shell cells.
        pml_a (jax.Array): Precomputed PML recurrence coefficient ``a``, gathered at the
            shell. Shape is (6, M). See :func:`compute_pml_coefficients`.
        pml_b (jax.Array): Precomputed PML recurrence coefficient ``b``, gathered at the
            shell. Shape is (6, M).
        pml_inv_kappa (jax.Array): Precomputed reciprocal stretching factor ``1 / kappa``,
            gathered at the shell. Shape is (6, M).
        pml_indices (jax.Array): Grid coordinates (ix, iy, iz) of the M shell cells, shape (3, M).
        simulate_boundaries (bool): Whether to update the PML auxiliary fields.

    Returns:
        jax.Array: The curl of E - an H-type field located on the faces of the grid
                  (half-integer grid points). Has same shape as input (3, nx, ny, nz).
    """

    shape = E_pad.shape[1] - 2, E_pad.shape[2] - 2, E_pad.shape[3] - 2
    dx_scale = _metric_scale(config, axis=0, shape=shape, stencil="forward")
    dy_scale = _metric_scale(config, axis=1, shape=shape, stencil="forward")
    dz_scale = _metric_scale(config, axis=2, shape=shape, stencil="forward")

    dyEz = (jnp.roll(E_pad[2], -1, axis=1) - E_pad[2])[1:-1, 1:-1, 1:-1] * dy_scale
    dzEy = (jnp.roll(E_pad[1], -1, axis=2) - E_pad[1])[1:-1, 1:-1, 1:-1] * dz_scale
    dzEx = (jnp.roll(E_pad[0], -1, axis=2) - E_pad[0])[1:-1, 1:-1, 1:-1] * dz_scale
    dxEz = (jnp.roll(E_pad[2], -1, axis=0) - E_pad[2])[1:-1, 1:-1, 1:-1] * dx_scale
    dxEy = (jnp.roll(E_pad[1], -1, axis=0) - E_pad[1])[1:-1, 1:-1, 1:-1] * dx_scale
    dyEx = (jnp.roll(E_pad[0], -1, axis=1) - E_pad[0])[1:-1, 1:-1, 1:-1] * dy_scale

    # Raw curl (kappa=1, psi=0): correct everywhere outside the PML shell.
    curl_x = dyEz - dzEy
    curl_y = dzEx - dxEz
    curl_z = dxEy - dyEx

    # Gather the six derivative terms at the M shell cells.
    ix, iy, iz = pml_indices[0], pml_indices[1], pml_indices[2]
    g_dyEz, g_dzEy = dyEz[ix, iy, iz], dzEy[ix, iy, iz]
    g_dzEx, g_dxEz = dzEx[ix, iy, iz], dxEz[ix, iy, iz]
    g_dxEy, g_dyEx = dxEy[ix, iy, iz], dyEx[ix, iy, iz]

    psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy = psi_H

    if simulate_boundaries:
        # Update auxiliary fields using precomputed H-field PML coefficients
        psi_Hxy = pml_b[4] * psi_Hxy + pml_a[4] * g_dyEz
        psi_Hxz = pml_b[5] * psi_Hxz + pml_a[5] * g_dzEy
        psi_Hyz = pml_b[5] * psi_Hyz + pml_a[5] * g_dzEx
        psi_Hyx = pml_b[3] * psi_Hyx + pml_a[3] * g_dxEz
        psi_Hzx = pml_b[3] * psi_Hzx + pml_a[3] * g_dxEy
        psi_Hzy = pml_b[4] * psi_Hzy + pml_a[4] * g_dyEx

    psi_H_updated = jnp.stack((psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy), axis=0)

    # Per-term PML correction (inv_kappa indices [1],[2],[0] as in the dense formulation)
    # so that raw_curl + correction == (inv_kappa * diff + psi) at the shell cells.
    corr_xy = (pml_inv_kappa[1] - 1.0) * g_dyEz + psi_Hxy
    corr_xz = (pml_inv_kappa[2] - 1.0) * g_dzEy + psi_Hxz
    corr_yz = (pml_inv_kappa[2] - 1.0) * g_dzEx + psi_Hyz
    corr_yx = (pml_inv_kappa[0] - 1.0) * g_dxEz + psi_Hyx
    corr_zx = (pml_inv_kappa[0] - 1.0) * g_dxEy + psi_Hzx
    corr_zy = (pml_inv_kappa[1] - 1.0) * g_dyEx + psi_Hzy

    # Fused scatter: assemble the raw curl (3, N) and the per-term corrections (3, M),
    # then apply a single scatter-add across all three components at the shell cells.
    # Equivalent to three per-component scatters but emits one scatter kernel instead.
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)
    corr = jnp.stack((corr_xy - corr_xz, corr_yz - corr_yx, corr_zx - corr_zy), axis=0)
    curl = curl.at[:, ix, iy, iz].add(corr)

    return curl, psi_H_updated


def curl_H(
    config: SimulationConfig,
    H_pad: jax.Array,
    psi_E: jax.Array,
    pml_a: jax.Array,
    pml_b: jax.Array,
    pml_inv_kappa: jax.Array,
    pml_indices: jax.Array,
    simulate_boundaries: bool,
) -> tuple[jax.Array, jax.Array]:
    """Transforms an H-type field into an E-type field by performing a curl operation.

    Computes the discrete curl of the magnetic field to obtain the corresponding
    electric field components. The input H-field is defined on the faces of the Yee grid
    cells (half-integer grid points), while the output E-field is defined on the edges
    (integer grid points).

    The CPML correction is applied sparsely: the full-volume curl is computed first
    (kappa=1, psi=0), then the per-cell PML correction is gathered/updated/scattered
    over the ``M`` shell cells given by ``pml_indices``.

    Args:
        config (SimulationConfig): Simulation configuration parameters.
        H_pad (jax.Array): Pre-padded magnetic field of shape (3, nx+2, ny+2, nz+2).
        psi_E (jax.Array): Auxiliary field for the electric field, stored sparsely.
            Shape is (6, M) for the 6 auxiliary fields over M shell cells.
        pml_a (jax.Array): Precomputed PML recurrence coefficient ``a``, gathered at the
            shell. Shape is (6, M). See :func:`compute_pml_coefficients`.
        pml_b (jax.Array): Precomputed PML recurrence coefficient ``b``, gathered at the
            shell. Shape is (6, M).
        pml_inv_kappa (jax.Array): Precomputed reciprocal stretching factor ``1 / kappa``,
            gathered at the shell. Shape is (6, M).
        pml_indices (jax.Array): Grid coordinates (ix, iy, iz) of the M shell cells, shape (3, M).
        simulate_boundaries (bool): Whether to update the PML auxiliary fields.

    Returns:
        jax.Array: The curl of H - an E-type field located on the edges of the grid
                  (integer grid points). Has same shape as input (3, nx, ny, nz).
    """

    shape = H_pad.shape[1] - 2, H_pad.shape[2] - 2, H_pad.shape[3] - 2
    dx_scale = _metric_scale(config, axis=0, shape=shape, stencil="backward")
    dy_scale = _metric_scale(config, axis=1, shape=shape, stencil="backward")
    dz_scale = _metric_scale(config, axis=2, shape=shape, stencil="backward")

    dyHz = (H_pad[2] - jnp.roll(H_pad[2], 1, axis=1))[1:-1, 1:-1, 1:-1] * dy_scale
    dzHy = (H_pad[1] - jnp.roll(H_pad[1], 1, axis=2))[1:-1, 1:-1, 1:-1] * dz_scale
    dzHx = (H_pad[0] - jnp.roll(H_pad[0], 1, axis=2))[1:-1, 1:-1, 1:-1] * dz_scale
    dxHz = (H_pad[2] - jnp.roll(H_pad[2], 1, axis=0))[1:-1, 1:-1, 1:-1] * dx_scale
    dxHy = (H_pad[1] - jnp.roll(H_pad[1], 1, axis=0))[1:-1, 1:-1, 1:-1] * dx_scale
    dyHx = (H_pad[0] - jnp.roll(H_pad[0], 1, axis=1))[1:-1, 1:-1, 1:-1] * dy_scale

    # Raw curl (kappa=1, psi=0): correct everywhere outside the PML shell.
    curl_x = dyHz - dzHy
    curl_y = dzHx - dxHz
    curl_z = dxHy - dyHx

    # Gather the six derivative terms at the M shell cells.
    ix, iy, iz = pml_indices[0], pml_indices[1], pml_indices[2]
    g_dyHz, g_dzHy = dyHz[ix, iy, iz], dzHy[ix, iy, iz]
    g_dzHx, g_dxHz = dzHx[ix, iy, iz], dxHz[ix, iy, iz]
    g_dxHy, g_dyHx = dxHy[ix, iy, iz], dyHx[ix, iy, iz]

    psi_Exy, psi_Exz, psi_Eyz, psi_Eyx, psi_Ezx, psi_Ezy = psi_E

    if simulate_boundaries:
        # Update auxiliary fields using precomputed E-field PML coefficients
        psi_Exy = pml_b[1] * psi_Exy + pml_a[1] * g_dyHz
        psi_Exz = pml_b[2] * psi_Exz + pml_a[2] * g_dzHy
        psi_Eyz = pml_b[2] * psi_Eyz + pml_a[2] * g_dzHx
        psi_Eyx = pml_b[0] * psi_Eyx + pml_a[0] * g_dxHz
        psi_Ezx = pml_b[0] * psi_Ezx + pml_a[0] * g_dxHy
        psi_Ezy = pml_b[1] * psi_Ezy + pml_a[1] * g_dyHx

    psi_E_updated = jnp.stack((psi_Exy, psi_Exz, psi_Eyz, psi_Eyx, psi_Ezx, psi_Ezy), axis=0)

    # Per-term PML correction (inv_kappa indices [1],[2],[0] as in the dense formulation)
    # so that raw_curl + correction == (inv_kappa * diff + psi) at the shell cells.
    corr_xy = (pml_inv_kappa[1] - 1.0) * g_dyHz + psi_Exy
    corr_xz = (pml_inv_kappa[2] - 1.0) * g_dzHy + psi_Exz
    corr_yz = (pml_inv_kappa[2] - 1.0) * g_dzHx + psi_Eyz
    corr_yx = (pml_inv_kappa[0] - 1.0) * g_dxHz + psi_Eyx
    corr_zx = (pml_inv_kappa[0] - 1.0) * g_dxHy + psi_Ezx
    corr_zy = (pml_inv_kappa[1] - 1.0) * g_dyHx + psi_Ezy

    # Fused scatter: assemble the raw curl (3, N) and the per-term corrections (3, M),
    # then apply a single scatter-add across all three components at the shell cells.
    # Equivalent to three per-component scatters but emits one scatter kernel instead.
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)
    corr = jnp.stack((corr_xy - corr_xz, corr_yz - corr_yx, corr_zx - corr_zy), axis=0)
    curl = curl.at[:, ix, iy, iz].add(corr)

    return curl, psi_E_updated
