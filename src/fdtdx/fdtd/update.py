import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.constants import eta0
from fdtdx.core.misc import expand_to_3x3, pad_fields
from fdtdx.core.physics.curl import curl_E, curl_H, interpolate_fields
from fdtdx.core.switch import OnOffSwitch
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.misc import (
    add_boundary_interfaces,
    avg_anisotropic_E_component,
    avg_anisotropic_H_component,
    collect_boundary_interfaces,
    compute_anisotropic_update_matrices,
    compute_anisotropic_update_matrices_reverse,
)
from fdtdx.objects.detectors.detector import Detector, DetectorState


def _source_uses_default_always_on_switch(source) -> bool:
    switch = getattr(source, "switch", None)
    return isinstance(switch, OnOffSwitch) and switch.is_default_always_on


def get_wrap_padding_axes(objects: ObjectContainer) -> tuple[bool, bool, bool]:
    """Determines which axes should use wrap (periodic) padding.

    Delegates to each boundary's `uses_wrap_padding` property, so no
    boundary-type-specific logic lives in the update loop.

    Args:
        objects (ObjectContainer): Container with simulation objects including boundaries

    Returns:
        tuple[bool, bool, bool]: Tuple indicating which axes (x,y,z) use wrap padding
    """
    wrap_axes = [False, False, False]
    for boundary in objects.boundary_objects:
        if boundary.uses_wrap_padding:
            wrap_axes[boundary.axis] = True
    return tuple(wrap_axes)  # type: ignore


def apply_boundary_post_E_update(
    E: jax.Array,
    objects: ObjectContainer,
) -> jax.Array:
    """Apply all boundary post-E-update enforcement.

    Delegates to each boundary's `apply_post_E_update` method, so
    boundary-specific logic (e.g. PEC tangential zeroing) lives in
    the boundary class, not here.

    Args:
        E: Electric field array of shape (3, Nx, Ny, Nz)
        objects: Container with simulation objects including boundaries

    Returns:
        E field with all boundary conditions enforced
    """
    for boundary in objects.boundary_objects:
        E = boundary.apply_post_E_update(E)
    return E


def apply_boundary_post_H_update(
    H: jax.Array,
    objects: ObjectContainer,
) -> jax.Array:
    """Apply all boundary post-H-update enforcement.

    Delegates to each boundary's `apply_post_H_update` method, so
    boundary-specific logic (e.g. PMC tangential zeroing) lives in
    the boundary class, not here.

    Args:
        H: Magnetic field array of shape (3, Nx, Ny, Nz)
        objects: Container with simulation objects including boundaries

    Returns:
        H field with all boundary conditions enforced
    """
    for boundary in objects.boundary_objects:
        H = boundary.apply_post_H_update(H)
    return H


def pad_fields_for_boundaries(
    fields: jax.Array,
    objects: ObjectContainer,
    config: SimulationConfig,
) -> jax.Array:
    """Pad fields and apply boundary-specific corrections.

    Combines wrap/constant padding with boundary-specific corrections
    (e.g. Bloch phase shifts) in a single call.

    Args:
        fields: Field array of shape (3, Nx, Ny, Nz)
        objects: Container with simulation objects including boundaries
        config: Simulation configuration. The scalar spacing argument is kept
            for boundary API compatibility; grid-aware boundaries should read
            physical metrics from ``config.grid`` when it is available.

    Returns:
        Padded fields of shape (3, Nx+2, Ny+2, Nz+2) with all corrections applied
    """
    periodic_axes = get_wrap_padding_axes(objects)
    padded = pad_fields(fields, periodic_axes)
    boundaries = objects.boundary_objects
    if boundaries:
        volume_shape = objects.volume.grid_shape
        if config.has_nonuniform_grid:
            assert config.resolved_grid is not None
            spacing = float(config.resolved_grid.min_spacing)
        else:
            spacing = config.uniform_spacing()
        for boundary in boundaries:
            padded = boundary.apply_pad_correction(padded, volume_shape, spacing)
    return padded


def get_anisotropic_averaging_widths(
    config: SimulationConfig,
) -> tuple[jax.Array, jax.Array, jax.Array] | None:
    """Build the per-axis cell widths that spacing-weight the off-diagonal anisotropic average.

    The result depends only on the run-fixed grid, so the averaging functions take it as a
    precomputed input and operate on arrays alone; under JIT it folds to a constant with no
    per-step cost. Each entry is the axis cell widths padded by replicating the edge cell (to
    line up with the field halo) and reshaped to broadcast along that axis.

    Args:
        config (SimulationConfig): Simulation configuration providing the resolved grid.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array] | None: Per-axis padded cell widths, or None on
            a uniform grid (where the averaging keeps its unweighted four-point mean).
    """
    if not config.has_nonuniform_grid:
        return None
    grid = config.resolved_grid
    assert grid is not None  # narrowed by has_nonuniform_grid
    widths = []
    for axis in range(3):
        axis_widths = grid.cell_widths(axis)
        padded = jnp.concatenate([axis_widths[:1], axis_widths, axis_widths[-1:]])
        broadcast_shape = [1, 1, 1]
        broadcast_shape[axis] = padded.shape[0]
        widths.append(padded.reshape(broadcast_shape))
    return (widths[0], widths[1], widths[2])


def update_E(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    simulate_boundaries: bool,
) -> ArrayContainer:
    """Updates the electric field (E) according to Maxwell's equations using the FDTD method.

    Implements the discretized form of dE/dt = (1/eps) curl(H) on the Yee grid. Updates include:
    1. PML/periodic boundary conditions if simulate_boundaries=True
    2. Source contributions for active sources
    3. Field updates based on curl of H field

    Args:
        time_step (jax.Array): Current simulation time step
        arrays (ArrayContainer): Container with E, H fields and material properties
        objects (ObjectContainer): Container with sources, boundaries and other simulation objects
        config (SimulationConfig): Simulation configuration parameters
        simulate_boundaries (bool): Whether to apply boundary conditions

    Returns:
        ArrayContainer: Updated ArrayContainer with new E field values
    """

    inv_eps = arrays.inv_permittivities
    sigma_E = arrays.electric_conductivity
    c = config.courant_number
    H_pad = pad_fields_for_boundaries(arrays.fields.H, objects, config)
    curl, psi_E = curl_H(
        config,
        H_pad,
        arrays.fields.psi_E,
        objects,
        simulate_boundaries,
    )
    arrays = arrays.aset("fields->psi_E", psi_E)

    # Check if we have full anisotropic tensors (shape[0] == 9)
    inv_eps_is_full_tensor = inv_eps.shape[0] == 9
    sigma_E_is_full_tensor = sigma_E is not None and sigma_E.shape[0] == 9

    if not inv_eps_is_full_tensor and not sigma_E_is_full_tensor:
        # Isotropic and diagonal anisotropic case
        factor = 1
        if sigma_E is not None:
            # update formula for lossy material. Simplifies to Noop for conductivity = 0
            # for details see Schneider, chapter 3.12
            # Component-wise multiplication: sigma_E[i, x, y, z] * inv_eps[i, x, y, z]
            factor = 1 - c * sigma_E * eta0 * inv_eps / 2

        # standard update formula using lossless material
        # Component-wise multiplication for diagonally anisotropic materials:
        # E[i, x, y, z] = factor * E[i, x, y, z] + c * curl[i, x, y, z] * inv_eps[i, x, y, z]
        E = factor * arrays.fields.E + c * curl * inv_eps

        # Dispersive (ADE) correction. Non-dispersive cells have c3 = 0, so P_hat =
        # c1*P_curr + c2*P_prev and (P_curr - P_hat) reduces to a purely historical
        # term that is also zero when P_curr and P_prev start at zero — so it's a
        # no-op outside dispersive regions. Only active when arrays are allocated.
        if arrays.fields.dispersive_P_curr is not None:
            P_curr = arrays.fields.dispersive_P_curr
            P_prev = arrays.fields.dispersive_P_prev
            disp_c1 = arrays.dispersive_c1
            disp_c2 = arrays.dispersive_c2
            disp_c3 = arrays.dispersive_c3
            disp_c4 = arrays.dispersive_c4
            assert P_prev is not None and disp_c1 is not None and disp_c2 is not None and disp_c3 is not None
            # disp_c* are (num_poles, 1|3, Nx, Ny, Nz) — the component axis is 1
            # (isotropic dispersion, broadcast) or 3 (per-axis anisotropic
            # dispersion); arrays.fields.E is (3, Nx, Ny, Nz). Right-aligned
            # broadcasting produces (num_poles, 3, Nx, Ny, Nz) either way without
            # an explicit newaxis — skip the reshape so the HLO stays flat.
            # P_hat is the explicit part of the recurrence (independent of E^{n+1}).
            P_hat = disp_c1 * P_curr + disp_c2 * P_prev + disp_c3 * arrays.fields.E
            delta_hat = jnp.sum(P_curr - P_hat, axis=0)
            E = E + inv_eps * delta_hat
            if disp_c4 is not None:
                # CCPR: the polarization couples to E^{n+1} through c4. Fold the
                # per-cell D_kappa = inv_eps*sum(c4) into the implicit divide
                # (alongside the conductivity loss factor), then reconstruct the
                # full P^{n+1} = P_hat + c4*E^{n+1} once E^{n+1} is known.
                divisor = 1 + inv_eps * jnp.sum(disp_c4, axis=0)
                if sigma_E is not None:
                    divisor = divisor + c * sigma_E * eta0 * inv_eps / 2
                E = E / divisor
                P_new = P_hat + disp_c4 * E
                arrays = arrays.aset("fields->dispersive_P_prev", P_curr)
                arrays = arrays.aset("fields->dispersive_P_curr", P_new)
            else:
                arrays = arrays.aset("fields->dispersive_P_prev", P_curr)
                arrays = arrays.aset("fields->dispersive_P_curr", P_hat)
                if sigma_E is not None:
                    # lossy update formula. Noop for conductivity = 0; see Schneider 3.12
                    E = E / (1 + c * sigma_E * eta0 * inv_eps / 2)
        elif sigma_E is not None:
            # update formula for lossy material. Simplifies to Noop for conductivity = 0
            # for details see Schneider, chapter 3.12
            E = E / (1 + c * sigma_E * eta0 * inv_eps / 2)

    else:
        # Full anisotropic case: expand inv_eps and sigma_E to (3, 3, Nx, Ny, Nz)
        inv_eps = expand_to_3x3(inv_eps)
        sigma_E = expand_to_3x3(sigma_E)

        # Compute A and B matrices for forward update
        # E^(n+1) = A @ E^(n) + B @ curl(H^(n+1/2))
        A, B = compute_anisotropic_update_matrices(inv_eps, sigma_E, c, eta0)

        # We need to pad the fields to account for ghost cells when computing the averages
        E_pad = pad_fields_for_boundaries(arrays.fields.E, objects, config)

        # Spacing weights for the off-diagonal average (None on a uniform grid).
        aniso_widths = get_anisotropic_averaging_widths(config)
        # Compute the averages of the fields and curl
        Ex_y_avg = avg_anisotropic_E_component(
            E_pad, component=0, location=1, aniso_widths=aniso_widths
        )  # calc Ex at location of Ey
        Ex_z_avg = avg_anisotropic_E_component(
            E_pad, component=0, location=2, aniso_widths=aniso_widths
        )  # calc Ex at location of Ez
        Ey_x_avg = avg_anisotropic_E_component(
            E_pad, component=1, location=0, aniso_widths=aniso_widths
        )  # calc Ey at location of Ex
        Ey_z_avg = avg_anisotropic_E_component(
            E_pad, component=1, location=2, aniso_widths=aniso_widths
        )  # calc Ey at location of Ez
        Ez_x_avg = avg_anisotropic_E_component(
            E_pad, component=2, location=0, aniso_widths=aniso_widths
        )  # calc Ez at location of Ex
        Ez_y_avg = avg_anisotropic_E_component(
            E_pad, component=2, location=1, aniso_widths=aniso_widths
        )  # calc Ez at location of Ey

        # Dispersive (ADE) correction. Same recurrence as the diagonal branch,
        # except the field coupling may be a full 3x3 tensor per pole (oriented
        # poles): each off-diagonal entry multiplies the neighboring E component
        # averaged to the target component's Yee location. The polarization term
        # delta enters the E update exactly like the curl (both are currents on
        # the RHS of Ampere's law): since B = c * M1^-1 @ inv_eps, folding
        # curl += delta / c before the curl averages yields M1^-1 @ inv_eps @ delta
        # including its off-diagonal spatial averaging — no extra solve needed.
        # CCPR c4 poles are rejected at initialization for this branch.
        if arrays.fields.dispersive_P_curr is not None:
            P_curr = arrays.fields.dispersive_P_curr
            P_prev = arrays.fields.dispersive_P_prev
            disp_c1 = arrays.dispersive_c1
            disp_c2 = arrays.dispersive_c2
            disp_c3 = arrays.dispersive_c3
            assert P_prev is not None and disp_c1 is not None and disp_c2 is not None and disp_c3 is not None
            assert arrays.dispersive_c4 is None
            if disp_c3.shape[1] == 9:
                # E at each row's Yee location: diagonal entries use the local
                # component, off-diagonal entries the averaged neighbor with
                # symmetrized pair weights restricted to material cells,
                #   w(r, s) = stencil(r, s) * (c(r) m(s) + m(r) c(s)) / 2,  m = (c != 0),
                # so the coupling blocks stay mutual adjoints where c varies.
                # Plain c(r) * avg(E) couples boundary cells to vacuum neighbors
                # that do not couple back, and the resulting non-normal operator
                # amplifies boundary modes of media with off-diagonal coupling and
                # mixed-sign permittivity. Uniform c reduces to the plain form.

                def _avg_offdiag(arr_pad: jax.Array, component: int, location: int):
                    # mirrors avg_anisotropic_E_component with the leading pole
                    # axis offsetting the spatial axes by one
                    la, ca = location + 1, component + 1
                    return (
                        arr_pad
                        + jnp.roll(arr_pad, -1, axis=la)
                        + jnp.roll(arr_pad, 1, axis=ca)
                        + jnp.roll(arr_pad, (-1, 1), axis=(la, ca))
                    )[:, 1:-1, 1:-1, 1:-1] / 4

                avg_at = {
                    (0, 1): Ey_x_avg,
                    (0, 2): Ez_x_avg,
                    (1, 0): Ex_y_avg,
                    (1, 2): Ez_y_avg,
                    (2, 0): Ex_z_avg,
                    (2, 1): Ey_z_avg,
                }
                rows = []
                for i in range(3):
                    row = disp_c3[:, 3 * i + i] * arrays.fields.E[i]
                    for j in range(3):
                        if j == i:
                            continue
                        cij = disp_c3[:, 3 * i + j]
                        if aniso_widths is None:
                            pad = ((0, 0), (1, 1), (1, 1), (1, 1))
                            cij_pad = jnp.pad(cij, pad, mode="edge")
                            mask_pad = (cij_pad != 0.0).astype(cij.dtype)
                            row = row + 0.5 * (
                                cij * _avg_offdiag(mask_pad * E_pad[j], component=j, location=i)
                                + (cij != 0.0) * _avg_offdiag(cij_pad * E_pad[j], component=j, location=i)
                            )
                        else:
                            # non-uniform grids keep the legacy form for now (the
                            # weighted symmetrization needs matching edge weights)
                            row = row + cij * avg_at[(i, j)]
                    rows.append(row)
                coupling = jnp.stack(rows, axis=1)
            else:
                coupling = disp_c3 * arrays.fields.E
            P_hat = disp_c1 * P_curr + disp_c2 * P_prev + coupling
            delta = jnp.sum(P_curr - P_hat, axis=0)
            curl = curl + delta / c
            arrays = arrays.aset("fields->dispersive_P_prev", P_curr)
            arrays = arrays.aset("fields->dispersive_P_curr", P_hat)

        curl_pad = pad_fields_for_boundaries(curl, objects, config)
        curlHx_y_avg = avg_anisotropic_E_component(
            curl_pad, component=0, location=1, aniso_widths=aniso_widths
        )  # calc curl(H)x at location of Ey
        curlHx_z_avg = avg_anisotropic_E_component(
            curl_pad, component=0, location=2, aniso_widths=aniso_widths
        )  # calc curl(H)x at location of Ez
        curlHy_x_avg = avg_anisotropic_E_component(
            curl_pad, component=1, location=0, aniso_widths=aniso_widths
        )  # calc curl(H)y at location of Ex
        curlHy_z_avg = avg_anisotropic_E_component(
            curl_pad, component=1, location=2, aniso_widths=aniso_widths
        )  # calc curl(H)y at location of Ez
        curlHz_x_avg = avg_anisotropic_E_component(
            curl_pad, component=2, location=0, aniso_widths=aniso_widths
        )  # calc curl(H)z at location of Ex
        curlHz_y_avg = avg_anisotropic_E_component(
            curl_pad, component=2, location=1, aniso_widths=aniso_widths
        )  # calc curl(H)z at location of Ey

        # K = curl(H)
        # Ex <= (Axx * Ex + Axy * x_avg(Ey) + Axz * x_avg(Ez)) +
        #       (Bxx * Kx + Bxy * x_avg(Ky) + Bxz * x_avg(Kz))
        Ex = (A[0, 0] * arrays.fields.E[0] + A[0, 1] * Ey_x_avg + A[0, 2] * Ez_x_avg) + (
            B[0, 0] * curl[0] + B[0, 1] * curlHy_x_avg + B[0, 2] * curlHz_x_avg
        )
        # Ey <= (Ayx * y_avg(Ex) + Ayy * Ey + Ayz * y_avg(Ez)) +
        #       (Byx * y_avg(Kx) + Byy * Ky + Byz * y_avg(Kz))
        Ey = (A[1, 0] * Ex_y_avg + A[1, 1] * arrays.fields.E[1] + A[1, 2] * Ez_y_avg) + (
            B[1, 0] * curlHx_y_avg + B[1, 1] * curl[1] + B[1, 2] * curlHz_y_avg
        )
        # Ez <= (Azx * z_avg(Ex) + Azy * z_avg(Ey) + Azz * Ez) +
        #       (Bzx * z_avg(Kx) + Bzy * z_avg(Ky) + Bzz * Kz)
        Ez = (A[2, 0] * Ex_z_avg + A[2, 1] * Ey_z_avg + A[2, 2] * arrays.fields.E[2]) + (
            B[2, 0] * curlHx_z_avg + B[2, 1] * curlHy_z_avg + B[2, 2] * curl[2]
        )

        E = jnp.stack((Ex, Ey, Ez), axis=0)

    for source in objects.sources:
        if _source_uses_default_always_on_switch(source):
            E = source.update_E(
                E=E,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=time_step,
                inverse=False,
            )
            continue

        def _update():
            adj_time_step = source.adjust_time_step_by_on_off(time_step)
            return source.update_E(
                E=E,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=adj_time_step,
                inverse=False,
            )

        E = jax.lax.cond(
            source.is_on_at_time_step(time_step),
            _update,
            lambda: E,
        )

    E = apply_boundary_post_E_update(E, objects)
    arrays = arrays.aset("fields->E", E)
    return arrays


def update_E_reverse(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
) -> ArrayContainer:
    """Reverse time step update for the electric field used in automatic differentiation.

    Implements the inverse update step that transforms the electromagnetic field state
    from time step t+1 to time step t, leveraging the time-reversibility property of
    Maxwell's equations.

    Args:
        time_step (jax.Array): Current simulation time step
        arrays (ArrayContainer): Container with E, H fields and material properties
        objects (ObjectContainer): Container with sources and other simulation objects
        config (SimulationConfig): Simulation configuration parameters

    Returns:
        ArrayContainer: Updated ArrayContainer with reversed E field values
    """
    E = arrays.fields.E
    for source in objects.sources:
        if _source_uses_default_always_on_switch(source):
            E = source.update_E(
                E,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=time_step,
                inverse=True,
            )
            continue

        def _update():
            adj_time_step = source.adjust_time_step_by_on_off(time_step)
            return source.update_E(
                E,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=adj_time_step,
                inverse=True,
            )

        E = jax.lax.cond(
            source.is_on_at_time_step(time_step),
            _update,
            lambda: E,
        )

    inv_eps = arrays.inv_permittivities
    sigma_E = arrays.electric_conductivity
    c = config.courant_number
    H_pad = pad_fields_for_boundaries(arrays.fields.H, objects, config)
    curl, _ = curl_H(
        config,
        H_pad,
        arrays.fields.psi_E,
        objects,
        False,
    )

    # Check if we have full anisotropic tensors (shape[0] == 9)
    inv_eps_is_full_tensor = inv_eps.shape[0] == 9
    sigma_E_is_full_tensor = sigma_E is not None and sigma_E.shape[0] == 9

    if not inv_eps_is_full_tensor and not sigma_E_is_full_tensor:
        # Isotropic and diagonal anisotropic case
        # Capture E^{n+1} (post source-inverse, pre field-recovery) — the CCPR
        # polarization inversion below needs it for the c4*E^{n+1} term. For the
        # E-recovery itself the c4/D_kappa contributions cancel exactly, so that
        # formula is unchanged from the non-CCPR case.
        E_np1 = E
        factor = 1
        if sigma_E is not None:
            E = E * (1 + c * sigma_E * eta0 * inv_eps / 2)
            factor = 1 - c * sigma_E * eta0 * inv_eps / 2

        # Dispersive (ADE) reverse correction. At reverse time, arrays.fields.dispersive_P_curr
        # holds P^(n+1) and arrays.fields.dispersive_P_prev holds P^n. The forward update added
        # inv_eps * sum(P^n - P^(n+1)) inside the lossy factor; subtract it here to
        # recover E^n. Non-dispersive cells (all zero arrays) contribute zero.
        if arrays.fields.dispersive_P_curr is not None:
            P_curr_r = arrays.fields.dispersive_P_curr
            P_prev_r = arrays.fields.dispersive_P_prev
            disp_c1_r = arrays.dispersive_c1
            disp_c3_r = arrays.dispersive_c3
            disp_c4_r = arrays.dispersive_c4
            disp_inv_c2_r = arrays.dispersive_inv_c2
            assert (
                P_prev_r is not None and disp_c1_r is not None and disp_c3_r is not None and disp_inv_c2_r is not None
            )
            delta_sum = jnp.sum(P_prev_r - P_curr_r, axis=0)
            E = (E - c * curl * inv_eps - inv_eps * delta_sum) / factor
            # Invert the polarization recurrence:
            # P^(n+1) = c1 * P^n + c2 * P^(n-1) + c3 * E^n + c4 * E^(n+1)
            # =>  P^(n-1) = (P^(n+1) - c1 * P^n - c3 * E^n - c4 * E^(n+1)) / c2
            # Using precomputed inv_c2 (with non-dispersive cells zeroed) replaces
            # a per-step jnp.where + division with a single multiply. Non-dispersive
            # cells have inv_c2 = 0 and P_curr = P_prev = 0, so the product is zero.
            # E is now the recovered E^n; the c4 term (CCPR only) uses E^(n+1).
            P_prev_new = P_curr_r - disp_c1_r * P_prev_r - disp_c3_r * E
            if disp_c4_r is not None:
                P_prev_new = P_prev_new - disp_c4_r * E_np1
            P_prev_new = P_prev_new * disp_inv_c2_r
            arrays = arrays.aset("fields->dispersive_P_curr", P_prev_r)
            arrays = arrays.aset("fields->dispersive_P_prev", P_prev_new)
        else:
            E = (E - c * curl * inv_eps) / factor

    else:
        if arrays.fields.dispersive_P_curr is not None:
            # The tensor-branch ADE has no closed-form time reversal (the
            # off-diagonal coupling mixes neighboring cells through the Yee
            # averages). Guarded at initialization; kept here defensively since
            # full_backward is public API.
            raise NotImplementedError(
                "Time-reversal of dispersion on the fully anisotropic update path is not supported; "
                "use the 'checkpointed' gradient method."
            )
        # Full anisotropic case: expand inv_eps and sigma_E to (3, 3, Nx, Ny, Nz)
        inv_eps = expand_to_3x3(inv_eps)
        sigma_E = expand_to_3x3(sigma_E)

        # Compute A and B matrices for reverse update
        # E^(n) = A @ E^(n+1) - B @ curl(H^(n+1/2))
        A, B = compute_anisotropic_update_matrices_reverse(inv_eps, sigma_E, c, eta0)

        # We need to pad the fields and curl to account for ghost cells when computing the averages
        E_pad = pad_fields_for_boundaries(E, objects, config)
        curl_pad = pad_fields_for_boundaries(curl, objects, config)

        # Spacing weights for the off-diagonal average (None on a uniform grid).
        aniso_widths = get_anisotropic_averaging_widths(config)
        # Compute the averages of the fields and curl
        Ex_y_avg = avg_anisotropic_E_component(
            E_pad, component=0, location=1, aniso_widths=aniso_widths
        )  # calc Ex at location of Ey
        Ex_z_avg = avg_anisotropic_E_component(
            E_pad, component=0, location=2, aniso_widths=aniso_widths
        )  # calc Ex at location of Ez
        Ey_x_avg = avg_anisotropic_E_component(
            E_pad, component=1, location=0, aniso_widths=aniso_widths
        )  # calc Ey at location of Ex
        Ey_z_avg = avg_anisotropic_E_component(
            E_pad, component=1, location=2, aniso_widths=aniso_widths
        )  # calc Ey at location of Ez
        Ez_x_avg = avg_anisotropic_E_component(
            E_pad, component=2, location=0, aniso_widths=aniso_widths
        )  # calc Ez at location of Ex
        Ez_y_avg = avg_anisotropic_E_component(
            E_pad, component=2, location=1, aniso_widths=aniso_widths
        )  # calc Ez at location of Ey
        curlHx_y_avg = avg_anisotropic_E_component(
            curl_pad, component=0, location=1, aniso_widths=aniso_widths
        )  # calc curl(H)x at location of Ey
        curlHx_z_avg = avg_anisotropic_E_component(
            curl_pad, component=0, location=2, aniso_widths=aniso_widths
        )  # calc curl(H)x at location of Ez
        curlHy_x_avg = avg_anisotropic_E_component(
            curl_pad, component=1, location=0, aniso_widths=aniso_widths
        )  # calc curl(H)y at location of Ex
        curlHy_z_avg = avg_anisotropic_E_component(
            curl_pad, component=1, location=2, aniso_widths=aniso_widths
        )  # calc curl(H)y at location of Ez
        curlHz_x_avg = avg_anisotropic_E_component(
            curl_pad, component=2, location=0, aniso_widths=aniso_widths
        )  # calc curl(H)z at location of Ex
        curlHz_y_avg = avg_anisotropic_E_component(
            curl_pad, component=2, location=1, aniso_widths=aniso_widths
        )  # calc curl(H)z at location of Ey

        # K = curl(H)
        # Ex <= (Axx * Ex + Axy * x_avg(Ey) + Axz * x_avg(Ez)) -
        #       (Bxx * Kx + Bxy * x_avg(Ky) + Bxz * x_avg(Kz))
        Ex = (A[0, 0] * E[0] + A[0, 1] * Ey_x_avg + A[0, 2] * Ez_x_avg) - (
            B[0, 0] * curl[0] + B[0, 1] * curlHy_x_avg + B[0, 2] * curlHz_x_avg
        )
        # Ey <= (Ayx * y_avg(Ex) + Ayy * Ey + Ayz * y_avg(Ez)) -
        #       (Byx * y_avg(Kx) + Byy * Ky + Byz * y_avg(Kz))
        Ey = (A[1, 0] * Ex_y_avg + A[1, 1] * E[1] + A[1, 2] * Ez_y_avg) - (
            B[1, 0] * curlHx_y_avg + B[1, 1] * curl[1] + B[1, 2] * curlHz_y_avg
        )
        # Ez <= (Azx * z_avg(Ex) + Azy * z_avg(Ey) + Azz * Ez) -
        #       (Bzx * z_avg(Kx) + Bzy * z_avg(Ky) + Bzz * Kz)
        Ez = (A[2, 0] * Ex_z_avg + A[2, 1] * Ey_z_avg + A[2, 2] * E[2]) - (
            B[2, 0] * curlHx_z_avg + B[2, 1] * curlHy_z_avg + B[2, 2] * curl[2]
        )

        E = jnp.stack((Ex, Ey, Ez), axis=0)

    E = apply_boundary_post_E_update(E, objects)
    arrays = arrays.aset("fields->E", E)

    return arrays


def update_H(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    simulate_boundaries: bool,
) -> ArrayContainer:
    """Updates the magnetic field (H) according to Maxwell's equations using the FDTD method.

    Implements the discretized form of dH/dt = -(1/mu) curl(E) on the Yee grid. Updates include:
    1. PML/periodic boundary conditions if simulate_boundaries=True
    2. Source contributions for active sources
    3. Field updates based on curl of E field

    The H field is updated at time points offset by half steps from the E field updates,
    following the Yee grid scheme.

    Args:
        time_step (jax.Array): Current simulation time step
        arrays (ArrayContainer): Container with E, H fields and material properties
        objects (ObjectContainer): Container with sources, boundaries and other simulation objects
        config (SimulationConfig): Simulation configuration parameters
        simulate_boundaries (bool): Whether to apply boundary conditions

    Returns:
        ArrayContainer: Updated ArrayContainer with new H field values
    """

    inv_mu = arrays.inv_permeabilities
    sigma_H = arrays.magnetic_conductivity
    c = config.courant_number
    E_pad = pad_fields_for_boundaries(arrays.fields.E, objects, config)
    curl, psi_H = curl_E(
        config,
        E_pad,
        arrays.fields.psi_H,
        objects,
        simulate_boundaries,
    )
    arrays = arrays.aset("fields->psi_H", psi_H)

    # Check if we have full anisotropic tensors (shape[0] == 9)
    # inv_mu can be a scalar (float) for non-magnetic materials
    inv_mu_shape = getattr(inv_mu, "shape", (0,))
    inv_mu_is_full_tensor = len(inv_mu_shape) > 0 and inv_mu_shape[0] == 9
    sigma_H_is_full_tensor = sigma_H is not None and sigma_H.shape[0] == 9

    if not inv_mu_is_full_tensor and not sigma_H_is_full_tensor:
        # Isotropic and diagonal anisotropic case
        factor = 1
        if sigma_H is not None:
            # update formula for lossy material. Simplifies to Noop for conductivity = 0
            # for details see Schneider, chapter 3.12
            factor = 1 - c * sigma_H / eta0 * inv_mu / 2

        # standard update formula for lossless material
        H = factor * arrays.fields.H - c * curl * inv_mu

        if sigma_H is not None:
            # update formula for lossy material. Simplifies to NoOp for conductivity = 0
            # for details see Schneider, chapter 3.12
            H = H / (1 + c * sigma_H / eta0 * inv_mu / 2)

    else:
        # Full anisotropic case: expand inv_mu and sigma_H to (3, 3, Nx, Ny, Nz)
        inv_mu = expand_to_3x3(inv_mu)
        sigma_H = expand_to_3x3(sigma_H)

        # Compute A and B matrices for forward update
        # H^(n+1/2) = A @ H^(n-1/2) - B @ curl(E^(n))
        A, B = compute_anisotropic_update_matrices(inv_mu, sigma_H, c, 1 / eta0)

        # We need to pad the fields and curl to account for ghost cells when computing the averages
        H_pad = pad_fields_for_boundaries(arrays.fields.H, objects, config)
        curl_pad = pad_fields_for_boundaries(curl, objects, config)

        # Spacing weights for the off-diagonal average (None on a uniform grid).
        aniso_widths = get_anisotropic_averaging_widths(config)
        # Compute the averages of the fields and curl
        Hx_y_avg = avg_anisotropic_H_component(
            H_pad, component=0, location=1, aniso_widths=aniso_widths
        )  # calc Hx at location of Hy
        Hx_z_avg = avg_anisotropic_H_component(
            H_pad, component=0, location=2, aniso_widths=aniso_widths
        )  # calc Hx at location of Hz
        Hy_x_avg = avg_anisotropic_H_component(
            H_pad, component=1, location=0, aniso_widths=aniso_widths
        )  # calc Hy at location of Hx
        Hy_z_avg = avg_anisotropic_H_component(
            H_pad, component=1, location=2, aniso_widths=aniso_widths
        )  # calc Hy at location of Hz
        Hz_x_avg = avg_anisotropic_H_component(
            H_pad, component=2, location=0, aniso_widths=aniso_widths
        )  # calc Hz at location of Hx
        Hz_y_avg = avg_anisotropic_H_component(
            H_pad, component=2, location=1, aniso_widths=aniso_widths
        )  # calc Hz at location of Hy
        curlEx_y_avg = avg_anisotropic_H_component(
            curl_pad, component=0, location=1, aniso_widths=aniso_widths
        )  # calc curl(E)x at location of Hy
        curlEx_z_avg = avg_anisotropic_H_component(
            curl_pad, component=0, location=2, aniso_widths=aniso_widths
        )  # calc curl(E)x at location of Hz
        curlEy_x_avg = avg_anisotropic_H_component(
            curl_pad, component=1, location=0, aniso_widths=aniso_widths
        )  # calc curl(E)y at location of Hx
        curlEy_z_avg = avg_anisotropic_H_component(
            curl_pad, component=1, location=2, aniso_widths=aniso_widths
        )  # calc curl(E)y at location of Hz
        curlEz_x_avg = avg_anisotropic_H_component(
            curl_pad, component=2, location=0, aniso_widths=aniso_widths
        )  # calc curl(E)z at location of Hx
        curlEz_y_avg = avg_anisotropic_H_component(
            curl_pad, component=2, location=1, aniso_widths=aniso_widths
        )  # calc curl(E)z at location of Hy

        # K = curl(E)
        # Hx <= (Axx * Hx + Axy * x_avg(Hy) + Axz * x_avg(Hz)) -
        #       (Bxx * Kx + Bxy * x_avg(Ky) + Bxz * x_avg(Kz))
        Hx = (A[0, 0] * arrays.fields.H[0] + A[0, 1] * Hy_x_avg + A[0, 2] * Hz_x_avg) - (
            B[0, 0] * curl[0] + B[0, 1] * curlEy_x_avg + B[0, 2] * curlEz_x_avg
        )
        # Hy <= (Ayx * y_avg(Hx) + Ayy * Hy + Ayz * y_avg(Hz)) -
        #       (Byx * y_avg(Kx) + Byy * Ky + Byz * y_avg(Kz))
        Hy = (A[1, 0] * Hx_y_avg + A[1, 1] * arrays.fields.H[1] + A[1, 2] * Hz_y_avg) - (
            B[1, 0] * curlEx_y_avg + B[1, 1] * curl[1] + B[1, 2] * curlEz_y_avg
        )
        # Hz <= (Azx * z_avg(Hx) + Azy * z_avg(Hy) + Azz * Hz) -
        #       (Bzx * z_avg(Kx) + Bzy * z_avg(Ky) + Bzz * Kz)
        Hz = (A[2, 0] * Hx_z_avg + A[2, 1] * Hy_z_avg + A[2, 2] * arrays.fields.H[2]) - (
            B[2, 0] * curlEx_z_avg + B[2, 1] * curlEy_z_avg + B[2, 2] * curl[2]
        )

        H = jnp.stack((Hx, Hy, Hz), axis=0)

    for source in objects.sources:
        if _source_uses_default_always_on_switch(source):
            H = source.update_H(
                H=H,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=time_step + 0.5,
                inverse=False,
            )
            continue

        def _update():
            adj_time_step = source.adjust_time_step_by_on_off(time_step)
            return source.update_H(
                H=H,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=adj_time_step + 0.5,
                inverse=False,
            )

        H = jax.lax.cond(
            source.is_on_at_time_step(time_step),
            _update,
            lambda: H,
        )

    H = apply_boundary_post_H_update(H, objects)
    arrays = arrays.aset("fields->H", H)
    return arrays


def update_H_reverse(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
) -> ArrayContainer:
    """Reverse time step update for the magnetic field used in automatic differentiation.

    Implements the inverse update step that transforms the electromagnetic field state
    from time step t+1 to time step t, leveraging the time-reversibility property of
    Maxwell's equations.

    Args:
        time_step (jax.Array): Current simulation time step
        arrays (ArrayContainer): Container with E, H fields and material properties
        objects (ObjectContainer): Container with sources and other simulation objects
        config (SimulationConfig): Simulation configuration parameters

    Returns:
        ArrayContainer: Updated ArrayContainer with reversed H field values
    """
    H = arrays.fields.H
    for source in objects.sources:
        if _source_uses_default_always_on_switch(source):
            H = source.update_H(
                H,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=time_step + 0.5,
                inverse=True,
            )
            continue

        def _update():
            adj_time_step = source.adjust_time_step_by_on_off(time_step)
            return source.update_H(
                H,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=adj_time_step + 0.5,
                inverse=True,
            )

        H = jax.lax.cond(
            source.is_on_at_time_step(time_step),
            _update,
            lambda: H,
        )

    inv_mu = arrays.inv_permeabilities
    sigma_H = arrays.magnetic_conductivity
    c = config.courant_number
    E_pad = pad_fields_for_boundaries(arrays.fields.E, objects, config)
    curl, _ = curl_E(
        config,
        E_pad,
        arrays.fields.psi_H,
        objects,
        False,
    )

    # Check if we have full anisotropic tensors (shape[0] == 9)
    # inv_mu can be a scalar (float) for non-magnetic materials
    inv_mu_shape = getattr(inv_mu, "shape", (0,))
    inv_mu_is_full_tensor = len(inv_mu_shape) > 0 and inv_mu_shape[0] == 9
    sigma_H_is_full_tensor = sigma_H is not None and sigma_H.shape[0] == 9

    if not inv_mu_is_full_tensor and not sigma_H_is_full_tensor:
        # Isotropic and diagonal anisotropic case
        factor = 1
        if sigma_H is not None:
            # lossy materials get gain when simulating backwards
            H = H * (1 + c * sigma_H / eta0 * inv_mu / 2)
            factor = 1 - c * sigma_H / eta0 * inv_mu / 2
        H = (H + c * curl * inv_mu) / factor

    else:
        # Full anisotropic case: expand inv_mu and sigma_H to (3, 3, Nx, Ny, Nz)
        inv_mu = expand_to_3x3(inv_mu)
        sigma_H = expand_to_3x3(sigma_H)

        # Compute A and B matrices for reverse update
        # H^(n-1/2) = A @ H^(n+1/2) + B @ curl(E^(n))
        A, B = compute_anisotropic_update_matrices_reverse(inv_mu, sigma_H, c, 1 / eta0)

        # We need to pad the fields and curl to account for ghost cells when computing the averages
        H_pad = pad_fields_for_boundaries(H, objects, config)
        curl_pad = pad_fields_for_boundaries(curl, objects, config)

        # Spacing weights for the off-diagonal average (None on a uniform grid).
        aniso_widths = get_anisotropic_averaging_widths(config)
        # Compute the averages of the fields and curl
        Hx_y_avg = avg_anisotropic_H_component(
            H_pad, component=0, location=1, aniso_widths=aniso_widths
        )  # calc Hx at location of Hy
        Hx_z_avg = avg_anisotropic_H_component(
            H_pad, component=0, location=2, aniso_widths=aniso_widths
        )  # calc Hx at location of Hz
        Hy_x_avg = avg_anisotropic_H_component(
            H_pad, component=1, location=0, aniso_widths=aniso_widths
        )  # calc Hy at location of Hx
        Hy_z_avg = avg_anisotropic_H_component(
            H_pad, component=1, location=2, aniso_widths=aniso_widths
        )  # calc Hy at location of Hz
        Hz_x_avg = avg_anisotropic_H_component(
            H_pad, component=2, location=0, aniso_widths=aniso_widths
        )  # calc Hz at location of Hx
        Hz_y_avg = avg_anisotropic_H_component(
            H_pad, component=2, location=1, aniso_widths=aniso_widths
        )  # calc Hz at location of Hy
        curlEx_y_avg = avg_anisotropic_H_component(
            curl_pad, component=0, location=1, aniso_widths=aniso_widths
        )  # calc curl(E)x at location of Hy
        curlEx_z_avg = avg_anisotropic_H_component(
            curl_pad, component=0, location=2, aniso_widths=aniso_widths
        )  # calc curl(E)x at location of Hz
        curlEy_x_avg = avg_anisotropic_H_component(
            curl_pad, component=1, location=0, aniso_widths=aniso_widths
        )  # calc curl(E)y at location of Hx
        curlEy_z_avg = avg_anisotropic_H_component(
            curl_pad, component=1, location=2, aniso_widths=aniso_widths
        )  # calc curl(E)y at location of Hz
        curlEz_x_avg = avg_anisotropic_H_component(
            curl_pad, component=2, location=0, aniso_widths=aniso_widths
        )  # calc curl(E)z at location of Hx
        curlEz_y_avg = avg_anisotropic_H_component(
            curl_pad, component=2, location=1, aniso_widths=aniso_widths
        )  # calc curl(E)z at location of Hy

        # K = curl(E)
        # Hx <= (Axx * Hx + Axy * x_avg(Hy) + Axz * x_avg(Hz)) +
        #       (Bxx * Kx + Bxy * x_avg(Ky) + Bxz * x_avg(Kz))
        Hx = (A[0, 0] * H[0] + A[0, 1] * Hy_x_avg + A[0, 2] * Hz_x_avg) + (
            B[0, 0] * curl[0] + B[0, 1] * curlEy_x_avg + B[0, 2] * curlEz_x_avg
        )
        # Hy <= (Ayx * y_avg(Hx) + Ayy * Hy + Ayz * y_avg(Hz)) +
        #       (Byx * y_avg(Kx) + Byy * Ky + Byz * y_avg(Kz))
        Hy = (A[1, 0] * Hx_y_avg + A[1, 1] * H[1] + A[1, 2] * Hz_y_avg) + (
            B[1, 0] * curlEx_y_avg + B[1, 1] * curl[1] + B[1, 2] * curlEz_y_avg
        )
        # Hz <= (Azx * z_avg(Hx) + Azy * z_avg(Hy) + Azz * Hz) +
        #       (Bzx * z_avg(Kx) + Bzy * z_avg(Ky) + Bzz * Kz)
        Hz = (A[2, 0] * Hx_z_avg + A[2, 1] * Hy_z_avg + A[2, 2] * H[2]) + (
            B[2, 0] * curlEx_z_avg + B[2, 1] * curlEy_z_avg + B[2, 2] * curl[2]
        )

        H = jnp.stack((Hx, Hy, Hz), axis=0)

    H = apply_boundary_post_H_update(H, objects)
    arrays = arrays.aset("fields->H", H)

    return arrays


def _check_updated_state_layout(detector: Detector, old: DetectorState, new: DetectorState) -> None:
    """Checks that a detector update kept the layout of its initialized state.

    A mismatch usually means the update sliced `self.grid_slice` on fields that were already
    restricted to the detector region (double slicing).

    Args:
        detector (Detector): Detector whose state was updated.
        old (DetectorState): Detector state before the update.
        new (DetectorState): Detector state returned by the update.

    Raises:
        Exception: If the updated state has different keys, shapes or dtypes than before.
    """
    problems = [f"state keys changed from {sorted(old)} to {sorted(new)}"] if set(old) != set(new) else []
    problems += [
        f"'{k}' expected shape {jnp.shape(old[k])} / dtype {jnp.result_type(old[k])}, "
        f"got {jnp.shape(new[k])} / {jnp.result_type(new[k])}"
        for k in old
        if k in new and (jnp.shape(old[k]) != jnp.shape(new[k]) or jnp.result_type(old[k]) != jnp.result_type(new[k]))
    ]
    if problems:
        raise Exception(
            f"Detector '{detector.name}': update() returned a state that does not match its initialized "
            f"layout: {'; '.join(problems)}. Note that fields and materials are passed to Detector.update() "
            "already restricted to the detector's grid_slice, so slicing self.grid_slice inside update() "
            "(double slicing) is the most common cause."
        )


def update_detector_states(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    H_prev: jax.Array,
    inverse: bool,
) -> ArrayContainer:
    """Updates detector states based on current field values.

    Handles field interpolation for accurate detector measurements. Interpolation
    is enabled by default, but can be disabled per detector for performance during
    optimization. Interpolation is needed due to the staggered nature of E and H
    fields on the Yee grid.

    Args:
        time_step (jax.Array): Current simulation time step
        arrays (ArrayContainer): Container with E, H fields and material properties
        objects (ObjectContainer): Container with detectors and other simulation objects
        H_prev (jax.Array): Previous H field values for interpolation
        inverse (bool): Whether this is a forward or reverse update

    Returns:
        ArrayContainer: Updated ArrayContainer with new detector states

    Notes:
        Each detector receives fields and materials already restricted to its `grid_slice`. Since
        the interpolation stencil only reaches the neighboring cell, a strictly interior detector
        is interpolated over its region plus a one-cell halo; the full-domain interpolation is
        only built as a shared fallback for detectors touching a domain edge, where the boundary
        padding matters.
    """
    state = arrays.detector_states
    to_update = objects.backward_detectors if inverse else objects.forward_detectors
    if not to_update:
        return arrays

    grid_shape = objects.volume.grid_shape

    def is_interior(detector: Detector) -> bool:
        # The co-location stencil reads domain indices [s-1 .. e]; interior iff that stays in-bounds.
        return all(s >= 1 and e <= grid_shape[a] - 1 for a, (s, e) in enumerate(detector.grid_slice_tuple))

    # The full-domain interpolation is only needed for exact detectors whose stencil reaches a domain edge.
    full = None
    if any(d.exact_interpolation and not is_interior(d) for d in to_update):
        full = interpolate_fields(
            E_pad=pad_fields_for_boundaries(arrays.fields.E, objects, config),
            H_pad=pad_fields_for_boundaries((H_prev + arrays.fields.H) / 2, objects, config),
            config=config,
        )

    def helper_fn(E: jax.Array, H: jax.Array, H_prev: jax.Array, detector: Detector) -> DetectorState:
        gs = detector.grid_slice
        if not detector.exact_interpolation:
            E_reg, H_reg = E[:, *gs], H[:, *gs]
        elif is_interior(detector):
            block = (slice(None), *(slice(s - 1, e + 1) for (s, e) in detector.grid_slice_tuple))
            H_avg = (H_prev[block] + H[block]) / 2
            E_reg, H_reg = interpolate_fields(E[block], H_avg, config=config, region_slice=detector.grid_slice_tuple)
        else:
            assert full is not None  # built above whenever an edge-touching exact detector exists
            E_reg, H_reg = full[0][:, *gs], full[1][:, *gs]
        # inv_permeabilities is a plain scalar when all materials are non-magnetic.
        inv_mu = arrays.inv_permeabilities
        try:
            new_state = detector.update(
                time_step=time_step,
                E=E_reg,
                H=H_reg,
                state=state[detector.name],
                inv_permittivity=arrays.inv_permittivities[:, *gs],
                inv_permeability=inv_mu[:, *gs] if isinstance(inv_mu, jax.Array) and inv_mu.ndim > 0 else inv_mu,
            )
        except Exception as e:
            raise Exception(
                f"Detector '{detector.name}': update() raised while recording (see exception above). Fields "
                "and materials are passed to Detector.update() already restricted to the detector's "
                "grid_slice, so slicing self.grid_slice inside update() (double slicing) is a common cause "
                "of shape errors here."
            ) from e
        _check_updated_state_layout(detector, state[detector.name], new_state)
        return new_state

    for d in to_update:
        # E already lives at the detector's integer time step; H lives at half steps, so exact
        # detectors time-center H as (H_prev + H) / 2 on their region inside the branch.
        state[d.name] = jax.lax.cond(
            d._is_on_at_time_step_arr[time_step],
            helper_fn,
            lambda e, h, h_prev, detector: state[detector.name],
            arrays.fields.E,
            arrays.fields.H,
            H_prev,
            d,
        )
    arrays = arrays.aset("detector_states", state)
    return arrays


def collect_interfaces(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
) -> ArrayContainer:
    """Collects field values at PML interfaces for gradient computation.

    Part of the memory-efficient automatic differentiation implementation.
    Saves field values at boundaries between PML and inner simulation volume
    since PML updates are not time-reversible.

    Args:
        time_step (jax.Array): Current simulation time step
        arrays (ArrayContainer): Container with fields and material properties
        objects (ObjectContainer): Container with PML and other simulation objects
        config (SimulationConfig): Simulation configuration with gradient settings
        key (jax.Array): Random key for compression

    Returns:
        ArrayContainer: Updated ArrayContainer with recorded interface values
    """
    if config.gradient_config is None or config.gradient_config.recorder is None:
        raise Exception("Need recorder to record boundaries")
    if arrays.recording_state is None:
        raise Exception("Need recording state to record boundaries")
    values = collect_boundary_interfaces(
        arrays=arrays,
        pml_objects=objects.pml_objects,
    )
    recording_state = config.gradient_config.recorder.compress(
        values=values,
        state=arrays.recording_state,
        time_step=time_step,
        key=key,
    )
    arrays = arrays.aset("recording_state", recording_state)
    return arrays


def add_interfaces(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
) -> ArrayContainer:
    """Adds previously collected interface values back to the fields.

    Part of the memory-efficient automatic differentiation implementation.
    Restores saved field values at PML boundaries during reverse propagation
    since PML updates are not time-reversible.

    Args:
        time_step (jax.Array): Current simulation time step
        arrays (ArrayContainer): Container with fields and material properties
        objects (ObjectContainer): Container with PML and other simulation objects
        config (SimulationConfig): Simulation configuration with gradient settings
        key (jax.Array): Random key for decompression

    Returns:
        ArrayContainer: Updated ArrayContainer with restored interface values
    """
    if config.gradient_config is None or config.gradient_config.recorder is None:
        raise Exception("Need recorder to record boundaries")
    if arrays.recording_state is None:
        raise Exception("Need recording state to record boundaries")

    values, state = config.gradient_config.recorder.decompress(
        state=arrays.recording_state,
        time_step=time_step,
        key=key,
    )
    arrays = arrays.aset("recording_state", state)

    container = add_boundary_interfaces(
        arrays=arrays,
        values=values,
        pml_objects=objects.pml_objects,
    )

    return container
