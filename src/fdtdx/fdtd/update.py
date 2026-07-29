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


def pad_offdiag_coefficients(cij: jax.Array, periodic_axes: tuple[bool, bool, bool]) -> jax.Array:
    """Coefficient halo for the symmetrized off-diagonal dispersive coupling.

    Matches the boundary semantics of the field halo: wrap on periodic axes so
    the pair weight across the boundary uses the true neighbor coefficient,
    edge replication elsewhere (the field halo is zero there, so the halo value
    is inert). Coefficients carry no Bloch phase. The leading axis is the pole
    axis; the three trailing axes are spatial.

    Args:
        cij: Coefficient array of shape ``(num_poles, Nx, Ny, Nz)``.
        periodic_axes: Which spatial axes use periodic (wrap) boundaries.

    Returns:
        Padded array of shape ``(num_poles, Nx+2, Ny+2, Nz+2)``.
    """
    padded = cij
    for axis, periodic in enumerate(periodic_axes):
        pad_width = [(0, 0)] * cij.ndim
        pad_width[axis + 1] = (1, 1)
        padded = jnp.pad(padded, pad_width, mode="wrap" if periodic else "edge")
    return padded


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

        # Dispersive (ADE) correction, marched in the delta (zeta = z - 1) basis
        #   dx1^n     = y2^n - a1 x1^n + b1 E^n      (= x1^{n+1} - x1^n)
        #   y2^{n+1}  = y2^n - a0 x1^n + b0 E^n
        #   x1^{n+1}  = x1^n + dx1^n
        #   p^{n+1}   = x1^{n+1} + c4 E^{n+1}
        # with the state y2 = x1 + x2 relative to the observer form. See
        # fdtdx.dispersion for the derivation and for why this basis — not
        # (c1, c2, beta1, c4, beta0) — is what keeps float32 usable: a0 is
        # O((omega_0 dt)^2) and is *stored*, where 1 - c1 - c2 would be pure
        # round-off. Non-dispersive cells have all coefficients zero, so
        # b1 = b0 = 0 pins x1/y2 at zero and the polarization term vanishes — a
        # no-op outside dispersive regions. Only active when arrays are allocated.
        if arrays.fields.dispersive_x1 is not None:
            x1 = arrays.fields.dispersive_x1
            y2 = arrays.fields.dispersive_y2
            disp_a1 = arrays.dispersive_a1
            disp_a0 = arrays.dispersive_a0
            disp_b1 = arrays.dispersive_b1
            disp_c4 = arrays.dispersive_c4
            disp_b0 = arrays.dispersive_b0
            assert y2 is not None and disp_a1 is not None and disp_a0 is not None and disp_b1 is not None
            # Coefficients are (num_poles, 1|3, Nx, Ny, Nz) — the component axis is
            # 1 (isotropic dispersion, broadcast) or 3 (per-axis anisotropic
            # dispersion); arrays.fields.E is (3, Nx, Ny, Nz). Right-aligned
            # broadcasting produces (num_poles, 3, Nx, Ny, Nz) either way without
            # an explicit newaxis — skip the reshape so the HLO stays flat.
            E_n = arrays.fields.E
            # b0 == b1 exactly when no pole is implicit (c4 = c5 = 0 => beta0 = 0),
            # which is precisely when the b0 array is not allocated.
            b0_eff = disp_b1 if disp_b0 is None else disp_b0
            # dx1 is fully explicit; the implicit c4*E^{n+1} of p^{n+1} is folded
            # into the divisor below.
            dx1 = y2 - disp_a1 * x1 + disp_b1 * E_n
            y2_new = y2 - disp_a0 * x1 + b0_eff * E_n
            x1_new = x1 + dx1
            # p^n - x1^{n+1} = c4 E^n - dx1. Computed from dx1 rather than by
            # differencing p^n against x1^{n+1}: those are two nearly equal large
            # numbers for a metal (|x1| ~ 400 |E|), and the subtraction would throw
            # away most of the increment's significant digits. The c4 E^n term is
            # what makes the E^n coefficient (1 - kappa + inv_eps*sum(c4)).
            delta_p = -dx1 if disp_c4 is None else disp_c4 * E_n - dx1
            E = E + inv_eps * jnp.sum(delta_p, axis=0)
            if disp_c4 is not None:
                # The polarization couples to E^{n+1} through c4. Fold the per-cell
                # inv_eps*sum(c4) into the implicit divide, alongside the
                # conductivity loss factor.
                divisor = 1 + inv_eps * jnp.sum(disp_c4, axis=0)
                if sigma_E is not None:
                    divisor = divisor + c * sigma_E * eta0 * inv_eps / 2
                E = E / divisor
            elif sigma_E is not None:
                # lossy update formula. Noop for conductivity = 0; see Schneider 3.12
                E = E / (1 + c * sigma_E * eta0 * inv_eps / 2)
            arrays = arrays.aset("fields->dispersive_x1", x1_new)
            arrays = arrays.aset("fields->dispersive_y2", y2_new)
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
        # Poles with a non-zero c4 (any dE/dt coupling, or any pole under the
        # "bilinear" integrator) are rejected at initialization for this branch, so
        # here c4 = beta0 = 0, hence b1 == b0 == c3 and p^n == x1^n.
        if arrays.fields.dispersive_x1 is not None:
            x1 = arrays.fields.dispersive_x1
            y2 = arrays.fields.dispersive_y2
            disp_a1 = arrays.dispersive_a1
            disp_a0 = arrays.dispersive_a0
            disp_b1 = arrays.dispersive_b1
            assert y2 is not None and disp_a1 is not None and disp_a0 is not None and disp_b1 is not None
            assert arrays.dispersive_c4 is None and arrays.dispersive_b0 is None
            if disp_b1.shape[1] == 9:
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

                # Initialization rejects oriented poles on non-uniform grids, so
                # the uniform 4-point stencil is always valid here.
                periodic_axes = get_wrap_padding_axes(objects)
                rows = []
                for i in range(3):
                    row = disp_b1[:, 3 * i + i] * arrays.fields.E[i]
                    for j in range(3):
                        if j == i:
                            continue
                        cij = disp_b1[:, 3 * i + j]
                        cij_pad = pad_offdiag_coefficients(cij, periodic_axes)
                        mask_pad = (cij_pad != 0.0).astype(cij.dtype)
                        row = row + 0.5 * (
                            cij * _avg_offdiag(mask_pad * E_pad[j], component=j, location=i)
                            + (cij != 0.0) * _avg_offdiag(cij_pad * E_pad[j], component=j, location=i)
                        )
                    rows.append(row)
                coupling = jnp.stack(rows, axis=1)
            else:
                coupling = disp_b1 * arrays.fields.E
            # Delta basis, as in the diagonal branch. b0 == b1 here, so the same
            # `coupling` term drives both states.
            dx1 = y2 - disp_a1 * x1 + coupling
            # c4 = 0 on this branch, so p^n == x1^n and p^{n+1} - p^n == dx1.
            delta = -jnp.sum(dx1, axis=0)
            curl = curl + delta / c
            arrays = arrays.aset("fields->dispersive_x1", x1 + dx1)
            arrays = arrays.aset("fields->dispersive_y2", y2 - disp_a0 * x1 + coupling)

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

    Raises:
        NotImplementedError: If the simulation contains dispersive materials. The ADE
            state has no supported time reversal, so neither this update nor the
            ``full_backward`` / ``backward`` API can reconstruct it.
    """
    if arrays.fields.dispersive_x1 is not None:
        raise NotImplementedError(
            "Dispersive time-reversible gradient computation under active development. "
            "Use GradientConfig(method='checkpointed') instead."
        )

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
        factor = 1
        if sigma_E is not None:
            E = E * (1 + c * sigma_E * eta0 * inv_eps / 2)
            factor = 1 - c * sigma_E * eta0 * inv_eps / 2

        E = (E - c * curl * inv_eps) / factor

    else:
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
