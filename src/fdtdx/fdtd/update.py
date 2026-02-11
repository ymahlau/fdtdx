import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.constants import eta0
from fdtdx.core.misc import expand_to_3x3, pad_fields
from fdtdx.core.physics.curl import curl_E, curl_H, interpolate_fields
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.misc import (
    add_boundary_interfaces,
    avg_anisotropic_E_component,
    avg_anisotropic_H_component,
    collect_boundary_interfaces,
    compute_anisotropic_update_matrices,
    compute_anisotropic_update_matrices_reverse,
)
from fdtdx.objects.boundaries.periodic import PeriodicBoundary
from fdtdx.objects.detectors.detector import Detector


def get_periodic_axes(objects: ObjectContainer) -> tuple[bool, bool, bool]:
    """Determines which axes have periodic boundary conditions.

    Args:
        objects (ObjectContainer): Container with simulation objects including boundaries

    Returns:
        tuple[bool, bool, bool]: Tuple indicating which axes (x,y,z) are periodic
    """
    periodic_axes = [False, False, False]
    for boundary in objects.boundary_objects:
        if isinstance(boundary, PeriodicBoundary):
            periodic_axes[boundary.axis] = True
    return tuple(periodic_axes)  # type: ignore


def update_E(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    simulate_boundaries: bool,
) -> ArrayContainer:
    """Updates the electric field (E) according to Maxwell's equations using the FDTD method.

    Implements the discretized form of dE/dt = (1/ε)∇×H on the Yee grid. Updates include:
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
    periodic_axes = get_periodic_axes(objects)
    curl, psi_E = curl_H(
        config,
        arrays.H,
        arrays.psi_E,
        arrays.alpha,
        arrays.kappa,
        arrays.sigma,
        simulate_boundaries,
        periodic_axes,
    )
    arrays = arrays.aset("psi_E", psi_E)

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
        E = factor * arrays.E + c * curl * inv_eps

        if sigma_E is not None:
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

        # We need to pad the fields and curl to account for ghost cells when computing the averages
        E_pad = pad_fields(arrays.E, periodic_axes)
        curl_pad = pad_fields(curl, periodic_axes)

        # Compute the averages of the fields and curl
        Ex_y_avg = avg_anisotropic_E_component(E_pad, component=0, location=1)  # calc Ex at location of Ey
        Ex_z_avg = avg_anisotropic_E_component(E_pad, component=0, location=2)  # calc Ex at location of Ez
        Ey_x_avg = avg_anisotropic_E_component(E_pad, component=1, location=0)  # calc Ey at location of Ex
        Ey_z_avg = avg_anisotropic_E_component(E_pad, component=1, location=2)  # calc Ey at location of Ez
        Ez_x_avg = avg_anisotropic_E_component(E_pad, component=2, location=0)  # calc Ez at location of Ex
        Ez_y_avg = avg_anisotropic_E_component(E_pad, component=2, location=1)  # calc Ez at location of Ey
        curlHx_y_avg = avg_anisotropic_E_component(curl_pad, component=0, location=1)  # calc curl(H)x at location of Ey
        curlHx_z_avg = avg_anisotropic_E_component(curl_pad, component=0, location=2)  # calc curl(H)x at location of Ez
        curlHy_x_avg = avg_anisotropic_E_component(curl_pad, component=1, location=0)  # calc curl(H)y at location of Ex
        curlHy_z_avg = avg_anisotropic_E_component(curl_pad, component=1, location=2)  # calc curl(H)y at location of Ez
        curlHz_x_avg = avg_anisotropic_E_component(curl_pad, component=2, location=0)  # calc curl(H)z at location of Ex
        curlHz_y_avg = avg_anisotropic_E_component(curl_pad, component=2, location=1)  # calc curl(H)z at location of Ey

        # K = curl(H)
        # Ex <= (Axx * Ex + Axy * x_avg(Ey) + Axz * x_avg(Ez)) +
        #       (Bxx * Kx + Bxy * x_avg(Ky) + Bxz * x_avg(Kz))
        Ex = (A[0, 0] * arrays.E[0] + A[0, 1] * Ey_x_avg + A[0, 2] * Ez_x_avg) + (
            B[0, 0] * curl[0] + B[0, 1] * curlHy_x_avg + B[0, 2] * curlHz_x_avg
        )
        # Ey <= (Ayx * y_avg(Ex) + Ayy * Ey + Ayz * y_avg(Ez)) +
        #       (Byx * y_avg(Kx) + Byy * Ky + Byz * y_avg(Kz))
        Ey = (A[1, 0] * Ex_y_avg + A[1, 1] * arrays.E[1] + A[1, 2] * Ez_y_avg) + (
            B[1, 0] * curlHx_y_avg + B[1, 1] * curl[1] + B[1, 2] * curlHz_y_avg
        )
        # Ez <= (Azx * z_avg(Ex) + Azy * z_avg(Ey) + Azz * Ez) +
        #       (Bzx * z_avg(Kx) + Bzy * z_avg(Ky) + Bzz * Kz)
        Ez = (A[2, 0] * Ex_z_avg + A[2, 1] * Ey_z_avg + A[2, 2] * arrays.E[2]) + (
            B[2, 0] * curlHx_z_avg + B[2, 1] * curlHy_z_avg + B[2, 2] * curl[2]
        )

        E = jnp.stack((Ex, Ey, Ez), axis=0)

    for source in objects.sources:

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

    arrays = arrays.at["E"].set(E)
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
    E = arrays.E
    for source in objects.sources:

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
    periodic_axes = get_periodic_axes(objects)
    curl, _ = curl_H(
        config,
        arrays.H,
        arrays.psi_E,
        arrays.alpha,
        arrays.kappa,
        arrays.sigma,
        False,
        periodic_axes,
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
        E = E / factor - c * curl * inv_eps

    else:
        # Full anisotropic case: expand inv_eps and sigma_E to (3, 3, Nx, Ny, Nz)
        inv_eps = expand_to_3x3(inv_eps)
        sigma_E = expand_to_3x3(sigma_E)

        # Compute A and B matrices for reverse update
        # E^(n) = A @ E^(n+1) - B @ curl(H^(n+1/2))
        A, B = compute_anisotropic_update_matrices_reverse(inv_eps, sigma_E, c, eta0)

        # We need to pad the fields and curl to account for ghost cells when computing the averages
        E_pad = pad_fields(E, periodic_axes)
        curl_pad = pad_fields(curl, periodic_axes)

        # Compute the averages of the fields and curl
        Ex_y_avg = avg_anisotropic_E_component(E_pad, component=0, location=1)  # calc Ex at location of Ey
        Ex_z_avg = avg_anisotropic_E_component(E_pad, component=0, location=2)  # calc Ex at location of Ez
        Ey_x_avg = avg_anisotropic_E_component(E_pad, component=1, location=0)  # calc Ey at location of Ex
        Ey_z_avg = avg_anisotropic_E_component(E_pad, component=1, location=2)  # calc Ey at location of Ez
        Ez_x_avg = avg_anisotropic_E_component(E_pad, component=2, location=0)  # calc Ez at location of Ex
        Ez_y_avg = avg_anisotropic_E_component(E_pad, component=2, location=1)  # calc Ez at location of Ey
        curlHx_y_avg = avg_anisotropic_E_component(curl_pad, component=0, location=1)  # calc curl(H)x at location of Ey
        curlHx_z_avg = avg_anisotropic_E_component(curl_pad, component=0, location=2)  # calc curl(H)x at location of Ez
        curlHy_x_avg = avg_anisotropic_E_component(curl_pad, component=1, location=0)  # calc curl(H)y at location of Ex
        curlHy_z_avg = avg_anisotropic_E_component(curl_pad, component=1, location=2)  # calc curl(H)y at location of Ez
        curlHz_x_avg = avg_anisotropic_E_component(curl_pad, component=2, location=0)  # calc curl(H)z at location of Ex
        curlHz_y_avg = avg_anisotropic_E_component(curl_pad, component=2, location=1)  # calc curl(H)z at location of Ey

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

    arrays = arrays.at["E"].set(E)

    return arrays


def update_H(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    simulate_boundaries: bool,
) -> ArrayContainer:
    """Updates the magnetic field (H) according to Maxwell's equations using the FDTD method.

    Implements the discretized form of dH/dt = -(1/μ)∇×E on the Yee grid. Updates include:
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
    periodic_axes = get_periodic_axes(objects)
    curl, psi_H = curl_E(
        config,
        arrays.E,
        arrays.psi_H,
        arrays.alpha,
        arrays.kappa,
        arrays.sigma,
        simulate_boundaries,
        periodic_axes,
    )
    arrays = arrays.aset("psi_H", psi_H)

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
        H = factor * arrays.H - c * curl * inv_mu

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
        H_pad = pad_fields(arrays.H, periodic_axes)
        curl_pad = pad_fields(curl, periodic_axes)

        # Compute the averages of the fields and curl
        Hx_y_avg = avg_anisotropic_H_component(H_pad, component=0, location=1)  # calc Hx at location of Hy
        Hx_z_avg = avg_anisotropic_H_component(H_pad, component=0, location=2)  # calc Hx at location of Hz
        Hy_x_avg = avg_anisotropic_H_component(H_pad, component=1, location=0)  # calc Hy at location of Hx
        Hy_z_avg = avg_anisotropic_H_component(H_pad, component=1, location=2)  # calc Hy at location of Hz
        Hz_x_avg = avg_anisotropic_H_component(H_pad, component=2, location=0)  # calc Hz at location of Hx
        Hz_y_avg = avg_anisotropic_H_component(H_pad, component=2, location=1)  # calc Hz at location of Hy
        curlEx_y_avg = avg_anisotropic_H_component(curl_pad, component=0, location=1)  # calc curl(E)x at location of Hy
        curlEx_z_avg = avg_anisotropic_H_component(curl_pad, component=0, location=2)  # calc curl(E)x at location of Hz
        curlEy_x_avg = avg_anisotropic_H_component(curl_pad, component=1, location=0)  # calc curl(E)y at location of Hx
        curlEy_z_avg = avg_anisotropic_H_component(curl_pad, component=1, location=2)  # calc curl(E)y at location of Hz
        curlEz_x_avg = avg_anisotropic_H_component(curl_pad, component=2, location=0)  # calc curl(E)z at location of Hx
        curlEz_y_avg = avg_anisotropic_H_component(curl_pad, component=2, location=1)  # calc curl(E)z at location of Hy

        # K = curl(E)
        # Hx <= (Axx * Hx + Axy * x_avg(Hy) + Axz * x_avg(Hz)) -
        #       (Bxx * Kx + Bxy * x_avg(Ky) + Bxz * x_avg(Kz))
        Hx = (A[0, 0] * arrays.H[0] + A[0, 1] * Hy_x_avg + A[0, 2] * Hz_x_avg) - (
            B[0, 0] * curl[0] + B[0, 1] * curlEy_x_avg + B[0, 2] * curlEz_x_avg
        )
        # Hy <= (Ayx * y_avg(Hx) + Ayy * Hy + Ayz * y_avg(Hz)) -
        #       (Byx * y_avg(Kx) + Byy * Ky + Byz * y_avg(Kz))
        Hy = (A[1, 0] * Hx_y_avg + A[1, 1] * arrays.H[1] + A[1, 2] * Hz_y_avg) - (
            B[1, 0] * curlEx_y_avg + B[1, 1] * curl[1] + B[1, 2] * curlEz_y_avg
        )
        # Hz <= (Azx * z_avg(Hx) + Azy * z_avg(Hy) + Azz * Hz) -
        #       (Bzx * z_avg(Kx) + Bzy * z_avg(Ky) + Bzz * Kz)
        Hz = (A[2, 0] * Hx_z_avg + A[2, 1] * Hy_z_avg + A[2, 2] * arrays.H[2]) - (
            B[2, 0] * curlEx_z_avg + B[2, 1] * curlEy_z_avg + B[2, 2] * curl[2]
        )

        H = jnp.stack((Hx, Hy, Hz), axis=0)

    for source in objects.sources:

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

    arrays = arrays.at["H"].set(H)
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
    H = arrays.H
    for source in objects.sources:

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
    periodic_axes = get_periodic_axes(objects)
    curl, _ = curl_E(
        config,
        arrays.E,
        arrays.psi_H,
        arrays.alpha,
        arrays.kappa,
        arrays.sigma,
        False,
        periodic_axes,
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
        H = H / factor + c * curl * inv_mu

    else:
        # Full anisotropic case: expand inv_mu and sigma_H to (3, 3, Nx, Ny, Nz)
        inv_mu = expand_to_3x3(inv_mu)
        sigma_H = expand_to_3x3(sigma_H)

        # Compute A and B matrices for reverse update
        # H^(n-1/2) = A @ H^(n+1/2) + B @ curl(E^(n))
        A, B = compute_anisotropic_update_matrices_reverse(inv_mu, sigma_H, c, 1 / eta0)

        # We need to pad the fields and curl to account for ghost cells when computing the averages
        H_pad = pad_fields(H, periodic_axes)
        curl_pad = pad_fields(curl, periodic_axes)

        # Compute the averages of the fields and curl
        Hx_y_avg = avg_anisotropic_H_component(H_pad, component=0, location=1)  # calc Hx at location of Hy
        Hx_z_avg = avg_anisotropic_H_component(H_pad, component=0, location=2)  # calc Hx at location of Hz
        Hy_x_avg = avg_anisotropic_H_component(H_pad, component=1, location=0)  # calc Hy at location of Hx
        Hy_z_avg = avg_anisotropic_H_component(H_pad, component=1, location=2)  # calc Hy at location of Hz
        Hz_x_avg = avg_anisotropic_H_component(H_pad, component=2, location=0)  # calc Hz at location of Hx
        Hz_y_avg = avg_anisotropic_H_component(H_pad, component=2, location=1)  # calc Hz at location of Hy
        curlEx_y_avg = avg_anisotropic_H_component(curl_pad, component=0, location=1)  # calc curl(E)x at location of Hy
        curlEx_z_avg = avg_anisotropic_H_component(curl_pad, component=0, location=2)  # calc curl(E)x at location of Hz
        curlEy_x_avg = avg_anisotropic_H_component(curl_pad, component=1, location=0)  # calc curl(E)y at location of Hx
        curlEy_z_avg = avg_anisotropic_H_component(curl_pad, component=1, location=2)  # calc curl(E)y at location of Hz
        curlEz_x_avg = avg_anisotropic_H_component(curl_pad, component=2, location=0)  # calc curl(E)z at location of Hx
        curlEz_y_avg = avg_anisotropic_H_component(curl_pad, component=2, location=1)  # calc curl(E)z at location of Hy

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

    arrays = arrays.at["H"].set(H)

    return arrays


def update_detector_states(
    time_step: jax.Array,
    arrays: ArrayContainer,
    objects: ObjectContainer,
    H_prev: jax.Array,
    inverse: bool,
) -> ArrayContainer:
    """Updates detector states based on current field values.

    Handles field interpolation for accurate detector measurements. By default,
    interpolation is disabled for performance during optimization, but can be
    enabled for final evaluation. Interpolation is needed due to the staggered
    nature of E and H fields on the Yee grid.

    Args:
        time_step (jax.Array): Current simulation time step
        arrays (ArrayContainer): Container with E, H fields and material properties
        objects (ObjectContainer): Container with detectors and other simulation objects
        H_prev (jax.Array): Previous H field values for interpolation
        inverse (bool): Whether this is a forward or reverse update

    Returns:
        ArrayContainer: Updated ArrayContainer with new detector states
    """
    periodic_axes = get_periodic_axes(objects)
    interpolated_E, interpolated_H = interpolate_fields(
        E_field=arrays.E,
        H_field=(H_prev + arrays.H) / 2,
        periodic_axes=periodic_axes,
    )

    def helper_fn(E_input, H_input, detector: Detector):
        return detector.update(
            time_step=time_step,
            E=E_input,
            H=H_input,
            state=arrays.detector_states[detector.name],
            inv_permittivity=arrays.inv_permittivities,
            inv_permeability=arrays.inv_permeabilities,
        )

    state = arrays.detector_states
    to_update = objects.backward_detectors if inverse else objects.forward_detectors
    for d in to_update:
        state[d.name] = jax.lax.cond(
            d._is_on_at_time_step_arr[time_step],
            helper_fn,
            lambda e, h, _: state[d.name],
            interpolated_E if d.exact_interpolation else arrays.E,
            interpolated_H if d.exact_interpolation else arrays.H,
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
