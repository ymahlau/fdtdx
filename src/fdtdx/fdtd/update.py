import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.constants import eps0, eta0
from fdtdx.core.physics.curl import curl_E, curl_H, interpolate_fields
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.misc import add_boundary_interfaces, collect_boundary_interfaces
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

    # Handle periodic boundaries by copying values from opposite sides
    if simulate_boundaries:
        E = arrays.E
        for boundary in objects.boundary_objects:
            if isinstance(boundary, PeriodicBoundary):
                E = E.at[..., boundary.boundary_slice].set(E[..., boundary.opposite_slice])
        arrays = arrays.at["E"].set(E)

    inv_eps = arrays.inv_permittivities
    c = config.courant_number
    sigma_E = arrays.electric_conductivity

    # Get periodic axes for curl operation
    periodic_axes = get_periodic_axes(objects)
    H_pad = arrays.H
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

    DyHz = (H_pad[2] - jnp.roll(H_pad[2], 1, axis=1))[1:-1, 1:-1, 1:-1]
    DzHy = (H_pad[1] - jnp.roll(H_pad[1], 1, axis=2))[1:-1, 1:-1, 1:-1]
    DzHx = (H_pad[0] - jnp.roll(H_pad[0], 1, axis=2))[1:-1, 1:-1, 1:-1]
    DxHz = (H_pad[2] - jnp.roll(H_pad[2], 1, axis=0))[1:-1, 1:-1, 1:-1]
    DxHy = (H_pad[1] - jnp.roll(H_pad[1], 1, axis=0))[1:-1, 1:-1, 1:-1]
    DyHx = (H_pad[0] - jnp.roll(H_pad[0], 1, axis=1))[1:-1, 1:-1, 1:-1]

    # Auxiliary fields
    psi_Exy = arrays.psi_E[0, :, :, :]
    psi_Exz = arrays.psi_E[1, :, :, :]
    psi_Eyz = arrays.psi_E[2, :, :, :]
    psi_Eyx = arrays.psi_E[3, :, :, :]
    psi_Ezx = arrays.psi_E[4, :, :, :]
    psi_Ezy = arrays.psi_E[5, :, :, :]

    if simulate_boundaries:
        # Get E-field PML coefficients
        b_x = (
            jnp.expm1(
                -c
                * config.resolution
                / c0
                / eps0
                * (arrays.sigma[0, :, :, :] / arrays.kappa[0, :, :, :] + arrays.alpha[0, :, :, :])
            )
            + 1
        )
        b_y = (
            jnp.expm1(
                -c
                * config.resolution
                / c0
                / eps0
                * (arrays.sigma[1, :, :, :] / arrays.kappa[1, :, :, :] + arrays.alpha[1, :, :, :])
            )
            + 1
        )
        b_z = (
            jnp.expm1(
                -c
                * config.resolution
                / c0
                / eps0
                * (arrays.sigma[2, :, :, :] / arrays.kappa[2, :, :, :] + arrays.alpha[2, :, :, :])
            )
            + 1
        )

        a_x = (
            (b_x - 1.0)
            * arrays.sigma[0, :, :, :]
            / (arrays.sigma[0, :, :, :] + arrays.alpha[0, :, :, :] * arrays.kappa[0, :, :, :])
            / arrays.kappa[0, :, :, :]
        )
        a_y = (
            (b_y - 1.0)
            * arrays.sigma[1, :, :, :]
            / (arrays.sigma[1, :, :, :] + arrays.alpha[1, :, :, :] * arrays.kappa[1, :, :, :])
            / arrays.kappa[1, :, :, :]
        )
        a_z = (
            (b_z - 1.0)
            * arrays.sigma[2, :, :, :]
            / (arrays.sigma[2, :, :, :] + arrays.alpha[2, :, :, :] * arrays.kappa[2, :, :, :])
            / arrays.kappa[2, :, :, :]
        )

        a_x = jnp.nan_to_num(a_x, nan=0.0, posinf=0.0, neginf=0.0)
        a_y = jnp.nan_to_num(a_y, nan=0.0, posinf=0.0, neginf=0.0)
        a_z = jnp.nan_to_num(a_z, nan=0.0, posinf=0.0, neginf=0.0)

        # Update auxiliary fields
        psi_Exy = b_y * psi_Exy + a_y * DyHz
        psi_Exz = b_z * psi_Exz + a_z * DzHy
        psi_Eyz = b_z * psi_Eyz + a_z * DzHx
        psi_Eyx = b_x * psi_Eyx + a_x * DxHz
        psi_Ezx = b_x * psi_Ezx + a_x * DxHy
        psi_Ezy = b_y * psi_Ezy + a_y * DyHx

        psi_E = jnp.stack((psi_Exy, psi_Exz, psi_Eyz, psi_Eyx, psi_Ezx, psi_Ezy), axis=0)
        arrays = arrays.aset("psi_E", psi_E)

    # Curl equations
    curl_x = (1.0 / arrays.kappa[1, :, :, :] * DyHz + psi_Exy) - (1.0 / arrays.kappa[2, :, :, :] * DzHy + psi_Exz)
    curl_y = (1.0 / arrays.kappa[2, :, :, :] * DzHx + psi_Eyz) - (1.0 / arrays.kappa[0, :, :, :] * DxHz + psi_Eyx)
    curl_z = (1.0 / arrays.kappa[0, :, :, :] * DxHy + psi_Ezx) - (1.0 / arrays.kappa[1, :, :, :] * DyHx + psi_Ezy)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)

    factor = 1
    if sigma_E is not None:
        # update formula for lossy material. Simplifies to Noop for conductivity = 0
        # for details see Schneider, chapter 3.12
        factor = 1 - c * sigma_E * eta0 * inv_eps / 2

    # standard update formula using lossless material
    E = factor * arrays.E + c * curl * inv_eps

    if sigma_E is not None:
        # update formula for lossy material. Simplifies to Noop for conductivity = 0
        # for details see Schneider, chapter 3.12
        E = E / (1 + c * sigma_E * eta0 * inv_eps / 2)

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

    # Get periodic axes for curl operation
    periodic_axes = get_periodic_axes(objects)
    curl = curl_H(arrays.H, periodic_axes)
    inv_eps = arrays.inv_permittivities
    c = config.courant_number
    sigma_E = arrays.electric_conductivity
    factor = 1

    if sigma_E is not None:
        E = E * (1 + c * sigma_E * eta0 * inv_eps / 2)
        factor = 1 - c * sigma_E * eta0 * inv_eps / 2

    E = E / factor - c * curl * inv_eps

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

    # Handle periodic boundaries by copying values from opposite sides
    if simulate_boundaries:
        H = arrays.H
        for boundary in objects.boundary_objects:
            if isinstance(boundary, PeriodicBoundary):
                H = H.at[..., boundary.boundary_slice].set(H[..., boundary.opposite_slice])
        arrays = arrays.at["H"].set(H)

    inv_mu = arrays.inv_permeabilities
    c = config.courant_number
    sigma_H = arrays.magnetic_conductivity

    # Get periodic axes for curl operation
    periodic_axes = get_periodic_axes(objects)
    E_pad = arrays.E
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

    DyEz = (jnp.roll(E_pad[2], -1, axis=1) - E_pad[2])[1:-1, 1:-1, 1:-1]
    DzEy = (jnp.roll(E_pad[1], -1, axis=2) - E_pad[1])[1:-1, 1:-1, 1:-1]
    DzEx = (jnp.roll(E_pad[0], -1, axis=2) - E_pad[0])[1:-1, 1:-1, 1:-1]
    DxEz = (jnp.roll(E_pad[2], -1, axis=0) - E_pad[2])[1:-1, 1:-1, 1:-1]
    DxEy = (jnp.roll(E_pad[1], -1, axis=0) - E_pad[1])[1:-1, 1:-1, 1:-1]
    DyEx = (jnp.roll(E_pad[0], -1, axis=1) - E_pad[0])[1:-1, 1:-1, 1:-1]

    # Auxiliary fields
    psi_Hxy = arrays.psi_H[0, :, :, :]
    psi_Hxz = arrays.psi_H[1, :, :, :]
    psi_Hyz = arrays.psi_H[2, :, :, :]
    psi_Hyx = arrays.psi_H[3, :, :, :]
    psi_Hzx = arrays.psi_H[4, :, :, :]
    psi_Hzy = arrays.psi_H[5, :, :, :]

    if simulate_boundaries:
        # Get H-field PML coefficients
        b_x = (
            jnp.expm1(
                -c
                * config.resolution
                / c0
                / eps0
                * (arrays.sigma[3, :, :, :] / arrays.kappa[3, :, :, :] + arrays.alpha[3, :, :, :])
            )
            + 1
        )
        b_y = (
            jnp.expm1(
                -c
                * config.resolution
                / c0
                / eps0
                * (arrays.sigma[4, :, :, :] / arrays.kappa[4, :, :, :] + arrays.alpha[4, :, :, :])
            )
            + 1
        )
        b_z = (
            jnp.expm1(
                -c
                * config.resolution
                / c0
                / eps0
                * (arrays.sigma[5, :, :, :] / arrays.kappa[5, :, :, :] + arrays.alpha[5, :, :, :])
            )
            + 1
        )

        a_x = (
            (b_x - 1.0)
            * arrays.sigma[3, :, :, :]
            / (arrays.sigma[3, :, :, :] + arrays.alpha[3, :, :, :] * arrays.kappa[3, :, :, :])
            / arrays.kappa[3, :, :, :]
        )
        a_y = (
            (b_y - 1.0)
            * arrays.sigma[4, :, :, :]
            / (arrays.sigma[4, :, :, :] + arrays.alpha[4, :, :, :] * arrays.kappa[4, :, :, :])
            / arrays.kappa[4, :, :, :]
        )
        a_z = (
            (b_z - 1.0)
            * arrays.sigma[5, :, :, :]
            / (arrays.sigma[5, :, :, :] + arrays.alpha[5, :, :, :] * arrays.kappa[5, :, :, :])
            / arrays.kappa[5, :, :, :]
        )

        a_x = jnp.nan_to_num(a_x, nan=0.0, posinf=0.0, neginf=0.0)
        a_y = jnp.nan_to_num(a_y, nan=0.0, posinf=0.0, neginf=0.0)
        a_z = jnp.nan_to_num(a_z, nan=0.0, posinf=0.0, neginf=0.0)

        # Update auxiliary fields
        psi_Hxy = b_y * psi_Hxy + a_y * DyEz
        psi_Hxz = b_z * psi_Hxz + a_z * DzEy
        psi_Hyz = b_z * psi_Hyz + a_z * DzEx
        psi_Hyx = b_x * psi_Hyx + a_x * DxEz
        psi_Hzx = b_x * psi_Hzx + a_x * DxEy
        psi_Hzy = b_y * psi_Hzy + a_y * DyEx

        psi_H = jnp.stack((psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy), axis=0)
        arrays = arrays.aset("psi_H", psi_H)

    # Curl equations
    curl_x = (1.0 / arrays.kappa[1, :, :, :] * DyEz + psi_Hxy) - (1.0 / arrays.kappa[2, :, :, :] * DzEy + psi_Hxz)
    curl_y = (1.0 / arrays.kappa[2, :, :, :] * DzEx + psi_Hyz) - (1.0 / arrays.kappa[0, :, :, :] * DxEz + psi_Hyx)
    curl_z = (1.0 / arrays.kappa[0, :, :, :] * DxEy + psi_Hzx) - (1.0 / arrays.kappa[1, :, :, :] * DyEx + psi_Hzy)
    curl = jnp.stack((curl_x, curl_y, curl_z), axis=0)

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

    # Get periodic axes for curl operation
    periodic_axes = get_periodic_axes(objects)
    curl = curl_E(arrays.E, periodic_axes)
    inv_mu = arrays.inv_permeabilities
    c = config.courant_number
    sigma_H = arrays.magnetic_conductivity
    factor = 1

    if sigma_H is not None:
        # lossy materials get gain when simulating backwards
        H = H * (1 + c * sigma_H / eta0 * inv_mu / 2)
        factor = 1 - c * sigma_H / eta0 * inv_mu / 2

    H = H / factor + c * curl * inv_mu

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
