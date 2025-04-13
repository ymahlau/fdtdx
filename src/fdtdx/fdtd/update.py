import jax
import pytreeclass as tc

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.curl import curl_E, curl_H, interpolate_fields
from fdtdx.fdtd.misc import add_boundary_interfaces, collect_boundary_interfaces
from fdtdx.objects.boundaries.periodic import PeriodicBoundary
from fdtdx.objects.detectors.detector import Detector


def get_periodic_axes(objects: ObjectContainer) -> tuple[bool, bool, bool]:
    """Determines which axes have periodic boundary conditions.

    Args:
        objects: Container with simulation objects including boundaries

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
        time_step: Current simulation time step
        arrays: Container with E, H fields and material properties
        objects: Container with sources, boundaries and other simulation objects
        config: Simulation configuration parameters
        simulate_boundaries: Whether to apply boundary conditions

    Returns:
        Updated ArrayContainer with new E field values
    """
    boundary_states = {}
    if simulate_boundaries:
        for boundary in objects.boundary_objects:
            boundary_states[boundary.name] = boundary.update_E_boundary_state(
                boundary_state=arrays.boundary_states[boundary.name],
                H=arrays.H,
            )

    # Get periodic axes for curl operation
    periodic_axes = get_periodic_axes(objects)
    E = arrays.E + config.courant_number * curl_H(arrays.H, periodic_axes) * arrays.inv_permittivities

    for source in objects.sources:

        def _update():
            return source.update_E(
                E=E,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=time_step,
                inverse=False,
            )

        E = _update()
        # E = jax.lax.cond(
        #     source._is_on_at_time_step_arr[time_step],
        #     _update,
        #     lambda: E,
        # )

    if simulate_boundaries:
        for boundary in objects.boundary_objects:
            E = boundary.update_E(
                E=E,
                boundary_state=boundary_states[boundary.name],
                inverse_permittivity=arrays.inv_permittivities,
            )

    arrays = arrays.at["E"].set(E)
    if simulate_boundaries:
        arrays = arrays.aset("boundary_states", boundary_states)

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
        time_step: Current simulation time step
        arrays: Container with E, H fields and material properties
        objects: Container with sources and other simulation objects
        config: Simulation configuration parameters

    Returns:
        Updated ArrayContainer with reversed E field values
    """
    E = arrays.E
    for source in objects.sources:

        def _update():
            return source.update_E(
                E,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=time_step,
                inverse=True,
            )

        E = _update()
        # E = jax.lax.cond(
        #     source._is_on_at_time_step_arr[time_step],
        #     _update,
        #     lambda: E,
        # )

    # Get periodic axes for curl operation
    periodic_axes = get_periodic_axes(objects)
    E = E - config.courant_number * curl_H(arrays.H, periodic_axes) * arrays.inv_permittivities

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
        time_step: Current simulation time step
        arrays: Container with E, H fields and material properties
        objects: Container with sources, boundaries and other simulation objects
        config: Simulation configuration parameters
        simulate_boundaries: Whether to apply boundary conditions

    Returns:
        Updated ArrayContainer with new H field values
    """
    boundary_states = {}
    if simulate_boundaries:
        for boundary in objects.boundary_objects:
            boundary_states[boundary.name] = boundary.update_H_boundary_state(
                boundary_state=arrays.boundary_states[boundary.name],
                E=arrays.E,
            )

    # Get periodic axes for curl operation
    periodic_axes = get_periodic_axes(objects)
    H = arrays.H - config.courant_number * curl_E(arrays.E, periodic_axes) * arrays.inv_permeabilities

    for source in objects.sources:

        def _update():
            return source.update_H(
                H=H,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=time_step + 0.5,
                inverse=False,
            )

        H = _update()
        # H = jax.lax.cond(
        #     source._is_on_at_time_step_arr[time_step],
        #     _update,
        #     lambda: H,
        # )

    if simulate_boundaries:
        for boundary in objects.boundary_objects:
            H = boundary.update_H(
                H=H,
                boundary_state=boundary_states[boundary.name],
                inverse_permeability=arrays.inv_permeabilities,
            )

    arrays = arrays.at["H"].set(H)
    if simulate_boundaries:
        arrays = arrays.aset("boundary_states", boundary_states)

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
        time_step: Current simulation time step
        arrays: Container with E, H fields and material properties
        objects: Container with sources and other simulation objects
        config: Simulation configuration parameters

    Returns:
        Updated ArrayContainer with reversed H field values
    """
    H = arrays.H
    for source in objects.sources:

        def _update():
            return source.update_H(
                H,
                inv_permittivities=arrays.inv_permittivities,
                inv_permeabilities=arrays.inv_permeabilities,
                time_step=time_step + 0.5,
                inverse=True,
            )

        H = _update()
        # H = jax.lax.cond(
        #     source._is_on_at_time_step_arr[time_step],
        #     _update,
        #     lambda: H,
        # )

    # Get periodic axes for curl operation
    periodic_axes = get_periodic_axes(objects)
    H = H + config.courant_number * curl_E(arrays.E, periodic_axes) * arrays.inv_permeabilities

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
        time_step: Current simulation time step
        arrays: Container with E, H fields and material properties
        objects: Container with detectors and other simulation objects
        H_prev: Previous H field values for interpolation
        inverse: Whether this is a forward or reverse update

    Returns:
        Updated ArrayContainer with new detector states
    """
    periodic_axes = get_periodic_axes(objects)
    interpolated_E, interpolated_H = interpolate_fields(
        E_field=arrays.E,
        H_field=(H_prev + arrays.H) / 2,
        periodic_axes=periodic_axes,
    )

    def helper_fn(E_input, H_input, detector: Detector):
        detector = tc.unfreeze(detector)
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
            tc.freeze(d),
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
        time_step: Current simulation time step
        arrays: Container with fields and material properties
        objects: Container with PML and other simulation objects
        config: Simulation configuration with gradient settings
        key: Random key for compression

    Returns:
        Updated ArrayContainer with recorded interface values

    Raises:
        Exception: If gradient_config or recorder is None, or if recording_state is None
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
        time_step: Current simulation time step
        arrays: Container with fields and material properties
        objects: Container with PML and other simulation objects
        config: Simulation configuration with gradient settings
        key: Random key for decompression

    Returns:
        Updated ArrayContainer with restored interface values

    Raises:
        Exception: If gradient_config or recorder is None, or if recording_state is None
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
