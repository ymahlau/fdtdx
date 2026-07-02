import math
import warnings
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from loguru import logger

from fdtdx import constants
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import QuasiUniformGrid, RectilinearGrid
from fdtdx.core.jax.default_key import default_key
from fdtdx.core.jax.guards import check_not_tracing
from fdtdx.core.jax.sharding import create_named_sharded_matrix
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.fdtd.container import ArrayContainer, FieldState, ObjectContainer, ParameterContainer
from fdtdx.fdtd.symmetry import apply_mode_symmetry, make_symmetry_walls, reduce_resolved_slices
from fdtdx.materials import (
    compute_allowed_dispersive_coefficients,
    compute_allowed_electric_conductivities,
    compute_allowed_magnetic_conductivities,
    compute_allowed_permeabilities,
    compute_allowed_permittivities,
    compute_pole_coefficients,
)
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.device.parameters.transform import ParameterType
from fdtdx.objects.object import (
    GridCoordinateConstraint,
    PositionConstraint,
    RealCoordinateConstraint,
    SimulationObject,
    SizeConstraint,
    SizeExtensionConstraint,
)
from fdtdx.objects.static_material.static import SimulationVolume, StaticMultiMaterialObject, UniformMaterialObject

DEFAULT_MAX_ITER = 1000


def _warn_if_simulation_volume_too_large(grid_shape: tuple[int, int, int]) -> None:
    num_cells = math.prod(grid_shape)
    if num_cells > constants.MAX_SIMULATION_VOLUME_CELLS:
        warnings.warn(
            f"Simulation volume has {num_cells:,} cells (grid shape {grid_shape}), "
            f"which exceeds the recommended limit of {constants.MAX_SIMULATION_VOLUME_CELLS:,}. "
            "Allocating FDTD field arrays may require excessive memory and fail.",
            UserWarning,
            stacklevel=3,
        )


AnyConstraint = (
    PositionConstraint | SizeConstraint | SizeExtensionConstraint | GridCoordinateConstraint | RealCoordinateConstraint
)


def _resolve_grid_from_volume(
    objects: Sequence[SimulationObject],
    config: SimulationConfig,
) -> SimulationConfig:
    """Resolve an unresolved grid policy using the volume's declared shape.

    If ``config.grid`` is already a ``RectilinearGrid`` this is a no-op.
    The volume's ``partial_grid_shape`` takes priority; ``partial_real_shape``
    is converted using the policy's per-axis spacing as a fallback.
    """
    if isinstance(config.grid, RectilinearGrid):
        return config
    object_map = {obj.name: obj for obj in objects}
    volume_obj = object_map[_resolve_volume_name(object_map)]
    pre_shape_list: list[int] = []
    for axis in range(3):
        n = volume_obj.partial_grid_shape[axis]
        if n is not None:
            pre_shape_list.append(n)
            continue
        length = volume_obj.partial_real_shape[axis]
        if length is not None:
            spacing = (
                config.grid.axis_spacing(axis) if isinstance(config.grid, QuasiUniformGrid) else config.grid.spacing
            )
            pre_shape_list.append(round(length / spacing))
            continue
        raise ValueError(
            f"SimulationVolume axis {axis} has neither partial_grid_shape nor "
            f"partial_real_shape. At least one must be specified so the grid "
            f"can be resolved before constraint solving."
        )
    pre_volume_shape: tuple[int, int, int] = (pre_shape_list[0], pre_shape_list[1], pre_shape_list[2])
    return config.aset("grid", config.grid.resolve(pre_volume_shape))


def place_objects(
    object_list: Sequence[SimulationObject],
    config: SimulationConfig,
    constraints: Sequence[AnyConstraint],
    key: jax.Array | None = None,
) -> tuple[
    ObjectContainer,
    ArrayContainer,
    ParameterContainer,
    SimulationConfig,
    dict[str, Any],
]:
    """Places simulation objects according to specified constraints and initializes containers.

    Args:
        objects (list[SimulationObject]): List of all simulation objects, including the simulation volume.
        config (SimulationConfig): Simulation configuration.
        constraints (Sequence[Constraint]): List of positioning/sizing constraints referencing object names.
        key (jax.Array | None): JAX random key for initialization.  When ``None``
            (the default) a deterministic key is derived from ``_DEFAULT_KEY_SEED``.

    Returns:
        tuple[ObjectContainer, ArrayContainer, ParameterContainer, SimulationConfig, dict[str, Any]]:
        A tuple containing:
            - ObjectContainer with placed simulation objects
            - ArrayContainer with initialized field arrays
            - ParameterContainer with device parameters
            - Updated SimulationConfig
            - Dictionary with additional initialization info

    Raises:
        ValueError: If constraint resolution fails for one or more objects.
    """
    key = default_key(key)
    # Step 0: Check if called inside a JIT trace
    check_not_tracing("fdtdx.place_objects")

    # Step 1a: Extract the volume's shape before constraint solving to resolve the grid early.
    # The volume defines the domain, so its shape must be determinable from its own fields alone —
    # either from partial_grid_shape (cell counts, direct) or partial_real_shape (metres, converted
    # using the policy spacing, which is available without knowing shape).
    config = _resolve_grid_from_volume(object_list, config)

    # Step 1b: Resolve constraints into grid slices
    resolved_slices, errors = resolve_object_constraints(
        objects=object_list,
        constraints=constraints,
        config=config,
    )

    # Step 2: Aggregate errors and raise if needed
    failed = {name: msg for name, msg in errors.items() if msg}
    if failed:
        formatted = "\n".join(f"  - {name}: {msg}" for name, msg in failed.items())
        raise ValueError(f"Failed to resolve object constraints:\n{formatted}")

    # Step 3: Convert name → object for placement
    object_map = {obj.name: obj for obj in object_list}
    volume_name = _resolve_volume_name(object_map)
    volume_obj = object_map[volume_name]

    # Step 4: Mirror-symmetry reduction. When config.symmetry has a nonzero entry, clip every
    # resolved slice onto the kept (upper) half along each symmetric axis, drop objects that fall
    # entirely in the discarded half, and remember the reduced volume shape (used in Step 5b to
    # build the PEC/PMC walls on the symmetry planes). This runs BEFORE the grid is resolved/pinned
    # so config.grid describes the reduced domain the FDTD actually runs on, not the full one. The
    # non-symmetric path is unchanged.
    dropped_names: set[str] = set()
    reduced_volume_shape = None
    if config.has_symmetry:
        resolved_slices, dropped_names, reduced_volume_shape = reduce_resolved_slices(
            resolved_slices=resolved_slices,
            object_map=object_map,
            config=config,
            volume_name=volume_name,
        )

    # Step 5: Re-resolve the grid onto the (possibly symmetry-reduced) volume shape and pin it.
    # For an explicit RectilinearGrid + symmetry, slice the edge arrays onto the kept upper half.
    # For policy grids, the pre-resolved grid from Step 1a already matches unless symmetry reduced
    # the shape, in which case we re-resolve onto the smaller domain.
    vol_slice = resolved_slices[volume_obj.name]
    volume_shape: tuple[int, int, int] = (
        vol_slice[0][1] - vol_slice[0][0],
        vol_slice[1][1] - vol_slice[1][0],
        vol_slice[2][1] - vol_slice[2][0],
    )
    if config.has_symmetry and isinstance(config.grid, RectilinearGrid):
        # Explicit non-uniform grid + symmetry: slice the edge arrays onto the kept upper half
        # (validating even cell count and mirror-symmetric widths) so the reduced grid matches the
        # reduced domain. The UniformGrid path below builds a uniform reduced grid via resolve_grid.
        grid = config.grid.reduce_symmetric(config.symmetry)
        config = config.aset("grid", grid)
    else:
        grid = config.resolve_grid(volume_shape)
        if not isinstance(config.grid, RectilinearGrid) or config.grid.shape != volume_shape:
            config = config.aset("grid", grid)
    if grid.shape != volume_shape:
        raise ValueError(f"Configured grid shape {grid.shape} does not match simulation volume shape {volume_shape}.")

    # Step 6: Place objects on grid based on resolved slice tuples
    placed_objects = []
    for name, slice_tuple in resolved_slices.items():
        if name == volume_obj.name or name in dropped_names:
            continue
        obj = object_map[name]
        assert key is not None
        key, subkey = jax.random.split(key)
        placed_objects.append(
            obj.place_on_grid(
                grid_slice_tuple=slice_tuple,
                config=config,
                key=subkey,
            )
        )

    # Step 7: Place volume first (index 0)
    assert key is not None
    key, subkey = jax.random.split(key)
    placed_objects.insert(
        0,
        volume_obj.place_on_grid(
            grid_slice_tuple=resolved_slices[volume_obj.name],
            config=config,
            key=subkey,
        ),
    )

    # Step 8: Insert the PEC/PMC symmetry walls and forward the per-axis condition to mode
    # sources/detectors, then warn that the simulation now runs on the reduced domain.
    if config.has_symmetry and reduced_volume_shape is not None:
        assert key is not None
        key, subkey = jax.random.split(key)
        walls = make_symmetry_walls(
            config=config,
            reduced_volume_shape=reduced_volume_shape,
            key=subkey,
            existing_names={o.name for o in placed_objects},
        )
        placed_objects.extend(walls)
        placed_objects = apply_mode_symmetry(placed_objects, config)
        # Volume is index 0 and may itself be a mode object in principle; keep it pinned.
        wall_names = [w.name for w in walls]
        logger.warning(
            f"Symmetry {config.symmetry} reduces the simulation to grid shape {reduced_volume_shape} "
            f"(walls added: {wall_names}; objects dropped: {sorted(dropped_names) or 'none'}). "
            f"Results are on the reduced domain — call fdtdx.unfold_detector_states / "
            f"fdtdx.unfold_fields to reconstruct the full domain."
        )

    # Step 9: Create object container
    objects_container = ObjectContainer(
        object_list=placed_objects,
        volume_idx=0,
    )

    # Step 10: Initialize parameters and arrays
    assert key is not None
    key, subkey = jax.random.split(key)
    params = _init_params(objects=objects_container, key=subkey)
    arrays, config, info = _init_arrays(objects=objects_container, config=config)

    # Step 11: Update object configs and apply objects if possible
    disp_c1 = None if arrays.dispersive_c1 is None else jax.lax.stop_gradient(arrays.dispersive_c1)
    disp_c2 = None if arrays.dispersive_c2 is None else jax.lax.stop_gradient(arrays.dispersive_c2)
    disp_c3 = None if arrays.dispersive_c3 is None else jax.lax.stop_gradient(arrays.dispersive_c3)
    disp_c4 = None if arrays.dispersive_c4 is None else jax.lax.stop_gradient(arrays.dispersive_c4)
    sigma_e = None if arrays.electric_conductivity is None else jax.lax.stop_gradient(arrays.electric_conductivity)
    new_object_list = []
    devices = objects_container.devices
    for obj in objects_container.objects:
        # Update object configs with compiled configuration
        obj = obj.aset("_config", config)

        # Apply objects which do not depend on any devices
        if not any([d.check_overlap(obj) for d in devices]):
            key, subkey = jax.random.split(key)
            obj = obj.apply(
                key=subkey,
                inv_permittivities=jax.lax.stop_gradient(arrays.inv_permittivities),
                inv_permeabilities=jax.lax.stop_gradient(arrays.inv_permeabilities),
                dispersive_c1=disp_c1,
                dispersive_c2=disp_c2,
                dispersive_c3=disp_c3,
                dispersive_c4=disp_c4,
                electric_conductivity=sigma_e,
            )
        new_object_list.append(obj)

    objects_container = ObjectContainer(
        object_list=new_object_list,
        volume_idx=0,
    )

    return objects_container, arrays, params, config, info


def apply_params(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    params: ParameterContainer,
    key: jax.Array | None = None,
    **transform_kwargs,
) -> tuple[ArrayContainer, ObjectContainer, dict[str, Any]]:
    """Applies parameters to devices and updates source states.

    Args:
        arrays (ArrayContainer): Container with field arrays
        objects (ObjectContainer): Container with simulation objects
        params (ParameterContainer): Container with device parameters
        key (jax.Array | None): JAX random key for source updates.  When ``None``
            (the default) a deterministic key is derived from ``_DEFAULT_KEY_SEED``.
        **transform_kwargs: Keyword arguments passed to the parameter transformation.
    Returns:
        tuple[ArrayContainer, ObjectContainer, dict[str, Any]]: A tuple containing:
            - Updated ArrayContainer with applied device parameters
            - Updated ObjectContainer with new source states
            - Dictionary with parameter application info
    """
    key = default_key(key)
    info = {}
    # Determine number of components from existing array shape
    num_perm_components = arrays.inv_permittivities.shape[0]
    isotropic = num_perm_components == 1
    diagonally_anisotropic = num_perm_components == 3

    num_dispersive_poles = arrays.dispersive_c1.shape[0] if arrays.dispersive_c1 is not None else 0

    if arrays.initial_inv_permittivities is not None:
        arrays = arrays.at["inv_permittivities"].set(arrays.initial_inv_permittivities)

    # apply parameter to devices
    for device in objects.devices:
        cur_material_indices = device(params[device.name], expand_to_sim_grid=True, **transform_kwargs)
        # allowed_perm_list is list of tuples with length 1 (isotropic) or 3 (diagonally anisotropic) or 9 (fully anisotropic)
        allowed_perm_array = jnp.asarray(
            compute_allowed_permittivities(
                device.materials,
                isotropic=isotropic,
                diagonally_anisotropic=diagonally_anisotropic,
            )
        )  # shape: (num_materials, num_components)

        # When any object in the sim is dispersive (num_dispersive_poles > 0) we
        # always write the coefficient stack into the device's grid_slice — even
        # when none of the device's materials are dispersive themselves. Otherwise
        # stale coefficients from an underlying dispersive region would survive
        # and keep evolving polarization in the device's voxels.
        # compute_allowed_dispersive_coefficients zero-pads non-dispersive materials.
        write_dispersive = num_dispersive_poles > 0
        # ``dispersive_c4`` only exists when a CCPR pole with non-zero dE/dt
        # coupling is present anywhere in the sim (gated at init time). When it
        # is None, every material's c4 is identically zero, so we simply skip
        # writing it.
        write_dispersive_c4 = write_dispersive and arrays.dispersive_c4 is not None

        # Initialise dispersive slots; populated below when write_dispersive is True.
        allowed_c1_arr = allowed_c2_arr = allowed_c3_arr = allowed_c4_arr = None
        new_c1_slice = new_c2_slice = new_c3_slice = new_c4_slice = None
        if write_dispersive:
            assert (
                arrays.dispersive_c1 is not None
                and arrays.dispersive_c2 is not None
                and arrays.dispersive_c3 is not None
            )
            dt = device._config.time_step_duration
            allowed_c1_np, allowed_c2_np, allowed_c3_np, allowed_c4_np = compute_allowed_dispersive_coefficients(
                device.materials,
                dt=dt,
                max_num_poles=num_dispersive_poles,
            )
            allowed_c1_arr = jnp.asarray(allowed_c1_np, dtype=arrays.dispersive_c1.dtype)
            allowed_c2_arr = jnp.asarray(allowed_c2_np, dtype=arrays.dispersive_c2.dtype)
            allowed_c3_arr = jnp.asarray(allowed_c3_np, dtype=arrays.dispersive_c3.dtype)
            if write_dispersive_c4:
                allowed_c4_arr = jnp.asarray(allowed_c4_np, dtype=arrays.dispersive_c4.dtype)

        if device.output_type == ParameterType.CONTINUOUS:
            # Linear interpolation between two materials via their permittivities
            # Add spatial broadcast dims for element-wise multiplication
            perm_bc = allowed_perm_array[:, :, None, None, None]
            if device.use_etching:
                # interpolate between existing background material and etch material
                perm_slice = _invert_property(arrays.inv_permittivities[:, *device.grid_slice])
                perm_slice = perm_slice + cur_material_indices * (perm_bc[0] - perm_slice)
            else:
                # interpolate between two device materials
                # cur_material_indices: (*grid_shape) broadcasts with (num_components, 1, 1, 1)
                perm_slice = perm_bc[0] + cur_material_indices * (perm_bc[1] - perm_bc[0])

            new_inv_perm_slice = _invert_property(perm_slice)

            if write_dispersive:
                assert allowed_c1_arr is not None and allowed_c2_arr is not None and allowed_c3_arr is not None
                # Linear interpolation of dispersive coefficients between the two bracketing materials.
                # allowed_cN_arr: (num_materials, num_poles) — here num_materials == 2.
                # reshape to (num_poles, 1, 1, 1, 1) for broadcast over (num_poles, 1, Nx, Ny, Nz)
                w0 = (1 - cur_material_indices)[None, None, ...]  # (1, 1, Nx, Ny, Nz)
                w1 = cur_material_indices[None, None, ...]
                c1_0 = allowed_c1_arr[0][:, None, None, None, None]  # (num_poles, 1, 1, 1, 1)
                c1_1 = allowed_c1_arr[1][:, None, None, None, None]
                c2_0 = allowed_c2_arr[0][:, None, None, None, None]
                c2_1 = allowed_c2_arr[1][:, None, None, None, None]
                c3_0 = allowed_c3_arr[0][:, None, None, None, None]
                c3_1 = allowed_c3_arr[1][:, None, None, None, None]
                new_c1_slice = w0 * c1_0 + w1 * c1_1
                new_c2_slice = w0 * c2_0 + w1 * c2_1
                new_c3_slice = w0 * c3_0 + w1 * c3_1
                if write_dispersive_c4:
                    assert allowed_c4_arr is not None
                    c4_0 = allowed_c4_arr[0][:, None, None, None, None]
                    c4_1 = allowed_c4_arr[1][:, None, None, None, None]
                    new_c4_slice = w0 * c4_0 + w1 * c4_1
        else:
            # Discrete material selection
            # Precompute inverse permittivities since the selection result is binary
            if isotropic or diagonally_anisotropic:
                inv_allowed = 1.0 / allowed_perm_array  # (num_materials, num_components)
            else:
                # Fully anisotropic: reshape to 3x3 matrix, invert, and flatten back to 9 elements
                inv_allowed = jnp.array([jnp.linalg.inv(perm.reshape(3, 3)).flatten() for perm in allowed_perm_array])

            # inv_allowed[indices] -> (*grid_shape, num_components), then moveaxis -> (num_components, *grid_shape)
            component_values = jnp.moveaxis(inv_allowed[cur_material_indices.astype(jnp.int32)], -1, 0)
            new_inv_perm_slice = straight_through_estimator(cur_material_indices, component_values)

            if write_dispersive:
                assert allowed_c1_arr is not None and allowed_c2_arr is not None and allowed_c3_arr is not None
                int_idx = cur_material_indices.astype(jnp.int32)
                # allowed_cN_arr[int_idx]: (Nx, Ny, Nz, num_poles) -> moveaxis -> (num_poles, Nx, Ny, Nz)
                new_c1_slice = jnp.moveaxis(allowed_c1_arr[int_idx], -1, 0)[:, None, ...]
                new_c2_slice = jnp.moveaxis(allowed_c2_arr[int_idx], -1, 0)[:, None, ...]
                new_c3_slice = jnp.moveaxis(allowed_c3_arr[int_idx], -1, 0)[:, None, ...]
                if write_dispersive_c4:
                    assert allowed_c4_arr is not None
                    new_c4_slice = jnp.moveaxis(allowed_c4_arr[int_idx], -1, 0)[:, None, ...]

        # Update all components of inv_permittivities array at once
        new_inv_perm = arrays.inv_permittivities.at[:, *device.grid_slice].set(new_inv_perm_slice)
        arrays = arrays.at["inv_permittivities"].set(new_inv_perm)

        if write_dispersive:
            assert (
                arrays.dispersive_c1 is not None
                and arrays.dispersive_c2 is not None
                and arrays.dispersive_c3 is not None
            )
            new_c1 = arrays.dispersive_c1.at[:, :, *device.grid_slice].set(new_c1_slice)
            new_c2 = arrays.dispersive_c2.at[:, :, *device.grid_slice].set(new_c2_slice)
            new_c3 = arrays.dispersive_c3.at[:, :, *device.grid_slice].set(new_c3_slice)
            # Recompute inv_c2 from the post-interpolation c2. Do NOT interpolate
            # inv_c2 directly: 1/avg(c2) != avg(1/c2), and the reverse-time ADE
            # relies on inv_c2 being the exact reciprocal of the stored c2.
            new_inv_c2 = jnp.where(new_c2 == 0, 0.0, 1.0 / new_c2)
            arrays = arrays.at["dispersive_c1"].set(new_c1)
            arrays = arrays.at["dispersive_c2"].set(new_c2)
            arrays = arrays.at["dispersive_c3"].set(new_c3)
            arrays = arrays.at["dispersive_inv_c2"].set(new_inv_c2)
            if write_dispersive_c4:
                assert arrays.dispersive_c4 is not None
                new_c4 = arrays.dispersive_c4.at[:, :, *device.grid_slice].set(new_c4_slice)
                arrays = arrays.at["dispersive_c4"].set(new_c4)

    # apply random key to sources. Source-side sampling of the dispersion
    # coefficients (used only for carrier-frequency impedance / energy
    # normalization) is stop_gradient'd to match the treatment of
    # ``inv_permittivities`` — the FDTD VJP itself still propagates gradient
    # through the coefficients, so this only avoids noise from the source
    # amplitude path.
    disp_c1 = None if arrays.dispersive_c1 is None else jax.lax.stop_gradient(arrays.dispersive_c1)
    disp_c2 = None if arrays.dispersive_c2 is None else jax.lax.stop_gradient(arrays.dispersive_c2)
    disp_c3 = None if arrays.dispersive_c3 is None else jax.lax.stop_gradient(arrays.dispersive_c3)
    disp_c4 = None if arrays.dispersive_c4 is None else jax.lax.stop_gradient(arrays.dispersive_c4)
    sigma_e = None if arrays.electric_conductivity is None else jax.lax.stop_gradient(arrays.electric_conductivity)
    new_objects = []
    devices = objects.devices
    for obj in objects.object_list:
        # Only need to apply objects that overlap with devices, the others were applied in place_objects
        if any([d.check_overlap(obj) for d in devices]):
            assert key is not None
            key, subkey = jax.random.split(key)
            obj = obj.apply(
                key=subkey,
                inv_permittivities=jax.lax.stop_gradient(arrays.inv_permittivities),
                inv_permeabilities=jax.lax.stop_gradient(arrays.inv_permeabilities),
                dispersive_c1=disp_c1,
                dispersive_c2=disp_c2,
                dispersive_c3=disp_c3,
                dispersive_c4=disp_c4,
                electric_conductivity=sigma_e,
            )
        new_objects.append(obj)
    new_objects = ObjectContainer(
        object_list=new_objects,
        volume_idx=objects.volume_idx,
    )

    return arrays, new_objects, info


def _init_arrays(
    objects: ObjectContainer,
    config: SimulationConfig,
) -> tuple[ArrayContainer, SimulationConfig, dict[str, Any]]:
    """Initializes field arrays and material properties for the simulation.

    Creates and initializes the E/H fields, permittivity/permeability arrays,
    detector states, boundary states and recording states based on the
    simulation objects and configuration.

    Args:
        objects (ObjectContainer): Container with simulation objects
        config (SimulationConfig): The simulation configuration

    Returns:
        tuple[ArrayContainer, SimulationConfig, dict[str, Any]]: A tuple containing:
            - ArrayContainer with initialized arrays and states
            - Updated SimulationConfig
            - Dictionary with initialization info
    """
    # create E/H fields
    volume_shape = objects.volume.grid_shape
    _warn_if_simulation_volume_too_large(volume_shape)
    grid = config.resolve_grid(volume_shape)
    if grid.shape != volume_shape:
        raise ValueError(f"Configured grid shape {grid.shape} does not match simulation volume shape {volume_shape}.")
    ext_shape = (3, *volume_shape)

    # Determine whether to use complex-valued fields
    needs_complex = any(isinstance(o, BlochBoundary) and o.needs_complex_fields for o in objects.boundary_objects)
    if config.use_complex_fields is None:
        # Auto-detect: promote to complex if any Bloch boundary has non-zero k
        use_complex = needs_complex
    else:
        use_complex = config.use_complex_fields
        if needs_complex and not use_complex:
            raise ValueError(
                "use_complex_fields=False but Bloch boundaries with non-zero "
                "wave vector are present. These require complex-valued fields."
            )

    if use_complex:
        field_dtype = jnp.complex64 if config.dtype == jnp.float32 else jnp.complex128
    else:
        field_dtype = config.dtype

    E = create_named_sharded_matrix(
        ext_shape,
        sharding_axis=1,
        value=0.0,
        dtype=field_dtype,
        backend=config.backend,
    )
    H = create_named_sharded_matrix(
        ext_shape,
        value=0.0,
        dtype=field_dtype,
        sharding_axis=1,
        backend=config.backend,
    )

    # PML auxiliary fields
    psi_E = {
        pml.name: (
            jnp.zeros(pml.grid_shape, dtype=field_dtype),
            jnp.zeros(pml.grid_shape, dtype=field_dtype),
        )
        for pml in objects.pml_objects
    }
    psi_H = {
        pml.name: (
            jnp.zeros(pml.grid_shape, dtype=field_dtype),
            jnp.zeros(pml.grid_shape, dtype=field_dtype),
        )
        for pml in objects.pml_objects
    }

    # Determine isotropy flags
    isotropic_permittivity = objects.all_objects_isotropic_permittivity
    isotropic_permeability = objects.all_objects_isotropic_permeability
    isotropic_electric_conductivity = objects.all_objects_isotropic_electric_conductivity
    isotropic_magnetic_conductivity = objects.all_objects_isotropic_magnetic_conductivity

    # Determine diagonally anisotropic flags
    diagonally_anisotropic_permittivity = objects.all_objects_diagonally_anisotropic_permittivity
    diagonally_anisotropic_permeability = objects.all_objects_diagonally_anisotropic_permeability
    diagonally_anisotropic_electric_conductivity = objects.all_objects_diagonally_anisotropic_electric_conductivity
    diagonally_anisotropic_magnetic_conductivity = objects.all_objects_diagonally_anisotropic_magnetic_conductivity

    # Get component counts for each property
    if isotropic_permittivity:
        num_perm_components = 1
    elif diagonally_anisotropic_permittivity:
        num_perm_components = 3
    else:
        num_perm_components = 9

    if isotropic_permeability:
        num_permeability_components = 1
    elif diagonally_anisotropic_permeability:
        num_permeability_components = 3
    else:
        num_permeability_components = 9

    if isotropic_electric_conductivity:
        num_electric_cond_components = 1
    elif diagonally_anisotropic_electric_conductivity:
        num_electric_cond_components = 3
    else:
        num_electric_cond_components = 9

    if isotropic_magnetic_conductivity:
        num_magnetic_cond_components = 1
    elif diagonally_anisotropic_magnetic_conductivity:
        num_magnetic_cond_components = 3
    else:
        num_magnetic_cond_components = 9

    # permittivity - shape (1, Nx, Ny, Nz) for isotropic, (3, Nx, Ny, Nz) for diagonally anisotropic, (9, Nx, Ny, Nz) for fully anisotropic
    inv_permittivities = create_named_sharded_matrix(
        (num_perm_components, *volume_shape),
        value=0.0,
        dtype=config.dtype,
        sharding_axis=1,
        backend=config.backend,
    )

    # permeability - scalar 1.0 if non-magnetic, else (1, Nx, Ny, Nz) for isotropic, (3, Nx, Ny, Nz) for diagonally anisotropic, (9, Nx, Ny, Nz) for fully anisotropic
    if objects.all_objects_non_magnetic:
        inv_permeabilities = 1.0
    else:
        inv_permeabilities = create_named_sharded_matrix(
            (num_permeability_components, *volume_shape),
            value=0.0,
            dtype=config.dtype,
            sharding_axis=1,
            backend=config.backend,
        )

    # electric conductivity - None if non-conductive, else (1, Nx, Ny, Nz) for isotropic, (3, Nx, Ny, Nz) for diagonally anisotropic, (9, Nx, Ny, Nz) for fully anisotropic
    electric_conductivity = None
    if not objects.all_objects_non_electrically_conductive:
        electric_conductivity = create_named_sharded_matrix(
            (num_electric_cond_components, *volume_shape),
            value=0.0,
            dtype=config.dtype,
            sharding_axis=1,
            backend=config.backend,
        )

    # magnetic conductivity - None if non-conductive, else (1, Nx, Ny, Nz) for isotropic, (3, Nx, Ny, Nz) for diagonally anisotropic, (9, Nx, Ny, Nz) for fully anisotropic
    magnetic_conductivity = None
    if not objects.all_objects_non_magnetically_conductive:
        magnetic_conductivity = create_named_sharded_matrix(
            (num_magnetic_cond_components, *volume_shape),
            value=0.0,
            dtype=config.dtype,
            sharding_axis=1,
            backend=config.backend,
        )
    conductivity_spacing = None
    if electric_conductivity is not None or magnetic_conductivity is not None:
        conductivity_spacing = constants.c * config.time_step_duration / config.courant_number

    # dispersive ADE auxiliary arrays - all None unless any material is dispersive.
    # Per-cell coefficients are broadcast over component via a size-1 axis.
    num_dispersive_poles = objects.max_num_dispersive_poles
    # ``dispersive_c4`` (the CCPR dE/dt coupling) is only allocated when at least
    # one pole in the sim has a non-zero ``coupling_edot``. Lorentz/Drude-only
    # sims leave it None so the ADE update takes the classic path unchanged.
    allocate_c4 = num_dispersive_poles > 0 and objects.has_dispersive_edot
    dispersive_P_curr = None
    dispersive_P_prev = None
    dispersive_c1 = None
    dispersive_c2 = None
    dispersive_c3 = None
    dispersive_c4 = None
    if num_dispersive_poles > 0:
        if not (isotropic_permittivity or diagonally_anisotropic_permittivity):
            raise NotImplementedError(
                "Dispersive materials cannot be combined with fully anisotropic "
                "(off-diagonal) permittivity tensors in v1."
            )
        dispersive_P_curr = create_named_sharded_matrix(
            (num_dispersive_poles, 3, *volume_shape),
            value=0.0,
            dtype=field_dtype,
            sharding_axis=2,
            backend=config.backend,
        )
        dispersive_P_prev = create_named_sharded_matrix(
            (num_dispersive_poles, 3, *volume_shape),
            value=0.0,
            dtype=field_dtype,
            sharding_axis=2,
            backend=config.backend,
        )
        dispersive_c1 = create_named_sharded_matrix(
            (num_dispersive_poles, 1, *volume_shape),
            value=0.0,
            dtype=config.dtype,
            sharding_axis=2,
            backend=config.backend,
        )
        dispersive_c2 = create_named_sharded_matrix(
            (num_dispersive_poles, 1, *volume_shape),
            value=0.0,
            dtype=config.dtype,
            sharding_axis=2,
            backend=config.backend,
        )
        dispersive_c3 = create_named_sharded_matrix(
            (num_dispersive_poles, 1, *volume_shape),
            value=0.0,
            dtype=config.dtype,
            sharding_axis=2,
            backend=config.backend,
        )
        if allocate_c4:
            dispersive_c4 = create_named_sharded_matrix(
                (num_dispersive_poles, 1, *volume_shape),
                value=0.0,
                dtype=config.dtype,
                sharding_axis=2,
                backend=config.backend,
            )

    # set permittivity/permeability/conductivity of static objects
    sorted_obj = sorted(
        objects.static_material_objects,
        key=lambda o: o.placement_order,
    )
    info = {}
    for o in sorted_obj:
        if isinstance(o, UniformMaterialObject):
            # Material properties are tuples (εxx, εxy, εxz, εyx, εyy, εyz, εzx, εzy, εzz)
            # Arrays have shape (num_components, Nx, Ny, Nz) where num_components is 1 (isotropic), 3 (diagonally anisotropic), or 9 (fully anisotropic)

            if num_perm_components == 1:
                # Isotropic: simple element-wise inversion
                perm_tuple = (o.material.permittivity[0],)
                inv_obj_permittivity = (1 / jnp.array(perm_tuple, dtype=config.dtype))[:, None, None, None]
                inv_permittivities = inv_permittivities.at[:, *o.grid_slice].set(inv_obj_permittivity)
            elif num_perm_components == 3:
                # Diagonally anisotropic: simple element-wise inversion
                perm_tuple = (o.material.permittivity[0], o.material.permittivity[4], o.material.permittivity[8])
                inv_obj_permittivity = (1 / jnp.array(perm_tuple, dtype=config.dtype))[:, None, None, None]
                inv_permittivities = inv_permittivities.at[:, *o.grid_slice].set(inv_obj_permittivity)
            else:
                # Fully anisotropic: reshape to 3x3 matrix, invert, and flatten back to 9 elements
                perm_tuple = o.material.permittivity
                perm_matrix = jnp.array(perm_tuple, dtype=config.dtype).reshape(3, 3)
                inv_perm_matrix = jnp.linalg.inv(perm_matrix)
                inv_obj_permittivity = inv_perm_matrix.flatten()[:, None, None, None]
                inv_permittivities = inv_permittivities.at[:, *o.grid_slice].set(inv_obj_permittivity)

            if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
                if num_permeability_components == 1:
                    # Isotropic: simple element-wise inversion
                    perm_tuple = (o.material.permeability[0],)
                    inv_obj_permeability = (1 / jnp.array(perm_tuple, dtype=config.dtype))[:, None, None, None]
                    inv_permeabilities = inv_permeabilities.at[:, *o.grid_slice].set(inv_obj_permeability)
                elif num_permeability_components == 3:
                    # Diagonally anisotropic: simple element-wise inversion
                    perm_tuple = (o.material.permeability[0], o.material.permeability[4], o.material.permeability[8])
                    inv_obj_permeability = (1 / jnp.array(perm_tuple, dtype=config.dtype))[:, None, None, None]
                    inv_permeabilities = inv_permeabilities.at[:, *o.grid_slice].set(inv_obj_permeability)
                else:
                    # Fully anisotropic: reshape to 3x3 matrix, invert, and flatten back to 9 elements
                    perm_tuple = o.material.permeability
                    perm_matrix = jnp.array(perm_tuple, dtype=config.dtype).reshape(3, 3)
                    inv_perm_matrix = jnp.linalg.inv(perm_matrix)
                    inv_obj_permeability = inv_perm_matrix.flatten()[:, None, None, None]
                    inv_permeabilities = inv_permeabilities.at[:, *o.grid_slice].set(inv_obj_permeability)

            if electric_conductivity is not None:
                if num_electric_cond_components == 1:
                    # Isotropic
                    cond_tuple = (o.material.electric_conductivity[0],)
                elif num_electric_cond_components == 3:
                    # Diagonally anisotropic
                    cond_tuple = (
                        o.material.electric_conductivity[0],
                        o.material.electric_conductivity[4],
                        o.material.electric_conductivity[8],
                    )
                else:
                    # Fully anisotropic
                    cond_tuple = o.material.electric_conductivity

                # Scale physical conductivity into the dimensionless update coefficient.
                # On uniform grids this equals the scalar grid spacing.  On stretched
                # grids it is the reference spacing implied by ``c0 * dt / courant``.
                assert conductivity_spacing is not None
                obj_electric_conductivity = (jnp.array(cond_tuple, dtype=config.dtype) * conductivity_spacing)[
                    :, None, None, None
                ]
                electric_conductivity = electric_conductivity.at[:, *o.grid_slice].set(obj_electric_conductivity)

            if magnetic_conductivity is not None:
                if num_magnetic_cond_components == 1:
                    # Isotropic
                    cond_tuple = (o.material.magnetic_conductivity[0],)
                elif num_magnetic_cond_components == 3:
                    # Diagonally anisotropic
                    cond_tuple = (
                        o.material.magnetic_conductivity[0],
                        o.material.magnetic_conductivity[4],
                        o.material.magnetic_conductivity[8],
                    )
                else:
                    # Fully anisotropic
                    cond_tuple = o.material.magnetic_conductivity

                # Scale physical conductivity into the dimensionless update coefficient.
                assert conductivity_spacing is not None
                obj_magnetic_conductivity = (jnp.array(cond_tuple, dtype=config.dtype) * conductivity_spacing)[
                    :, None, None, None
                ]
                magnetic_conductivity = magnetic_conductivity.at[:, *o.grid_slice].set(obj_magnetic_conductivity)

            if num_dispersive_poles > 0:
                # Always write the full pole-coefficient stack — zero-padded for
                # non-dispersive materials — so later placements deterministically
                # overwrite earlier coefficients across the object's grid_slice.
                # Without this, a non-dispersive UniformMaterialObject stacked over
                # a dispersive one would leave stale pole coefficients in the overlap
                # and drive an ADE update on cells that shouldn't have one.
                assert dispersive_c1 is not None and dispersive_c2 is not None and dispersive_c3 is not None
                poles = o.material.dispersion.poles if o.material.dispersion is not None else ()
                c1_vals, c2_vals, c3_vals, c4_vals = compute_pole_coefficients(poles, config.time_step_duration)
                n = len(poles)
                c1_padded = jnp.zeros(num_dispersive_poles, dtype=config.dtype)
                c2_padded = jnp.zeros(num_dispersive_poles, dtype=config.dtype)
                c3_padded = jnp.zeros(num_dispersive_poles, dtype=config.dtype)
                c4_padded = jnp.zeros(num_dispersive_poles, dtype=config.dtype)
                if n > 0:
                    c1_padded = c1_padded.at[:n].set(jnp.asarray(c1_vals, dtype=config.dtype))
                    c2_padded = c2_padded.at[:n].set(jnp.asarray(c2_vals, dtype=config.dtype))
                    c3_padded = c3_padded.at[:n].set(jnp.asarray(c3_vals, dtype=config.dtype))
                    c4_padded = c4_padded.at[:n].set(jnp.asarray(c4_vals, dtype=config.dtype))
                # Broadcast (num_poles,) → (num_poles, 1, Nx, Ny, Nz) over grid_slice
                slice_shape = dispersive_c1[:, :, *o.grid_slice].shape
                c1_block = jnp.broadcast_to(c1_padded[:, None, None, None, None], slice_shape)
                c2_block = jnp.broadcast_to(c2_padded[:, None, None, None, None], slice_shape)
                c3_block = jnp.broadcast_to(c3_padded[:, None, None, None, None], slice_shape)
                dispersive_c1 = dispersive_c1.at[:, :, *o.grid_slice].set(c1_block)
                dispersive_c2 = dispersive_c2.at[:, :, *o.grid_slice].set(c2_block)
                dispersive_c3 = dispersive_c3.at[:, :, *o.grid_slice].set(c3_block)
                if dispersive_c4 is not None:
                    c4_block = jnp.broadcast_to(c4_padded[:, None, None, None, None], slice_shape)
                    dispersive_c4 = dispersive_c4.at[:, :, *o.grid_slice].set(c4_block)

        elif isinstance(o, (StaticMultiMaterialObject)):
            indices = o.get_material_mapping()
            mask = o.get_voxel_mask_for_shape()

            # compute_allowed_permittivities returns list of tuples with length 1 (isotropic), 3 (diagonally anisotropic), or 9 (fully anisotropic)
            allowed_perms = jnp.asarray(
                compute_allowed_permittivities(
                    o.materials,
                    isotropic=isotropic_permittivity,
                    diagonally_anisotropic=diagonally_anisotropic_permittivity,
                )
            )

            # allowed_perms[indices] -> (*grid_shape, num_components)
            # After moveaxis -> (num_components, *grid_shape)
            component_values = jnp.moveaxis(allowed_perms[indices], -1, 0)
            perm_slice = _invert_property(inv_permittivities[:, *o.grid_slice])

            # Linearly interpolate in the forward domain
            perm_slice = perm_slice + mask * (component_values - perm_slice)
            inv_permittivities = inv_permittivities.at[:, *o.grid_slice].set(_invert_property(perm_slice))

            if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
                allowed_perms = jnp.asarray(
                    compute_allowed_permeabilities(
                        o.materials,
                        isotropic=isotropic_permeability,
                        diagonally_anisotropic=diagonally_anisotropic_permeability,
                    )
                )

                component_values = jnp.moveaxis(allowed_perms[indices], -1, 0)
                perm_slice = _invert_property(inv_permeabilities[:, *o.grid_slice])

                perm_slice = perm_slice + mask * (component_values - perm_slice)
                inv_permeabilities = inv_permeabilities.at[:, *o.grid_slice].set(_invert_property(perm_slice))

            if electric_conductivity is not None:
                allowed_conds = jnp.asarray(
                    compute_allowed_electric_conductivities(
                        o.materials,
                        isotropic=isotropic_electric_conductivity,
                        diagonally_anisotropic=diagonally_anisotropic_electric_conductivity,
                    )
                )

                assert conductivity_spacing is not None
                component_values = jnp.moveaxis(allowed_conds[indices], -1, 0) * conductivity_spacing
                diff = component_values - electric_conductivity[:, *o.grid_slice]
                electric_conductivity = electric_conductivity.at[:, *o.grid_slice].add(mask * diff)

            if magnetic_conductivity is not None:
                allowed_conds = jnp.asarray(
                    compute_allowed_magnetic_conductivities(
                        o.materials,
                        isotropic=isotropic_magnetic_conductivity,
                        diagonally_anisotropic=diagonally_anisotropic_magnetic_conductivity,
                    )
                )

                assert conductivity_spacing is not None
                component_values = jnp.moveaxis(allowed_conds[indices], -1, 0) * conductivity_spacing
                diff = component_values - magnetic_conductivity[:, *o.grid_slice]
                magnetic_conductivity = magnetic_conductivity.at[:, *o.grid_slice].add(mask * diff)

            # Always run when dispersive arrays exist in the sim: a non-dispersive
            # StaticMultiMaterialObject layered over a dispersive region must
            # zero the inherited coefficients in its mask. compute_allowed_dispersive_coefficients
            # zero-pads non-dispersive materials, so this still cleanly overwrites.
            if num_dispersive_poles > 0:
                assert dispersive_c1 is not None and dispersive_c2 is not None and dispersive_c3 is not None
                allowed_c1, allowed_c2, allowed_c3, allowed_c4 = compute_allowed_dispersive_coefficients(
                    o.materials,
                    dt=config.time_step_duration,
                    max_num_poles=num_dispersive_poles,
                )
                # Shape (num_materials, num_poles) -> index by (Nx, Ny, Nz) ->
                # (Nx, Ny, Nz, num_poles) -> moveaxis -> (num_poles, Nx, Ny, Nz)
                c1_voxels = jnp.moveaxis(jnp.asarray(allowed_c1, dtype=config.dtype)[indices], -1, 0)
                c2_voxels = jnp.moveaxis(jnp.asarray(allowed_c2, dtype=config.dtype)[indices], -1, 0)
                c3_voxels = jnp.moveaxis(jnp.asarray(allowed_c3, dtype=config.dtype)[indices], -1, 0)
                # broadcast over component axis
                c1_voxels = c1_voxels[:, None, ...]
                c2_voxels = c2_voxels[:, None, ...]
                c3_voxels = c3_voxels[:, None, ...]
                mask_bc = mask[None, None, ...]
                diff = c1_voxels - dispersive_c1[:, :, *o.grid_slice]
                dispersive_c1 = dispersive_c1.at[:, :, *o.grid_slice].add(mask_bc * diff)
                diff = c2_voxels - dispersive_c2[:, :, *o.grid_slice]
                dispersive_c2 = dispersive_c2.at[:, :, *o.grid_slice].add(mask_bc * diff)
                diff = c3_voxels - dispersive_c3[:, :, *o.grid_slice]
                dispersive_c3 = dispersive_c3.at[:, :, *o.grid_slice].add(mask_bc * diff)
                if dispersive_c4 is not None:
                    c4_voxels = jnp.moveaxis(jnp.asarray(allowed_c4, dtype=config.dtype)[indices], -1, 0)[:, None, ...]
                    diff = c4_voxels - dispersive_c4[:, :, *o.grid_slice]
                    dispersive_c4 = dispersive_c4.at[:, :, *o.grid_slice].add(mask_bc * diff)
        else:
            raise Exception(f"Unknown object type: {o}")

    # detector states
    detector_states = {}
    for d in objects.detectors:
        detector_states[d.name] = d.init_state()

    # interfaces
    recording_state = None
    if config.gradient_config is not None and config.gradient_config.recorder is not None:
        input_shape_dtypes = {}
        for boundary in objects.pml_objects:
            cur_shape = boundary.interface_grid_shape()
            extended_shape = (3, *cur_shape)
            input_shape_dtypes[f"{boundary.name}_E"] = jax.ShapeDtypeStruct(shape=extended_shape, dtype=field_dtype)
            input_shape_dtypes[f"{boundary.name}_H"] = jax.ShapeDtypeStruct(shape=extended_shape, dtype=field_dtype)
        recorder = config.gradient_config.recorder
        recorder, recording_state = recorder.init_state(
            input_shape_dtypes=input_shape_dtypes,
            max_time_steps=config.time_steps_total,
            backend=config.backend,
        )
        grad_cfg = config.gradient_config.aset(
            "recorder",
            recorder,
        )
        config = config.aset("gradient_config", grad_cfg)

    # Cache 1/c2 with non-dispersive cells zeroed so update_E_reverse can replace
    # its ``jnp.where(c2 == 0, ..., / c2)`` pair with a single multiply.
    dispersive_inv_c2 = None
    if dispersive_c2 is not None:
        dispersive_inv_c2 = jnp.where(dispersive_c2 == 0, 0.0, 1.0 / dispersive_c2)

    # Save backup of initial inv_permittivities when using etched_devices
    using_etching = any(d.use_etching for d in objects.devices)
    initial_inv_permittivities = jnp.copy(inv_permittivities) if using_etching else None

    arrays = ArrayContainer(
        fields=FieldState(
            E=E,
            H=H,
            psi_E=psi_E,
            psi_H=psi_H,
            dispersive_P_curr=dispersive_P_curr,
            dispersive_P_prev=dispersive_P_prev,
        ),
        inv_permittivities=inv_permittivities,
        inv_permeabilities=inv_permeabilities,
        detector_states=detector_states,
        recording_state=recording_state,
        electric_conductivity=electric_conductivity,
        magnetic_conductivity=magnetic_conductivity,
        dispersive_c1=dispersive_c1,
        dispersive_c2=dispersive_c2,
        dispersive_c3=dispersive_c3,
        dispersive_c4=dispersive_c4,
        dispersive_inv_c2=dispersive_inv_c2,
        initial_inv_permittivities=initial_inv_permittivities,
    )
    return arrays, config, info


def _init_params(
    objects: ObjectContainer,
    key: jax.Array,
) -> ParameterContainer:
    """Initializes parameters for simulation devices.

    Args:
        objects (ObjectContainer): Container with simulation objects
        key (jax.Array): JAX random key for parameter initialization

    Returns:
        ParameterContainer: ParameterContainer with initialized device parameters
    """
    params = {}
    for d in objects.devices:
        key, subkey = jax.random.split(key)
        cur_dict = d.init_params(key=subkey)
        params[d.name] = cur_dict
    return params


def resolve_object_constraints(
    objects: Sequence[SimulationObject],
    constraints: Sequence[AnyConstraint],
    config: SimulationConfig,
    max_iter: int = DEFAULT_MAX_ITER,
) -> tuple[dict, dict]:
    """Resolve object constraints into grid slices and shapes."""
    # Sanity check: Ensure all objects have unique names
    object_names = [obj.name for obj in objects]
    duplicates = {name for name in object_names if object_names.count(name) > 1}
    invalid_objects = [obj for obj in objects if not isinstance(obj, SimulationObject)]
    if duplicates:
        raise Exception(
            f"Duplicate object names detected: {', '.join(sorted(duplicates))}. "
            "Each object must have a unique name before resolving constraints into grid slices."
        )
    if invalid_objects:
        raise ValueError(
            f"Invalid object types detected: {', '.join(sorted(invalid_objects))}. "
            "All objects must be instances or subclasses of SimulationObject."
        )
    _check_objects_names_from_constraints(
        constraints=constraints,
        object_names=object_names,
    )

    # Resolve grid before applying constraints
    config = _resolve_grid_from_volume(objects, config)

    # Apply constraints iteratively
    resolved, errors = _apply_constraints_iteratively(
        objects=list(objects),
        constraints=constraints,
        config=config,
        max_iter=max_iter,
    )

    # Convert shape_dict and slice_dict from object references to object names
    resolved_slices = {}
    for obj_name, slice_list in resolved.items():
        resolved_slices[obj_name] = tuple([(axis_slice_list[0], axis_slice_list[1]) for axis_slice_list in slice_list])

    # Get volume bounds from resolved slices
    volume_name = _resolve_volume_name({obj.name: obj for obj in objects})
    volume_slice = resolved_slices.get(volume_name)

    # If the volume itself failed to resolve, skip bounds checks
    if volume_slice is not None:
        volume_bounds = tuple((s1, s2) for s1, s2 in volume_slice)

        # Validate all non-volume objects are within simulation volume bounds
        for obj_name, slice_tuple in resolved_slices.items():
            if obj_name == volume_name:
                continue  # Skip the volume itself

            # Check for unresolved bounds first
            unresolved_axes = []
            for axis in range(3):
                s1, s2 = slice_tuple[axis]
                if s1 is None or s2 is None:
                    unresolved_axes.append(axis)

            if unresolved_axes:
                # Ensure unresolved objects are flagged in errors
                if not errors.get(obj_name):
                    errors[obj_name] = (
                        f"Object '{obj_name}' has unresolved bounds on axes {unresolved_axes}. Slice: {slice_tuple}"
                    )
                continue

            # Check bounds violations
            msgs = []
            for axis in range(3):
                s1, s2 = slice_tuple[axis]
                vol_s1, vol_s2 = volume_bounds[axis]

                if s1 < vol_s1:
                    msgs.append(f"axis {axis}: lower bound {s1} < volume lower bound {vol_s1}")
                if s2 > vol_s2:
                    msgs.append(f"axis {axis}: upper bound {s2} > volume upper bound {vol_s2}")
                if s2 <= s1:
                    msgs.append(f"axis {axis}: invalid size (lower bound {s1} >= upper bound {s2})")

            if msgs:
                prev = errors.get(obj_name) or ""
                errors[obj_name] = (
                    (prev + "; " if prev else "")
                    + f"Object '{obj_name}' out of bounds ({slice_tuple} vs volume {volume_bounds}): "
                    + "; ".join(msgs)
                )

    return resolved_slices, errors


def _center_to_bounds(
    real_pos: float,
    resolution: float,
    size: int,
    volume_size: int,
) -> tuple[int, int]:
    """Convert a center-relative real-space position into grid bounds.

    The coordinate origin (0,0,0) is interpreted as the center of the
    simulation volume, not the lower-left simulation corner.
    """

    # convert physical coordinate to grid coordinate relative to volume center
    volume_center = volume_size / 2

    grid_center = round(real_pos / resolution + volume_center)

    lower = round(grid_center - size / 2)
    upper = lower + size

    return lower, upper


def _real_length_to_grid_size(config: SimulationConfig, axis: int, length: float) -> int:
    """Convert a physical length to a grid-cell count.

    For uniform grids, uses nearest snapping.
    For non-uniform grids, uses upper snapping but adjusts for exact edge alignment
    to avoid off-by-one errors while ensuring coverage of the requested length.
    """
    grid = config.resolved_grid
    if grid is None:
        raise ValueError(
            "_real_length_to_grid_size requires a resolved RectilinearGrid. "
            "Ensure place_objects has resolved the grid before constraint solving."
        )

    # Uniform grids: use nearest snapping (no edge alignment issues)
    if not config.has_nonuniform_grid:
        return grid.length_to_cell_count(axis, length, snap="nearest")

    # Non-uniform grids: handle edge alignment and coverage
    edges = grid.edges(axis)
    end_coord = float(edges[0] + length)  # Object starts at the first edge
    end_index = grid.coord_to_index(axis, end_coord, snap="nearest")

    # If the object's end lands exactly on a grid edge, use the nearest index
    # to avoid upper-snap overshooting (e.g., 2.0 in [0.0, 2.0, 5.0] -> index 1, not 2)
    if abs(end_coord - edges[end_index]) < 1e-6 * grid.min_spacing:
        return end_index

    # Otherwise, use upper snapping to ensure coverage, then clamp to grid size
    cell_count = grid.length_to_cell_count(axis, length, snap="upper")
    return min(cell_count, grid.shape[axis])


def _real_coord_to_edge_index(config: SimulationConfig, axis: int, coord: float) -> int:
    """Snap a physical coordinate to a grid edge index.

    Requires ``config.grid`` to already be a resolved ``RectilinearGrid``.
    """
    grid = config.resolved_grid
    if grid is None:
        raise ValueError(
            "_real_coord_to_edge_index requires a resolved RectilinearGrid. "
            "Ensure place_objects has resolved the grid before constraint solving."
        )
    return grid.coord_to_index(axis, coord, snap="nearest")


def _center_to_bounds_for_grid(config: SimulationConfig, axis: int, real_pos: float, size: int) -> tuple[int, int]:
    """Convert a center-relative position to edge bounds on the resolved grid.

    ``real_pos`` is a physical coordinate relative to the simulation domain
    center (0 = center, negative = lower half, positive = upper half).
    ``config.grid`` must already be a resolved ``RectilinearGrid``.
    """
    grid = config.resolved_grid
    if grid is None:
        raise ValueError(
            "_center_to_bounds_for_grid requires a resolved RectilinearGrid. "
            "Ensure place_objects has resolved the grid before constraint solving."
        )
    edges = grid.edges(axis)
    domain_center = (float(edges[0]) + float(edges[-1])) / 2.0
    return grid.bounds_for_center(axis, real_pos + domain_center, size)


def _raise_for_nonuniform_grid_offsets(config: SimulationConfig, values: Sequence[int | None], name: str):
    """Reject index-space distance offsets when a grid is non-uniform.

    Zero and ``None`` are accepted as no-ops for backwards-compatible helper
    defaults.  Non-zero grid distances do not have a metric meaning on stretched
    grids and must be expressed in metres instead.
    """
    if not config.has_nonuniform_grid:
        return
    if any(v not in (None, 0) for v in values):
        raise ValueError(f"{name} are index-space distances and are not supported on non-uniform grids.")


def _resolve_static_positions_initial(
    object_map: dict[str, SimulationObject],
    slice_dict: dict[str, list[list[int | None]]],
    shape_dict: dict[str, list[int | None]],
    config: SimulationConfig,
):
    """Fill in static or directly defined positions from partial_real_position during initial setup.

    The partial_real_position represents the center position of the object.

    Coordinates are interpreted relative to the center of the simulation
    volume, i.e. partial_real_position=(0,0,0) places an object at the
    geometric center of the simulation domain.

    This function converts center-relative real coordinates into positive
    grid coordinates and computes slice boundaries if the object's size
    is known.
    """

    for obj_name, obj in object_map.items():
        if hasattr(obj, "partial_real_position") and obj.partial_real_position is not None:
            for axis in range(3):
                real_position = obj.partial_real_position[axis]

                if real_position is None:
                    continue

                size = shape_dict[obj_name][axis]

                # Need object size to compute centered bounds
                if size is None:
                    continue

                lower, upper = _center_to_bounds_for_grid(
                    config=config,
                    axis=axis,
                    real_pos=real_position,
                    size=size,
                )

                slice_dict[obj_name][axis][0] = lower
                slice_dict[obj_name][axis][1] = upper

    return slice_dict


def _resolve_static_positions_iterative(
    object_map: dict[str, SimulationObject],
    slice_dict: dict[str, list[list[int | None]]],
    shape_dict: dict[str, list[int | None]],
    config: SimulationConfig,
    errors: dict[str, str | None],
):
    """Iteratively resolve positions from partial_real_position when size becomes known.

    The partial_real_position represents the center position of the object.

    Coordinates are interpreted relative to the center of the simulation
    volume, i.e. partial_real_position=(0,0,0) places an object at the
    geometric center of the simulation domain.

    This function is called in each iteration of constraint resolution so
    that positions can be computed as soon as the object size becomes known.

    Returns:
        tuple:
            - resolved_something: Whether new positions were resolved
            - updated slice_dict
            - updated errors
    """

    resolved_something = False

    for obj_name, obj in object_map.items():
        if hasattr(obj, "partial_real_position") and obj.partial_real_position is not None:
            for axis in range(3):
                real_position = obj.partial_real_position[axis]

                if real_position is None:
                    continue

                # Current bounds
                b0, b1 = slice_dict[obj_name][axis]

                # Already fully resolved
                if b0 is not None and b1 is not None:
                    continue

                # Need object size to compute centered bounds
                size = shape_dict[obj_name][axis]

                if size is None:
                    continue

                lower, upper = _center_to_bounds_for_grid(
                    config=config,
                    axis=axis,
                    real_pos=real_position,
                    size=size,
                )

                # Set or validate lower bound
                if b0 is None:
                    slice_dict[obj_name][axis][0] = lower
                    resolved_something = True

                elif b0 != lower:
                    errors[obj_name] = (
                        f"Inconsistent position for {obj_name} "
                        f"axis {axis}: partial_real_position implies "
                        f"lower bound {lower}, but constraint set it "
                        f"to {b0}"
                    )

                # Set or validate upper bound
                if b1 is None:
                    slice_dict[obj_name][axis][1] = upper
                    resolved_something = True

                elif b1 != upper:
                    errors[obj_name] = (
                        f"Inconsistent position for {obj_name} "
                        f"axis {axis}: partial_real_position implies "
                        f"upper bound {upper}, but constraint set it "
                        f"to {b1}"
                    )

    return resolved_something, slice_dict, errors


def _check_objects_names_from_constraints(
    constraints: Sequence[AnyConstraint],
    object_names: list[str],
):
    """Collect object names mentioned in constraints and verify they exist."""
    all_names = set()
    for c in constraints:
        for name in [getattr(c, "object", None), getattr(c, "other_object", None)]:
            if name and name not in object_names:
                raise ValueError(f"Unknown object name in constraint: {name}")
            if name:
                all_names.add(name)
    return list(all_names)


def _apply_constraints_iteratively(
    objects: list[SimulationObject],
    constraints: Sequence[AnyConstraint],
    config: SimulationConfig,
    max_iter: int = DEFAULT_MAX_ITER,
) -> tuple[dict, dict]:
    """
    Iteratively apply all constraints until shapes and positions converge.
    """
    # Convert objects list to object_map dictionary
    object_map = {}
    for obj in objects:
        object_map[obj.name] = obj
    volume_name = _resolve_volume_name(object_map)

    # Initialize shape_dict and slice_dict with object references as keys
    shape_dict = {}
    slice_dict = {}
    for obj in objects:
        shape_dict[obj.name] = [None, None, None]
        slice_dict[obj.name] = [[None, None], [None, None], [None, None]]
    for axis in range(3):
        slice_dict[volume_name][axis][0] = 0

    errors: dict[str, str | None] = {obj.name: None for obj in objects}

    # handle static shapes
    shape_dict = _resolve_static_shapes(
        object_map=object_map,
        shape_dict=shape_dict,
        config=config,
    )

    slice_dict = _resolve_static_positions_initial(
        object_map=object_map,
        slice_dict=slice_dict,
        shape_dict=shape_dict,
        config=config,
    )

    # iterate
    for iteration in range(max_iter):
        changed = False

        # check if we already resolved everything
        if all(
            [
                all([shape_dict[o][i] is not None for i in range(3)])
                and all([all([slice_dict[o][i][s] is not None for s in range(2)]) for i in range(3)])
                for o in object_map.keys()
            ]
        ):
            break

        # Try to resolve positions from partial_real_position if size is now known
        resolved, slice_dict, errors = _resolve_static_positions_iterative(
            object_map=object_map,
            slice_dict=slice_dict,
            shape_dict=shape_dict,
            config=config,
            errors=errors,
        )
        changed = changed or resolved

        # update the grid slices based on static shape and partial known positions
        resolved, slice_dict, errors = _update_grid_slices_from_shapes(
            object_map=object_map,
            shape_dict=shape_dict,
            slice_dict=slice_dict,
            errors=errors,
        )
        changed = changed or resolved

        # update grid shapes based on grid slices
        resolved, shape_dict, errors = _update_grid_shapes_from_slices(
            object_map=object_map,
            shape_dict=shape_dict,
            slice_dict=slice_dict,
            errors=errors,
        )
        changed = changed or resolved

        # go through all constraints
        for c in constraints:
            try:
                if isinstance(c, GridCoordinateConstraint):
                    resolved, slice_dict = _apply_grid_coordinate_constraint(
                        constraint=c,
                        object_map=object_map,
                        slice_dict=slice_dict,
                        config=config,
                    )
                elif isinstance(c, RealCoordinateConstraint):
                    resolved, slice_dict = _apply_real_coordinate_constraint(
                        constraint=c,
                        object_map=object_map,
                        slice_dict=slice_dict,
                        config=config,
                    )
                elif isinstance(c, PositionConstraint):
                    resolved, slice_dict = _apply_position_constraint(
                        constraint=c,
                        object_map=object_map,
                        config=config,
                        shape_dict=shape_dict,
                        slice_dict=slice_dict,
                    )
                elif isinstance(c, SizeConstraint):
                    resolved, shape_dict = _apply_size_constraint(
                        constraint=c,
                        object_map=object_map,
                        config=config,
                        shape_dict=shape_dict,
                        slice_dict=slice_dict,
                    )
                elif isinstance(c, SizeExtensionConstraint):
                    resolved, slice_dict = _apply_size_extension_constraint(
                        constraint=c,
                        object_map=object_map,
                        config=config,
                        slice_dict=slice_dict,
                        volume_name=volume_name,
                    )
                else:
                    raise ValueError(f"Unknown constraint type: {type(c).__name__}")
            except Exception as e:
                errors[c.object] = f"Error applying {type(c).__name__}: {e}"
            changed = changed or resolved

        # Extend objects to infinity if possible
        if not changed:
            changed, slice_dict = _extend_to_inf_if_possible(
                constraints=constraints,
                object_map=object_map,
                slice_dict=slice_dict,
                shape_dict=shape_dict,
                volume_name=volume_name,
            )

        # check for misspecification
        if not changed:
            errors = _handle_unresolved_objects(object_map=object_map, slice_dict=slice_dict, errors=errors)
            break
    else:
        # max_iter reached without convergence
        # Ensure all unresolved objects are flagged
        errors = _handle_unresolved_objects(object_map=object_map, slice_dict=slice_dict, errors=errors)

    return slice_dict, errors


def _resolve_volume_name(
    object_map: dict[str, SimulationObject],
) -> str:
    volume_objects = [o for o in object_map.values() if isinstance(o, SimulationVolume)]
    if not volume_objects:
        raise ValueError("No SimulationVolume object found in the provided objects list.")
    elif len(volume_objects) > 1:
        raise ValueError(
            f"Multiple SimulationVolume objects found ({[o.name for o in volume_objects]}). "
            "There must be exactly one simulation volume."
        )
    return volume_objects[0].name


def _resolve_static_shapes(
    object_map: dict[str, SimulationObject],
    shape_dict: dict[str, list[int | None]],
    config: SimulationConfig,
):
    """Fill in static or directly defined shapes."""
    for obj_name, obj in object_map.items():
        for axis in range(3):
            if obj.partial_grid_shape[axis] is not None:
                shape_dict[obj_name][axis] = obj.partial_grid_shape[axis]
            elif obj.partial_real_shape[axis] is not None:
                cur_grid_shape = _real_length_to_grid_size(config, axis, obj.partial_real_shape[axis])  # type: ignore
                shape_dict[obj_name][axis] = cur_grid_shape
    return shape_dict


def _record_shape_bound_conflict(
    obj_name: str,
    axis: int,
    bound_size: int,
    obj: SimulationObject,
    shape_dict: dict[str, list[int | None]],
    errors: dict[str, str | None],
) -> bool:
    """Record a conflict where shape_dict and bound-derived size disagree. Always an error."""
    errors[obj_name] = (
        f"Inconsistent grid shape for object: {shape_dict[obj_name][axis]} != {bound_size} "
        f"for axis={axis}, {obj.name} ({obj.__class__.__name__}). "
        f"Check partial_real_shape, partial_grid_shape, and any SizeConstraints for this object. "
        f"If the shape is derived from geometry (e.g. radius), a conflicting constraint was applied."
    )
    return False


def _update_grid_slices_from_shapes(
    object_map: dict[str, SimulationObject],
    shape_dict: dict[str, list[int | None]],
    slice_dict: dict[str, list[list[int | None]]],
    errors: dict[str, str | None],
):
    resolved_something = False
    for obj_name, s in shape_dict.items():
        obj = object_map[obj_name]
        for axis in range(3):
            s_axis = s[axis]
            if s_axis is None:
                continue
            b0, b1 = slice_dict[obj_name][axis]
            if b0 is None and b1 is None:
                continue
            elif b0 is not None and b1 is not None:
                if s_axis != b1 - b0:
                    errors[obj_name] = (
                        f"Inconsistent grid shape for object: {s_axis} != {b1 - b0}, {obj.name} ({obj.__class__})."
                    )
            elif b0 is not None:
                slice_dict[obj_name][axis][1] = b0 + s_axis
                resolved_something = True
            elif b1 is not None:
                slice_dict[obj_name][axis][0] = b1 - s_axis
                resolved_something = True
    return resolved_something, slice_dict, errors


def _update_grid_shapes_from_slices(
    object_map: dict[str, SimulationObject],
    shape_dict: dict[str, list[int | None]],
    slice_dict: dict[str, list[list[int | None]]],
    errors: dict[str, str | None],
):
    resolved_something = False
    for obj_name, b in slice_dict.items():
        obj = object_map[obj_name]
        s = shape_dict[obj_name]
        for axis in range(3):
            b0, b1 = b[axis]
            s_axis = s[axis]
            if b0 is not None and b1 is not None:
                if s_axis is None:
                    shape_dict[obj_name][axis] = b1 - b0
                    resolved_something = True
                elif s_axis is not None and b1 - b0 != s_axis:
                    errors[obj_name] = (
                        f"Inconsistent grid shape for object: {s_axis} != {b1 - b0}, {obj.name} ({obj.__class__})."
                    )
    return resolved_something, shape_dict, errors


def _apply_grid_coordinate_constraint(
    constraint: GridCoordinateConstraint,
    object_map: dict[str, SimulationObject],
    slice_dict: dict[str, list[list[int | None]]],
    config: SimulationConfig | None = None,
):
    if config is not None and config.has_nonuniform_grid:
        raise ValueError(
            "GridCoordinateConstraint is an index-space placement API and is not supported on non-uniform grids."
        )
    obj_name = constraint.object
    obj = object_map[obj_name]
    resolved_something = False
    for axis_idx, axis in enumerate(constraint.axes):
        edge_index = constraint.coordinates[axis_idx]
        b_idx = 0 if constraint.sides[axis_idx] == "-" else 1
        if slice_dict[obj_name][axis][b_idx] is None:
            slice_dict[obj_name][axis][b_idx] = edge_index
            resolved_something = True
        elif slice_dict[obj_name][axis][b_idx] != edge_index:
            raise Exception(
                f"Inconsistent grid coordinates for object: "
                f"{slice_dict[obj_name][axis][b_idx]} != {edge_index} for {axis=} {obj.name} ({obj.__class__}). "
            )
    return resolved_something, slice_dict


def _apply_real_coordinate_constraint(
    constraint: RealCoordinateConstraint,
    object_map: dict[str, SimulationObject],
    slice_dict: dict[str, list[list[int | None]]],
    config: SimulationConfig,
):
    obj_name = constraint.object
    obj = object_map[obj_name]
    resolved_something = False
    for axis_idx, axis in enumerate(constraint.axes):
        edge_index = _real_coord_to_edge_index(config, axis, constraint.coordinates[axis_idx])
        b_idx = 0 if constraint.sides[axis_idx] == "-" else 1
        if slice_dict[obj_name][axis][b_idx] is None:
            slice_dict[obj_name][axis][b_idx] = edge_index
            resolved_something = True
        elif slice_dict[obj_name][axis][b_idx] != edge_index:
            raise Exception(
                f"Inconsistent grid coordinates for object: "
                f"{slice_dict[obj_name][axis][b_idx]} != {edge_index} for {axis=} {obj.name} ({obj.__class__}). "
            )
    return resolved_something, slice_dict


def _apply_position_constraint(
    constraint: PositionConstraint,
    object_map: dict[str, SimulationObject],
    config: SimulationConfig,
    shape_dict: dict[str, list[int | None]],
    slice_dict: dict[str, list[list[int | None]]],
):
    """Apply a position constraint between two objects."""
    grid = config.resolved_grid
    if grid is None:
        raise ValueError(
            "_apply_position_constraint requires a resolved RectilinearGrid. "
            "Ensure place_objects has resolved the grid before constraint solving."
        )
    obj_name, other_name = constraint.object, constraint.other_object
    obj = object_map[obj_name]
    resolved_something = False
    # go through axes of constraint
    for axis_idx, axis in enumerate(constraint.axes):
        grid_margin = constraint.grid_margins[axis_idx]
        real_margin = constraint.margins[axis_idx]
        _raise_for_nonuniform_grid_offsets(config, (grid_margin,), "grid_margins")
        # check if other knows their position
        other_b0, other_b1 = slice_dict[other_name][axis]
        if other_b0 is None or other_b1 is None:
            continue
        # check if object knows their size
        object_size = shape_dict[obj_name][axis]
        if object_size is None:
            continue
        other_anchor = grid.anchor_coordinate(
            axis,
            (other_b0, other_b1),
            constraint.other_object_positions[axis_idx],
        )
        if real_margin is not None:
            other_anchor += real_margin
        if grid_margin:
            # grid_margin is in cell units; nonzero values were rejected for non-uniform grids above,
            # so a zero/None margin must not require uniform_spacing() (which raises on stretched grids).
            other_anchor += grid_margin * config.uniform_spacing()
        b0, b1 = grid.bounds_for_anchor(
            axis,
            object_size,
            other_anchor,
            constraint.object_positions[axis_idx],
        )
        # update position or check consistency
        old_b0, old_b1 = slice_dict[obj_name][axis]
        if old_b0 is None:
            slice_dict[obj_name][axis][0] = b0
            resolved_something = True
        elif old_b0 != b0:
            raise Exception(
                f"Inconsistent grid shape (may be due to extension to infinity) at lower bound: "
                f"{old_b0} != {b0} for {axis=}, {obj.name} ({obj.__class__}). "
                f"Object has a position constraint that puts the lower boundary at {b0}, "
                f"but the lower bound was alreay computed to be at {old_b0}. "
                f"This could be due to a missing size constraint/specification, "
                f"or another constraint on this object."
            )
        if old_b1 is None:
            slice_dict[obj_name][axis][1] = b1
            resolved_something = True
        elif old_b1 != b1:
            raise Exception(
                f"Inconsistent grid shape (may be due to extension to infinity) at lower bound: "
                f"{old_b1} != {b1} for {axis=}, {obj.name} ({obj.__class__}). "
                f"Object has a position constraint that puts the upper boundary at {b1}, "
                f"but the lower bound was alreay computed to be at {old_b1}. "
                f"This could be either due to a missing size constraint/specification, "
                f"or another constraint on this object."
            )
    return resolved_something, slice_dict


def _apply_size_constraint(
    constraint: SizeConstraint,
    object_map: dict[str, SimulationObject],
    config: SimulationConfig,
    shape_dict: dict[str, list[int | None]],
    slice_dict: dict[str, list[list[int | None]]] | None = None,
):
    """Resolve a size relationship between objects."""
    grid = config.resolved_grid
    if grid is None:
        raise ValueError(
            "_apply_size_constraint requires a resolved RectilinearGrid. "
            "Ensure place_objects has resolved the grid before constraint solving."
        )
    obj_name, other_name = constraint.object, constraint.other_object
    obj = object_map[obj_name]
    resolved_something = False
    # iterate through axes of the constraint
    for axis_idx, axis in enumerate(constraint.axes):
        _raise_for_nonuniform_grid_offsets(config, (constraint.grid_offsets[axis_idx],), "grid_offsets")
        other_axes = constraint.other_axes[axis_idx]
        # check if other object knows their shape
        other_shape = shape_dict[other_name][other_axes]
        if other_shape is None:
            continue
        # calculate objects shape
        proportion = constraint.proportions[axis_idx]
        assert slice_dict is not None, "_apply_size_constraint requires slice_dict"
        other_b0, other_b1 = slice_dict[other_name][other_axes]
        if other_b0 is None or other_b1 is None:
            continue
        other_length = grid.axis_extent(other_axes, (other_b0, other_b1))
        target_length = other_length * proportion
        if constraint.offsets[axis_idx] is not None:
            target_length += constraint.offsets[axis_idx]
        if constraint.grid_offsets[axis_idx]:
            # grid_offsets are in cell units; nonzero values were rejected for non-uniform grids above,
            # so a zero/None offset must not require uniform_spacing() (which raises on stretched grids).
            target_length += constraint.grid_offsets[axis_idx] * config.uniform_spacing()
        object_shape = _real_length_to_grid_size(config, axis, target_length)
        # update or check consistency
        if shape_dict[obj_name][axis] is None:
            shape_dict[obj_name][axis] = object_shape
            resolved_something = True
        elif shape_dict[obj_name][axis] != object_shape:
            raise Exception(
                f"Inconsistent grid shape for object: "
                f"{shape_dict[obj_name][axis]} != {object_shape} for axis={axis}, "
                f"{obj.name} ({obj.__class__.__name__}). "
                f"Check partial_real_shape, partial_grid_shape, and any SizeConstraints for this object. "
                f"If the shape is derived from geometry (e.g. radius), a conflicting SizeConstraint was applied."
            )
    return resolved_something, shape_dict


def _apply_size_extension_constraint(
    constraint: SizeExtensionConstraint,
    object_map: dict[str, SimulationObject],
    config: SimulationConfig,
    slice_dict: dict[str, list[list[int | None]]],
    volume_name: str,
):
    grid = config.resolved_grid
    if grid is None:
        raise ValueError(
            "_apply_size_extension_constraint requires a resolved RectilinearGrid. "
            "Ensure place_objects has resolved the grid before constraint solving."
        )
    obj_name, other_name = constraint.object, constraint.other_object
    obj = object_map[obj_name]
    dir_idx = 0 if constraint.direction == "-" else 1
    resolved_something = False
    _raise_for_nonuniform_grid_offsets(config, (constraint.grid_offset,), "grid_offset")
    # calculate anchor point
    if other_name is not None:
        # check if other knows their position
        other_b0, other_b1 = slice_dict[other_name][constraint.axis]
        if other_b0 is None or other_b1 is None:
            return False, slice_dict
        other_anchor_coord = grid.anchor_coordinate(
            constraint.axis,
            (other_b0, other_b1),
            constraint.other_position,
        )
        if constraint.offset is not None:
            other_anchor_coord += constraint.offset
        if constraint.grid_offset:
            # grid_offset is in cell units; nonzero values were rejected for non-uniform grids above,
            # so a zero/None offset must not require uniform_spacing() (which raises on stretched grids).
            other_anchor_coord += constraint.grid_offset * config.uniform_spacing()
        other_anchor = grid.coord_to_index(constraint.axis, other_anchor_coord, snap="nearest")
    else:
        # if other is not specified, extend to boundary of simulation volume
        other_anchor = slice_dict[volume_name][constraint.axis][dir_idx]
        if other_anchor is None:
            raise Exception(f"This should never happen: Simulation volume not specified: {volume_name}")
    # update position or check consistency
    old_val = slice_dict[obj_name][constraint.axis][dir_idx]
    if old_val is None:
        slice_dict[obj_name][constraint.axis][dir_idx] = other_anchor
        resolved_something = True
    elif old_val != other_anchor:
        raise Exception(
            f"Inconsistent grid shape at bound {constraint.direction}: "
            f"{old_val} != {other_anchor} for {constraint.axis=}, "
            f"{obj.name} ({obj.__class__})."
        )
    return resolved_something, slice_dict


def _extend_to_inf_if_possible(
    constraints: Sequence[AnyConstraint],
    object_map: dict[str, SimulationObject],
    slice_dict: dict[str, list[list[int | None]]],
    shape_dict: dict[str, list[int | None]],
    volume_name: str,
):
    # Extend objects to infinity, which fulfill the properties:
    # - do not already have both boundaries specified
    # - are not constrained by extension constraints in that direction
    # Note: Objects with known size but no position will extend from 0
    # Note: Size constraints alone don't prevent extension - they just constrain the size
    resolved_something = False
    for axis in range(3):
        extension_obj = [(o, 0) for o in object_map.keys()] + [(o, 1) for o in object_map.keys()]

        # Remove objects that are in extension constraints (not size constraints!)
        # Size constraints only constrain the size, not the position
        for c in constraints:
            if isinstance(c, SizeExtensionConstraint) and axis == c.axis:
                direction = 0 if c.direction == "-" else 1
                if (c.object, direction) in extension_obj:
                    extension_obj.remove((c.object, direction))

            # Do not extend objects that have a pending PositionConstraint on this axis.
            # If the referenced object's bounds are still unknown the constraint cannot resolve
            # yet, and locking position=0 now will conflict when the constraint resolves later.
            if isinstance(c, PositionConstraint):
                for c_axis in c.axes:
                    if c_axis != axis:
                        continue
                    other_b0, other_b1 = slice_dict[c.other_object][axis]
                    if other_b0 is None or other_b1 is None:
                        if (c.object, 0) in extension_obj:
                            extension_obj.remove((c.object, 0))
                        if (c.object, 1) in extension_obj:
                            extension_obj.remove((c.object, 1))

        # For each object, determine what can be extended
        for o in object_map.keys():
            b0, b1 = slice_dict[o][axis]
            size = shape_dict[o][axis]

            # Both boundaries known - don't extend either
            if b0 is not None and b1 is not None:
                if (o, 0) in extension_obj:
                    extension_obj.remove((o, 0))
                if (o, 1) in extension_obj:
                    extension_obj.remove((o, 1))
            # Lower bound known but upper not - can compute upper if size known
            elif b0 is not None and b1 is None and size is not None:
                if (o, 1) in extension_obj:
                    extension_obj.remove((o, 1))
            # Upper bound known but lower not - can compute lower if size known
            elif b1 is not None and b0 is None and size is not None:
                if (o, 0) in extension_obj:
                    extension_obj.remove((o, 0))
            # No boundaries known but size is known - extend lower from 0, upper can be computed
            elif b0 is None and b1 is None and size is not None:
                # Keep lower (0) in extension_obj so it extends from 0
                # Remove upper from extension_obj since it will be computed
                if (o, 1) in extension_obj:
                    extension_obj.remove((o, 1))

        # Apply extensions
        for o, direction in extension_obj:
            if slice_dict[o][axis][direction] is not None:
                continue
            resolved_something = True
            if direction == 0:
                slice_dict[o][axis][0] = 0
            else:
                slice_dict[o][axis][1] = shape_dict[volume_name][axis]
    return resolved_something, slice_dict


def _handle_unresolved_objects(
    object_map: dict[str, SimulationObject],
    slice_dict: dict[str, list[list[int | None]]],
    errors: dict[str, str | None],
):
    for obj_name, obj in object_map.items():
        if any([slice_dict[obj_name][a][0] is None or slice_dict[obj_name][a][1] is None for a in range(3)]):
            errors[obj_name] = f"Could not resolve position/size of {obj.name} ({obj.__class__})."
    return errors


def _invert_property(arr: jax.Array):
    """Inverts a property array, e.g. inv_permittivities of shape (num_comp, *grid_shape)."""
    num_components = arr.shape[0]
    assert arr.ndim == 4 and num_components in [1, 3, 9], (
        f"Expecting shape (num_comp, *grid_shape), got shape {arr.shape}"
    )

    if num_components in (1, 3):
        return 1.0 / arr
    else:
        # Full tensor inversion: move 9-component axis to the end, reshape, invert, and flatten back
        arr_reshaped = jnp.moveaxis(arr, 0, -1)
        spatial_shape = arr_reshaped.shape[:-1]
        matrices = arr_reshaped.reshape(*spatial_shape, 3, 3)
        inv_matrices = jnp.linalg.inv(matrices)
        inv_flattened = inv_matrices.reshape(*spatial_shape, 9)
        return jnp.moveaxis(inv_flattened, -1, 0)
