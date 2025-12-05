from typing import Any, Sequence

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.sharding import create_named_sharded_matrix
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, ParameterContainer
from fdtdx.materials import (
    compute_allowed_electric_conductivities,
    compute_allowed_magnetic_conductivities,
    compute_allowed_permeabilities,
    compute_allowed_permittivities,
)
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


def place_objects(
    object_list: list[SimulationObject],
    config: SimulationConfig,
    constraints: Sequence[(PositionConstraint | SizeConstraint | SizeExtensionConstraint | GridCoordinateConstraint)],
    key: jax.Array,
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
        key (jax.Array): JAX random key for initialization.

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

    # Step 1: Resolve constraints into grid slices
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

    # Step 4: Place objects on grid based on resolved slice tuples
    placed_objects = []
    for name, slice_tuple in resolved_slices.items():
        if name == volume_obj.name:
            continue
        obj = object_map[name]
        key, subkey = jax.random.split(key)
        placed_objects.append(
            obj.place_on_grid(
                grid_slice_tuple=slice_tuple,
                config=config,
                key=subkey,
            )
        )

    # Step 5: Place volume first (index 0)
    key, subkey = jax.random.split(key)
    placed_objects.insert(
        0,
        volume_obj.place_on_grid(
            grid_slice_tuple=resolved_slices[volume_obj.name],
            config=config,
            key=subkey,
        ),
    )

    # Step 6: Create object container
    objects_container = ObjectContainer(
        object_list=placed_objects,
        volume_idx=0,
    )

    # Step 7: Initialize parameters and arrays
    params = _init_params(objects=objects_container, key=key)
    arrays, config, info = _init_arrays(objects=objects_container, config=config)

    # Step 8: Update object configs with compiled configuration
    new_object_list = []
    for o in objects_container.objects:
        o = o.aset("_config", config)
        new_object_list.append(o)

    objects_container = ObjectContainer(
        object_list=new_object_list,
        volume_idx=0,
    )

    return objects_container, arrays, params, config, info


def apply_params(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    params: ParameterContainer,
    key: jax.Array,
    **transform_kwargs,
) -> tuple[ArrayContainer, ObjectContainer, dict[str, Any]]:
    """Applies parameters to devices and updates source states.

    Args:
        arrays (ArrayContainer): Container with field arrays
        objects (ObjectContainer): Container with simulation objects
        params (ParameterContainer): Container with device parameters
        key (jax.Array): JAX random key for source updates
        **transform_kwargs: Keyword arguments passed to the parameter transformation.
    Returns:
        tuple[ArrayContainer, ObjectContainer, dict[str, Any]]: A tuple containing:
            - Updated ArrayContainer with applied device parameters
            - Updated ObjectContainer with new source states
            - Dictionary with parameter application info
    """
    info = {}
    # apply parameter to devices
    for device in objects.devices:
        cur_material_indices = device(params[device.name], expand_to_sim_grid=True, **transform_kwargs)
        # allowed_perm_list is now list of tuples: [(εx, εy, εz), ...]
        allowed_perm_list = compute_allowed_permittivities(device.materials)
        allowed_perm_array = jnp.asarray(allowed_perm_list)  # shape: (num_materials, 3)

        # Process each component separately
        for i in range(3):
            if device.output_type == ParameterType.CONTINUOUS:
                # Linear interpolation between two materials for component i
                first_term = (1 - cur_material_indices) * (1.0 / allowed_perm_array[0, i])
                second_term = cur_material_indices * (1.0 / allowed_perm_array[1, i])
                new_perm_slice = first_term + second_term
            else:
                # Discrete material selection for component i
                component_values = allowed_perm_array[:, i][cur_material_indices.astype(jnp.int32)]
                component_values = straight_through_estimator(cur_material_indices, component_values)
                new_perm_slice = 1 / component_values
            # Update component i of inv_permittivities array
            new_perm = arrays.inv_permittivities.at[i, *device.grid_slice].set(new_perm_slice)
            arrays = arrays.at["inv_permittivities"].set(new_perm)

    # apply random key to sources
    new_objects = []
    for obj in objects.object_list:
        key, subkey = jax.random.split(key)
        new_obj = obj.apply(
            key=subkey,
            inv_permittivities=jax.lax.stop_gradient(arrays.inv_permittivities),
            inv_permeabilities=jax.lax.stop_gradient(arrays.inv_permeabilities),
        )
        new_objects.append(new_obj)
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
    ext_shape = (3, *volume_shape)
    E = create_named_sharded_matrix(
        ext_shape,
        sharding_axis=1,
        value=0.0,
        dtype=config.dtype,
        backend=config.backend,
    )
    H = create_named_sharded_matrix(
        ext_shape,
        value=0.0,
        dtype=config.dtype,
        sharding_axis=1,
        backend=config.backend,
    )

    # permittivity - shape (3, Nx, Ny, Nz) for anisotropic materials
    # First dimension is for (x, y, z) components
    shape_params = (3, *volume_shape)
    inv_permittivities = create_named_sharded_matrix(
        shape_params,
        value=0.0,
        dtype=config.dtype,
        sharding_axis=1,
        backend=config.backend,
    )

    # permeability - shape (3, Nx, Ny, Nz) for anisotropic materials
    # Use scalar 1.0 only if all materials are non-magnetic AND isotropic
    if objects.all_objects_non_magnetic:
        inv_permeabilities = 1.0
    else:
        inv_permeabilities = create_named_sharded_matrix(
            shape_params,
            value=0.0,
            dtype=config.dtype,
            sharding_axis=1,
            backend=config.backend,
        )

    # electric conductivity - shape (3, Nx, Ny, Nz) for anisotropic materials
    electric_conductivity = None
    if not objects.all_objects_non_electrically_conductive:
        electric_conductivity = create_named_sharded_matrix(
            shape_params,
            value=0.0,
            dtype=config.dtype,
            sharding_axis=1,
            backend=config.backend,
        )

    # magnetic conductivity - shape (3, Nx, Ny, Nz) for anisotropic materials
    magnetic_conductivity = None
    if not objects.all_objects_non_magnetically_conductive:
        magnetic_conductivity = create_named_sharded_matrix(
            shape_params,
            value=0.0,
            dtype=config.dtype,
            sharding_axis=1,
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
            # Material properties are tuples (εx, εy, εz)
            # Arrays have shape (3, Nx, Ny, Nz) where first axis is component
            # Set each component: inv_permittivities[i, x, y, z] = 1/ε_i
            for i in range(3):
                inv_perm_value = 1.0 / o.material.permittivity[i]
                inv_permittivities = inv_permittivities.at[i, *o.grid_slice].set(inv_perm_value)
            if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
                # inv_permeabilities is a jax.Array with shape (3, Nx, Ny, Nz)
                inv_perm_array: jax.Array = inv_permeabilities
                for i in range(3):
                    inv_perm_value = 1.0 / o.material.permeability[i]
                    inv_perm_array = inv_perm_array.at[i, *o.grid_slice].set(inv_perm_value)
                inv_permeabilities = inv_perm_array

            if electric_conductivity is not None:
                # scale by grid size
                for i in range(3):
                    cond = o.material.electric_conductivity[i] * config.resolution
                    electric_conductivity = electric_conductivity.at[i, *o.grid_slice].set(cond)

            if magnetic_conductivity is not None:
                # scale by grid size
                for i in range(3):
                    cond = o.material.magnetic_conductivity[i] * config.resolution
                    magnetic_conductivity = magnetic_conductivity.at[i, *o.grid_slice].set(cond)
        elif isinstance(o, (StaticMultiMaterialObject)):
            indices = o.get_material_mapping()
            mask = o.get_voxel_mask_for_shape()

            # compute_allowed_permittivities returns list of tuples: [(εx, εy, εz), ...]
            # Convert to array of shape (num_materials, 3) then transpose to (3, num_materials)
            allowed_perms = jnp.asarray(compute_allowed_permittivities(o.materials))  # shape: (num_materials, 3)
            allowed_inv_perms = 1 / allowed_perms  # shape: (num_materials, 3)

            # For each component i: inv_permittivities[i, :, :, :] needs updating
            for i in range(3):
                # Select component i from all materials: allowed_inv_perms[:, i]
                component_values = allowed_inv_perms[:, i][indices]  # shape: grid_slice shape
                diff = component_values - inv_permittivities[i, *o.grid_slice]
                inv_permittivities = inv_permittivities.at[i, *o.grid_slice].add(mask * diff)

            if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
                # inv_permeabilities is a jax.Array with shape (3, Nx, Ny, Nz)
                inv_perm_array: jax.Array = inv_permeabilities
                allowed_perms = jnp.asarray(compute_allowed_permeabilities(o.materials))  # shape: (num_materials, 3)
                allowed_inv_perms = 1 / allowed_perms

                for i in range(3):
                    component_values = allowed_inv_perms[:, i][indices]
                    diff = component_values - inv_perm_array[i, *o.grid_slice]
                    inv_perm_array = inv_perm_array.at[i, *o.grid_slice].add(mask * diff)
                inv_permeabilities = inv_perm_array

            if electric_conductivity is not None:
                allowed_conds = jnp.asarray(compute_allowed_electric_conductivities(o.materials))  # shape: (num_materials, 3)

                for i in range(3):
                    component_values = allowed_conds[:, i][indices] * config.resolution
                    diff = component_values - electric_conductivity[i, *o.grid_slice]
                    electric_conductivity = electric_conductivity.at[i, *o.grid_slice].add(mask * diff)

            if magnetic_conductivity is not None:
                allowed_conds = jnp.asarray(compute_allowed_magnetic_conductivities(o.materials))  # shape: (num_materials, 3)

                for i in range(3):
                    component_values = allowed_conds[:, i][indices] * config.resolution
                    diff = component_values - magnetic_conductivity[i, *o.grid_slice]
                    magnetic_conductivity = magnetic_conductivity.at[i, *o.grid_slice].add(mask * diff)
        else:
            raise Exception(f"Unknown object type: {o}")

    # detector states
    detector_states = {}
    for d in objects.detectors:
        detector_states[d.name] = d.init_state()

    # boundary states
    boundary_states = {}
    for boundary in objects.boundary_objects:
        boundary_states[boundary.name] = boundary.init_state()

    # interfaces
    recording_state = None
    if config.gradient_config is not None and config.gradient_config.recorder is not None:
        input_shape_dtypes = {}
        for boundary in objects.pml_objects:
            cur_shape = boundary.interface_grid_shape()
            extended_shape = (3, *cur_shape)
            input_shape_dtypes[f"{boundary.name}_E"] = jax.ShapeDtypeStruct(shape=extended_shape, dtype=config.dtype)
            input_shape_dtypes[f"{boundary.name}_H"] = jax.ShapeDtypeStruct(shape=extended_shape, dtype=config.dtype)
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

    arrays = ArrayContainer(
        E=E,
        H=H,
        inv_permittivities=inv_permittivities,
        inv_permeabilities=inv_permeabilities,
        boundary_states=boundary_states,
        detector_states=detector_states,
        recording_state=recording_state,
        electric_conductivity=electric_conductivity,
        magnetic_conductivity=magnetic_conductivity,
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
    objects: list[SimulationObject],
    constraints: Sequence[PositionConstraint | SizeConstraint | SizeExtensionConstraint | GridCoordinateConstraint],
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

    # Apply constraints iteratively
    resolved, errors = _apply_constraints_iteratively(
        objects=objects,
        constraints=constraints,
        config=config,
        max_iter=max_iter,
    )

    # Convert shape_dict and slice_dict from object references to object names
    resolved_slices = {}
    for obj_name, slice_list in resolved.items():
        resolved_slices[obj_name] = tuple([(axis_slice_list[0], axis_slice_list[1]) for axis_slice_list in slice_list])

    return resolved_slices, errors


def _check_objects_names_from_constraints(
    constraints: Sequence[PositionConstraint | SizeConstraint | SizeExtensionConstraint | GridCoordinateConstraint],
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
    constraints: Sequence[PositionConstraint | SizeConstraint | SizeExtensionConstraint | GridCoordinateConstraint],
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

    # iterate
    for _ in range(max_iter):
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
            if obj.partial_real_shape[axis] is not None:
                cur_grid_shape = round(
                    obj.partial_real_shape[axis] / config.resolution  # type: ignore
                )
                shape_dict[obj_name][axis] = cur_grid_shape
    return shape_dict


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
):
    obj_name = constraint.object
    obj = object_map[obj_name]
    resolved_something = False
    for axis_idx, axis in enumerate(constraint.axes):
        cur_size = constraint.coordinates[axis_idx]
        b_idx = 0 if constraint.sides[axis_idx] == "-" else 1
        if slice_dict[obj_name][axis][b_idx] is None:
            slice_dict[obj_name][axis][b_idx] = cur_size
            resolved_something = True
        elif slice_dict[obj_name][axis][b_idx] != cur_size:
            raise Exception(
                f"Inconsistent grid coordinates for object: "
                f"{slice_dict[obj_name][axis][b_idx]} != {cur_size} for {axis=} {obj.name} ({obj.__class__}). "
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
        cur_size = round(constraint.coordinates[axis_idx] / config.resolution)
        b_idx = 0 if constraint.sides[axis_idx] == "-" else 1
        if slice_dict[obj_name][axis][b_idx] is None:
            slice_dict[obj_name][axis][b_idx] = cur_size
            resolved_something = True
        elif slice_dict[obj_name][axis][b_idx] != cur_size:
            raise Exception(
                f"Inconsistent grid coordinates for object: "
                f"{slice_dict[obj_name][axis][b_idx]} != {cur_size} for {axis=} {obj.name} ({obj.__class__}). "
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
    obj_name, other_name = constraint.object, constraint.other_object
    obj = object_map[obj_name]
    resolved_something = False
    # go through axes of constraint
    for axis_idx, axis in enumerate(constraint.axes):
        grid_margin = constraint.grid_margins[axis_idx]
        real_margin = constraint.margins[axis_idx]
        # check if other knows their position
        other_b0, other_b1 = slice_dict[other_name][axis]
        if other_b0 is None or other_b1 is None:
            continue
        # check if object knows their size
        object_size = shape_dict[obj_name][axis]
        if object_size is None:
            continue
        # calculate anchor of other
        other_pos = constraint.other_object_positions[axis_idx]
        other_midpoint = (other_b1 + other_b0) / 2
        factor = (other_b1 - other_b0) / 2
        other_offset = 0
        if grid_margin is not None:
            other_offset += grid_margin
        if real_margin is not None:
            other_offset += real_margin / config.resolution
        other_anchor = other_midpoint + factor * other_pos + other_offset
        # calculate position of object
        obj_pos = constraint.object_positions[axis_idx]
        obj_factor = object_size / 2
        object_midpoint = other_anchor - obj_pos * obj_factor
        b0 = round(object_midpoint - obj_factor)
        # Important: do not round twice to exactly preserve object size
        b1 = b0 + object_size
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
):
    """Resolve a size relationship between objects."""
    obj_name, other_name = constraint.object, constraint.other_object
    obj = object_map[obj_name]
    resolved_something = False
    # iterate through axes of the constraint
    for axis_idx, axis in enumerate(constraint.axes):
        other_axes = constraint.other_axes[axis_idx]
        # check if other object knows their shape
        other_shape = shape_dict[other_name][other_axes]
        if other_shape is None:
            continue
        # calculate objects shape
        proportion = constraint.proportions[axis_idx]
        grid_offset = 0
        if constraint.grid_offsets[axis_idx] is not None:
            grid_offset += constraint.grid_offsets[axis_idx]
        if constraint.offsets[axis_idx] is not None:
            grid_offset += constraint.offsets[axis_idx] / config.resolution
        object_shape = round(other_shape * proportion + grid_offset)
        # update or check consistency
        if shape_dict[obj_name][axis] is None:
            shape_dict[obj_name][axis] = object_shape
            resolved_something = True
        elif shape_dict[obj_name][axis] != object_shape:
            raise Exception(
                "Inconsistent grid shape for object: ",
                f"{shape_dict[obj_name][axis]} != {object_shape} for {axis=}, {obj.name} ({obj.__class__}). ",
                "Please check if there are multiple constraints or sizes specified for the object.",
            )
    return resolved_something, shape_dict


def _apply_size_extension_constraint(
    constraint: SizeExtensionConstraint,
    object_map: dict[str, SimulationObject],
    config: SimulationConfig,
    slice_dict: dict[str, list[list[int | None]]],
    volume_name: str,
):
    obj_name, other_name = constraint.object, constraint.other_object
    obj = object_map[obj_name]
    dir_idx = 0 if constraint.direction == "-" else 1
    resolved_something = False
    # calculate anchor point
    if other_name is not None:
        # check if other knows their position
        other_b0, other_b1 = slice_dict[other_name][constraint.axis]
        if other_b0 is None or other_b1 is None:
            return False, slice_dict
        # calculate anchor of other position
        other_midpoint = (other_b1 + other_b0) / 2
        factor = (other_b1 - other_b0) / 2
        other_offset = 0
        if constraint.grid_offset is not None:
            other_offset += constraint.grid_offset
        if constraint.offset is not None:
            other_offset += constraint.offset / config.resolution
        other_anchor = round(other_midpoint + factor * constraint.other_position + other_offset)
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
    constraints: Sequence[PositionConstraint | SizeConstraint | SizeExtensionConstraint | GridCoordinateConstraint],
    object_map: dict[str, SimulationObject],
    slice_dict: dict[str, list[list[int | None]]],
    shape_dict: dict[str, list[int | None]],
    volume_name: str,
):
    # Extend objects to infinity, which fulfull the properties:
    # - do not already have a specified shape
    # - are not object in a size constraint/extend_to
    resolved_something = False
    for axis in range(3):
        extension_obj = [(o, 0) for o in object_map.keys()] + [(o, 1) for o in object_map.keys()]
        for c in constraints:
            if isinstance(c, SizeConstraint) and axis in c.axes:
                if (c.object, 0) in extension_obj:
                    extension_obj.remove((c.object, 0))
                if (c.object, 1) in extension_obj:
                    extension_obj.remove((c.object, 1))
            if isinstance(c, SizeExtensionConstraint) and axis == c.axis:
                direction = 0 if c.direction == "-" else 1
                if (c.object, direction) in extension_obj:
                    extension_obj.remove((c.object, direction))
        for o in object_map.keys():
            if shape_dict[o][axis] is not None:
                if (o, 0) in extension_obj:
                    extension_obj.remove((o, 0))
                if (o, 1) in extension_obj:
                    extension_obj.remove((o, 1))
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
