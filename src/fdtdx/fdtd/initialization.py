from typing import Any, Optional, Sequence, Tuple

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
    SimulationObject,
    SizeConstraint,
    SizeExtensionConstraint,
)
from fdtdx.objects.static_material.static import SimulationVolume, StaticMultiMaterialObject, UniformMaterialObject
from fdtdx.typing import SliceTuple3D

DEFAULT_MAX_ITER = 1000

def place_objects(
    objects: list[SimulationObject],
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
    # Step 1: Resolve constraints into grid slices
    resolved_slices, errors = resolve_object_constraints(
        objects=objects,
        constraints=constraints,
        config=config,
    )

    # Step 2: Aggregate errors and raise if needed
    failed = {name: msg for name, msg in errors.items() if msg}
    if failed:
        formatted = "\n".join(f"  - {name}: {msg}" for name, msg in failed.items())
        raise ValueError(f"Failed to resolve object constraints:\n{formatted}")

    # Step 3: Convert name → object for placement
    object_map = {obj.name: obj for obj in objects}
    volume_objects = [o for o in objects if isinstance(o, SimulationVolume)]
    if not volume_objects:
        raise ValueError("No SimulationVolume object found in the provided objects list.")
    elif len(volume_objects) > 1:
        raise ValueError(
            f"Multiple SimulationVolume objects found ({[o.name for o in volume_objects]}). "
            "There must be exactly one simulation volume."
        )

    volume_obj = volume_objects[0]

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
        allowed_perm_list = compute_allowed_permittivities(device.materials)
        if device.output_type == ParameterType.CONTINUOUS:
            first_term = (1 - cur_material_indices) * (1 / allowed_perm_list[0])
            second_term = cur_material_indices * (1 / allowed_perm_list[1])
            new_perm_slice = first_term + second_term
        else:
            new_perm_slice = jnp.asarray(allowed_perm_list)[cur_material_indices.astype(jnp.int32)]
            new_perm_slice = straight_through_estimator(cur_material_indices, new_perm_slice)
            new_perm_slice = 1 / new_perm_slice
        new_perm = arrays.inv_permittivities.at[*device.grid_slice].set(new_perm_slice)
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

    # permittivity
    shape_params = volume_shape
    inv_permittivities = create_named_sharded_matrix(
        shape_params,
        value=0.0,
        dtype=config.dtype,
        sharding_axis=1,
        backend=config.backend,
    )

    # permeability
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

    # electric conductivity
    electric_conductivity = None
    if not objects.all_objects_non_electrically_conductive:
        electric_conductivity = create_named_sharded_matrix(
            shape_params,
            value=0.0,
            dtype=config.dtype,
            sharding_axis=1,
            backend=config.backend,
        )

    # magnetic conductivity
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
            inv_permittivities = inv_permittivities.at[*o.grid_slice].set(1 / o.material.permittivity)
            if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
                inv_permeabilities = inv_permeabilities.at[*o.grid_slice].set(1 / o.material.permeability)
            if electric_conductivity is not None:
                # scale by grid size
                cond = o.material.electric_conductivity * config.resolution
                electric_conductivity = electric_conductivity.at[*o.grid_slice].set(cond)
            if magnetic_conductivity is not None:
                # scale by grid size
                cond = o.material.magnetic_conductivity * config.resolution
                magnetic_conductivity = magnetic_conductivity.at[*o.grid_slice].set(cond)
        elif isinstance(o, (StaticMultiMaterialObject)):
            indices = o.get_material_mapping()
            mask = o.get_voxel_mask_for_shape()

            allowed_inv_perms = 1 / jnp.asarray(compute_allowed_permittivities(o.materials))
            diff = allowed_inv_perms[indices] - inv_permittivities[*o.grid_slice]
            inv_permittivities = inv_permittivities.at[*o.grid_slice].add(mask * diff)

            if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
                allowed_inv_perms = 1 / jnp.asarray(compute_allowed_permeabilities(o.materials))
                diff = allowed_inv_perms[indices] - inv_permeabilities[*o.grid_slice]
                inv_permeabilities = inv_permeabilities.at[*o.grid_slice].add(mask * diff)

            if electric_conductivity is not None:
                allowed_conds = jnp.asarray(compute_allowed_electric_conductivities(o.materials))
                update = allowed_conds[indices] * config.resolution
                diff = update - electric_conductivity[*o.grid_slice]
                electric_conductivity = electric_conductivity.at[*o.grid_slice].add(mask * diff)

            if magnetic_conductivity is not None:
                allowed_conds = jnp.asarray(compute_allowed_magnetic_conductivities(o.materials))
                update = allowed_conds[indices] * config.resolution
                diff = update - magnetic_conductivity[*o.grid_slice]
                magnetic_conductivity = magnetic_conductivity.at[*o.grid_slice].add(mask * diff)
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
            cur_shape = boundary.boundary_interface_grid_shape()
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
        constraints: Sequence[
            PositionConstraint | SizeConstraint | SizeExtensionConstraint | GridCoordinateConstraint
            ],
        config: SimulationConfig,
) -> tuple[dict, dict]:
    """Resolve object constraints into grid slices and shapes."""

    # Convert objects list to object_map dictionary
    object_map = {}
    for obj in objects:
        object_map[obj.name] = obj
        print(f"Added to object_map: {obj.name} -> {type(obj).__name__}")  # Debug

    # Initialize shape_dict and slice_dict with object references as keys
    shape_dict = {}
    slice_dict = {}
    for obj in objects:
        shape_dict[obj] = [None, None, None]
        slice_dict[obj] = [[None, None], [None, None], [None, None]]

    errors = {obj.name: None for obj in objects}

    # Debug: print constraints before applying
    print(f"Number of constraints: {len(constraints)}")
    for i, c in enumerate(constraints):
        obj_field = getattr(c, "object", None)
        other_field = getattr(c, "other_object", None)
        print(f"Constraint {i}: object={type(obj_field)}, other_object={type(other_field)}")

    # Apply constraints iteratively
    _apply_constraints_iteratively(object_map, constraints, shape_dict, slice_dict, errors)

    # Convert shape_dict and slice_dict from object references to object names
    resolved_slices = {}
    for obj, slices in slice_dict.items():
        resolved_slices[obj.name] = tuple(
            (start, end) for start, end in slices
        )

    return resolved_slices, errors


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------


def _collect_objects_from_constraints(constraints, object_map):
    """Collect object names mentioned in constraints and verify they exist."""
    all_names = set()
    for c in constraints:
        for name in [getattr(c, "object_name", None), getattr(c, "other_name", None)]:
            if name and name not in object_map:
                raise ValueError(f"Unknown object name in constraint: {name}")
            if name:
                all_names.add(name)
    return list(all_names)


def _initialize_shape_dicts(objects):
    """Initialize empty shape and slice dictionaries."""
    shape_dict = {o: [None, None, None] for o in objects}
    slice_dict = {o: [[None, None], [None, None], [None, None]] for o in objects}
    return shape_dict, slice_dict


def _resolve_static_shapes(objects, shape_dict, config, resolution, errors):
    """Fill in static or directly defined shapes."""
    for o in objects:
        for axis in range(3):
            if getattr(o, "partial_grid_shape", [None] * 3)[axis] is not None:
                shape_dict[o][axis] = o.partial_grid_shape[axis]
            elif getattr(o, "partial_real_shape", [None] * 3)[axis] is not None:
                try:
                    cur_grid_shape = round(o.partial_real_shape[axis] / resolution)
                    shape_dict[o][axis] = cur_grid_shape
                except Exception as e:
                    errors[o.name] = f"Invalid real shape at axis {axis}: {e}"


def _apply_constraints_iteratively(
        object_map: dict[str, SimulationObject],  # This should now be a proper dictionary
        constraints: Sequence[
            PositionConstraint | SizeConstraint | SizeExtensionConstraint | GridCoordinateConstraint
            ],
        shape_dict: dict["SimulationObject", list[int | None]],
        slice_dict: dict["SimulationObject", list[list[int | None]]],
        errors: dict[str, Optional[str]],
        max_iter: int = DEFAULT_MAX_ITER,
) -> None:
    """
    Iteratively apply all constraints until shapes and positions converge.
    """

    for iteration in range(max_iter):
        changed = False

        for c in constraints:
            # Get object references from constraints - handle mixed types
            obj = None
            other = None

            # Handle the 'object' field
            obj_field = getattr(c, "object", None)
            if isinstance(obj_field, str):
                # If it's a string (name), look it up
                obj = object_map.get(obj_field)
            elif obj_field and hasattr(obj_field, 'name'):
                # If it's an object instance, look it up by name
                obj = object_map.get(obj_field.name)

            # Handle the 'other_object' field
            other_field = getattr(c, "other_object", None)
            if isinstance(other_field, str):
                # If it's a string (name), look it up
                other = object_map.get(other_field)
            elif other_field and hasattr(other_field, 'name'):
                # If it's an object instance, look it up by name
                other = object_map.get(other_field.name)

            if obj is None:
                # Provide meaningful error message
                obj_id = "unknown"
                if isinstance(obj_field, str):
                    obj_id = obj_field
                elif hasattr(obj_field, 'name'):
                    obj_id = obj_field.name
                errors[obj_id] = f"Unknown object '{obj_id}' in constraint."
                continue

            handled = False
            try:
                if isinstance(c, PositionConstraint):
                    changed |= _apply_position_constraint(c, obj, other, shape_dict, slice_dict)
                    handled = True

                elif isinstance(c, SizeConstraint):
                    changed |= _apply_size_constraint(c, obj, other, shape_dict)
                    handled = True

                elif isinstance(c, SizeExtensionConstraint):
                    changed |= _apply_size_extension_constraint(c, obj, other, shape_dict)
                    handled = True

                elif isinstance(c, GridCoordinateConstraint):
                    changed |= _apply_grid_coordinate_constraint(c, obj, shape_dict, slice_dict)
                    handled = True

                if not handled:
                    raise ValueError(f"Unknown constraint type: {type(c).__name__}")

            except Exception as e:
                obj_id = obj.name if hasattr(obj, 'name') else "unknown"
                errors[obj_id] = f"Error applying {type(c).__name__}: {e}"

        # Check if everything is resolved
        all_done = True
        for o in object_map.values():
            if any(v is None for v in shape_dict[o]) or any(any(v is None for v in pair) for pair in slice_dict[o]):
                all_done = False
                break

        if all_done:
            break

        if not changed:
            for o in object_map.values():
                if errors.get(o.name) is None:
                    errors[o.name] = "Unresolved: constraints could not converge."
            break
    else:
        for o in object_map.values():
            if errors.get(o.name) is None:
                errors[o.name] = "Unresolved after max iterations."


def _apply_position_constraint(
        constraint: PositionConstraint,
        obj: SimulationObject,
        other: SimulationObject | None,
        shape_dict: dict["SimulationObject", list[int | None]],
        slice_dict: dict["SimulationObject", list[list[int | None]]],
) -> bool:
    """Apply a position constraint between two objects."""
    if other is None:
        return False

    changed = False
    # Use constraint.axes instead of constraint.axis
    for i, axis in enumerate(constraint.axes):
        # Apply position constraint logic here
        # This is a simplified version - you'll need to implement the actual positioning logic
        obj_positions = constraint.object_positions[i] if i < len(constraint.object_positions) else 0
        other_positions = constraint.other_object_positions[i] if i < len(constraint.other_object_positions) else 0

        # Basic position alignment logic
        if shape_dict[obj][axis] is not None and shape_dict[other][axis] is not None:
            # Calculate target position based on constraint
            # This is simplified
            target_position = 0  # Placeholder
            current_position = slice_dict[obj][axis][0] or 0

            if current_position != target_position:
                slice_dict[obj][axis] = [target_position, target_position + shape_dict[obj][axis]]
                changed = True

    return changed


def _apply_size_constraint(c, obj, other, shape_dict) -> bool:
    """Resolve a size relationship between objects."""
    axis = c.axis
    ratio = getattr(c, "ratio", 1.0)
    changed = False

    if other and shape_dict[other][axis] is not None:
        new_size = int(round(shape_dict[other][axis] * ratio))
        if shape_dict[obj][axis] != new_size:
            shape_dict[obj][axis] = new_size
            changed = True
    return changed


def _apply_size_extension_constraint(c, obj, other, shape_dict) -> bool:
    """Extend object size relative to another object's dimension."""
    axis = c.axis
    extension = getattr(c, "extension", 0)
    changed = False

    if other and shape_dict[other][axis] is not None:
        new_size = shape_dict[other][axis] + extension
        if shape_dict[obj][axis] != new_size:
            shape_dict[obj][axis] = new_size
            changed = True
    return changed


def _apply_grid_coordinate_constraint(
    c,
    obj,
    shape_dict,
    slice_dict,
    config=None,
) -> bool:
    """Fix an object's grid coordinate directly using its center position.

    The coordinate system origin is at the center of the simulation volume.

    Args:
        c: GridCoordinateConstraint with attribute `center` (float)
        obj: SimulationObject to modify
        shape_dict: Dict mapping object → [sx, sy, sz] (grid sizes)
        slice_dict: Dict mapping object → [[x0, x1], [y0, y1], [z0, z1]]
        config: SimulationConfig (optional, used to determine grid size)

    Returns:
        bool: True if object position or shape changed
    """
    axis = c.axis
    center = getattr(c, "center", None)
    changed = False

    # Determine total grid size (so we can translate coordinates)
    total_size = None
    if config is not None and hasattr(config, "grid_shape"):
        total_size = config.grid_shape[axis]

    # --- Convert center coordinate to grid indices ---
    # User provides center relative to simulation origin (0 at volume center)
    # So we map [-Nx/2, +Nx/2] → [0, Nx] by shifting by total_size / 2
    if total_size is not None and center is not None:
        center_idx = int(round(center + total_size / 2))
    else:
        center_idx = None

    # --- Compute start / end indices from center and object size ---
    cur_size = shape_dict[obj][axis]
    if cur_size is None:
        # Can't compute slice without size yet
        return False

    if center_idx is not None:
        start_idx = int(round(center_idx - cur_size / 2))
        end_idx = start_idx + cur_size

        if slice_dict[obj][axis] != [start_idx, end_idx]:
            slice_dict[obj][axis] = [start_idx, end_idx]
            changed = True

    return changed


def _apply_single_constraint(constraint, shape_dict, slice_dict):
    """Stub for applying one constraint; extend as needed."""
    # This is where PositionConstraint, SizeConstraint, etc. logic should go.
    # For clarity, each constraint type could have its own helper like:
    # _apply_position_constraint, _apply_size_constraint, etc.
    pass


def _finalize_resolution(objects, shape_dict, slice_dict, errors):
    """Convert lists into SliceTuple3D and fill None for unresolved entries."""
    resolved: dict[str, SliceTuple3D | None] = {}
    for o in objects:
        if errors[o.name] is not None:
            resolved[o.name] = None
            continue
        if any(v is None for v in shape_dict[o]) or any(any(v is None for v in pair) for pair in slice_dict[o]):
            resolved[o.name] = None
            errors[o.name] = "Incomplete constraint resolution"
        else:
            # Explicitly create the 3D slice tuple structure
            x_slice = (slice_dict[o][0][0], slice_dict[o][0][1])
            y_slice = (slice_dict[o][1][0], slice_dict[o][1][1])
            z_slice = (slice_dict[o][2][0], slice_dict[o][2][1])
            resolved[o.name] = (x_slice, y_slice, z_slice)
    return resolved
