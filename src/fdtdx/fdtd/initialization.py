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
from fdtdx.objects.static_material.static import StaticMultiMaterialObject, UniformMaterialObject
from fdtdx.typing import SliceTuple3D


def place_objects(
    objects: list["SimulationObject"],
    config: "SimulationConfig",
    constraints: Sequence[
        ("PositionConstraint" | "SizeConstraint" | "SizeExtensionConstraint" | "GridCoordinateConstraint")
    ],
    key: "jax.Array",
) -> tuple[
    "ObjectContainer",
    "ArrayContainer",
    "ParameterContainer",
    "SimulationConfig",
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
    resolved_slices, errors = _resolve_object_constraints(
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
    volume_obj = next((o for o in objects if o.name.lower() == "volume"), None)
    if not volume_obj:
        raise ValueError("No volume object found in objects list.")

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


def _resolve_object_constraints(
    objects: list["SimulationObject"],
    constraints: Sequence[
        "PositionConstraint" | "SizeConstraint" | "SizeExtensionConstraint" | "GridCoordinateConstraint"
    ],
    config: "SimulationConfig",
) -> Tuple[dict[str, "SliceTuple3D" | None], dict[str, Optional[str]]]:
    """
    Cleaned version of `_resolve_object_constraints`.

    Resolves positioning and sizing constraints between simulation objects.
    Returns dictionaries for resolved slices and error messages.

    Args:
        objects: List of simulation objects to resolve.
        constraints: List of constraints defining object relations.
        config: Simulation configuration object.

    Returns:
        resolved: dict mapping object names to SliceTuple3D or None
        errors: dict mapping object names to error messages (None if OK)
    """
    resolution = config.resolution

    # Initialize structures
    errors: dict[str, Optional[str]] = {obj.name: None for obj in objects}

    all_objects = _collect_objects_from_constraints(objects, constraints)
    shape_dict, slice_dict = _initialize_shape_dicts(all_objects)

    _resolve_static_shapes(all_objects, shape_dict, config, resolution, errors)
    _apply_constraints_iteratively(all_objects, constraints, shape_dict, slice_dict, errors)

    resolved = _finalize_resolution(all_objects, shape_dict, slice_dict, errors)
    return resolved, errors


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
    object_map: dict[str, "SimulationObject"],
    constraints: Sequence[
        "PositionConstraint" | "SizeConstraint" | "SizeExtensionConstraint" | "GridCoordinateConstraint"
    ],
    shape_dict: dict["SimulationObject", list[int | None]],
    slice_dict: dict["SimulationObject", list[list[int | None]]],
    errors: dict[str, Optional[str]],
) -> None:
    """
    Iteratively apply all constraints until shapes and positions converge.

    Args:
        object_map: Dict mapping object names to SimulationObject instances.
        constraints: Sequence of constraint instances referencing object names.
        shape_dict: Dict tracking (x, y, z) shape values in grid units.
        slice_dict: Dict tracking grid slice start/end positions for each axis.
        errors: Dict collecting user-friendly error messages.
    """
    MAX_ITER = 1000
    stable_iterations = 0

    for iteration in range(MAX_ITER):
        changed = False

        for c in constraints:
            obj_name = getattr(c, "object_name", None)
            other_name = getattr(c, "other_name", None)
            obj = object_map.get(obj_name)
            other = object_map.get(other_name) if other_name else None

            if obj is None:
                errors[obj_name] = f"Unknown object '{obj_name}' in constraint."
                continue

            try:
                if isinstance(c, PositionConstraint):
                    changed |= _apply_position_constraint(c, obj, other, shape_dict, slice_dict)

                elif isinstance(c, SizeConstraint):
                    changed |= _apply_size_constraint(c, obj, other, shape_dict)

                elif isinstance(c, SizeExtensionConstraint):
                    changed |= _apply_size_extension_constraint(c, obj, other, shape_dict)

                elif isinstance(c, GridCoordinateConstraint):
                    changed |= _apply_grid_coordinate_constraint(c, obj, shape_dict, slice_dict)

            except Exception as e:
                errors[obj_name] = f"Error applying {type(c).__name__}: {e}"

        # Check if everything is resolved
        all_done = True
        for o in object_map.values():
            if any(v is None for v in shape_dict[o]) or any(any(v is None for v in pair) for pair in slice_dict[o]):
                all_done = False
                break

        if all_done:
            break

        # Stop early if no changes in several iterations (stuck)
        if not changed:
            stable_iterations += 1
            if stable_iterations > 10:
                for o in object_map.values():
                    if errors[o.name] is None:
                        errors[o.name] = "Unresolved: constraints could not converge."
                break
        else:
            stable_iterations = 0
    else:
        # Exceeded maximum iterations
        for o in object_map.values():
            if errors[o.name] is None:
                errors[o.name] = "Unresolved after max iterations."


def _apply_position_constraint(c, obj, other, shape_dict, slice_dict) -> bool:
    """Resolve a position constraint along a given axis."""
    axis = c.axis
    offset = getattr(c, "offset", 0)
    changed = False

    if other:
        other_slice = slice_dict[other][axis]
        if all(v is not None for v in other_slice):
            start = other_slice[1] + offset
            end = start + (shape_dict[obj][axis] or 0)
            if slice_dict[obj][axis] != [start, end]:
                slice_dict[obj][axis] = [start, end]
                changed = True
    else:
        # Absolute position (e.g., grid start)
        start = offset
        end = start + (shape_dict[obj][axis] or 0)
        if slice_dict[obj][axis] != [start, end]:
            slice_dict[obj][axis] = [start, end]
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


def _apply_grid_coordinate_constraint(c, obj, shape_dict, slice_dict) -> bool:
    """Fix an object's grid coordinate directly."""
    axis = c.axis
    start = getattr(c, "start", None)
    end = getattr(c, "end", None)
    changed = False

    if start is not None and slice_dict[obj][axis][0] != start:
        slice_dict[obj][axis][0] = start
        changed = True
    if end is not None and slice_dict[obj][axis][1] != end:
        slice_dict[obj][axis][1] = end
        changed = True

    if start is not None and end is not None:
        new_size = end - start
        if shape_dict[obj][axis] != new_size:
            shape_dict[obj][axis] = new_size
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
    resolved: dict[str, "SliceTuple3D" | None] = {}
    for o in objects:
        if errors[o.name] is not None:
            resolved[o.name] = None
            continue
        if any(v is None for v in shape_dict[o]) or any(any(v is None for v in pair) for pair in slice_dict[o]):
            resolved[o.name] = None
            errors[o.name] = "Incomplete constraint resolution"
        else:
            resolved[o.name] = tuple(tuple(pair) for pair in slice_dict[o])
    return resolved
