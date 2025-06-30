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
from fdtdx.objects.static_material.static import StaticMultiMaterialObject, UniformMaterialObject
from fdtdx.typing import SliceTuple3D


def place_objects(
    volume: SimulationObject,
    config: SimulationConfig,
    constraints: Sequence[
        (
            PositionConstraint
            | SizeConstraint
            | SizeExtensionConstraint
            | GridCoordinateConstraint
            | RealCoordinateConstraint
        )
    ],
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
        volume (SimulationObject): The volume object defining the simulation boundaries
        config (SimulationConfig): The simulation configuration
        constraints (Sequence[PositionConstraint| SizeConstraint| SizeExtensionConstraint| GridCoordinateConstraint| RealCoordinateConstraint]):
            Sequence of positioning and sizing constraints for objects
        key (jax.Array): JAX random key for initialization

    Returns:
        tuple[ObjectContainer, ArrayContainer, ParameterContainer, SimulationConfig, dict[str, Any]]: A tuple containing
            - ObjectContainer with placed simulation objects
            - ArrayContainer with initialized field arrays
            - ParameterContainer with device parameters
            - Updated SimulationConfig
            - Dictionary with additional initialization info
    """
    slice_tuple_dict = _resolve_object_constraints(
        volume=volume,
        constraints=constraints,
        config=config,
    )
    obj_list = list(slice_tuple_dict.keys())

    # place objects on computed grid positions
    placed_objects = []
    for o in obj_list:
        if o == volume:
            continue
        key, subkey = jax.random.split(key)
        placed_objects.append(
            o.place_on_grid(
                grid_slice_tuple=slice_tuple_dict[o],
                config=config,
                key=subkey,
            )
        )
    key, subkey = jax.random.split(key)
    placed_objects.insert(
        0,
        volume.place_on_grid(
            grid_slice_tuple=slice_tuple_dict[volume],
            config=config,
            key=subkey,
        ),
    )

    # create container
    objects = ObjectContainer(
        object_list=placed_objects,
        volume_idx=0,
    )
    params = _init_params(
        objects=objects,
        key=key,
    )
    arrays, config, info = _init_arrays(
        objects=objects,
        config=config,
    )

    # replace config in objects with compiled config
    new_object_list = []
    for o in objects.objects:
        o = o.aset("_config", config)
        new_object_list.append(o)
    objects = ObjectContainer(
        object_list=new_object_list,
        volume_idx=0,
    )

    return objects, arrays, params, config, info


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


def _resolve_object_constraints(
    volume: SimulationObject,
    constraints: Sequence[
        (
            PositionConstraint
            | SizeConstraint
            | SizeExtensionConstraint
            | GridCoordinateConstraint
            | RealCoordinateConstraint
        )
    ],
    config: SimulationConfig,
) -> dict[SimulationObject, SliceTuple3D]:
    """Resolves positioning and sizing constraints between simulation objects.

    Iteratively resolves the constraints between objects to determine their
    final positions and sizes in the simulation grid. Handles absolute and
    relative positioning, size relationships, and grid alignments.

    Args:
        volume (SimulationObject): The volume object defining simulation boundaries
        constraints (Sequence[PositionConstraint| SizeConstraint| SizeExtensionConstraint | GridCoordinateConstraint | RealCoordinateConstraint]):
            Sequence of positioning and sizing constraints
        config (SimulationConfig): The simulation configuration

    Returns:
        dict[SimulationObject, SliceTuple3D]: Dictionary mapping objects to their resolved grid slice tuples
    """
    resolution = config.resolution
    # split constraints into seperate lists
    obj_list: list[SimulationObject] = [volume]

    # collect objects
    for c in constraints:
        if isinstance(
            c,
            (
                PositionConstraint,
                SizeConstraint,
                SizeExtensionConstraint,
            ),
        ):
            if c.other_object is not None and c.other_object not in obj_list:
                obj_list.append(c.other_object)
            if c.object not in obj_list:
                obj_list.append(c.object)
        elif isinstance(
            c,
            (
                GridCoordinateConstraint,
                RealCoordinateConstraint,
            ),
        ):
            if c.object not in obj_list:
                obj_list.append(c.object)

    # init shape and position dict
    shape_dict: dict[SimulationObject, list[int | None]] = {o: [None, None, None] for o in obj_list}
    slice_dict: dict[SimulationObject, list[list[int | None]]] = {
        o: [[None, None], [None, None], [None, None]] for o in obj_list
    }

    # calculate static shapes
    for o in obj_list:
        for axis in range(3):
            if o.partial_grid_shape[axis] is not None:
                shape_dict[o][axis] = o.partial_grid_shape[axis]
            if o.partial_real_shape[axis] is not None:
                cur_grid_shape = round(
                    o.partial_real_shape[axis] / resolution  # type: ignore
                )
                shape_dict[o][axis] = cur_grid_shape

    for axis in range(3):
        slice_dict[volume][axis][0] = 0

    # resolve constraints, break condition below
    while True:
        if all(
            [
                all([shape_dict[o][i] is not None for i in range(3)])
                and all([all([slice_dict[o][i][s] is not None for s in range(2)]) for i in range(3)])
                for o in obj_list
            ]
        ):
            # everything is resolved
            break
        # prevent infinite loop when constraints are underspecified
        resolved_something = False
        # update grid-slices based on grid shapes
        for o, s in shape_dict.items():
            for axis in range(3):
                s_axis = s[axis]
                if s_axis is None:
                    continue
                b0, b1 = slice_dict[o][axis]
                if b0 is None and b1 is None:
                    continue
                elif b0 is not None and b1 is not None:
                    if s_axis != b1 - b0:
                        raise Exception(
                            f"Inconsistent grid shape for object: {s_axis} != {b1 - b0}, {o.name} ({o.__class__})."
                        )
                elif b0 is not None:
                    slice_dict[o][axis][1] = b0 + s_axis
                    resolved_something = True
                elif b1 is not None:
                    slice_dict[o][axis][0] = b1 - s_axis
                    resolved_something = True
        # update grid-shapes based on grid-slices
        for o, b in slice_dict.items():
            s = shape_dict[o]
            for axis in range(3):
                b0, b1 = b[axis]
                s_axis = s[axis]
                if b0 is not None and b1 is not None:
                    if s_axis is None:
                        shape_dict[o][axis] = b1 - b0
                        resolved_something = True
                    elif s_axis is not None and b1 - b0 != s_axis:
                        raise Exception(
                            f"Inconsistent grid shape for object: {s_axis} != {b1 - b0}, {o.name} ({o.__class__})."
                        )
        # iterate over all constraints
        for c in constraints:
            # absolute grid coordinate constraints
            if isinstance(c, GridCoordinateConstraint):
                for axis_idx, axis in enumerate(c.axes):
                    cur_size = c.coordinates[axis_idx]
                    o = c.object
                    b_idx = 0 if c.sides[axis_idx] == "-" else 1
                    if slice_dict[o][axis][b_idx] is None:
                        slice_dict[o][axis][b_idx] = cur_size
                        resolved_something = True
                    elif slice_dict[o][axis][b_idx] != cur_size:
                        raise Exception(
                            f"Inconsistent grid coordinates for object: "
                            f"{slice_dict[o][axis][b_idx]} != {cur_size} for {axis=} {o.name} ({o.__class__}). "
                        )
            # absolute real coordinate constraints
            if isinstance(c, RealCoordinateConstraint):
                for axis_idx, axis in enumerate(c.axes):
                    cur_size = round(c.coordinates[axis_idx] / resolution)
                    o = c.object
                    b_idx = 0 if c.sides[axis_idx] == "-" else 1
                    if slice_dict[o][axis][b_idx] is None:
                        slice_dict[o][axis][b_idx] = cur_size
                        resolved_something = True
                    elif slice_dict[o][axis][b_idx] != cur_size:
                        raise Exception(
                            f"Inconsistent grid coordinates for object: "
                            f"{slice_dict[o][axis][b_idx]} != {cur_size} for {axis=} {o.name} ({o.__class__}). "
                        )
            # size constraints
            if isinstance(c, SizeConstraint):
                for axis_idx, axis in enumerate(c.axes):
                    other_axes = c.other_axes[axis_idx]
                    o, other = c.object, c.other_object
                    # check if other object knows their shape
                    other_shape = shape_dict[other][other_axes]
                    if other_shape is None:
                        continue
                    # calculate objects shape
                    proportion = c.proportions[axis_idx]
                    grid_offset = 0
                    if c.grid_offsets[axis_idx] is not None:
                        grid_offset += c.grid_offsets[axis_idx]
                    if c.offsets[axis_idx] is not None:
                        grid_offset += c.offsets[axis_idx] / resolution
                    object_shape = round(other_shape * proportion + grid_offset)
                    # update or check consistency
                    if shape_dict[o][axis] is None:
                        shape_dict[o][axis] = object_shape
                        resolved_something = True
                    elif shape_dict[o][axis] != object_shape:
                        raise Exception(
                            "Inconsistent grid shape for object: ",
                            f"{shape_dict[o][axis]} != {object_shape} for {axis=}, {o.name} ({o.__class__}). ",
                            "Please check if there are multiple constraints or sizes specified for the object.",
                        )
            # positional constraints
            if isinstance(c, PositionConstraint):
                for axis_idx, axis in enumerate(c.axes):
                    o, other = c.object, c.other_object
                    grid_margin = c.grid_margins[axis_idx]
                    real_margin = c.margins[axis_idx]
                    # check if other knows their position
                    other_b0, other_b1 = slice_dict[other][axis]
                    if other_b0 is None or other_b1 is None:
                        continue
                    # check if object knows their size
                    object_size = shape_dict[o][axis]
                    if object_size is None:
                        continue
                    # calculate anchor of other
                    other_pos = c.other_object_positions[axis_idx]
                    other_midpoint = (other_b1 + other_b0) / 2
                    factor = (other_b1 - other_b0) / 2
                    other_offset = 0
                    if grid_margin is not None:
                        other_offset += grid_margin
                    if real_margin is not None:
                        other_offset += real_margin / resolution
                    other_anchor = other_midpoint + factor * other_pos + other_offset
                    # calculate position of object
                    obj_pos = c.object_positions[axis_idx]
                    obj_factor = object_size / 2
                    object_midpoint = other_anchor - obj_pos * obj_factor
                    b0 = round(object_midpoint - obj_factor)
                    # Important: do not round twice to exactly preserve object size
                    b1 = b0 + object_size
                    # update position or check consistency
                    old_b0, old_b1 = slice_dict[o][axis]
                    if old_b0 is None:
                        slice_dict[o][axis][0] = b0
                        resolved_something = True
                    elif old_b0 != b0:
                        raise Exception(
                            f"Inconsistent grid shape (may be due to extension to infinity) at lower bound: "
                            f"{old_b0} != {b0} for {axis=}, {o.name} ({o.__class__}). "
                            f"Object has a position constraint that puts the lower boundary at {b0}, "
                            f"but the lower bound was alreay computed to be at {old_b0}. "
                            f"This could be due to a missing size constraint/specification, "
                            f"which resulted in an expansion of the object to the simulation boundary (default size) "
                            f"or another constraint on this object."
                        )
                    if old_b1 is None:
                        slice_dict[o][axis][1] = b1
                        resolved_something = True
                    elif old_b1 != b1:
                        raise Exception(
                            f"Inconsistent grid shape (may be due to extension to infinity) at lower bound: "
                            f"{old_b1} != {b1} for {axis=}, {o.name} ({o.__class__}). "
                            f"Object has a position constraint that puts the upper boundary at {b1}, "
                            f"but the lower bound was alreay computed to be at {old_b1}. "
                            f"This could be either due to a missing size constraint/specification, "
                            f"which resulted in an expansion of the object to the simulation boundary (default size) "
                            f"or another constraint on this object."
                        )
            # size extension constraints
            if isinstance(c, SizeExtensionConstraint):
                o, other = c.object, c.other_object
                dir_idx = 0 if c.direction == "-" else 1
                # calculate anchor point
                if other is not None:
                    # check if other knows their position
                    other_b0, other_b1 = slice_dict[other][c.axis]
                    if other_b0 is None or other_b1 is None:
                        continue
                    # calculate anchor of other position
                    other_midpoint = (other_b1 + other_b0) / 2
                    factor = (other_b1 - other_b0) / 2
                    other_offset = 0
                    if c.grid_offset is not None:
                        other_offset += c.grid_offset
                    if c.offset is not None:
                        other_offset += c.offset / resolution
                    other_anchor = round(other_midpoint + factor * c.other_position + other_offset)
                else:
                    # if other is not specified, extend to boundary of simulation volume
                    other_anchor = slice_dict[volume][c.axis][dir_idx]
                    if other_anchor is None:
                        raise Exception(f"This should never happen: Simulation volume not specified: {volume}")
                # update position or check consistency
                old_val = slice_dict[o][c.axis][dir_idx]
                if old_val is None:
                    slice_dict[o][c.axis][dir_idx] = other_anchor
                    resolved_something = True
                elif old_val != other_anchor:
                    raise Exception(
                        f"Inconsistent grid shape at bound {c.direction}: "
                        f"{old_val} != {other_anchor} for {c.axis=}, "
                        f"{o.name} ({o.__class__})."
                    )
        # Extend objects to infinity, which fulfull the properties:
        # - do not already have a specified shape
        # - are not object in a size constraint/extend_to
        if not resolved_something:
            for axis in range(3):
                extension_obj = [(o, 0) for o in obj_list] + [(o, 1) for o in obj_list]
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
                for o in obj_list:
                    if shape_dict[o][axis] is not None:
                        if (o, 0) in extension_obj:
                            extension_obj.remove((o, 0))
                        if (o, 1) in extension_obj:
                            extension_obj.remove((o, 1))
                a = 1
                for o, direction in extension_obj:
                    if slice_dict[o][axis][direction] is not None:
                        continue
                    resolved_something = True
                    if direction == 0:
                        slice_dict[o][axis][0] = 0
                    else:
                        slice_dict[o][axis][1] = shape_dict[volume][axis]
        # if we still have not resolved something, the object is not specified properly
        if not resolved_something:
            to_resolve_str = [
                f"{o.__class__} ({o.name}): {slice_dict[o]}"
                for o in obj_list
                if any([slice_dict[o][a][0] is None or slice_dict[o][a][1] is None for a in range(3)])
            ]
            # error message
            raise Exception(f"Could not resolve position/size of objects: \n {to_resolve_str}")
    # create slice dictionary
    result = {}
    for o, s in slice_dict.items():
        slices = []
        for a in range(3):
            s0, s1 = s[a]
            if s0 is None or s1 is None:
                raise Exception(f"This should never happen: object not specified: {o=}, {s0=}, {s1=}")
            slices.append((s0, s1))
        result[o] = tuple(slices)
    return result
