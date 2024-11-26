from typing import Any, Sequence

import jax
from fdtdx.core.config import SimulationConfig
from fdtdx.core.jax.sharding import create_named_sharded_matrix
from fdtdx.core.jax.typing import PartialGridShape3D, PartialSliceTuple3D, SliceTuple3D
from fdtdx.objects.container import ArrayContainer, ObjectContainer, ParameterContainer
from fdtdx.objects.object import PositionConstraint, SimulationObject, SizeConstraint


def place_objects(
    volume: SimulationObject,
    config: SimulationConfig,
    constraints: Sequence[PositionConstraint | SizeConstraint],
    key: jax.Array,
) -> tuple[
    ObjectContainer,
    ArrayContainer,
    ParameterContainer,
    SimulationConfig,
    dict[str, Any],
]:
    obj_list, pos_list, size_list = _gather_objects(volume, constraints)
    # compute static grid shapes
    partial_grid_shape_dict = _calculate_grid_shapes(
        obj_list=obj_list,
        size_list=size_list,
        config=config,
    )
    
    # compute all remaining grid shapes and positions
    slice_tuple_dict = _calculate_grid_positons(
        volume=volume,
        partial_grid_shape_dict=partial_grid_shape_dict,
        pos_constraints=pos_list,
        config=config,
    )
    
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
        )
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
) -> tuple[
    ArrayContainer, 
    ObjectContainer,
    dict[str, Any]
]:
    info = {}
    # apply parameter to devices
    for device in objects.devices:
        prev_slice = arrays.inv_permittivities[*device.grid_slice]
        inv_perm_at_slice, cur_info = device.get_inv_permittivity(
            prev_inv_permittivity=prev_slice,
            params=params[device.name],
        )
        info.update(cur_info)
        new_perm = arrays.inv_permittivities.at[*device.grid_slice].set(
            inv_perm_at_slice
        )
        arrays = arrays.at["inv_permittivities"].set(new_perm)
    
    # apply random key to sources
    new_sources = []
    for source in objects.sources:
        key, subkey = jax.random.split(key)
        new_source = source.apply(
            key=subkey,
            inv_permittivities=jax.lax.stop_gradient(arrays.inv_permittivities),
            inv_permeabilities=jax.lax.stop_gradient(arrays.inv_permeabilities),
        )
        new_sources.append(new_source)
    objects = objects.replace_sources(new_sources)
    
    return arrays, objects, info


def _init_arrays(
    objects: ObjectContainer,
    config: SimulationConfig,
) -> tuple[
    ArrayContainer, 
    SimulationConfig,
    dict[str, Any]
]:
    
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

    # permittivity / permeability
    shape_params = volume_shape
    inv_permittivities = create_named_sharded_matrix(
        shape_params,
        value=0.0,
        dtype=config.dtype,
        sharding_axis=1,
        backend=config.backend,
    )
    inv_permeabilities = create_named_sharded_matrix(
        shape_params,
        value=0.0,
        dtype=config.dtype,
        sharding_axis=1,
        backend=config.backend,
    )
    
    # set permittivity/permeability of static objects
    sorted_obj = sorted(
        objects.static_material_objects,
        key=lambda o: o.placement_order,
    )
    info = {}
    for o in sorted_obj:
        prev_slice = inv_permittivities[*o.grid_slice]
        update, cur_info = o.get_inv_permittivity(
            prev_inv_permittivity=prev_slice,
            params=None,
        )
        inv_permittivities = (
            inv_permittivities
            .at[*o.grid_slice]
            .set(update)
        )
        info.update(cur_info)
        
        prev_slice = inv_permeabilities[*o.grid_slice]
        update, cur_info = o.get_inv_permeability(
            prev_inv_permeability=prev_slice,
            params=None,
        )
        inv_permeabilities = (
            inv_permeabilities
            .at[*o.grid_slice]
            .set(update)
        )
        info.update(cur_info)
        
    # detector states
    detector_states = {}
    for d in objects.detectors:
        detector_states[d.name] = d.init_state()
    
    # boundary states
    boundary_states = {}
    for pml in objects.pml_objects:
        boundary_states[pml.name] = pml.init_state()
        
    # interfaces
    recording_state = None
    if config.gradient_config is not None and config.gradient_config.recorder is not None:
        input_shape_dtypes = {}
        for pml in objects.pml_objects:
            cur_shape = pml.boundary_interface_grid_shape()
            extended_shape = (3, *cur_shape)
            input_shape_dtypes[f"{pml.name}_E"] = jax.ShapeDtypeStruct(
                shape=extended_shape,
                dtype=config.dtype
            )
            input_shape_dtypes[f"{pml.name}_H"] = jax.ShapeDtypeStruct(
                shape=extended_shape,
                dtype=config.dtype
            )
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
    )
    return arrays, config, info


def _init_params(
    objects: ObjectContainer,
    key: jax.Array,
) -> ParameterContainer:
    params = {}
    for d in objects.devices:
    # device and aparameters
        key, subkey = jax.random.split(key)
        cur_dict = d.init_params(
            key=subkey,
        )
        params[d.name] = cur_dict
    return params


def _gather_objects(
    volume: SimulationObject,
    constraints: Sequence[PositionConstraint | SizeConstraint],
) -> tuple[
    list[SimulationObject],
    list[PositionConstraint],
    list[SizeConstraint],
]:
    obj_list: list[SimulationObject] = [volume]
    pos_list: list[PositionConstraint] = [
        c for c in constraints if isinstance(c, PositionConstraint)
    ]
    size_list: list[SizeConstraint] = [
        c for c in constraints if isinstance(c, SizeConstraint)
    ]
    
    # collect objects
    for c in constraints:
        if c.owner not in obj_list:
            obj_list.append(c.owner)
        if c.target not in obj_list:
            obj_list.append(c.target)
    
    return obj_list, pos_list, size_list


def _filter_size_constraints(
    constraints: Sequence[SizeConstraint],
    object: SimulationObject,
) -> list[SizeConstraint]:
    return [c for c in constraints if c.owner == object or c.target == object]

def _filter_pos_constraints(
    constraints: Sequence[PositionConstraint],
    object: SimulationObject,
) -> list[PositionConstraint]:
    return [c for c in constraints if c.owner == object or c.target == object]

def _calculate_grid_shapes(
    obj_list: list[SimulationObject],
    size_list: list[SizeConstraint],
    config: SimulationConfig,
) -> dict[SimulationObject, PartialGridShape3D]:
    result_dict: dict[SimulationObject, PartialGridShape3D]= {
        o: (None, None, None) for o in obj_list
    }
    to_resolve = [o for o in obj_list]
    while to_resolve:
        to_remove = []
        for o in to_resolve:
            # check if size constraints specify either real or grid shape
            for c in [c for c in _filter_size_constraints(size_list, o) if c.target == o]:
                for axis_idx, axis in enumerate(c.axes):
                    # case: owner has defined grid shape
                    old_grid_shape = result_dict[o][axis]
                    owner_grid_shape = result_dict[c.owner][c.other_axes[axis_idx]]
                    if owner_grid_shape is not None:
                        new_grid_shape = owner_grid_shape * c.proportions[axis_idx]
                        new_grid_shape += (
                            c.offsets[axis_idx] / config.resolution
                            + c.grid_offsets[axis_idx]
                        )
                        new_grid_shape = round(new_grid_shape)
                        if (
                            old_grid_shape is not None
                            and old_grid_shape != new_grid_shape
                        ):
                            raise Exception(
                                f"Size mismatch for {o} at axis {axis}: {old_grid_shape=} {new_grid_shape=}"
                            )
                        partial_list = list(result_dict[o])
                        partial_list[axis] = new_grid_shape
                        result_dict[o] = (
                            partial_list[0],
                            partial_list[1],
                            partial_list[2],
                        )
                        continue
                    # case: owner has defined real shape
                    owner_real_shape = c.owner.partial_real_shape[axis]
                    if owner_real_shape is not None:
                        new_grid_shape = (
                            owner_real_shape
                            * c.proportions[axis_idx]
                            / config.resolution
                        )
                        new_grid_shape += (
                            c.offsets[axis_idx] / config.resolution
                            + c.grid_offsets[axis_idx]
                        )
                        new_grid_shape = round(new_grid_shape)
                        if (
                            old_grid_shape is not None
                            and old_grid_shape != new_grid_shape
                        ):
                            raise Exception(f"Size mismatch for {o} at axis {axis}")
                        partial_list = list(result_dict[o])
                        partial_list[axis] = new_grid_shape
                        result_dict[o] = (
                            partial_list[0],
                            partial_list[1],
                            partial_list[2],
                        )
            # convert real shapes to grid shapes
            for axis in range(3):
                axis_real_shape = o.partial_real_shape[axis]
                if axis_real_shape is not None:
                    new_grid_shape = round(axis_real_shape / config.resolution)
                    old_grid_shape = result_dict[o][axis]
                    if old_grid_shape is not None and old_grid_shape != new_grid_shape:
                        raise Exception(
                            f"Cannot specify both real and grid shape for {axis=} in object {o}"
                        )
                    partial_list = list(result_dict[o])
                    partial_list[axis] = new_grid_shape
                    result_dict[o] = (
                        partial_list[0],
                        partial_list[1],
                        partial_list[2],
                    )
            # move grid shapes to dict
            for axis in range(3):
                axis_grid_shape = o.partial_grid_shape[axis]
                if axis_grid_shape is not None:
                    old_grid_shape = result_dict[o][axis]
                    if old_grid_shape is not None and old_grid_shape != axis_grid_shape:
                        raise Exception(
                            f"Cannot specify both real and grid shape for {axis=} in object {o}"
                        )
                    partial_list = list(result_dict[o])
                    partial_list[axis] = axis_grid_shape
                    result_dict[o] = (
                        partial_list[0],
                        partial_list[1],
                        partial_list[2],
                    )
            # fully specified by grid shape
            if all(a is not None for a in result_dict[o]):
                to_remove.append(o)
        if not to_remove:
            # we resolved all possible sizes, everything left has theoretical infinite size
            break
        for o in to_remove:
            to_resolve.remove(o)
    return result_dict


def _calculate_grid_positons(
    volume: SimulationObject,
    partial_grid_shape_dict: dict[SimulationObject, PartialGridShape3D],
    pos_constraints: list[PositionConstraint],
    config: SimulationConfig,
) -> dict[SimulationObject, SliceTuple3D]:
    objects = list(partial_grid_shape_dict.keys())
    partial_slice_dict: dict[SimulationObject, PartialSliceTuple3D] = {
        o: (None, None, None) for o in objects
    }
    to_resolve_pos: list[SimulationObject] = [o for o in partial_grid_shape_dict.keys()]
    to_resolve_shape: list[SimulationObject] = [
        o for o, v in partial_grid_shape_dict.items() if any(s is None for s in v)
    ]
    # first set position of setup as fixed
    vx, vy, vz = partial_grid_shape_dict[volume]
    if vx is None or vy is None or vz is None:
        raise Exception(f"Size of Simulation Volume is not specified: {vx}, {vy}, {vz}")
    volume_shape: tuple[int, int, int] = vx, vy, vz
    partial_slice_dict[volume] = (
        (0, vx),
        (0, vy),
        (0, vz),
    )
    while to_resolve_pos or to_resolve_shape:
        to_remove_pos: list[SimulationObject] = []
        to_remove_shape: list[SimulationObject] = []
        # resolve all positional constraints
        for o in to_resolve_pos:
            object_constraints = _filter_pos_constraints(pos_constraints, o)
            target_constraints = [c for c in object_constraints if c.target == o]
            if not target_constraints and o != volume:
                raise Exception(
                    f"Object {o} does not have constraints specifying position"
                )
            for c in target_constraints:
                for axis_idx, axis in enumerate(c.axes):
                    # check if we already know position in axis
                    if partial_slice_dict[o][axis] is not None:
                        continue
                    # check if both owner and target know their sizes
                    owner_axis_size = partial_grid_shape_dict[c.owner][axis]
                    target_axis_size = partial_grid_shape_dict[o][axis]
                    if owner_axis_size is None or target_axis_size is None:
                        continue
                    # check if owner knows their position in axis
                    owner_axis_slice = partial_slice_dict[c.owner][axis]
                    if owner_axis_slice is None:
                        continue
                    owner_midpoint = (owner_axis_slice[1] + owner_axis_slice[0]) / 2
                    owner_point = (
                        c.owner_positions[axis_idx] * owner_axis_size / 2
                        + owner_midpoint
                    )
                    target_point = owner_point + c.grid_margins[axis_idx]
                    target_point += c.margins[axis_idx] / config.resolution
                    target_lower = round(
                        target_point
                        - (1 + c.target_positions[axis_idx]) * target_axis_size / 2
                    )
                    # update target slice
                    slice_list = list(partial_slice_dict[o])
                    slice_list[axis] = (target_lower, target_lower + target_axis_size)
                    partial_slice_dict[o] = (slice_list[0], slice_list[1], slice_list[2])
            # check if positions are fully resolved
            if all(s is not None for s in partial_slice_dict[o]):
                to_remove_pos.append(o)
        # resolve possible shape constraints
        for o in to_resolve_shape:
            for axis in [a for a in range(3) if partial_grid_shape_dict[o][a] is None]:
                object_constraints = _filter_pos_constraints(pos_constraints, o)
                target_constraints = [c for c in object_constraints if c.target == o]
                axis_constraints = [c for c in target_constraints if axis in c.axes]
                if len(axis_constraints) > 2:
                    raise Exception(f"Object {o} is overconstrained in axis {axis}")
                if not axis_constraints:
                    # target does not have a constraint in this axis. Make as large as setup
                    shape_list = list(partial_grid_shape_dict[o])
                    shape_list[axis] = partial_grid_shape_dict[volume][axis]
                    new_shape = (shape_list[0], shape_list[1], shape_list[2])
                    partial_grid_shape_dict[o] = new_shape
                    target_slice = (0, volume_shape[axis])
                    # update position slice
                    slice_list = list(partial_slice_dict[o])
                    slice_list[axis] = target_slice
                    new_tuple = (slice_list[0], slice_list[1], slice_list[2])
                    partial_slice_dict[o] = new_tuple
                    continue
                # check if owner has position and shape specified for axis.
                c = axis_constraints[0]
                axis_idx = c.axes.index(axis)
                old_owner_slice = partial_slice_dict[c.owner][axis]
                owner_axis_size = partial_grid_shape_dict[c.owner][axis]
                if old_owner_slice is None or owner_axis_size is None:
                    continue
                # Calculate target shape and position
                owner_midpoint = (old_owner_slice[1] + old_owner_slice[0]) / 2
                owner_point = round(
                    c.owner_positions[axis_idx] * owner_axis_size / 2 + owner_midpoint
                )
                target_pos = c.target_positions[axis_idx]
                if target_pos == -1:
                    target_slice: tuple[int, int] = (
                        owner_point,
                        volume_shape[axis],
                    )
                elif target_pos == 1:
                    target_slice = (0, owner_point)
                else:
                    target_slice = (0, volume_shape[axis])
                # update shape
                target_size = target_slice[1] - target_slice[0]
                shape_list = list(partial_grid_shape_dict[o])
                shape_list[axis] = target_size
                new_tuple = (shape_list[0], shape_list[1], shape_list[2])
                partial_grid_shape_dict[o] = new_tuple
                # update position slice
                slice_list = list(partial_slice_dict[o])
                slice_list[axis] = target_slice
                new_tuple = (slice_list[0], slice_list[1], slice_list[2])
                partial_slice_dict[o] = new_tuple
            # check if shape is fully resolved
            if all(a is not None for a in partial_slice_dict[o]):
                to_remove_shape.append(o)
        if not to_remove_shape and not to_remove_pos:
            raise Exception("Could not resolve all positions and shapes")
        for o in to_remove_pos:
            to_resolve_pos.remove(o)
        for o in to_remove_shape:
            to_resolve_shape.remove(o)
    # check if we could resolve all shapes
    slice_dict: dict[SimulationObject, SliceTuple3D] = {}
    for o, v in partial_slice_dict.items():
        sx, sy, sz = v[0], v[1], v[2]
        if sx is None or sy is None or sz is None:
            raise Exception(f"Could not compute position for {o}: {v}")
        slice_dict[o] = (sx, sy, sz)
    return slice_dict