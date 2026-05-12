import jax

import fdtdx


def test_place_at_center_of_infinite_object():
    """
    Placing an object above and centered on a substrate that is infinite in x, y
    (partial_real_shape has None for those axes) must succeed.

    Bug (before fix): _extend_to_inf_if_possible sets cube's x/y lower bound to 0
    because substrate's x/y are still unknown at that point. Later, when substrate's
    x/y are finally extended to volume bounds, place_at_center computes b0=8 but
    finds old_b0=0, raising "Inconsistent grid shape".

    Root cause: _extend_to_inf_if_possible does not check whether an object has a
    pending PositionConstraint on the axis being extended. It eagerly locks position=0
    even when a constraint will later demand a different position.
    """
    config = fdtdx.SimulationConfig(resolution=100e-9, time=10e-15)
    constraints, objects = [], []

    volume = fdtdx.SimulationVolume(partial_real_shape=(2e-6, 2e-6, 2e-6))
    objects.append(volume)

    substrate = fdtdx.UniformMaterialObject(
        partial_real_shape=(None, None, 0.6e-6),
        name="substrate",
        color=fdtdx.colors.DARK_GREY,
        material=fdtdx.Material(permittivity=fdtdx.constants.relative_permittivity_silica),
    )
    objects.append(substrate)
    constraints.append(
        substrate.place_relative_to(
            volume,
            axes=2,
            own_positions=-1,
            other_positions=-1,
            margins=0,
            grid_margins=0,
        )
    )

    cube = fdtdx.UniformMaterialObject(
        name="cube",
        color=fdtdx.colors.GREEN,
        partial_real_shape=(0.5e-6, 0.5e-6, 0.5e-6),
        material=fdtdx.Material(permittivity=fdtdx.constants.relative_permittivity_silicon),
    )
    objects.append(cube)
    constraints.append(cube.place_above(substrate))
    constraints.append(cube.place_at_center(substrate, axes=(0, 1)))

    key = jax.random.PRNGKey(0)
    objects_out, _arrays, _params, _config_out, _info = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    # cube sits directly above substrate (substrate occupies z=[0,6], cube z size=5)
    cube_slice = objects_out["cube"].grid_slice_tuple
    assert cube_slice[2][0] == 6
    assert cube_slice[2][1] == 11
    # cube is centered in x and y (volume=20 cells, cube=5, center offset=(20-5)/2=7.5→8)
    assert cube_slice[0][0] == 8
    assert cube_slice[1][0] == 8
