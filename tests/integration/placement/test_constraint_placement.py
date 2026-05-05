"""Tests for object placement constraint combinations.

All tests call resolve_object_constraints() directly — no field arrays are
initialized — so they run without GPU resources and complete in milliseconds.

Volume: 2 um x 2 um x 2 um at 100 nm resolution → 20 x 20 x 20 grid cells.
"""

import fdtdx
from fdtdx.fdtd.initialization import resolve_object_constraints
from fdtdx.objects.object import RealCoordinateConstraint
from fdtdx.objects.static_material.cylinder import Cylinder
from fdtdx.objects.static_material.polygon import ExtrudedPolygon
from fdtdx.objects.static_material.sphere import Sphere

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

RESOLUTION = 100e-9  # 100 nm → 1 cell
VOL_SIDE = 2e-6  # 20 cells per axis
N = 20  # grid cells per axis


def _cfg():
    return fdtdx.SimulationConfig(resolution=RESOLUTION, time=10e-15)


def _volume():
    return fdtdx.SimulationVolume(partial_real_shape=(VOL_SIDE, VOL_SIDE, VOL_SIDE))


def _box(real_shape=(None, None, None), *, name=None):
    kwargs = dict(partial_real_shape=real_shape, material=fdtdx.Material())
    if name is not None:
        kwargs["name"] = name
    return fdtdx.UniformMaterialObject(**kwargs)


def _resolve(volume, extra_objects, constraints):
    """Resolve constraints and assert no errors; return slices dict."""
    slices, errors = resolve_object_constraints(
        objects=[volume, *extra_objects],
        constraints=constraints,
        config=_cfg(),
    )
    failed = {n: m for n, m in errors.items() if m}
    assert not failed, "Constraint resolution failed:\n" + "\n".join(f"  {n}: {m}" for n, m in failed.items())
    return slices


def _sl(slices, obj):
    """Return (x, y, z) as ((x0,x1),(y0,y1),(z0,z1))."""
    return slices[obj.name]


# ---------------------------------------------------------------------------
# Position constraints
# ---------------------------------------------------------------------------


def test_same_position_all_axes():
    """Centering a 10x10x10 box in a 20x20x20 volume gives [5,15] on each axis."""
    vol = _volume()
    box = _box((1e-6, 1e-6, 1e-6), name="box")
    constraints = [box.same_position(vol)]
    sl = _sl(_resolve(vol, [box], constraints), box)
    assert sl == ((5, 15), (5, 15), (5, 15))


def test_same_position_single_axis():
    """Centering along z only leaves x/y extending from 0."""
    vol = _volume()
    box = _box((1e-6, 1e-6, 1e-6), name="box")
    constraints = [box.same_position(vol, axes=2)]
    sl = _sl(_resolve(vol, [box], constraints), box)
    # z centered: [5,15]; x/y: size=10, no position → extend from 0
    assert sl[2] == (5, 15)
    assert sl[0] == (0, 10)
    assert sl[1] == (0, 10)


def test_place_relative_to_left_edge():
    """Aligning box's left side with volume's left side gives x=[0,10]."""
    vol = _volume()
    box = _box((1e-6, 1e-6, 1e-6), name="box")
    constraints = [box.place_relative_to(vol, axes=0, own_positions=-1, other_positions=-1)]
    sl = _sl(_resolve(vol, [box], constraints), box)
    assert sl[0] == (0, 10)


def test_place_relative_to_right_edge():
    """Aligning box's right side with volume's right side gives x=[10,20]."""
    vol = _volume()
    box = _box((1e-6, 1e-6, 1e-6), name="box")
    constraints = [box.place_relative_to(vol, axes=0, own_positions=1, other_positions=1)]
    sl = _sl(_resolve(vol, [box], constraints), box)
    assert sl[0] == (10, 20)


def test_place_above():
    """Box placed directly above a substrate sits flush at the substrate top."""
    vol = _volume()
    # 0.5 µm substrate (5 cells) at the volume bottom
    substrate = _box((None, None, 0.5e-6), name="substrate")
    cube = _box((0.5e-6, 0.5e-6, 0.5e-6), name="cube")
    constraints = [
        substrate.place_relative_to(vol, axes=2, own_positions=-1, other_positions=-1),
        cube.place_above(substrate),
    ]
    sl = _sl(_resolve(vol, [substrate, cube], constraints), cube)
    # substrate z=[0,5], cube directly above → z=[5,10]
    assert sl[2] == (5, 10)


def test_place_below():
    """Box placed directly below a ceiling slab is flush at its bottom."""
    vol = _volume()
    # 0.5 µm ceiling (5 cells) at the volume top
    ceiling = _box((None, None, 0.5e-6), name="ceiling")
    box = _box((0.5e-6, 0.5e-6, 0.5e-6), name="box")
    constraints = [
        ceiling.place_relative_to(vol, axes=2, own_positions=1, other_positions=1),
        box.place_below(ceiling),
    ]
    sl = _sl(_resolve(vol, [ceiling, box], constraints), box)
    # ceiling z=[15,20], box directly below → z=[10,15]
    assert sl[2] == (10, 15)


def test_face_to_face_positive_direction():
    """face_to_face_positive_direction places left side of box at right side of other."""
    vol = _volume()
    wall = _box((None, None, 0.8e-6), name="wall")
    box = _box((1e-6, 1e-6, 1e-6), name="box")
    constraints = [
        wall.place_relative_to(vol, axes=2, own_positions=-1, other_positions=-1),  # wall z=[0,8]
        box.face_to_face_positive_direction(wall, axes=2),
    ]
    sl = _sl(_resolve(vol, [wall, box], constraints), box)
    # wall z=[0,8], box left side (+z) at wall right side → z=[8,18]
    assert sl[2] == (8, 18)


def test_face_to_face_negative_direction():
    """face_to_face_negative_direction places right side of box at left side of other."""
    vol = _volume()
    wall = _box((None, None, 0.8e-6), name="wall")
    box = _box((1e-6, 1e-6, 1e-6), name="box")
    constraints = [
        wall.place_relative_to(vol, axes=2, own_positions=1, other_positions=1),  # wall z=[12,20]
        box.face_to_face_negative_direction(wall, axes=2),
    ]
    sl = _sl(_resolve(vol, [wall, box], constraints), box)
    # wall z=[12,20], box right side at wall left side → z=[2,12]
    assert sl[2] == (2, 12)


def test_position_real_margin():
    """Real-space margin offsets the placed position by the correct number of cells."""
    vol = _volume()
    substrate = _box((None, None, 0.5e-6), name="sub")
    box = _box((0.5e-6, 0.5e-6, 0.5e-6), name="box")
    margin = 0.5e-6  # 5 cells
    constraints = [
        substrate.place_relative_to(vol, axes=2, own_positions=-1, other_positions=-1),
        box.place_above(substrate, margins=margin),
    ]
    sl = _sl(_resolve(vol, [substrate, box], constraints), box)
    # substrate z=[0,5], gap=5, box z=[10,15]
    assert sl[2] == (10, 15)


def test_position_grid_margin():
    """Grid-space margin (in cells) offsets the placed position correctly."""
    vol = _volume()
    substrate = _box((None, None, 0.5e-6), name="sub")
    box = _box((0.5e-6, 0.5e-6, 0.5e-6), name="box")
    constraints = [
        substrate.place_relative_to(vol, axes=2, own_positions=-1, other_positions=-1),
        box.place_above(substrate, grid_margins=3),
    ]
    sl = _sl(_resolve(vol, [substrate, box], constraints), box)
    # substrate z=[0,5], 3-cell gap, box lower=8 → z=[8,13]
    assert sl[2] == (8, 13)


def test_position_chain_abc():
    """Chained constraints: A at bottom, B above A, C above B."""
    vol = _volume()
    a = _box((None, None, 0.4e-6), name="a")
    b = _box((None, None, 0.4e-6), name="b")
    c = _box((None, None, 0.4e-6), name="c")
    constraints = [
        a.place_relative_to(vol, axes=2, own_positions=-1, other_positions=-1),  # a z=[0,4]
        b.place_above(a),  # b z=[4,8]
        c.place_above(b),  # c z=[8,12]
    ]
    slices = _resolve(vol, [a, b, c], constraints)
    assert _sl(slices, a)[2] == (0, 4)
    assert _sl(slices, b)[2] == (4, 8)
    assert _sl(slices, c)[2] == (8, 12)


def test_place_at_center_multiple_axes():
    """place_at_center on (0,1) centers in x and y independently."""
    vol = _volume()
    box = _box((1e-6, 0.6e-6, 1e-6), name="box")  # 10x6x10 cells
    constraints = [
        box.place_at_center(vol, axes=(0, 1)),
        box.same_position(vol, axes=2),
    ]
    sl = _sl(_resolve(vol, [box], constraints), box)
    # x: center=10, size=10 → [5,15]
    # y: center=10, size=6 → round(10-3)=7 → [7,13]
    # z: center=10, size=10 → [5,15]
    assert sl[0] == (5, 15)
    assert sl[1] == (7, 13)
    assert sl[2] == (5, 15)


# ---------------------------------------------------------------------------
# Size constraints
# ---------------------------------------------------------------------------


def test_same_size_all_axes():
    """same_size on all axes makes box exactly as large as the volume."""
    vol = _volume()
    box = _box(name="box")  # all None
    constraints = [box.same_size(vol), box.same_position(vol)]
    sl = _sl(_resolve(vol, [box], constraints), box)
    assert sl == ((0, N), (0, N), (0, N))


def test_same_size_single_axis_with_centering():
    """same_size on z only; box has fixed x/y, centered everywhere."""
    vol = _volume()
    box = _box((1e-6, 1e-6, None), name="box")  # z shape from constraint
    constraints = [box.same_size(vol, axes=2), box.same_position(vol)]
    sl = _sl(_resolve(vol, [box], constraints), box)
    # z same as volume → 20 cells, centered at [0,20]
    # x,y: 10 cells centered at [5,15]
    assert sl[2] == (0, 20)
    assert sl[0] == (5, 15)
    assert sl[1] == (5, 15)


def test_size_relative_to_half():
    """size_relative_to with proportion=0.5 gives half the volume side."""
    vol = _volume()
    box = _box(name="box")
    constraints = [
        box.size_relative_to(vol, axes=(0, 1, 2), proportions=(0.5, 0.5, 0.5)),
        box.same_position(vol),
    ]
    sl = _sl(_resolve(vol, [box], constraints), box)
    # shape = 10x10x10, centered in 20x20x20 → [5,15]
    assert sl == ((5, 15), (5, 15), (5, 15))


def test_size_relative_to_with_grid_offset():
    """size_relative_to with negative grid_offset shrinks the size by that many cells."""
    vol = _volume()
    box = _box(name="box")
    constraints = [
        box.size_relative_to(vol, axes=2, grid_offsets=-4),
        box.same_position(vol, axes=2),
        box.same_size(vol, axes=(0, 1)),
        box.same_position(vol, axes=(0, 1)),
    ]
    sl = _sl(_resolve(vol, [box], constraints), box)
    # z: 20 - 4 = 16 cells, centered in 20 → round(10 - 8) = 2 → [2,18]
    assert sl[2] == (2, 18)
    assert sl[0] == (0, 20)
    assert sl[1] == (0, 20)


def test_same_position_and_size():
    """same_position_and_size is a shortcut for both constraints simultaneously."""
    vol = _volume()
    box = _box(name="box")
    pos_c, size_c = box.same_position_and_size(vol)
    sl = _sl(_resolve(vol, [box], [pos_c, size_c]), box)
    assert sl == ((0, N), (0, N), (0, N))


def test_size_with_real_offset():
    """size_relative_to with real offset adjusts size by converted cell count."""
    vol = _volume()
    box = _box(name="box")
    constraints = [
        # z size = volume_z - 1 µm (= 20 - 10 = 10 cells)
        box.size_relative_to(vol, axes=2, offsets=-1e-6),
        box.same_position(vol, axes=2),
        box.same_size(vol, axes=(0, 1)),
        box.same_position(vol, axes=(0, 1)),
    ]
    sl = _sl(_resolve(vol, [box], constraints), box)
    # z: 10 cells, centered in 20 → [5,15]
    assert sl[2] == (5, 15)
    assert sl[0] == (0, N)
    assert sl[1] == (0, N)


# ---------------------------------------------------------------------------
# Size extension constraints
# ---------------------------------------------------------------------------


def test_extend_to_volume_positive():
    """extend_to None in '+' direction extends the object to the volume upper bound."""
    vol = _volume()
    box = _box((1e-6, 1e-6, None), name="box")
    constraints = [
        box.set_grid_coordinates(axes=2, sides="-", coordinates=5),
        box.extend_to(None, axis=2, direction="+"),
        box.same_position(vol, axes=(0, 1)),
    ]
    sl = _sl(_resolve(vol, [box], constraints), box)
    assert sl[2] == (5, 20)


def test_extend_to_volume_negative():
    """extend_to None in '-' direction extends the object to the volume lower bound."""
    vol = _volume()
    box = _box((1e-6, 1e-6, None), name="box")
    constraints = [
        box.set_grid_coordinates(axes=2, sides="+", coordinates=15),
        box.extend_to(None, axis=2, direction="-"),
        box.same_position(vol, axes=(0, 1)),
    ]
    sl = _sl(_resolve(vol, [box], constraints), box)
    assert sl[2] == (0, 15)


def test_extend_to_object_positive():
    """extend_to(other, direction='+') extends the upper bound to the other's lower bound."""
    vol = _volume()
    wall = _box((None, None, 0.8e-6), name="wall")  # 8 cells
    box = _box((1e-6, 1e-6, None), name="box")
    constraints = [
        wall.place_relative_to(vol, axes=2, own_positions=1, other_positions=1),  # wall z=[12,20]
        box.set_grid_coordinates(axes=2, sides="-", coordinates=5),
        box.extend_to(wall, axis=2, direction="+"),  # upper = wall lower = 12
        box.same_position(vol, axes=(0, 1)),
    ]
    sl = _sl(_resolve(vol, [wall, box], constraints), box)
    assert sl[2] == (5, 12)


def test_extend_to_object_negative():
    """extend_to(other, direction='-') extends the lower bound to the other's upper bound."""
    vol = _volume()
    floor = _box((None, None, 0.8e-6), name="floor")  # 8 cells
    box = _box((1e-6, 1e-6, None), name="box")
    constraints = [
        floor.place_relative_to(vol, axes=2, own_positions=-1, other_positions=-1),  # floor z=[0,8]
        box.set_grid_coordinates(axes=2, sides="+", coordinates=15),
        box.extend_to(floor, axis=2, direction="-"),  # lower = floor upper = 8
        box.same_position(vol, axes=(0, 1)),
    ]
    sl = _sl(_resolve(vol, [floor, box], constraints), box)
    assert sl[2] == (8, 15)


def test_extend_to_object_with_grid_offset():
    """grid_offset shifts the extension anchor inward by that many cells."""
    vol = _volume()
    wall = _box((None, None, 0.8e-6), name="wall")  # z=[12,20]
    box = _box((1e-6, 1e-6, None), name="box")
    constraints = [
        wall.place_relative_to(vol, axes=2, own_positions=1, other_positions=1),
        box.set_grid_coordinates(axes=2, sides="-", coordinates=5),
        # extend to wall but 2 cells inside (toward center) → anchor = 12 + 2 = 14
        box.extend_to(wall, axis=2, direction="+", grid_offset=2),
        box.same_position(vol, axes=(0, 1)),
    ]
    sl = _sl(_resolve(vol, [wall, box], constraints), box)
    # wall lower=12, other_pos=-1, offset=+2 → anchor=12+2=14
    assert sl[2] == (5, 14)


def test_extend_to_object_custom_other_position():
    """other_position=0 extends to the center of the target object."""
    vol = _volume()
    wall = _box((None, None, 1e-6), name="wall")  # 10 cells
    box = _box((1e-6, 1e-6, None), name="box")
    constraints = [
        wall.place_relative_to(vol, axes=2, own_positions=1, other_positions=1),  # wall z=[10,20]
        box.set_grid_coordinates(axes=2, sides="-", coordinates=0),
        box.extend_to(wall, axis=2, direction="+", other_position=0),  # center of wall = 15
        box.same_position(vol, axes=(0, 1)),
    ]
    sl = _sl(_resolve(vol, [wall, box], constraints), box)
    assert sl[2] == (0, 15)


def test_extend_to_volume_both_sides():
    """Extending both '+' and '-' to volume boundary fills the full axis."""
    vol = _volume()
    box = _box((1e-6, None, None), name="box")
    constraints = [
        box.extend_to(None, axis=2, direction="+"),
        box.extend_to(None, axis=2, direction="-"),
        box.same_position(vol, axes=(0, 1)),
    ]
    sl = _sl(_resolve(vol, [box], constraints), box)
    assert sl[2] == (0, 20)


def test_extend_between_two_objects():
    """Box fills the gap between a floor and ceiling object."""
    vol = _volume()
    floor = _box((None, None, 0.4e-6), name="floor")  # 4 cells at bottom
    ceiling = _box((None, None, 0.4e-6), name="ceiling")  # 4 cells at top
    box = _box((1e-6, 1e-6, None), name="box")
    constraints = [
        floor.place_relative_to(vol, axes=2, own_positions=-1, other_positions=-1),  # z=[0,4]
        ceiling.place_relative_to(vol, axes=2, own_positions=1, other_positions=1),  # z=[16,20]
        box.extend_to(floor, axis=2, direction="-"),  # box lower = 4 (floor upper)
        box.extend_to(ceiling, axis=2, direction="+"),  # box upper = 16 (ceiling lower)
        box.same_position(vol, axes=(0, 1)),
    ]
    sl = _sl(_resolve(vol, [floor, ceiling, box], constraints), box)
    assert sl[2] == (4, 16)


# ---------------------------------------------------------------------------
# Grid coordinate constraints
# ---------------------------------------------------------------------------


def test_set_grid_coordinates_lower_side():
    """set_grid_coordinates with '-' side fixes the lower boundary."""
    vol = _volume()
    box = _box((1e-6, 1e-6, 1e-6), name="box")  # 10x10x10
    constraints = [box.set_grid_coordinates(axes=0, sides="-", coordinates=3)]
    sl = _sl(_resolve(vol, [box], constraints), box)
    assert sl[0] == (3, 13)


def test_set_grid_coordinates_upper_side():
    """set_grid_coordinates with '+' side fixes the upper boundary."""
    vol = _volume()
    box = _box((1e-6, 1e-6, 1e-6), name="box")  # 10x10x10
    constraints = [box.set_grid_coordinates(axes=0, sides="+", coordinates=18)]
    sl = _sl(_resolve(vol, [box], constraints), box)
    assert sl[0] == (8, 18)


def test_set_grid_coordinates_multiple_axes():
    """set_grid_coordinates can fix coordinates on multiple axes at once."""
    vol = _volume()
    box = _box((1e-6, 1e-6, 1e-6), name="box")  # 10x10x10
    constraints = [box.set_grid_coordinates(axes=(0, 2), sides=("-", "+"), coordinates=(2, 17))]
    sl = _sl(_resolve(vol, [box], constraints), box)
    assert sl[0] == (2, 12)
    assert sl[2] == (7, 17)


# ---------------------------------------------------------------------------
# Real coordinate constraints (direct dataclass construction)
# ---------------------------------------------------------------------------


def test_real_coordinate_constraint_lower():
    """RealCoordinateConstraint with '-' side pins the lower bound in real space."""
    vol = _volume()
    box = _box((1e-6, 1e-6, 1e-6), name="box")  # 10x10x10
    c = RealCoordinateConstraint(
        object=box.name,
        axes=(0,),
        sides=("-",),
        coordinates=(0.5e-6,),  # 5 cells
    )
    sl = _sl(_resolve(vol, [box], [c]), box)
    assert sl[0] == (5, 15)


def test_real_coordinate_constraint_upper():
    """RealCoordinateConstraint with '+' side pins the upper bound in real space."""
    vol = _volume()
    box = _box((1e-6, 1e-6, 1e-6), name="box")  # 10x10x10
    c = RealCoordinateConstraint(
        object=box.name,
        axes=(2,),
        sides=("+",),
        coordinates=(1.8e-6,),  # 18 cells
    )
    sl = _sl(_resolve(vol, [box], [c]), box)
    assert sl[2] == (8, 18)


def test_real_coordinate_constraint_multiple_axes():
    """RealCoordinateConstraint can constrain multiple axes simultaneously."""
    vol = _volume()
    box = _box((1e-6, 1e-6, 1e-6), name="box")  # 10x10x10
    c = RealCoordinateConstraint(
        object=box.name,
        axes=(0, 1),
        sides=("-", "+"),
        coordinates=(0.3e-6, 1.5e-6),  # x lower=3, y upper=15
    )
    sl = _sl(_resolve(vol, [box], [c]), box)
    assert sl[0] == (3, 13)
    assert sl[1] == (5, 15)


# ---------------------------------------------------------------------------
# Infinity extension (auto-extension to volume bounds)
# ---------------------------------------------------------------------------


def test_infinity_extension_unconstrained_axes():
    """An object with None shape and no constraint on some axes extends to volume."""
    vol = _volume()
    slab = _box((None, None, 1e-6), name="slab")  # z fixed, x/y → infinity
    constraints = [slab.same_position(vol, axes=2)]  # only z positioned
    sl = _sl(_resolve(vol, [slab], constraints), slab)
    # z: 10 cells, centered at [5,15]
    assert sl[2] == (5, 15)
    # x,y: no shape, no constraints → extend to volume [0,20]
    assert sl[0] == (0, N)
    assert sl[1] == (0, N)


def test_infinity_extension_known_size_no_position():
    """Object with known size but no position extends from 0."""
    vol = _volume()
    box = _box((1e-6, 1e-6, 1e-6), name="box")  # 10x10x10, no position constraint
    sl = _sl(_resolve(vol, [box], []), box)
    # Size known (10), no position → b0=0, b1=10 on each axis
    assert sl == ((0, 10), (0, 10), (0, 10))


def test_infinity_extension_fully_unconstrained():
    """Object with no shape and no constraints extends to fill the full volume."""
    vol = _volume()
    box = _box(name="box")  # all None, no constraints
    sl = _sl(_resolve(vol, [box], []), box)
    assert sl == ((0, N), (0, N), (0, N))


def test_infinity_extension_pending_position_not_premature():
    """Infinity extension must not lock x/y=0 when a PositionConstraint will resolve later.

    Regression: substrate has None x/y shape; cube is centered on substrate. If the
    infinity-extension fires before the centering constraint resolves, it would lock
    cube x/y lower=0, which later conflicts with the centering result.
    """
    vol = _volume()
    substrate = _box((None, None, 0.6e-6), name="substrate")
    cube = _box((0.5e-6, 0.5e-6, 0.5e-6), name="cube")
    constraints = [
        substrate.place_relative_to(vol, axes=2, own_positions=-1, other_positions=-1),
        cube.place_above(substrate),
        cube.place_at_center(substrate, axes=(0, 1)),
    ]
    sl_cube = _sl(_resolve(vol, [substrate, cube], constraints), cube)
    # substrate x/y fills volume [0,20], cube (5 cells) centered → 8
    assert sl_cube[2] == (6, 11)
    assert sl_cube[0] == (8, 13)
    assert sl_cube[1] == (8, 13)


# ---------------------------------------------------------------------------
# partial_real_position (center-based absolute positioning)
# ---------------------------------------------------------------------------


def test_partial_real_position_centers_object():
    """partial_real_position places the object center at the given real coordinate."""
    vol = _volume()
    # Place a 10x10x10 box with center at (0.5µm, 1.0µm, 1.5µm) = (5, 10, 15) cells
    box = fdtdx.UniformMaterialObject(
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        partial_real_position=(0.5e-6, 1.0e-6, 1.5e-6),
        material=fdtdx.Material(),
        name="box",
    )
    sl = _sl(_resolve(vol, [box], []), box)
    # center=(5,10,15), half-size=5 → lower=(0,5,10), upper=(10,15,20)
    assert sl[0] == (0, 10)
    assert sl[1] == (5, 15)
    assert sl[2] == (10, 20)


def test_partial_real_position_partial_axes():
    """partial_real_position with None entries only constrains specified axes."""
    vol = _volume()
    box = fdtdx.UniformMaterialObject(
        partial_real_shape=(1e-6, 1e-6, 1e-6),
        partial_real_position=(None, 1.0e-6, None),  # only y centered at 1µm = 10 cells
        material=fdtdx.Material(),
        name="box",
    )
    sl = _sl(_resolve(vol, [box], []), box)
    # y: center=10, size=10 → [5,15]
    assert sl[1] == (5, 15)
    # x, z: size=10, no position → extend from 0
    assert sl[0] == (0, 10)
    assert sl[2] == (0, 10)


# ---------------------------------------------------------------------------
# Mixed / combined scenarios
# ---------------------------------------------------------------------------


def test_size_constraint_plus_extend_to():
    """Size comes from same_size; position from set_grid_coordinates + extend_to."""
    vol = _volume()
    box = _box(name="box")
    constraints = [
        box.same_size(vol, axes=(0, 1)),  # x,y → 20 cells
        box.set_grid_coordinates(axes=2, sides="-", coordinates=6),
        box.extend_to(None, axis=2, direction="+"),  # z upper → 20
    ]
    sl = _sl(_resolve(vol, [box], constraints), box)
    # x,y: 20 cells, extend from 0 → [0,20]
    # z: [6,20]
    assert sl[0] == (0, N)
    assert sl[1] == (0, N)
    assert sl[2] == (6, 20)


def test_position_and_size_from_different_constraints():
    """Size from size_relative_to, position from place_relative_to."""
    vol = _volume()
    ref = _box((None, None, 1.0e-6), name="ref")  # 10-cell z slab
    box = _box(name="box")
    constraints = [
        ref.place_relative_to(vol, axes=2, own_positions=-1, other_positions=-1),  # ref z=[0,10]
        box.size_relative_to(ref, axes=2, proportions=0.5),  # box z = 5 cells
        box.place_above(ref),  # box z lower = 10
        box.same_size(vol, axes=(0, 1)),
        box.same_position(vol, axes=(0, 1)),
    ]
    sl = _sl(_resolve(vol, [ref, box], constraints), box)
    assert sl[2] == (10, 15)
    assert sl[0] == (0, N)
    assert sl[1] == (0, N)


def test_multi_constraint_stacked_layers():
    """Four equal-height layers stacked with all extension constraints resolved."""
    vol = _volume()
    height = 0.5e-6  # 5 cells per layer
    a = _box((None, None, height), name="a")
    b = _box((None, None, height), name="b")
    c = _box((None, None, height), name="c")
    d = _box((None, None, height), name="d")
    constraints = [
        a.place_relative_to(vol, axes=2, own_positions=-1, other_positions=-1),
        b.place_above(a),
        c.place_above(b),
        d.place_above(c),
    ]
    slices = _resolve(vol, [a, b, c, d], constraints)
    assert _sl(slices, a)[2] == (0, 5)
    assert _sl(slices, b)[2] == (5, 10)
    assert _sl(slices, c)[2] == (10, 15)
    assert _sl(slices, d)[2] == (15, 20)


def test_size_from_one_object_position_from_another():
    """Box gets its size from object A and its position from object B."""
    vol = _volume()
    a = _box((None, None, 0.8e-6), name="a")  # 8-cell z
    b = _box((None, None, 0.6e-6), name="b")  # 6-cell z at top
    box = _box(name="box")
    constraints = [
        a.place_relative_to(vol, axes=2, own_positions=-1, other_positions=-1),  # a z=[0,8]
        b.place_relative_to(vol, axes=2, own_positions=1, other_positions=1),  # b z=[14,20]
        box.size_relative_to(a, axes=2),  # box z size = 8
        box.place_below(b),  # box upper z = b lower = 14
        box.same_size(vol, axes=(0, 1)),
        box.same_position(vol, axes=(0, 1)),
    ]
    sl = _sl(_resolve(vol, [a, b, box], constraints), box)
    # box z: size=8, upper=14 → [6,14]
    assert sl[2] == (6, 14)


def test_extend_to_plus_same_size_on_other_axes():
    """Combining extend_to on one axis with same_size on others."""
    vol = _volume()
    box = _box(name="box")
    constraints = [
        box.same_size(vol, axes=(0, 1)),
        box.same_position(vol, axes=(0, 1)),
        box.set_grid_coordinates(axes=2, sides="-", coordinates=3),
        box.extend_to(None, axis=2, direction="+"),
    ]
    sl = _sl(_resolve(vol, [box], constraints), box)
    assert sl[0] == (0, N)
    assert sl[1] == (0, N)
    assert sl[2] == (3, N)


# ---------------------------------------------------------------------------
# Auto-size from geometry (Cylinder / Sphere / ExtrudedPolygon)
# ---------------------------------------------------------------------------

_MATS = {"si": fdtdx.Material(permittivity=12.25)}


def test_cylinder_auto_size_from_radius():
    """Cylinder cross-section is sized automatically from radius without partial_real_shape."""
    vol = _volume()
    radius = 500e-9  # = 5 grid cells at 100 nm resolution
    cyl = Cylinder(name="cyl", radius=radius, axis=2, materials=_MATS, material_name="si")

    constraints = [
        cyl.same_position(vol, axes=(0, 1)),
        cyl.extend_to(None, axis=2, direction="+"),
        cyl.extend_to(None, axis=2, direction="-"),
    ]
    sl = _sl(_resolve(vol, [cyl], constraints), cyl)

    expected = round(2 * radius / RESOLUTION)  # = 10
    assert sl[0] == (N // 2 - expected // 2, N // 2 + expected // 2)
    assert sl[1] == (N // 2 - expected // 2, N // 2 + expected // 2)
    assert sl[2] == (0, N)


def test_cylinder_constraint_overrides_auto_size():
    """Explicit GridCoordinateConstraints on cross-section axes win over auto-size."""
    vol = _volume()
    radius = 500e-9
    cyl = Cylinder(name="cyl", radius=radius, axis=2, materials=_MATS, material_name="si")

    constraints = [
        cyl.set_grid_coordinates(axes=0, sides="-", coordinates=4),
        cyl.set_grid_coordinates(axes=0, sides="+", coordinates=16),
        cyl.set_grid_coordinates(axes=1, sides="-", coordinates=4),
        cyl.set_grid_coordinates(axes=1, sides="+", coordinates=16),
        cyl.extend_to(None, axis=2, direction="+"),
        cyl.extend_to(None, axis=2, direction="-"),
    ]
    sl = _sl(_resolve(vol, [cyl], constraints), cyl)

    # Constraint says 12 cells, auto-size would say 10 — constraint wins.
    assert sl[0] == (4, 16)
    assert sl[1] == (4, 16)


def test_sphere_auto_size_all_axes():
    """Sphere bounding box is sized automatically on all three axes."""
    vol = _volume()
    radius = 500e-9  # = 5 grid cells
    sphere = Sphere(name="sphere", radius=radius, materials=_MATS, material_name="si")

    constraints = [sphere.same_position(vol)]
    sl = _sl(_resolve(vol, [sphere], constraints), sphere)

    expected = round(2 * radius / RESOLUTION)  # = 10
    lo, hi = N // 2 - expected // 2, N // 2 + expected // 2
    assert sl[0] == (lo, hi)
    assert sl[1] == (lo, hi)
    assert sl[2] == (lo, hi)


def test_extruded_polygon_auto_size_from_vertices():
    """ExtrudedPolygon cross-section is sized from vertex bounding box."""
    import numpy as np

    vol = _volume()
    half = 500e-9  # 5 cells → 10-cell square
    verts = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    poly = ExtrudedPolygon(name="poly", axis=2, material_name="si", materials=_MATS, vertices=verts)

    constraints = [
        poly.same_position(vol, axes=(0, 1)),
        poly.extend_to(None, axis=2, direction="+"),
        poly.extend_to(None, axis=2, direction="-"),
    ]
    sl = _sl(_resolve(vol, [poly], constraints), poly)

    expected = round(2 * half / RESOLUTION)  # = 10
    assert sl[0] == (N // 2 - expected // 2, N // 2 + expected // 2)
    assert sl[1] == (N // 2 - expected // 2, N // 2 + expected // 2)
    assert sl[2] == (0, N)
