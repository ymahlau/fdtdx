"""Unit tests for objects/object.py.

Tests UniqueName, constraint dataclasses, SimulationObject, and OrderableObject.
Uses minimal concrete subclasses where abstract classes are involved.
"""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit
from fdtdx.objects.object import (
    GridCoordinateConstraint,
    OrderableObject,
    PositionConstraint,
    RealCoordinateConstraint,
    SimulationObject,
    SizeConstraint,
    SizeExtensionConstraint,
    UniqueName,
)

# ---------------------------------------------------------------------------
# Minimal concrete implementations for testing
# ---------------------------------------------------------------------------


@autoinit
class _ConcreteObject(SimulationObject):
    """Minimal concrete SimulationObject for testing."""


@autoinit
class _ConcreteOrderable(OrderableObject):
    """Minimal concrete OrderableObject for testing."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(name=None, **kwargs) -> _ConcreteObject:
    return _ConcreteObject(name=name, **kwargs)


def _place(obj, config, key, slices=((0, 10), (0, 10), (0, 10))):
    return obj.place_on_grid(grid_slice_tuple=slices, config=config, key=key)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return SimulationConfig(
        time=100e-15,
        resolution=50e-9,
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=None,
    )


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# UniqueName
# ---------------------------------------------------------------------------


class TestUniqueName:
    def test_returns_provided_name(self):
        un = UniqueName()
        assert un("my_name") == "my_name"

    def test_generates_name_when_none(self):
        un = UniqueName()
        name = un(None)
        assert name.startswith("Object_")
        assert name[7:].isdigit()

    def test_consecutive_calls_yield_different_names(self):
        un = UniqueName()
        n1 = un(None)
        n2 = un(None)
        assert n1 != n2

    def test_explicit_name_does_not_increment_counter(self):
        import fdtdx.objects.object as _mod

        before = _mod._GLOBAL_COUNTER
        UniqueName()("explicit_name")
        after = _mod._GLOBAL_COUNTER
        assert before == after


# ---------------------------------------------------------------------------
# Constraint dataclasses
# ---------------------------------------------------------------------------


class TestPositionConstraint:
    def test_creation(self):
        pc = PositionConstraint(
            object="obj1",
            other_object="obj2",
            axes=(0,),
            object_positions=(0.0,),
            other_object_positions=(1.0,),
            margins=(0.0,),
            grid_margins=(0,),
        )
        assert pc.object == "obj1"
        assert pc.other_object == "obj2"
        assert pc.axes == (0,)
        assert pc.object_positions == (0.0,)
        assert pc.other_object_positions == (1.0,)

    def test_frozen(self):
        pc = PositionConstraint(
            object="o",
            other_object="p",
            axes=(0,),
            object_positions=(0.0,),
            other_object_positions=(0.0,),
            margins=(0.0,),
            grid_margins=(0,),
        )
        with pytest.raises(Exception):
            pc.axes = (1,)  # type: ignore[misc]


class TestSizeConstraint:
    def test_creation(self):
        sc = SizeConstraint(
            object="child",
            other_object="parent",
            axes=(1, 2),
            other_axes=(1, 2),
            proportions=(1.0, 0.5),
            offsets=(0.0, 1e-9),
            grid_offsets=(0, 2),
        )
        assert sc.proportions == (1.0, 0.5)
        assert sc.grid_offsets == (0, 2)


class TestSizeExtensionConstraint:
    def test_creation(self):
        sec = SizeExtensionConstraint(
            object="obj",
            other_object="ref",
            axis=0,
            direction="+",
            other_position=-1.0,
            offset=0.0,
            grid_offset=0,
        )
        assert sec.axis == 0
        assert sec.direction == "+"
        assert sec.other_object == "ref"

    def test_none_other_object(self):
        sec = SizeExtensionConstraint(
            object="obj",
            other_object=None,
            axis=1,
            direction="-",
            other_position=1.0,
            offset=0.0,
            grid_offset=0,
        )
        assert sec.other_object is None


class TestGridCoordinateConstraint:
    def test_creation(self):
        gcc = GridCoordinateConstraint(
            object="obj",
            axes=(0,),
            sides=("+",),
            coordinates=(5,),
        )
        assert gcc.axes == (0,)
        assert gcc.sides == ("+",)
        assert gcc.coordinates == (5,)


class TestRealCoordinateConstraint:
    def test_creation(self):
        rcc = RealCoordinateConstraint(
            object="obj",
            axes=(1,),
            sides=("-",),
            coordinates=(1e-6,),
        )
        assert rcc.axes == (1,)
        assert rcc.coordinates == (1e-6,)


# ---------------------------------------------------------------------------
# SimulationObject – basic properties before placement
# ---------------------------------------------------------------------------


class TestSimulationObjectDefaults:
    def test_name_auto_assigned(self):
        obj = _make()
        assert obj.name.startswith("Object_")

    def test_name_manual(self):
        obj = _make(name="test_obj")
        assert obj.name == "test_obj"

    def test_color_defaults_none(self):
        obj = _make()
        assert obj.color is None

    def test_max_random_offsets_default_zero(self):
        obj = _make()
        assert obj.max_random_real_offsets == (0, 0, 0)
        assert obj.max_random_grid_offsets == (0, 0, 0)

    def test_grid_slice_tuple_raises_before_placement(self):
        obj = _make()
        with pytest.raises(Exception, match="not yet initialized"):
            _ = obj.grid_slice_tuple

    def test_grid_shape_raises_before_placement(self):
        obj = _make()
        with pytest.raises(Exception, match="non-initialized"):
            _ = obj.grid_shape


# ---------------------------------------------------------------------------
# SimulationObject – place_on_grid
# ---------------------------------------------------------------------------


class TestPlaceOnGrid:
    def test_sets_grid_slice_tuple(self, config, key):
        obj = _make()
        placed = _place(obj, config, key, ((0, 10), (0, 20), (0, 30)))
        assert placed.grid_slice_tuple == ((0, 10), (0, 20), (0, 30))

    def test_grid_shape_after_placement(self, config, key):
        obj = _make()
        placed = _place(obj, config, key, ((0, 5), (0, 8), (0, 12)))
        assert placed.grid_shape == (5, 8, 12)

    def test_real_shape_after_placement(self, config, key):
        obj = _make()
        placed = _place(obj, config, key, ((0, 4), (0, 6), (0, 10)))
        res = config.resolution
        assert placed.real_shape == pytest.approx((4 * res, 6 * res, 10 * res))

    def test_grid_slice_returns_python_slices(self, config, key):
        obj = _make()
        placed = _place(obj, config, key, ((2, 8), (3, 9), (1, 7)))
        s = placed.grid_slice
        assert s == (slice(2, 8), slice(3, 9), slice(1, 7))

    def test_place_twice_raises(self, config, key):
        obj = _make()
        placed = _place(obj, config, key)
        with pytest.raises(Exception, match="already compiled"):
            placed.place_on_grid(((0, 10), (0, 10), (0, 10)), config, key)

    def test_negative_start_raises(self, config, key):
        obj = _make()
        with pytest.raises(Exception, match="Invalid placement"):
            obj.place_on_grid(((-1, 5), (0, 5), (0, 5)), config, key)

    def test_inverted_slice_raises(self, config, key):
        obj = _make()
        with pytest.raises(Exception, match="Invalid placement"):
            obj.place_on_grid(((5, 3), (0, 5), (0, 5)), config, key)

    def test_zero_length_slice_raises(self, config, key):
        obj = _make()
        with pytest.raises(Exception, match="Invalid placement"):
            obj.place_on_grid(((5, 5), (0, 5), (0, 5)), config, key)

    def test_apply_returns_self(self, config, key):
        obj = _make(name="test_apply")
        placed = _place(obj, config, key)
        result = placed.apply(key, jnp.ones((3, 10, 10, 10)), 1.0)
        assert result.name == placed.name


# ---------------------------------------------------------------------------
# place_relative_to
# ---------------------------------------------------------------------------


class TestPlaceRelativeTo:
    def test_basic_single_axis(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.place_relative_to(other=b, axes=0, own_positions=0.0, other_positions=0.5)
        assert isinstance(c, PositionConstraint)
        assert c.object == "a"
        assert c.other_object == "b"
        assert c.axes == (0,)
        assert c.object_positions == (0.0,)
        assert c.other_object_positions == (0.5,)

    def test_multiple_axes(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.place_relative_to(
            other=b,
            axes=(0, 1),
            own_positions=(0.0, 0.0),
            other_positions=(0.5, -0.5),
        )
        assert c.axes == (0, 1)
        assert c.other_object_positions == (0.5, -0.5)

    def test_with_margins(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.place_relative_to(
            other=b,
            axes=0,
            own_positions=0.0,
            other_positions=0.0,
            margins=1e-9,
            grid_margins=3,
        )
        assert c.margins == (1e-9,)
        assert c.grid_margins == (3,)

    def test_default_margins_zero(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.place_relative_to(other=b, axes=0, own_positions=0.0, other_positions=0.0)
        assert c.margins == (0,)
        assert c.grid_margins == (0,)

    def test_length_mismatch_raises(self):
        a = _make(name="a")
        b = _make(name="b")
        with pytest.raises(Exception, match="same lengths"):
            a.place_relative_to(
                other=b,
                axes=(0, 1),
                own_positions=(0.0,),
                other_positions=(0.0, 0.0),
            )


# ---------------------------------------------------------------------------
# size_relative_to
# ---------------------------------------------------------------------------


class TestSizeRelativeTo:
    def test_default_proportion_and_offsets(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.size_relative_to(other=b, axes=0)
        assert isinstance(c, SizeConstraint)
        assert c.proportions == (1.0,)
        assert c.offsets == (0,)
        assert c.grid_offsets == (0,)

    def test_explicit_proportion(self):
        a = _make(name="a")
        b = _make(name="b")
        # scalar proportion wraps to single-element tuple, must match single axis
        c = a.size_relative_to(other=b, axes=0, proportions=0.5)
        assert c.proportions == (0.5,)

    def test_default_other_axes_matches_axes(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.size_relative_to(other=b, axes=(0, 2))
        assert c.other_axes == (0, 2)

    def test_custom_other_axes(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.size_relative_to(other=b, axes=(0,), other_axes=(2,))
        assert c.other_axes == (2,)

    def test_length_mismatch_raises(self):
        a = _make(name="a")
        b = _make(name="b")
        with pytest.raises(Exception, match="same lengths"):
            a.size_relative_to(other=b, axes=(0,), proportions=(1.0, 2.0))


# ---------------------------------------------------------------------------
# Convenience constraint methods
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    def test_same_size_all_axes(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.same_size(other=b)
        assert isinstance(c, SizeConstraint)
        assert c.axes == (0, 1, 2)
        assert all(p == 1 for p in c.proportions)

    def test_same_size_single_axis(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.same_size(other=b, axes=1)
        assert c.axes == (1,)
        assert c.proportions == (1,)

    def test_place_at_center(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.place_at_center(other=b, axes=(0, 2))
        assert isinstance(c, PositionConstraint)
        assert c.object_positions == (0, 0)
        assert c.other_object_positions == (0, 0)

    def test_same_position_delegates_to_place_at_center(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.same_position(other=b, axes=0)
        assert isinstance(c, PositionConstraint)
        assert c.object_positions == (0,)

    def test_same_position_and_size_returns_pair(self):
        a = _make(name="a")
        b = _make(name="b")
        pos_c, size_c = a.same_position_and_size(other=b, axes=(0, 1))
        assert isinstance(pos_c, PositionConstraint)
        assert isinstance(size_c, SizeConstraint)

    def test_face_to_face_positive(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.face_to_face_positive_direction(other=b, axes=0)
        assert c.object_positions == (-1,)
        assert c.other_object_positions == (1,)

    def test_face_to_face_negative(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.face_to_face_negative_direction(other=b, axes=1)
        assert c.object_positions == (1,)
        assert c.other_object_positions == (-1,)

    def test_place_above_uses_axis2(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.place_above(other=b)
        assert c.axes == (2,)
        assert c.object_positions == (-1,)
        assert c.other_object_positions == (1,)

    def test_place_below_uses_axis2(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.place_below(other=b)
        assert c.axes == (2,)
        assert c.object_positions == (1,)
        assert c.other_object_positions == (-1,)


# ---------------------------------------------------------------------------
# set_grid_coordinates
# ---------------------------------------------------------------------------


class TestSetGridCoordinates:
    def test_single_axis(self):
        obj = _make(name="obj")
        c = obj.set_grid_coordinates(axes=0, sides="+", coordinates=5)
        assert isinstance(c, GridCoordinateConstraint)
        assert c.axes == (0,)
        assert c.sides == ("+",)
        assert c.coordinates == (5,)

    def test_multiple_axes(self):
        obj = _make(name="obj")
        c = obj.set_grid_coordinates(axes=(0, 2), sides=("+", "-"), coordinates=(5, 10))
        assert c.axes == (0, 2)
        assert c.coordinates == (5, 10)

    def test_length_mismatch_raises(self):
        obj = _make(name="obj")
        with pytest.raises(Exception, match="same lengths"):
            obj.set_grid_coordinates(axes=(0, 1), sides="+", coordinates=5)


# ---------------------------------------------------------------------------
# extend_to
# ---------------------------------------------------------------------------


class TestExtendTo:
    def test_default_other_position_for_plus(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.extend_to(other=b, axis=0, direction="+")
        assert isinstance(c, SizeExtensionConstraint)
        assert c.other_position == -1

    def test_default_other_position_for_minus(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.extend_to(other=b, axis=0, direction="-")
        assert c.other_position == 1

    def test_custom_other_position(self):
        a = _make(name="a")
        b = _make(name="b")
        c = a.extend_to(other=b, axis=2, direction="+", other_position=0.5)
        assert c.other_position == 0.5

    def test_extend_to_none_other(self):
        a = _make(name="a")
        c = a.extend_to(other=None, axis=1, direction="-")
        assert c.other_object is None

    def test_extend_to_none_with_real_offset_raises(self):
        a = _make(name="a")
        with pytest.raises(Exception, match="offset when extending"):
            a.extend_to(other=None, axis=0, direction="+", offset=1e-6)

    def test_extend_to_none_with_grid_offset_raises(self):
        a = _make(name="a")
        with pytest.raises(Exception, match="offset when extending"):
            a.extend_to(other=None, axis=0, direction="+", grid_offset=2)

    def test_other_object_name_captured(self):
        a = _make(name="a")
        b = _make(name="ref_obj")
        c = a.extend_to(other=b, axis=0, direction="+")
        assert c.other_object == "ref_obj"


# ---------------------------------------------------------------------------
# __eq__ and __hash__
# ---------------------------------------------------------------------------


class TestEquality:
    def test_same_name_equal(self):
        a = _make(name="same")
        b = _make(name="same")
        assert a == b

    def test_different_names_not_equal(self):
        a = _make(name="foo")
        b = _make(name="bar")
        assert a != b

    def test_not_equal_to_non_object(self):
        a = _make(name="x")
        assert a != "x"
        assert a != 42
        assert a != None  # noqa: E711

    def test_hash_equal_for_same_name(self):
        a = _make(name="x")
        b = _make(name="x")
        assert hash(a) == hash(b)

    def test_usable_in_set(self):
        a = _make(name="x")
        b = _make(name="y")
        c = _make(name="x")
        s = {a, b, c}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# check_overlap
# ---------------------------------------------------------------------------


class TestCheckOverlap:
    def test_overlapping_objects(self, config, key):
        a = _make(name="a")
        b = _make(name="b")
        placed_a = _place(a, config, key, ((0, 10), (0, 10), (0, 10)))
        placed_b = _place(b, config, key, ((5, 15), (5, 15), (5, 15)))
        assert placed_a.check_overlap(placed_b)

    def test_non_overlapping_objects(self, config, key):
        a = _make(name="a")
        b = _make(name="b")
        placed_a = _place(a, config, key, ((0, 5), (0, 5), (0, 5)))
        placed_b = _place(b, config, key, ((10, 15), (10, 15), (10, 15)))
        assert not placed_a.check_overlap(placed_b)

    def test_touching_objects_reported_as_overlapping(self, config, key):
        a = _make(name="a")
        b = _make(name="b")
        placed_a = _place(a, config, key, ((0, 5), (0, 5), (0, 5)))
        # b starts at 5, which equals s_end of a – boundary condition
        placed_b = _place(b, config, key, ((5, 10), (0, 5), (0, 5)))
        # 5 <= 5 <= 10: True for axis 0 → reports overlap
        assert placed_a.check_overlap(placed_b)


# ---------------------------------------------------------------------------
# OrderableObject
# ---------------------------------------------------------------------------


class TestOrderableObject:
    def test_default_placement_order(self):
        obj = _ConcreteOrderable()
        assert obj.placement_order == 0

    def test_custom_placement_order(self):
        obj = _ConcreteOrderable(placement_order=7)
        assert obj.placement_order == 7
