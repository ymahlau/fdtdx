"""Unit tests for fdtdx.core.jax.pytrees module."""

import pytest

from fdtdx.core.jax.pytrees import (
    TreeClass,
    TreeClassField,
    autoinit,
    field,
    frozen_field,
    frozen_private_field,
    private_field,
    safe_hasattr,
)
from fdtdx.core.null import NULL

# ---- safe_hasattr ----


class TestSafeHasattr:
    """Tests for the safe_hasattr helper."""

    def test_returns_true_for_existing_attr(self):
        assert safe_hasattr("hello", "upper") is True

    def test_returns_false_for_missing_attr(self):
        assert safe_hasattr("hello", "nonexistent") is False

    def test_returns_false_on_keyerror(self):
        """safe_hasattr should catch KeyError from __getattr__ and return False."""

        class BadAttr:
            def __getattr__(self, name):
                raise KeyError("simulated")

        assert safe_hasattr(BadAttr(), "anything") is False


# ---- TreeClassField ----


class TestTreeClassField:
    """Tests for the TreeClassField dataclass."""

    def test_default_values(self):
        f = TreeClassField(name="x", type=int)
        assert f.name == "x"
        assert f.type is int
        assert f.default == (NULL,)
        assert f.init is True
        assert f.repr is True
        assert f.kind == "POS_OR_KW"
        assert f.metadata is None
        assert f.on_setattr == ()
        assert f.on_getattr == ()
        assert f.alias is None
        assert f.value is NULL

    def test_iter_yields_field_name_value_pairs(self):
        f = TreeClassField(name="x", type=int, init=False)
        d = dict(f)
        assert d["name"] == "x"
        assert d["type"] is int
        assert d["init"] is False

    def test_frozen(self):
        f = TreeClassField(name="x", type=int)
        with pytest.raises(AttributeError):
            f.name = "y"


# ---- _parse_operations ----


class TestParseOperations:
    """Tests for TreeClass._parse_operations static method."""

    fn = staticmethod(TreeClass._parse_operations)

    def test_single_attribute(self):
        assert self.fn("single") == [("single", "attribute")]

    def test_chained_attributes(self):
        result = self.fn("a->b->c")
        assert result == [("a", "attribute"), ("b", "attribute"), ("c", "attribute")]

    def test_integer_index(self):
        result = self.fn("a->[0]")
        assert result == [("a", "attribute"), (0, "index")]

    def test_negative_index(self):
        result = self.fn("a->[-5]")
        assert result == [("a", "attribute"), (-5, "index")]

    def test_string_key(self):
        result = self.fn("a->['name']")
        assert result == [("a", "attribute"), ("name", "key")]

    def test_mixed_operations(self):
        result = self.fn("a->b->[0]->['name']")
        assert result == [
            ("a", "attribute"),
            ("b", "attribute"),
            (0, "index"),
            ("name", "key"),
        ]

    def test_string_key_with_spaces(self):
        result = self.fn("data->['hello world']->result")
        assert result == [
            ("data", "attribute"),
            ("hello world", "key"),
            ("result", "attribute"),
        ]

    # Error cases

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Empty string"):
            self.fn("")

    def test_ends_with_arrow_raises(self):
        with pytest.raises(ValueError, match="ends with"):
            self.fn("a->")

    def test_starts_with_arrow_raises(self):
        with pytest.raises(ValueError, match="Empty attribute at position"):
            self.fn("->b")

    def test_unclosed_bracket_raises(self):
        with pytest.raises(ValueError, match="Unclosed bracket"):
            self.fn("a->[")

    def test_invalid_bracket_content_raises(self):
        with pytest.raises(ValueError, match="Invalid bracket content"):
            self.fn("a->[invalid]")

    def test_invalid_attribute_name_raises(self):
        with pytest.raises(ValueError, match="Invalid attribute name"):
            self.fn("a->123invalid")

    def test_brackets_inside_string_key_raises(self):
        with pytest.raises(ValueError, match="Invalid bracket content"):
            self.fn("a->['has [brackets]']")

    def test_quotes_inside_string_key_raises(self):
        with pytest.raises(ValueError, match="cannot contain single quotes"):
            self.fn("a->['it's bad']")

    def test_missing_arrow_separator_raises(self):
        """After a bracket op, the next char must be '->' if there's more to parse."""
        with pytest.raises(ValueError, match="Expected '->' at position"):
            self.fn("[0]x")


# ---- TreeClass with autoinit ----


@autoinit
class SimpleTree(TreeClass):
    x: int
    y: str = field(default="hello")


@autoinit
class NestedTree(TreeClass):
    child: SimpleTree
    value: float = field(default=1.0)


@autoinit
class TreeWithPrivate(TreeClass):
    public_val: int
    _hidden: str = private_field(default="secret")


@autoinit
class TreeWithFrozen(TreeClass):
    data: tuple = frozen_field(default=(1, 2, 3))


@autoinit
class TreeWithFrozenPrivate(TreeClass):
    internal: tuple = frozen_private_field(default=(("a", 1),))


# ---- field functions ----


class TestFieldFunctions:
    """Tests for field, private_field, frozen_field, frozen_private_field."""

    def test_field_default(self):
        obj = SimpleTree(x=10)
        assert obj.x == 10
        assert obj.y == "hello"

    def test_private_field_not_in_init(self):
        obj = TreeWithPrivate(public_val=42)
        assert obj.public_val == 42
        assert obj._hidden == "secret"

    def test_frozen_field_default(self):
        obj = TreeWithFrozen()
        assert obj.data == (1, 2, 3)

    def test_frozen_private_field_default(self):
        obj = TreeWithFrozenPrivate()
        assert obj.internal == (("a", 1),)


# ---- TreeClass.get_class_fields / get_public_fields ----


class TestTreeClassFields:
    """Tests for get_class_fields and get_public_fields."""

    def test_get_class_fields_returns_all(self):
        obj = TreeWithPrivate(public_val=42)
        fields = obj.get_class_fields()
        names = [f.name for f in fields]
        assert "public_val" in names
        assert "_hidden" in names

    def test_get_class_fields_returns_tree_class_field_instances(self):
        obj = SimpleTree(x=1)
        fields = obj.get_class_fields()
        assert all(isinstance(f, TreeClassField) for f in fields)

    def test_get_public_fields_excludes_private(self):
        obj = TreeWithPrivate(public_val=42)
        public = obj.get_public_fields()
        names = [f.name for f in public]
        assert "public_val" in names
        assert "_hidden" not in names

    def test_get_public_fields_include_value(self):
        obj = SimpleTree(x=99, y="world")
        public = obj.get_public_fields()
        values = {f.name: f.value for f in public}
        assert values["x"] == 99
        assert values["y"] == "world"


# ---- TreeClass.aset ----


class TestAset:
    """Tests for the aset method (non-recursive attribute setting)."""

    def test_set_simple_attribute(self):
        obj = SimpleTree(x=1, y="a")
        updated = obj.aset("x", 42)
        assert updated.x == 42
        assert updated.y == "a"

    def test_set_nested_attribute(self):
        child = SimpleTree(x=1, y="a")
        parent = NestedTree(child=child)
        updated = parent.aset("child->x", 99)
        assert updated.child.x == 99

    def test_set_with_list_index(self):
        @autoinit
        class WithList(TreeClass):
            items: list = frozen_field(default=None)

        obj = WithList(items=[10, 20, 30])
        updated = obj.aset("items->[1]", 99)
        assert updated.items[1] == 99
        assert updated.items[0] == 10

    def test_set_with_dict_key(self):
        @autoinit
        class WithDict(TreeClass):
            data: dict = frozen_field(default=None)

        obj = WithDict(data={"key": "old"})
        updated = obj.aset("data->['key']", "new")
        assert updated.data["key"] == "new"

    def test_nonexistent_attribute_raises(self):
        obj = SimpleTree(x=1)
        with pytest.raises(Exception, match="does not exist"):
            obj.aset("nonexistent", 42)

    def test_create_new_ok(self):
        obj = SimpleTree(x=1)
        updated = obj.aset("new_attr", 42, create_new_ok=True)
        assert updated.new_attr == 42

    def test_original_unchanged(self):
        obj = SimpleTree(x=1, y="a")
        _ = obj.aset("x", 42)
        assert obj.x == 1

    def test_missing_dict_key_raises(self):
        @autoinit
        class WithDict(TreeClass):
            data: dict = frozen_field(default=None)

        obj = WithDict(data={"a": 1})
        with pytest.raises(Exception, match="does not exist"):
            obj.aset("data->['missing']", 99)

    def test_missing_dict_key_create_new_ok(self):
        @autoinit
        class WithDict(TreeClass):
            data: dict = frozen_field(default=None)

        obj = WithDict(data={"a": 1})
        updated = obj.aset("data->['new_key']", 99, create_new_ok=True)
        assert updated.data["new_key"] == 99

    def test_index_on_non_subscriptable_raises(self):
        obj = SimpleTree(x=1)
        with pytest.raises(Exception, match="does not implement __getitem__"):
            obj.aset("x->[0]", 42)

    def test_key_on_non_subscriptable_raises(self):
        """Traversing a string key on a non-dict-like object should raise."""
        obj = SimpleTree(x=1)
        with pytest.raises(Exception, match="does not implement __getitem__"):
            obj.aset("x->['key']", 42)

    def test_set_attr_on_non_treeclass_raises(self):
        """Setting attribute through a non-TreeClass intermediate should raise in backward pass."""

        class Plain:
            def __init__(self):
                self.val = 42

        @autoinit
        class Holder(TreeClass):
            inner: object = field(default=None)

        obj = Holder(inner=Plain())
        with pytest.raises(Exception, match="Can only set attribute on"):
            obj.aset("inner->val", 99)

    def test_set_index_on_immutable_sequence_raises(self):
        """Setting by index on tuple (no __setitem__) in backward pass should raise."""

        @autoinit
        class WithTuple(TreeClass):
            items: tuple = field(default=(10, 20, 30))

        obj = WithTuple()
        with pytest.raises(Exception, match="__setitem__ is implemented"):
            obj.aset("items->[0]", 99)


# ---- autoinit ----


class TestAutoinit:
    """Tests for the autoinit decorator."""

    def test_preserves_custom_init(self):
        class CustomInit(TreeClass):
            def __init__(self, val):
                self.val = val

        decorated = autoinit(CustomInit)
        assert decorated is CustomInit
