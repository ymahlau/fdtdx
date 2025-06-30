from typing import Any, Self, Sequence, TypeVar, overload

import pytreeclass as tc
from pytreeclass._src.code_build import (
    NULL,
    ArgKindType,
    Field,
    build_init_method,
    convert_hints_to_fields,
    dataclass_transform,
)
from pytreeclass._src.code_build import (
    field as tc_field,
)
from pytreeclass._src.tree_base import TreeClassIndexer


def safe_hasattr(obj, name) -> bool:
    try:
        result = hasattr(obj, name)
        return result
    except KeyError:
        return False


class ExtendedTreeClassIndexer(TreeClassIndexer):
    """Extended indexer for tree class that preserves type information.

    Extends TreeClassIndexer to properly handle type hints and return Self type.
    """

    def __getitem__(self, where: Any) -> Self:
        return super().__getitem__(where)  # type: ignore


class TreeClass(tc.TreeClass):
    """Extended tree class with improved attribute setting functionality.

    Extends TreeClass to provide more flexible attribute setting capabilities,
    particularly for handling non-recursive attribute updates.
    """

    @property
    def at(self) -> ExtendedTreeClassIndexer:
        """Gets the extended indexer for this tree.

        Returns:
            ExtendedTreeClassIndexer: Indexer that preserves type information
        """
        return super().at  # type: ignore

    def _aset(
        self,
        attr_name: str,
        val: Any,
    ):
        setattr(self, attr_name, val)

    @staticmethod
    def _parse_operations(s: str) -> list[tuple[str, str]]:
        if not s:
            raise ValueError("Empty string is not valid")

        operations = []
        i = 0

        while i < len(s):
            if i > 0:
                # Expect "->" separator
                if not s[i:].startswith("->"):
                    raise ValueError(f"Expected '->' at position {i}")
                i += 2  # Skip "->"

                if i >= len(s):
                    raise ValueError("String ends with '->'")

            # Parse the next operation
            if s[i] == "[":
                # Find the closing bracket
                j = i + 1
                while j < len(s) and s[j] != "]":
                    j += 1

                if j >= len(s):
                    raise ValueError(f"Unclosed bracket starting at position {i}")

                bracket_content = s[i + 1 : j].strip()

                # Determine if it's an integer or string
                if bracket_content.isdigit() or (bracket_content.startswith("-") and bracket_content[1:].isdigit()):
                    operations.append((int(bracket_content), "index"))
                elif bracket_content.startswith("'") and bracket_content.endswith("'"):
                    # Extract string content
                    if len(bracket_content) < 2:
                        raise ValueError(f"Invalid string format in brackets: [{bracket_content}]")

                    string_content = bracket_content[1:-1]

                    # Check for forbidden characters
                    if "'" in string_content:
                        raise ValueError(f"String keys cannot contain single quotes: '{string_content}'")
                    if "[" in string_content or "]" in string_content:
                        raise ValueError(f"String keys cannot contain square brackets: '{string_content}'")

                    operations.append((string_content, "key"))
                else:
                    raise ValueError(f"Invalid bracket content: [{bracket_content}]")

                i = j + 1
            else:
                # Parse attribute name
                j = i
                while j < len(s) and s[j : j + 2] != "->":
                    j += 1

                attr_name = s[i:j]

                # Validate attribute name
                if not attr_name:
                    raise ValueError(f"Empty attribute at position {i}")

                # Check if it's a valid Python identifier
                if not attr_name.isidentifier():
                    raise ValueError(f"Invalid attribute name: '{attr_name}'")

                operations.append((attr_name, "attribute"))
                i = j

        return operations

    def aset(
        self,
        attr_name: str,
        val: Any,
        create_new_ok: bool = False,
    ) -> Self:
        """Sets an attribute of this class. In contrast to the classical .at[].set(), this method updates the class
        attribute directly and does not only operate on jax pytree leaf nodes. Instead, replaces the full attribute
        with the new value.

        The attribute can either be the attribute name of this class, or for nested classes it can also be the
        attribute name of a class, which itself is an attribute of this class. The syntax for this operation could
        look like this: "a->b->[0]->['name']". Here, the current class has an attribute a, which has an attribute b,
        which is a list, which we index at index 0, which is an element of type dictionary, which we index using
        the dictionary key 'name'.

        Note that dictionary keys cannot contain square brackets or single quotes (even if they are escaped).

        Args:
            attr_name (str): Name of attribute to set
            val (Any): Value to set the attribute to
            create_new_ok (bool, optional): If false (default), throw an error if the attribute does not exist.
                If true, creates a new attribute if the attribute name does not exist yet.

        Returns:
            Self: Updated instance with new attribute value
        """
        # parse operations
        ops = self._parse_operations(attr_name)

        # find final attribute and save intermediate attributes
        attr_list = [self]
        current_parent = self
        for idx, (op, op_type) in enumerate(ops):
            if op_type == "attribute":
                if not safe_hasattr(current_parent, op):
                    if idx != len(ops) - 1 or not create_new_ok:
                        raise Exception(f"Attribute: {op} does not exist for {current_parent.__class__}")
                    current_parent = None
                else:
                    current_parent = getattr(current_parent, op)
            elif op_type == "index":
                if "__getitem__" not in dir(current_parent):
                    raise Exception(f"{current_parent.__class__} does not implement __getitem__")
                current_parent = current_parent[int(op)]  # type: ignore
            elif op_type == "key":
                if "__getitem__" not in dir(current_parent):
                    raise Exception(f"{current_parent.__class__} does not implement __getitem__")
                if op not in current_parent:  # type: ignore
                    if idx != len(ops) - 1 or not create_new_ok:
                        raise Exception(f"Key: {op} does not exist for {current_parent}")
                    current_parent = None
                else:
                    current_parent = current_parent[op]  # type: ignore
            else:
                raise Exception(f"Invalid operation type: {op_type}. This is an internal bug!")
            if idx != len(ops) - 1:
                attr_list.append(current_parent)  # type: ignore

        # from bottom-up set attributes and update
        cur_attr = val
        for idx in list(range(len(attr_list)))[::-1]:
            op, op_type = ops[idx]
            current_parent = attr_list[idx]
            if op_type == "attribute":
                if not isinstance(current_parent, TreeClass):
                    raise Exception(f"Can only set attribute on ExtendedTreeClass, but got {current_parent.__class__}")
                _, cur_attr = current_parent.at["_aset"](op, cur_attr)
            elif op_type == "index":
                if "__setitem__" not in dir(current_parent):
                    raise Exception(
                        f"Can only update by index if __setitem__ is implemented, but got {current_parent.__class__}"
                    )
                cpy = current_parent.copy()  # type: ignore
                cpy[int(op)] = cur_attr  # type: ignore
                cur_attr = cpy
            elif op_type == "key":
                if "__setitem__" not in dir(current_parent):
                    raise Exception(
                        f"Can only update by index if __setitem__ is implemented, but got {current_parent.__class__}"
                    )
                cpy = current_parent.copy()  # type: ignore
                cpy[op] = cur_attr  # type: ignore
                cur_attr = cpy
            else:
                raise Exception(f"Invalid operation type: {op_type}. This is an internal bug!")

        assert cur_attr.__class__ == self.__class__
        return cur_attr


T = TypeVar("T")


@overload
def field(
    *,
    default: T,
    init: bool = True,
    repr: bool = True,
    kind: ArgKindType = "KW_ONLY",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> T: ...


@overload
def field(
    *,
    init: bool = True,
    repr: bool = True,
    kind: ArgKindType = "KW_ONLY",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> Any: ...


def field(
    *,
    default: Any = NULL,
    init: bool = True,
    repr: bool = True,
    kind: ArgKindType = "KW_ONLY",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> Any:
    """
    A wrapper for pytreeclass fields. Allows specification of more advanced features.

    Args:
        default (Any, optional): The default value for the field. Defaults to None.
        init (bool, optional): Whether to include the field in __init__. Defaults to True.
        repr (bool, optional): Whether to include the field in __repr__. Defaults to True.
        kind (ArgKindType, optional): The argument kind (POS_ONLY, POS_OR_KW, etc.). Defaults to KW_ONLY.
        metadata (dict[str, Any] | None, optional): Additional metadata for the field. Defaults to None.
        on_setattr (Sequence[Any], optional): Additional setattr callbacks. Defaults to no callbacks.
        on_getattr (Sequence[Any], optional): Additional getattr callbacks. Defaults to no callbacks.
        alias (str | None, optional): Alternative name for the field in __init__. Defaults to None

    Returns:
        Any: A Field instance configured with freeze/unfreeze behavior
    """
    return tc_field(
        default=default,
        init=init,
        repr=repr,
        kind=kind,
        metadata=metadata,
        on_setattr=on_setattr,
        on_getattr=on_getattr,
        alias=alias,
    )


@overload
def private_field(
    *,
    default: T,
    init: bool = False,
    repr: bool = True,
    kind: ArgKindType = "KW_ONLY",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> T: ...


@overload
def private_field(
    *,
    init: bool = False,
    repr: bool = True,
    kind: ArgKindType = "KW_ONLY",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> Any: ...


def private_field(
    *,
    default: Any = NULL,
    init: bool = False,
    repr: bool = True,
    kind: ArgKindType = "KW_ONLY",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> Any:
    """
    Creates a field that sets the default to None and init to False.

    Args:
        default (Any, optional): The default value for the field. Defaults to None.
        init (bool, optional): Whether to include the field in __init__. Defaults to False.
        repr (bool, optional): Whether to include the field in __repr__. Defaults to True.
        kind (ArgKindType, optional): The argument kind (POS_ONLY, POS_OR_KW, etc.). Defaults to KW_ONLY.
        metadata (dict[str, Any] | None, optional): Additional metadata for the field. Defaults to None.
        on_setattr (Sequence[Any], optional): Additional setattr callbacks. Defaults to no callbacks.
        on_getattr (Sequence[Any], optional): Additional getattr callbacks. Defaults to no callbacks.
        alias (str | None, optional): Alternative name for the field in __init__. Defaults to None

    Returns:
        Any: A private field instance.
    """
    return tc_field(
        default=default,
        init=init,
        repr=repr,
        kind=kind,
        metadata=metadata,
        on_setattr=on_setattr,
        on_getattr=on_getattr,
        alias=alias,
    )


@overload
def frozen_field(
    *,
    default: T,
    init: bool = True,
    repr: bool = True,
    kind: ArgKindType = "KW_ONLY",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> T: ...


@overload
def frozen_field(
    *,
    init: bool = True,
    repr: bool = True,
    kind: ArgKindType = "KW_ONLY",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> Any: ...


def frozen_field(
    *,
    default: Any = NULL,
    init: bool = True,
    repr: bool = True,
    kind: ArgKindType = "KW_ONLY",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> Any:
    """Creates a field that automatically freezes on set and unfreezes on get.

    This field behaves like a regular pytreeclass field but ensures values are
    frozen when stored and unfrozen when accessed.

    Args:
        default (Any, optional): The default value for the field. Defaults to None.
        init (bool, optional): Whether to include the field in __init__. Defaults to True.
        repr (bool, optional): Whether to include the field in __repr__. Defaults to True.
        kind (ArgKindType, optional): The argument kind (POS_ONLY, POS_OR_KW, etc.). Defaults to KW_ONLY.
        metadata (dict[str, Any] | None, optional): Additional metadata for the field. Defaults to None.
        on_setattr (Sequence[Any], optional): Additional setattr callbacks (applied after freezing).
            Defaults to no callbacks.
        on_getattr (Sequence[Any], optional): Additional getattr callbacks (applied after unfreezing).
            Defaults to no callbacks.
        alias (str | None, optional): Alternative name for the field in __init__. Defaults to None

    Returns:
        Any: A Field instance configured with freeze/unfreeze behavior
    """
    return tc_field(
        default=default,
        init=init,
        repr=repr,
        kind=kind,
        metadata=metadata,
        on_setattr=list(on_setattr) + [tc.freeze],
        on_getattr=[tc.unfreeze] + list(on_getattr),
        alias=alias,
    )


@overload
def frozen_private_field(
    *,
    default: T,
    init: bool = False,
    repr: bool = True,
    kind: ArgKindType = "KW_ONLY",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> T: ...


@overload
def frozen_private_field(
    *,
    init: bool = False,
    repr: bool = True,
    kind: ArgKindType = "KW_ONLY",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> Any: ...


def frozen_private_field(
    *,
    default: Any = None,
    init: bool = False,
    repr: bool = True,
    kind: ArgKindType = "KW_ONLY",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> Any:
    """Creates a field that automatically freezes on set and unfreezes on get,
    sets the default to None and init to False.

    This field behaves like a regular pytreeclass field but ensures values are
    frozen when stored and unfrozen when accessed.

    Args:
        default (Any, optional): The default value for the field. Defaults to None.
        init (bool, optional): Whether to include the field in __init__. Defaults to False.
        repr (bool, optional): Whether to include the field in __repr__. Defaults to True.
        kind (ArgKindType, optional): The argument kind (POS_ONLY, POS_OR_KW, etc.). Defaults to KW_ONLY.
        metadata (dict[str, Any] | None, optional): Additional metadata for the field. Defaults to None.
        on_setattr (Sequence[Any], optional): Additional setattr callbacks (applied after freezing).
            Defaults to no callbacks.
        on_getattr (Sequence[Any], optional): Additional getattr callbacks (applied after unfreezing).
            Defaults to no callbacks.
        alias (str | None, optional): Alternative name for the field in __init__. Defaults to None.

    Returns:
        Any: A Field instance configured with freeze/unfreeze behavior
    """
    return frozen_field(
        default=default,
        init=init,
        repr=repr,
        kind=kind,
        metadata=metadata,
        on_setattr=on_setattr,
        on_getattr=on_getattr,
        alias=alias,
    )


@dataclass_transform(
    field_specifiers=(Field, tc_field, frozen_field, frozen_private_field, field, private_field),
    kw_only_default=True,
)
def autoinit(klass: type[T]) -> type[T]:
    """Wrapper around tc.autoinit that preserves parameter requirement information"""
    return (
        klass
        # if the class already has a user-defined __init__ method
        # then return the class as is without any modification
        if "__init__" in vars(klass)
        # first convert the current class hints to fields
        # then build the __init__ method from the fields of the current class
        # and any base classes that are decorated with `autoinit`
        else build_init_method(convert_hints_to_fields(klass))
    )
