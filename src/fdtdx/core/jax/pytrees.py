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


class ExtendedTreeClassIndexer(TreeClassIndexer):
    """Extended indexer for tree class that preserves type information.

    Extends TreeClassIndexer to properly handle type hints and return Self type.
    """

    def __getitem__(self, where: Any) -> Self:
        """Gets item at specified index while preserving type information.

        Args:
            where: Index or key to access

        Returns:
            Self: The indexed item with proper type information preserved
        """
        return super().__getitem__(where)  # type: ignore


class ExtendedTreeClass(tc.TreeClass):
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
        """Internal helper for setting attributes directly.

        Args:
            attr_name: Name of attribute to set
            val: Value to set the attribute to
        """
        setattr(self, attr_name, val)

    def aset(
        self,
        attr_name: str,
        val: Any,
    ) -> Self:
        """Sets an attribute directly without recursive application.

        Similar to Self.at[attr_name].set(val), but without recursively
        applying to each tree leaf. Instead, replaces the full attribute
        with the new value.

        Args:
            attr_name: Name of attribute to set
            val: Value to set the attribute to

        Returns:
            Self: Updated instance with new attribute value
        """
        _, self = self.at["_aset"](attr_name, val)
        return self


T = TypeVar("T")


@overload
def field(
    *,
    default: T,
    init: bool = True,
    repr: bool = True,
    kind: ArgKindType = "POS_OR_KW",
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
    kind: ArgKindType = "POS_OR_KW",
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
    kind: ArgKindType = "POS_OR_KW",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> Any:
    """Creates a field that automatically freezes on set and unfreezes on get.

    This field behaves like a regular pytreeclass field but ensures values are
    frozen when stored and unfrozen when accessed.

    Args:
        default: The default value for the field
        init: Whether to include the field in __init__
        repr: Whether to include the field in __repr__
        kind: The argument kind (POS_ONLY, POS_OR_KW, etc.)
        metadata: Additional metadata for the field
        on_setattr: Additional setattr callbacks (applied after freezing)
        on_getattr: Additional getattr callbacks (applied after unfreezing)
        alias: Alternative name for the field in __init__

    Returns:
        A Field instance configured with freeze/unfreeze behavior
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
    kind: ArgKindType = "POS_OR_KW",
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
    kind: ArgKindType = "POS_OR_KW",
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
    kind: ArgKindType = "POS_OR_KW",
    metadata: dict[str, Any] | None = None,
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> Any:
    """Creates a field that automatically freezes on set and unfreezes on get.

    This field behaves like a regular pytreeclass field but ensures values are
    frozen when stored and unfrozen when accessed.

    Args:
        default: The default value for the field
        init: Whether to include the field in __init__
        repr: Whether to include the field in __repr__
        kind: The argument kind (POS_ONLY, POS_OR_KW, etc.)
        metadata: Additional metadata for the field
        on_setattr: Additional setattr callbacks (applied after freezing)
        on_getattr: Additional getattr callbacks (applied after unfreezing)
        alias: Alternative name for the field in __init__

    Returns:
        A Field instance configured with freeze/unfreeze behavior
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
    kind: ArgKindType = "POS_OR_KW",
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
    kind: ArgKindType = "POS_OR_KW",
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
    kind: ArgKindType = "POS_OR_KW",
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
        default: The default value for the field
        init: Whether to include the field in __init__
        repr: Whether to include the field in __repr__
        kind: The argument kind (POS_ONLY, POS_OR_KW, etc.)
        metadata: Additional metadata for the field
        on_setattr: Additional setattr callbacks (applied after freezing)
        on_getattr: Additional getattr callbacks (applied after unfreezing)
        alias: Alternative name for the field in __init__

    Returns:
        A Field instance configured with freeze/unfreeze behavior
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


@dataclass_transform(field_specifiers=(Field, tc_field, frozen_field, frozen_private_field))
def extended_autoinit(klass: type[T]) -> type[T]:
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
