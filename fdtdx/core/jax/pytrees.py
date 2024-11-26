from typing import Any, Callable, Optional, Self, Sequence, TypeVar, TypedDict, Unpack, overload
import pytreeclass as tc
from pytreeclass._src.tree_base import TreeClassIndexer
from pytreeclass._src.code_build import (
    NULL, ArgKind, ArgKindType, Field, build_init_method, convert_hints_to_fields,
    dataclass_transform, field
)

class ExtendedTreeClassIndexer(TreeClassIndexer):
    def __getitem__(self, where: Any) -> Self:
        return super().__getitem__(where)  # type: ignore

class ExtendedTreeClass(tc.TreeClass):
    
    @property
    def at(self) -> ExtendedTreeClassIndexer:
        return super().at  # type: ignore
    
    def _aset(
        self,
        attr_name: str,
        val: Any,
    ):
        setattr(self, attr_name, val)
    
    def aset(
        self,
        attr_name: str,
        val: Any,
    ) -> Self:
        """
        Self.at[attr_name].set(val), but without recursive application
        to each tree leaf. Instead, the full attribute is replaced with
        the new value
        """
        _, self = self.at["_aset"](attr_name, val)
        return self

T = TypeVar('T')

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
    return tc.field(
        default=default,
        init=init,
        repr=repr,
        kind=kind,
        metadata=metadata,
        on_setattr=[tc.freeze] + list(on_setattr),
        on_getattr=[tc.unfreeze] + list(on_getattr),
        alias=alias,
    )

@dataclass_transform(field_specifiers=(Field, field, frozen_field))
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
