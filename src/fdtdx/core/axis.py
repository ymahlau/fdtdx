from numbers import Integral
from typing import Any


def validate_axis(axis: Any, name: str = "axis") -> int:
    """Validate a Cartesian axis index and return it as an ``int``."""
    if not isinstance(axis, Integral) or isinstance(axis, bool) or axis not in (0, 1, 2):
        raise ValueError(f"{name} must be 0, 1, or 2.")
    return int(axis)


def get_transverse_axes(axis: int) -> tuple[int, int]:
    """Return the two axes perpendicular to *axis* in ascending order."""
    axis = validate_axis(axis)
    result = [a for a in range(3) if a != axis]
    return result[0], result[1]


def get_oriented_transverse_axes(axis: int) -> tuple[int, int]:
    """Return the two axes perpendicular to *axis* in right-hand cyclic order."""
    axis = validate_axis(axis)
    return (axis + 1) % 3, (axis + 2) % 3
