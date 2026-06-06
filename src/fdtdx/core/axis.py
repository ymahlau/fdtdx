def get_transverse_axes(axis: int) -> tuple[int, int]:
    """Return the two axes perpendicular to *axis* in ascending order."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    result = [a for a in range(3) if a != axis]
    return result[0], result[1]


def get_oriented_transverse_axes(axis: int) -> tuple[int, int]:
    """Return the two axes perpendicular to *axis* in right-hand cyclic order."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    return (axis + 1) % 3, (axis + 2) % 3
