def get_transverse_axes(axis: int) -> tuple[int, int]:
    """Return the two axes perpendicular to *axis* in ascending order."""
    result = [a for a in range(3) if a != axis]
    return result[0], result[1]


def get_oriented_transverse_axes(axis: int) -> tuple[int, int]:
    """Return the two axes perpendicular to *axis* in right-hand cyclic order."""
    return (axis + 1) % 3, (axis + 2) % 3
