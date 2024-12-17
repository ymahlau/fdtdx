"""Type definitions for 3D shapes, grids and slices used in FDTD simulations.

This module defines type aliases and constants for working with three-dimensional
shapes, grids and array slicing in FDTD simulations. It includes types for both
real-valued (physical) and integer-valued (grid) coordinates.
"""

from typing import Optional

# Real-valued shapes (physical dimensions)
RealShape3D = tuple[float, float, float]
"""3D shape with real-valued (physical) dimensions in meters."""

OptionalAxisSize = Optional[float]
"""Optional real-valued size for a single axis."""

PartialRealShape3D = tuple[OptionalAxisSize, OptionalAxisSize, OptionalAxisSize]
"""Partial 3D shape where some physical dimensions may be undefined (None)."""

# Grid-based shapes (discrete points)
GridShape3D = tuple[int, int, int]
"""3D shape with integer dimensions in grid points."""

OptionalGridAxisSize = Optional[int]
"""Optional integer size for a single grid axis."""

PartialGridShape3D = tuple[OptionalGridAxisSize, OptionalGridAxisSize, OptionalGridAxisSize]
"""Partial 3D grid shape where some dimensions may be undefined (None)."""

# Common shape constants
INVALID_SHAPE_3D = (-1, -1, -1)  # both real or grid
"""Invalid 3D shape marker, usable for both real and grid shapes."""

UNDEFINED_SHAPE_3D = (None, None, None)  # both real or grid
"""Undefined 3D shape marker, usable for both real and grid shapes."""

# Slice tuple types for array indexing
SliceTuple3D = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
"""3D slice specification using (start, stop) integer tuples for each axis."""

OptionalAxisSliceTuple = Optional[tuple[int, int]]
"""Optional (start, stop) tuple for slicing a single axis."""

PartialSliceTuple3D = tuple[OptionalAxisSliceTuple, OptionalAxisSliceTuple, OptionalAxisSliceTuple]
"""Partial 3D slice where some axes may be undefined (None)."""

INVALID_SLICE_TUPLE_3D: SliceTuple3D = ((-1, -1), (-1, -1), (-1, -1))
"""Invalid 3D slice marker using -1 for all bounds."""

# Python slice object types
Slice3D = tuple[slice, slice, slice]
"""3D slice specification using Python slice objects for each axis."""

OptionalAxisSlice = Optional[slice]
"""Optional Python slice object for a single axis."""

PartialSlice3D = tuple[OptionalAxisSlice, OptionalAxisSlice, OptionalAxisSlice]
"""Partial 3D slice where some axes may be undefined (None)."""
