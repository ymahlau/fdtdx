from typing import Optional
import jax

# real shapes
RealShape3D = tuple[float, float, float]
OptionalAxisSize = Optional[float]
PartialRealShape3D = tuple[OptionalAxisSize, OptionalAxisSize, OptionalAxisSize]

# grid shapes
GridShape3D = tuple[int, int, int]
OptionalGridAxisSize = Optional[int]
PartialGridShape3D = tuple[OptionalGridAxisSize, OptionalGridAxisSize, OptionalGridAxisSize]

INVALID_SHAPE_3D = (-1, -1, -1)  # both real or grid
UNDEFINED_SHAPE_3D = (None, None, None)  # both real or grid


# slice tuples
SliceTuple3D = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
OptionalAxisSliceTuple = Optional[tuple[int, int]]
PartialSliceTuple3D = tuple[OptionalAxisSliceTuple, OptionalAxisSliceTuple, OptionalAxisSliceTuple]
INVALID_SLICE_TUPLE_3D: SliceTuple3D = ((-1, -1), (-1, -1), (-1, -1))

# slices
Slice3D = tuple[slice, slice, slice]
OptionalAxisSlice = Optional[slice]
PartialSlice3D = tuple[OptionalAxisSlice, OptionalAxisSlice, OptionalAxisSlice]


