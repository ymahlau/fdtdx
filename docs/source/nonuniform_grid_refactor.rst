Non-Uniform Grid Refactor
=========================

This document tracks the staged implementation plan for non-uniform rectilinear
grids.  The main design rule is that uniform grids are a special case of the
same data structure: fdtdx internals should consume :class:`fdtdx.GridSpec`
instead of branching between a scalar-resolution solver and a non-uniform solver.

Stage 1: Grid Representation and Uniform Compatibility
------------------------------------------------------

Status: implemented.

The first stage introduces ``GridSpec`` as the canonical grid representation.
It stores physical edge coordinates for x, y, and z and derives cell widths,
centers, extents, face areas, and volumes from those coordinates.

``SimulationConfig.resolution`` remains available as a compatibility constructor,
but compiled simulations attach a concrete ``SimulationConfig.grid``.  Code paths
that still require a scalar spacing now call ``require_uniform_grid()`` so they
fail explicitly for non-uniform grids instead of silently applying the wrong
metric.

Stage 2: Coordinate-Aware Placement
-----------------------------------

Replace placement math that converts metres to indices through one scalar
spacing.  Object placement should use grid edge coordinates and explicit snapping
rules.  Grid-distance APIs should either be removed from public non-uniform use
or rejected clearly when ``grid.is_uniform`` is false.

Tests to add:

* real coordinate constraints snap to the expected edge on stretched grids
* real shape constraints choose intervals that cover the requested physical size
* object ``real_shape`` is the physical edge extent of its grid slice
* grid margins, grid offsets, and grid-coordinate constraints reject non-uniform grids

Stage 3: Solver Metrics and CFL
-------------------------------

Make Yee curl/update coefficients dimensionally explicit.  Derivatives should
divide by the local E/H staggered spacing, and update equations should use the
physical time step rather than hiding spacing inside a scalar Courant number.

Tests to add:

* analytic finite differences on stretched grids
* ``curl(grad(phi))`` is approximately zero
* one-step E/H updates match hand-computed local-metric coefficients
* uniform-grid updates remain numerically equivalent to the current solver

Stage 4: PML and Boundary Metrics
---------------------------------

PML grading should be based on physical depth into the boundary, not the number
of cells.  Bloch and periodic phase corrections should use physical domain length
from ``GridSpec`` edges.

Tests to add:

* uniform PML profile parity
* stretched-grid PML profile follows physical distance
* min and max side profiles mirror correctly
* Bloch phase uses edge extent rather than ``shape * spacing``

Stage 5: Weighted Detectors and Mode Coordinates
------------------------------------------------

Detector reductions must become physical integrals.  Flux detectors need face
area weights, energy detectors need volume weights, and mode overlap should use
transverse area weights.  Tidy3D mode solving should receive transverse coordinate
arrays from ``GridSpec`` rather than generated uniform coordinates.

Tests to add:

* constant Poynting flux integrates to ``S * area``
* constant energy density integrates to ``u * volume``
* mode solver coordinates match the selected transverse slice
* uniform-grid detector behavior is either preserved or intentionally migrated
  with documented unit changes

Stage 6: Rasterization, Export, and Visualization
-------------------------------------------------

Geometry masks for spheres, cylinders, and polygons should sample physical cell
coordinates or use fill fractions.  VTI image export should reject non-uniform
grids or be replaced by a rectilinear-grid export.  Plotting should use physical
coordinates from grid edges.

Performance Notes
-----------------

The metric-aware path can introduce overhead if helpers allocate large broadcast
arrays repeatedly inside JIT-compiled update loops.  The likely follow-up
optimizations are:

* cache ``dx``, ``dy``, ``dz`` and common broadcast shapes in a solver metrics object
* precompute PML physical-depth profiles once during initialization
* avoid materializing full 3D area/volume arrays when separable 1D weights are enough
* keep uniform grids on the same API path, but allow JAX/compiler constants to
  simplify equal-spacing metric arrays
* profile memory pressure from storing multiple staggered metric arrays before
  adding convenience caches
