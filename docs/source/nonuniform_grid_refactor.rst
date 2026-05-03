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

Status: implemented for real-coordinate and real-shape placement.

Placement math that converts metres to indices through one scalar spacing now
uses grid edge coordinates and explicit snapping rules.  Grid-distance APIs that
would be ambiguous on stretched grids reject non-uniform grids clearly.

Tests to add:

* real coordinate constraints snap to the expected edge on stretched grids
* real shape constraints choose intervals that cover the requested physical size
* object ``real_shape`` is the physical edge extent of its grid slice
* grid margins, grid offsets, and grid-coordinate constraints reject non-uniform grids

Stage 3: Solver Metrics and CFL
-------------------------------

Status: partially implemented.

The CFL time step now uses the smallest spacing on each axis.  Yee curl terms use
local metric factors for forward E curls and backward H curls while preserving
uniform-grid behavior.  Field interpolation for exact detector sampling also uses
distance-weighted center-to-edge averages on stretched grids.

Remaining work: PML auxiliary update coefficients and source/TFSF corrections
still need full non-uniform metric support before complete simulations with those
features should be considered supported.

Tests to add:

* analytic finite differences on stretched grids
* ``curl(grad(phi))`` is approximately zero
* one-step E/H updates match hand-computed local-metric coefficients
* uniform-grid updates remain numerically equivalent to the current solver

Stage 4: PML and Boundary Metrics
---------------------------------

Status: partially implemented.

PML grading should be based on physical depth into the boundary, not the number
of cells.  Bloch and periodic phase corrections should use physical domain length
from ``GridSpec`` edges.

Physical-depth PML profile construction is implemented.  The time-domain PML
auxiliary coefficient update now uses ``SimulationConfig.time_step_duration``
instead of reconstructing ``dt`` from a scalar spacing.  Bloch phase correction
still uses uniform spacing and remains a non-uniform boundary blocker.

Tests to add:

* uniform PML profile parity
* stretched-grid PML profile follows physical distance
* min and max side profiles mirror correctly
* Bloch phase uses edge extent rather than ``shape * spacing``

Stage 5: Weighted Detectors and Mode Coordinates
------------------------------------------------

Status: implemented for Poynting flux, energy, reduced field/phasor averages,
mode overlap, Tidy3D mode-solver coordinates, and mode-source coordinates.

Detector reductions must become physical integrals.  Flux detectors need face
area weights, energy detectors need volume weights, and mode overlap should use
transverse area weights.  Tidy3D mode solving should receive transverse coordinate
arrays from ``GridSpec`` rather than generated uniform coordinates.

Diffractive detectors remain uniform-only because the current FFT-based order
decomposition assumes uniform transverse samples.  Supporting stretched grids
there likely needs either resampling or a non-uniform Fourier transform strategy.

Tests to add:

* constant Poynting flux integrates to ``S * area``
* constant energy density integrates to ``u * volume``
* mode solver coordinates match the selected transverse slice
* uniform-grid detector behavior is either preserved or intentionally migrated
  with documented unit changes

Stage 6: Rasterization, Export, and Visualization
-------------------------------------------------

Status: partially implemented.

Geometry masks for spheres, cylinders, and polygons should sample physical cell
coordinates or use fill fractions.  VTI image export should reject non-uniform
grids or be replaced by a rectilinear-grid export.  Plotting should use physical
coordinates from grid edges.

Sphere, cylinder, and extruded-polygon masks now sample physical cell centers on
rectilinear grids.  Setup/material plots, diffractive detectors, generic
linearly polarized plane sources, random TFSF offsets, and device
parameterization now reject non-uniform grids explicitly where the old behavior
would have silently used scalar spacing.  Fill fractions/subpixel smoothing,
rectilinear plotting/export, and true non-uniform source correction remain open.

Remaining Uniform-Only Surfaces
-------------------------------

The remaining calls to ``require_uniform_grid()`` are intentional markers.  They
cluster around:

* PML auxiliary update coefficients inside the curl functions
* Bloch boundary phase correction
* generic TFSF/source profile sampling and correction metrics
* diffractive detector FFT order decomposition, currently guarded
* plotting and image/video export, currently guarded for setup/material plots
* device parameterization helpers that assume one voxel size, currently guarded
* fallback paths used before a concrete ``GridSpec`` is attached

Performance Notes
-----------------

The metric-aware path can introduce overhead if helpers allocate large broadcast
arrays repeatedly inside JIT-compiled update loops.  The likely follow-up
optimizations are:

* cache ``dx``, ``dy``, ``dz`` and common broadcast shapes in a solver metrics object
* precompute PML physical-depth profiles once during initialization
* cache detector face-area and cell-volume weights at placement/init time instead
  of rebuilding them during each detector update
* avoid materializing full 3D area/volume arrays when separable 1D weights are enough
* keep uniform grids on the same API path, but allow JAX/compiler constants to
  simplify equal-spacing metric arrays
* profile memory pressure from storing multiple staggered metric arrays before
  adding convenience caches
