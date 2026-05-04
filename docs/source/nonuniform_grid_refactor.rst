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

Status: implemented.

The CFL time step now uses the smallest spacing on each axis.  Yee curl terms use
local metric factors for forward E curls and backward H curls while preserving
uniform-grid behavior.  Field interpolation for exact detector sampling also uses
distance-weighted center-to-edge averages on stretched grids.

Uniform-grid parity tests now cover both scalar-resolution configs and explicit
uniform ``GridSpec`` configs so future solver changes have a regression fence.

Tests to add:

* analytic finite differences on stretched grids
* ``curl(grad(phi))`` is approximately zero
* one-step E/H updates match hand-computed local-metric coefficients
* uniform-grid updates remain numerically equivalent to the current solver

Stage 4: PML and Boundary Metrics
---------------------------------

Status: implemented for physical-depth PML profiles, PML update time scaling,
and Bloch phase lengths.

PML grading should be based on physical depth into the boundary, not the number
of cells.  Bloch and periodic phase corrections should use physical domain length
from ``GridSpec`` edges.

Physical-depth PML profile construction is implemented.  The time-domain PML
auxiliary coefficient update now uses ``SimulationConfig.time_step_duration``
instead of reconstructing ``dt`` from a scalar spacing.  Bloch phase correction
uses the physical axis extent from ``GridSpec`` edges, and the boundary padding
path no longer requires a scalar grid before calling grid-aware boundary
corrections.

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

Diffractive detectors resample stretched transverse planes onto a uniform plane
before FFT order analysis.  This preserves the existing FFT backend while making
the coordinate assumption explicit at the detector boundary.

Tests to add:

* constant Poynting flux integrates to ``S * area``
* constant energy density integrates to ``u * volume``
* mode solver coordinates match the selected transverse slice
* uniform-grid detector behavior is either preserved or intentionally migrated
  with documented unit changes

Stage 6: Rasterization, Export, and Visualization
-------------------------------------------------

Status: implemented for rectilinear-grid support.

Geometry masks for spheres, cylinders, and polygons should sample physical cell
coordinates or use fill fractions.  VTI image export should reject non-uniform
grids or be replaced by a rectilinear-grid export.  Plotting should use physical
coordinates from grid edges.

Sphere, cylinder, and extruded-polygon masks now sample physical cell centers on
rectilinear grids.  Detector, setup, and material plots can use rectilinear
physical axes.  VTR export writes explicit ``GridSpec`` edge coordinates for
non-uniform grids; VTI remains a uniform image-data export and rejects stretched
grids with a pointer to VTR.

Linearly polarized plane sources use rectilinear Gaussian profile sampling,
physical-coordinate tilted projection, random TFSF offsets in physical
transverse units, and grid-aware Yee time offsets.  Device parameterization
supports both simulation-cell-count design voxels and physical design voxels;
physical design voxels are mapped to simulation cells by volume-overlap
averaging instead of center sampling.  Fill fractions/subpixel smoothing remain
future accuracy improvements for geometry rasterization, not blockers for
rectilinear-grid execution.

Remaining Guarded Surfaces
--------------------------

The remaining calls to ``require_uniform_grid()`` are compatibility fallbacks or
intentional guards.  They cluster around:

* fallback paths used before a concrete ``GridSpec`` is attached
* legacy VTI image-data export, which cannot encode rectilinear spacing
* index-space placement APIs whose semantics are ambiguous on stretched grids

Performance Notes
-----------------

The metric-aware path can introduce overhead if helpers allocate large broadcast
arrays repeatedly inside JIT-compiled update loops.  The likely follow-up
optimizations are:

* cache ``dx``, ``dy``, ``dz`` and common broadcast shapes in a solver metrics object
* precompute PML physical-depth profiles once during initialization
* extend the detector face-area and cell-volume cache pattern to curl/source
  metric arrays that are still rebuilt inside hot paths
* cache detector/source plot edge coordinates and source profile coordinates at
  placement time when plotting or source re-application becomes hot
* avoid materializing full 3D area/volume arrays when separable 1D weights are enough
* keep uniform grids on the same API path, but allow JAX/compiler constants to
  simplify equal-spacing metric arrays
* profile memory pressure from storing multiple staggered metric arrays before
  adding convenience caches
