import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.path import Path

from fdtdx import constants
from fdtdx.core.axis import get_transverse_axes
from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field, frozen_private_field
from fdtdx.core.misc import validate_symmetric_axis_cells


@autoinit
class UniformGrid(TreeClass):
    """Unresolved policy for a uniform rectilinear grid.

    ``UniformGrid`` is user intent, not the solver mesh itself.  It records the
    physical cell spacing while the final simulation shape may still be unknown.
    Object placement resolves this policy to a concrete ``RectilinearGrid`` once
    the volume shape is known.

    Keeping uniform spacing here avoids a second scalar discretization source on
    ``SimulationConfig``.  Uniform grids and explicitly non-uniform grids both
    enter the solver through the same realized ``RectilinearGrid`` structure.

    The grid origin is at the **center** of the simulation domain.  Edge arrays
    therefore span ``[-N/2 * spacing, +N/2 * spacing]`` along each axis, giving
    the domain symmetric negative and positive coordinates.  ``center`` shifts
    this physical center away from the geometric origin when non-zero.
    """

    #: Physical cell spacing in metres. Must be positive.
    spacing: float = frozen_field()
    #: Physical coordinate of the domain center in metres. Defaults to (0, 0, 0).
    center: tuple[float, float, float] = frozen_field(default=(0, 0, 0))

    def __post_init__(self):
        if self.spacing <= 0:
            raise ValueError(f"Uniform grid spacing must be positive, got {self.spacing}.")

    def resolve(self, shape: tuple[int, int, int]) -> "RectilinearGrid":
        """Return a concrete solver grid for ``shape``."""
        origin = (
            self.center[0] - shape[0] * self.spacing / 2.0,
            self.center[1] - shape[1] * self.spacing / 2.0,
            self.center[2] - shape[2] * self.spacing / 2.0,
        )
        return RectilinearGrid.uniform(shape=shape, spacing=self.spacing, origin=origin)

    @property
    def is_uniform(self) -> bool:
        """Uniform policies always represent equal cell widths."""
        return True

    @property
    def min_spacing(self) -> float:
        """Smallest cell width implied by this policy."""
        return self.spacing

    @property
    def uniform_spacing(self) -> float:
        """Scalar cell width in metres."""
        return self.spacing

    def axis_extent(self, axis: int, bounds: tuple[int, int]) -> float:
        """Physical length covered by an index interval on one axis."""
        del axis
        lower, upper = bounds
        return (upper - lower) * self.spacing

    def slice_extent(
        self, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ) -> tuple[float, float, float]:
        """Physical side lengths covered by a 3D grid slice."""
        return (
            self.axis_extent(0, slice_tuple[0]),
            self.axis_extent(1, slice_tuple[1]),
            self.axis_extent(2, slice_tuple[2]),
        )

    def coord_to_index(self, axis: int, coord: float, snap: str = "nearest") -> int:
        """Map a physical coordinate to a uniform-grid edge index.

        Because unresolved policies do not yet know ``shape``, this helper uses
        a center-relative basis: ``coord`` is interpreted relative to
        ``self.center[axis]`` and the returned index is a center-relative edge
        offset. Use :meth:`RectilinearGrid.coord_to_index` on the resolved grid
        when you need absolute edge indices.
        """
        origin_offset = coord - self.center[axis]
        scaled = origin_offset / self.spacing
        if snap == "nearest":
            return round(scaled)
        if snap == "lower":
            return int(np.floor(scaled))
        if snap == "upper":
            return int(np.ceil(scaled))
        raise ValueError(f"Unknown snapping rule: {snap}")

    def length_to_cell_count(self, axis: int, length: float, snap: str = "nearest") -> int:
        """Convert a physical length to a uniform-grid cell count."""
        return self.coord_to_index(axis, self.center[axis] + length, snap=snap)

    def bounds_for_center(self, axis: int, center: float, size: int) -> tuple[int, int]:
        """Convert a physical center and grid size to edge bounds."""
        grid_center = self.coord_to_index(axis, center, snap="nearest")
        lower = round(grid_center - size / 2)
        return lower, lower + size

    def anchor_coordinate(self, axis: int, bounds: tuple[int, int], position: float) -> float:
        """Return a physical anchor coordinate inside a uniform interval."""
        lower, upper = bounds
        lower_coord = self.center[axis] + lower * self.spacing
        upper_coord = self.center[axis] + upper * self.spacing
        return lower_coord + 0.5 * (position + 1.0) * (upper_coord - lower_coord)

    def bounds_for_anchor(self, axis: int, size: int, anchor: float, position: float) -> tuple[int, int]:
        """Choose a uniform-grid interval from an object anchor."""
        anchor_cell = self.coord_to_index(axis, anchor, snap="nearest")
        offset = round(0.5 * (position + 1.0) * size)
        lower = anchor_cell - offset
        return lower, lower + size

    def cell_volume(self, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> jax.Array:
        """Return per-cell volumes for a slice on this uniform policy."""
        shape = tuple(upper - lower for lower, upper in slice_tuple)
        return jnp.ones(shape) * self.spacing**3

    def face_area(self, axis: int, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> jax.Array:
        """Return per-face areas for a slice on this uniform policy."""
        shape = tuple(upper - lower for lower, upper in slice_tuple)
        area_shape = tuple(shape[i] for i in range(3) if i != axis)
        return jnp.ones(area_shape) * self.spacing**2


@autoinit
class QuasiUniformGrid(TreeClass):
    """Unresolved policy for a rectilinear grid with independent per-axis spacings.

    ``QuasiUniformGrid`` generalises ``UniformGrid`` to allow different cell
    widths along x, y, and z while keeping each axis internally uniform.  This
    is sometimes called a *quasi-uniform* or *anisotropic-uniform* mesh: the
    grid is rectilinear and axis-aligned, but the aspect ratio is not 1 : 1 : 1.

    Like ``UniformGrid``, this is user intent rather than the solver mesh.
    Calling :meth:`resolve` converts the policy to a concrete
    ``RectilinearGrid`` once the simulation shape is known.  The resulting grid
    is centered at ``center`` so that coordinates span symmetrically into both
    negative and positive values along every axis.

    Example::

        grid = QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=20e-9)
        resolved = grid.resolve(shape=(100, 100, 50))
        # x, y edges span [-500 nm, +500 nm]; z edges span [-500 nm, +500 nm]
    """

    #: Cell width along x in metres. Must be positive.
    dx: float = frozen_field()
    #: Cell width along y in metres. Must be positive.
    dy: float = frozen_field()
    #: Cell width along z in metres. Must be positive.
    dz: float = frozen_field()
    #: Physical coordinate of the domain center in metres. Defaults to (0, 0, 0).
    center: tuple[float, float, float] = frozen_field(default=(0, 0, 0))

    def __post_init__(self):
        for name, val in (("dx", self.dx), ("dy", self.dy), ("dz", self.dz)):
            if val <= 0:
                raise ValueError(f"QuasiUniformGrid spacing {name} must be positive, got {val}.")

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, shape: tuple[int, int, int]) -> "RectilinearGrid":
        """Return a concrete ``RectilinearGrid`` for ``shape``.

        Edge arrays are built independently for each axis using the per-axis
        spacing and the requested number of cells.  The domain is centered at
        ``self.center``.

        Args:
            shape: Number of cells in ``(x, y, z)``.

        Returns:
            A ``RectilinearGrid`` whose edge arrays are piecewise-uniform (one
            constant spacing per axis) and span symmetrically around
            ``self.center``.

        Raises:
            ValueError: If any axis has an odd cell count.  The center-origin
                convention requires even cell counts on every axis so the domain
                center always lands on a cell edge.  An odd count silently shifts
                which Yee component sits at object boundaries, changing the
                effective simulated length by one cell.
        """
        for axis, n in enumerate(shape):
            if n % 2 != 0:
                raise ValueError(
                    f"QuasiUniformGrid requires an even cell count on every axis (center-origin "
                    f"convention). Axis {axis} has {n} cells (odd). Adjust the simulation "
                    f"volume size so every axis has an even number of cells."
                )
        spacings = (self.dx, self.dy, self.dz)
        edge_arrays = []
        for a in range(3):
            s = spacings[a]
            n = shape[a]
            lower = self.center[a] - n * s / 2.0
            edge_arrays.append(lower + s * jnp.arange(n + 1))
        return RectilinearGrid(
            x_edges=edge_arrays[0],
            y_edges=edge_arrays[1],
            z_edges=edge_arrays[2],
        )

    # ------------------------------------------------------------------
    # Convenience properties (mirror UniformGrid's interface)
    # ------------------------------------------------------------------

    @property
    def is_uniform(self) -> bool:
        """True only when all three spacings are equal."""
        return self.dx == self.dy == self.dz

    @property
    def min_spacing(self) -> float:
        """Smallest cell width across all three axes in metres."""
        return min(self.dx, self.dy, self.dz)

    def axis_spacing(self, axis: int) -> float:
        """Return the cell width for a single axis."""
        return (self.dx, self.dy, self.dz)[axis]

    def axis_extent(self, axis: int, bounds: tuple[int, int]) -> float:
        """Physical length covered by an index interval on one axis."""
        lower, upper = bounds
        return (upper - lower) * self.axis_spacing(axis)

    def slice_extent(
        self, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ) -> tuple[float, float, float]:
        """Physical side lengths covered by a 3D grid slice."""
        return (
            self.axis_extent(0, slice_tuple[0]),
            self.axis_extent(1, slice_tuple[1]),
            self.axis_extent(2, slice_tuple[2]),
        )

    def coord_to_index(self, axis: int, coord: float, snap: str = "nearest") -> int:
        """Map a physical coordinate to a grid edge index along ``axis``.

        Coordinates are measured from ``self.center[axis]``.
        """
        s = self.axis_spacing(axis)
        scaled = (coord - self.center[axis]) / s
        if snap == "nearest":
            return round(scaled)
        if snap == "lower":
            return int(np.floor(scaled))
        if snap == "upper":
            return int(np.ceil(scaled))
        raise ValueError(f"Unknown snapping rule: {snap}")

    def length_to_cell_count(self, axis: int, length: float, snap: str = "nearest") -> int:
        """Convert a physical length to a cell count along ``axis``."""
        return self.coord_to_index(axis, self.center[axis] + length, snap=snap)

    def cell_volume(self, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> jax.Array:
        """Return per-cell volume weights broadcast to a 3D slice shape."""
        shape = tuple(upper - lower for lower, upper in slice_tuple)
        return jnp.ones(shape) * (self.dx * self.dy * self.dz)

    def face_area(self, axis: int, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> jax.Array:
        """Return per-face area weights for a detector plane normal to ``axis``."""
        spacings = (self.dx, self.dy, self.dz)
        transverse = [a for a in range(3) if a != axis]
        shape = tuple(slice_tuple[a][1] - slice_tuple[a][0] for a in range(3))
        area_shape = tuple(shape[a] for a in transverse)
        return jnp.ones(area_shape) * spacings[transverse[0]] * spacings[transverse[1]]


@autoinit
class RectilinearGrid(TreeClass):
    """Realized rectilinear simulation grid described by physical cell edges.

    This is the canonical solver-facing grid representation used by fdtdx
    internals.  A uniform grid is represented by equally spaced edge arrays, not
    by a separate scalar code path.  Keeping one realized representation is
    important for the non-uniform grid migration: placement, PML profiles,
    mode-solver coordinates, detector weights, and Yee update metrics should all
    ask the grid for physical distances instead of deriving them from a global
    ``resolution`` value.

    The arrays store cell *edges* in metres.  For a grid with ``nx`` cells along
    x, ``x_edges`` has shape ``(nx + 1,)`` and must be strictly increasing.  Cell
    widths, centers, face areas, and volumes are derived from these arrays.

    Notes:
        This class intentionally does not encode automatic mesh generation
        policy.  Future policy objects such as ``AutoGrid`` or
        ``QuasiUniformGrid`` should resolve to ``RectilinearGrid`` before the
        solver runs.
    """

    #: Physical edge coordinates along x in metres, shape ``(nx + 1,)``.
    x_edges: jax.Array = field()
    #: Physical edge coordinates along y in metres, shape ``(ny + 1,)``.
    y_edges: jax.Array = field()
    #: Physical edge coordinates along z in metres, shape ``(nz + 1,)``.
    z_edges: jax.Array = field()
    _min_spacings: tuple[float, float, float] = frozen_private_field()
    _is_uniform: bool = frozen_private_field()
    _uniform_spacing: float | None = frozen_private_field()

    def __post_init__(self):
        object.__setattr__(self, "x_edges", jnp.asarray(self.x_edges))
        object.__setattr__(self, "y_edges", jnp.asarray(self.y_edges))
        object.__setattr__(self, "z_edges", jnp.asarray(self.z_edges))
        for axis, edges in enumerate((self.x_edges, self.y_edges, self.z_edges)):
            if edges.ndim != 1:
                raise ValueError(f"Grid edge coordinates for axis {axis} must be one-dimensional.")
            if edges.shape[0] < 2:
                raise ValueError(f"Grid edge coordinates for axis {axis} must contain at least two entries.")
            if bool(jnp.any(jnp.diff(edges) <= 0)):
                raise ValueError(f"Grid edge coordinates for axis {axis} must be strictly increasing.")
        edge_arrays_np = tuple(np.asarray(edges) for edges in (self.x_edges, self.y_edges, self.z_edges))
        width_arrays = tuple(np.diff(edges) for edges in edge_arrays_np)
        min_spacings = tuple(float(np.min(widths)) for widths in width_arrays)
        spacing = float(width_arrays[0][0])
        # Uniformity is a *relative* property. The absolute width jitter of a perfectly uniform
        # float grid grows with the edge-coordinate magnitude (~eps * max|edge|), so a fixed
        # absolute tolerance (np.allclose's default atol=1e-8) is wrong at physical scales: it
        # labels nanometre-spaced grids "uniform" even with tens-of-percent variation, and could
        # label very large coarse grids "non-uniform". Compare widths to the grid spacing with a
        # small relative tolerance plus a roundoff floor that tracks the float dtype and domain
        # size, so genuine (>~0.01%) non-uniformity is detected at any scale while a truly uniform
        # grid stays uniform regardless of cell count.
        is_uniform = True
        for edges_np, widths in zip(edge_arrays_np, width_arrays):
            eps = float(np.finfo(edges_np.dtype).eps) if np.issubdtype(edges_np.dtype, np.floating) else 0.0
            roundoff = 8.0 * eps * float(np.max(np.abs(edges_np)))
            if float(np.max(np.abs(widths - spacing))) > 1e-4 * abs(spacing) + roundoff:
                is_uniform = False
                break
        object.__setattr__(self, "_min_spacings", min_spacings)
        object.__setattr__(self, "_is_uniform", is_uniform)
        object.__setattr__(self, "_uniform_spacing", float(np.round(spacing, decimals=14)) if is_uniform else None)

    @classmethod
    def uniform(
        cls,
        shape: tuple[int, int, int],
        spacing: float,
        origin: tuple[float, float, float] | None = None,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Create a realized rectilinear grid for a uniform grid.

        The grid is centered at ``center`` by default, so edge arrays span
        ``[center[a] - shape[a] * spacing / 2, center[a] + shape[a] * spacing / 2]``
        along each axis.  Negative and positive coordinates are used symmetrically
        around the center of the simulation domain.

        Passing an explicit ``origin`` (lower-corner coordinate) overrides
        ``center`` and restores the legacy lower-corner behaviour.  This is used
        internally by ``UniformGrid.resolve`` after it has already computed the
        lower corner from the center.

        Args:
            shape: Number of cells in ``(x, y, z)``.
            spacing: Uniform cell width in metres.
            center: Physical coordinate of the domain center. Defaults to
                ``(0, 0, 0)`` so the domain spans equally into negative and
                positive coordinates.
            origin: Physical coordinate of the **lower** domain corner.  When
                provided this takes priority over ``center``.

        Returns:
            A grid whose edge arrays are equally spaced and centered on
            ``center`` (or anchored at ``origin`` when given).
        """
        if spacing <= 0:
            raise ValueError(f"Uniform grid spacing must be positive, got {spacing}.")
        if any(n <= 0 for n in shape):
            raise ValueError(f"Uniform grid shape entries must be positive, got {shape}.")
        if origin is not None:
            lower_corner = origin
        else:
            lower_corner = tuple(center[a] - shape[a] * spacing / 2.0 for a in range(3))
        edge_arrays = tuple(lower_corner[axis] + spacing * jnp.arange(shape[axis] + 1) for axis in range(3))
        return cls(x_edges=edge_arrays[0], y_edges=edge_arrays[1], z_edges=edge_arrays[2])

    @classmethod
    def custom(
        cls,
        x_edges: jax.Array,
        y_edges: jax.Array,
        z_edges: jax.Array,
    ):
        """Create a realized rectilinear grid from explicit edge arrays.

        This constructor is equivalent to calling ``RectilinearGrid(...)``
        directly, but it makes the user-facing intent explicit: the caller is
        supplying the final grid coordinates, not an automatic meshing policy.
        """
        return cls(x_edges=x_edges, y_edges=y_edges, z_edges=z_edges)

    def reduce_symmetric(self, symmetry: tuple[int, int, int]) -> "RectilinearGrid":
        """Return the grid reduced onto the kept (upper) half along each symmetric axis.

        Used by ``place_objects`` when ``config.symmetry`` is set on a non-uniform grid: the
        simulation runs on the reduced (half/quarter/octant) domain and the result is unfolded
        afterwards. For every axis with ``symmetry[a] != 0`` this keeps the upper-half edges
        ``edges(a)[n // 2:]`` (absolute coordinates preserved — the FDTD metrics depend only on
        cell widths, which are translation-invariant). Non-symmetric axes are returned unchanged.

        Two conditions must hold on each symmetric axis so that mirroring the kept half exactly
        reconstructs the full domain:

        * an even cell count, so the split lands on a cell edge, and
        * mirror-symmetric cell widths about the center (``dx[i] == dx[n - 1 - i]``), so the
          discarded lower half is the exact mirror of the kept half.

        Args:
            symmetry (tuple[int, int, int]): Per-axis symmetry condition ``(x, y, z)``; ``0`` means
                no reduction on that axis (any nonzero value reduces it).

        Returns:
            RectilinearGrid: The reduced grid (a new instance; the original is unchanged).

        Raises:
            ValueError: If a symmetric axis has an odd (or < 2) cell count, or cell widths that are
                not mirror-symmetric about the center.
        """
        axis_names = ("x", "y", "z")
        new_edges = []
        for a in range(3):
            edges = self.edges(a)
            if symmetry[a] == 0:
                new_edges.append(edges)
                continue
            n = self.shape[a]
            validate_symmetric_axis_cells(n, axis_names[a], subject="grid")
            widths = self.cell_widths(a)
            # rtol is loose enough to tolerate the float32 cumsum/diff roundoff of a grid that is
            # mathematically symmetric, but far tighter than any genuinely asymmetric profile.
            if not bool(jnp.allclose(widths, widths[::-1], rtol=1e-4, atol=0.0)):
                raise ValueError(
                    f"Cannot apply symmetry on axis {axis_names[a]}: the cell widths must be "
                    f"mirror-symmetric about the center (dx[i] == dx[n-1-i]) so the discarded half is "
                    f"the exact mirror of the kept half. Provide a grid whose spacing is symmetric "
                    f"about the {axis_names[a]} center plane, or drop symmetry on this axis."
                )
            new_edges.append(edges[n // 2 :])
        return RectilinearGrid.custom(x_edges=new_edges[0], y_edges=new_edges[1], z_edges=new_edges[2])

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of cells along each axis."""
        return (self.x_edges.shape[0] - 1, self.y_edges.shape[0] - 1, self.z_edges.shape[0] - 1)

    @property
    def dx(self) -> jax.Array:
        """Cell widths along x in metres."""
        return jnp.diff(self.x_edges)

    @property
    def dy(self) -> jax.Array:
        """Cell widths along y in metres."""
        return jnp.diff(self.y_edges)

    @property
    def dz(self) -> jax.Array:
        """Cell widths along z in metres."""
        return jnp.diff(self.z_edges)

    @property
    def min_spacing(self) -> float:
        """Smallest cell width in the grid.

        This value is the conservative spacing used for staged CFL migration.
        The full non-uniform update should eventually use explicit local metric
        arrays, but stability remains controlled by the smallest cell.
        """
        return min(self._min_spacings)

    @property
    def min_spacings(self) -> tuple[float, float, float]:
        """Smallest cell width along each axis in metres."""
        return self._min_spacings

    def cfl_time_step(self, courant_factor: float) -> float:
        """Return the CFL-limited time step for a rectilinear 3D grid.

        The stability limit for an orthogonal FDTD grid is controlled by the
        smallest spacing on each axis:

        ``dt <= courant_factor / (c * sqrt(1/dx_min^2 + 1/dy_min^2 + 1/dz_min^2))``.

        For uniform grids this is exactly the existing ``courant_factor/sqrt(3)``
        behavior.  For anisotropic or stretched grids it avoids using one global
        spacing for all three axes.
        """
        if self._is_uniform and self._uniform_spacing is not None:
            return (courant_factor / float(np.sqrt(3.0))) * self._uniform_spacing / constants.c
        dx_min, dy_min, dz_min = self.min_spacings
        inv_metric = (1 / dx_min**2) + (1 / dy_min**2) + (1 / dz_min**2)
        return courant_factor / (constants.c * float(np.sqrt(inv_metric)))

    @property
    def is_uniform(self) -> bool:
        """Whether all cell widths match a single spacing within numerical tolerance."""
        return self._is_uniform

    @property
    def uniform_spacing(self) -> float:
        """Return the scalar spacing for a uniform grid or raise for non-uniform grids.

        This compatibility escape hatch should only be used by code that has not
        yet been migrated to metric-aware helpers.  It deliberately raises for
        non-uniform grids so unsupported paths fail loudly.
        """
        if self._uniform_spacing is None:
            raise ValueError("This operation still requires a uniform grid.")
        return self._uniform_spacing

    def edges(self, axis: int) -> jax.Array:
        """Return edge coordinates for ``axis``."""
        return (self.x_edges, self.y_edges, self.z_edges)[axis]

    def cell_widths(self, axis: int) -> jax.Array:
        """Return cell widths for ``axis``."""
        return (self.dx, self.dy, self.dz)[axis]

    def centers(self, axis: int) -> jax.Array:
        """Return cell-center coordinates for ``axis``."""
        edges = self.edges(axis)
        return 0.5 * (edges[:-1] + edges[1:])

    def axis_extent(self, axis: int, bounds: tuple[int, int]) -> float:
        """Physical length covered by an index interval on one axis."""
        lower, upper = bounds
        edges = self.edges(axis)
        return float(edges[upper] - edges[lower])

    def slice_extent(
        self, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ) -> tuple[float, float, float]:
        """Physical side lengths covered by a 3D grid slice."""
        return (
            self.axis_extent(0, slice_tuple[0]),
            self.axis_extent(1, slice_tuple[1]),
            self.axis_extent(2, slice_tuple[2]),
        )

    def subgrid(
        self,
        grid_slice: tuple[slice, slice, slice],
    ):
        """Convenience wrapper to get the sub-grid of a placed fdtdx.SimulationObject given its grid_slice"""
        subgrid = self.custom(
            self.x_edges[slice(grid_slice[0].start, grid_slice[0].stop + 1)],
            self.y_edges[slice(grid_slice[1].start, grid_slice[1].stop + 1)],
            self.z_edges[slice(grid_slice[2].start, grid_slice[2].stop + 1)],
        )
        return subgrid

    def coord_to_index(self, axis: int, coord: float, snap: str = "nearest") -> int:
        """Map a physical coordinate to a grid edge index.

        Args:
            axis: Grid axis.
            coord: Coordinate in metres.
            snap: Snapping rule. ``"nearest"`` chooses the closest edge,
                ``"lower"`` chooses the previous edge, and ``"upper"`` chooses
                the next edge.

        Returns:
            Edge index after applying the requested snapping rule.
        """
        edges = np.asarray(self.edges(axis))
        if snap == "nearest":
            return int(np.argmin(np.abs(edges - coord)))
        if snap == "lower":
            return int(np.searchsorted(edges, coord, side="right") - 1)
        if snap == "upper":
            return int(np.searchsorted(edges, coord, side="left"))
        raise ValueError(f"Unknown snapping rule: {snap}")

    def length_to_cell_count(self, axis: int, length: float, snap: str = "nearest") -> int:
        """Convert a physical length to a number of cells from the lower domain edge.

        This helper preserves the old uniform-grid behavior when ``snap`` is
        ``"nearest"``.  For non-uniform placement, ``"upper"`` is usually the
        safer rule because it chooses enough cells to cover the requested metric
        size from the lower domain edge.
        """
        if length < 0:
            raise ValueError(f"Length must be non-negative, got {length}.")
        return self.coord_to_index(axis, float(self.edges(axis)[0]) + length, snap=snap)

    def bounds_for_center(self, axis: int, center: float, size: int) -> tuple[int, int]:
        """Choose a cell interval whose physical center is closest to ``center``.

        Args:
            axis: Grid axis.
            center: Desired physical center coordinate in metres.
            size: Number of cells in the interval.

        Returns:
            ``(lower, upper)`` edge indices with ``upper - lower == size``.

        Notes:
            This operation is used by object placement when a physical center
            position and an already-resolved grid-cell size are known.  On a
            non-uniform grid there is no exact analogue of ``round(x / dx)``;
            selecting the closest physical interval center gives deterministic
            snapping while preserving the requested grid-cell size.
        """
        if size <= 0:
            raise ValueError(f"Interval size must be positive, got {size}.")
        edges = np.asarray(self.edges(axis))
        max_lower = edges.shape[0] - size - 1
        if max_lower < 0:
            raise ValueError(f"Interval of size {size} does not fit on axis {axis} with shape {self.shape[axis]}.")
        lower_candidates = np.arange(max_lower + 1)
        interval_centers = 0.5 * (edges[lower_candidates] + edges[lower_candidates + size])
        lower = int(lower_candidates[np.argmin(np.abs(interval_centers - center))])
        return lower, lower + size

    def anchor_coordinate(self, axis: int, bounds: tuple[int, int], position: float) -> float:
        """Return a physical anchor coordinate inside an interval.

        ``position`` follows fdtdx object-anchor convention: ``-1`` is the lower
        side, ``0`` is the center, and ``+1`` is the upper side.
        """
        lower, upper = bounds
        edges = np.asarray(self.edges(axis))
        lower_coord = edges[lower]
        upper_coord = edges[upper]
        return float(lower_coord + 0.5 * (position + 1.0) * (upper_coord - lower_coord))

    def bounds_for_anchor(self, axis: int, size: int, anchor: float, position: float) -> tuple[int, int]:
        """Choose a cell interval whose object anchor is closest to ``anchor``.

        Args:
            axis: Grid axis.
            size: Number of cells in the interval.
            anchor: Desired physical anchor coordinate in metres.
            position: Object-relative anchor position, where ``-1`` is lower
                side, ``0`` is center, and ``+1`` is upper side.

        Returns:
            ``(lower, upper)`` edge indices with ``upper - lower == size``.
        """
        if size <= 0:
            raise ValueError(f"Interval size must be positive, got {size}.")
        edges = np.asarray(self.edges(axis))
        max_lower = edges.shape[0] - size - 1
        if max_lower < 0:
            raise ValueError(f"Interval of size {size} does not fit on axis {axis} with shape {self.shape[axis]}.")
        lower_candidates = np.arange(max_lower + 1)
        lower_edges = edges[lower_candidates]
        upper_edges = edges[lower_candidates + size]
        anchors = lower_edges + 0.5 * (position + 1.0) * (upper_edges - lower_edges)
        lower = int(lower_candidates[np.argmin(np.abs(anchors - anchor))])
        return lower, lower + size

    def face_area(self, axis: int, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> jax.Array:
        """Return per-cell face-area weights for a detector plane.

        Args:
            axis: Normal axis of the face.
            slice_tuple: Grid slice containing the detector volume.  The normal
                axis is expected to have width one for a plane detector.

        Returns:
            Area weights broadcast to the detector's 3D slice shape.
        """
        transverse_axes = get_transverse_axes(axis)
        widths = []
        for transverse_axis in transverse_axes:
            lower, upper = slice_tuple[transverse_axis]
            widths.append(self.cell_widths(transverse_axis)[lower:upper])
        area_2d = widths[0][:, None] * widths[1][None, :]
        shape = [1, 1, 1]
        shape[transverse_axes[0]] = area_2d.shape[0]
        shape[transverse_axes[1]] = area_2d.shape[1]
        return area_2d.reshape(shape)

    def cell_volume(self, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> jax.Array:
        """Return per-cell volume weights broadcast to a 3D slice shape."""
        x0, x1 = slice_tuple[0]
        y0, y1 = slice_tuple[1]
        z0, z1 = slice_tuple[2]
        return self.dx[x0:x1, None, None] * self.dy[None, y0:y1, None] * self.dz[None, None, z0:z1]


def calculate_spatial_offsets_yee() -> tuple[jax.Array, jax.Array]:
    offset_E = jnp.stack(
        [
            jnp.asarray([0.5, 0, 0])[None, None, None, :],
            jnp.asarray([0, 0.5, 0])[None, None, None, :],
            jnp.asarray([0, 0, 0.5])[None, None, None, :],
        ]
    )
    offset_H = jnp.stack(
        [
            jnp.asarray([0, 0.5, 0.5])[None, None, None, :],
            jnp.asarray([0.5, 0, 0.5])[None, None, None, :],
            jnp.asarray([0.5, 0.5, 0])[None, None, None, :],
        ]
    )
    return offset_E, offset_H


def calculate_time_offset_yee(
    center: jax.Array,
    wave_vector: jax.Array,
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array | float,
    resolution: float,
    time_step_duration: float,
    effective_index: jax.Array | float | None = None,
    e_polarization: jax.Array | None = None,
    h_polarization: jax.Array | None = None,
    coordinate_edges: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    center_physical: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    if inv_permittivities.ndim == 4:
        # Extract spatial shape from (1, Nx, Ny, Nz) or (3, Nx, Ny, Nz) or (9, Nx, Ny, Nz)
        spatial_shape = inv_permittivities.shape[1:]
    elif inv_permittivities.ndim == 3:
        # Legacy shape (Nx, Ny, Nz)
        spatial_shape = inv_permittivities.shape
    else:
        raise Exception(f"Invalid permittivity shape: {inv_permittivities.shape=}")

    if 1 not in spatial_shape:
        raise Exception(f"Expected one spatial axis to be one, but got {spatial_shape}")

    propagation_axis = spatial_shape.index(1)

    if coordinate_edges is None:
        # Build uniform coordinate edges from scalar resolution so the rest of
        # the function has one physical-space code path.
        e = [jnp.arange(spatial_shape[ax] + 1, dtype=jnp.float32) * resolution for ax in range(3)]
        resolved_edges: tuple[jax.Array, jax.Array, jax.Array] = (e[0], e[1], e[2])
        c0 = jnp.asarray(center[0], dtype=wave_vector.dtype) * resolution
        c1 = jnp.asarray(center[1], dtype=wave_vector.dtype) * resolution
        center_parts: list[jax.Array] = [c0, c1]
        center_parts.insert(propagation_axis, jnp.zeros((), dtype=wave_vector.dtype))
        center_physical = jnp.stack(center_parts)
    else:
        if center_physical is None:
            raise ValueError("center_physical must be provided with coordinate_edges")
        resolved_edges = coordinate_edges

    def component_positions(axis: int, offset: float) -> jax.Array:
        edges = resolved_edges[axis]
        centers = 0.5 * (edges[:-1] + edges[1:])
        if offset == 0:
            return edges[:-1]
        if offset == 0.5:
            return centers
        raise ValueError(f"Unsupported Yee offset: {offset}")

    def xyz_for_offsets(offsets: tuple[float, float, float]) -> jax.Array:
        coords = [component_positions(ax, offsets[ax]) for ax in range(3)]
        x, y, z = jnp.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
        return jnp.stack([x, y, z], axis=-1) - center_physical[None, None, None, :]

    xyz_E = jnp.stack(
        [
            xyz_for_offsets((0.5, 0, 0)),
            xyz_for_offsets((0, 0.5, 0)),
            xyz_for_offsets((0, 0, 0.5)),
        ]
    )
    xyz_H = jnp.stack(
        [
            xyz_for_offsets((0, 0.5, 0.5)),
            xyz_for_offsets((0.5, 0, 0.5)),
            xyz_for_offsets((0.5, 0.5, 0)),
        ]
    )
    distance_scale = 1.0

    travel_offset_E = -jnp.dot(xyz_E, wave_vector)
    travel_offset_H = -jnp.dot(xyz_H, wave_vector)

    if effective_index is not None:
        refractive_idx = effective_index * jnp.ones(spatial_shape)
    else:
        # adjust speed for material and calculate time offset
        if inv_permittivities.ndim == 4:
            # Remove when anisotropic case is verified
            if inv_permittivities.shape[0] == 3 or inv_permittivities.shape[0] == 9:
                # Check if diagonal isotropic (shape[0] == 3)
                is_diagonal_isotropic = (
                    (inv_permittivities.shape[0] == 3)
                    & jnp.allclose(inv_permittivities[0], inv_permittivities[1])
                    & jnp.allclose(inv_permittivities[1], inv_permittivities[2])
                )

                # Check if full tensor isotropic (shape[0] == 9)
                is_full_isotropic = (
                    (inv_permittivities.shape[0] == 9)
                    & jnp.allclose(inv_permittivities[0], inv_permittivities[4])
                    & jnp.allclose(inv_permittivities[4], inv_permittivities[8])
                    & jnp.allclose(inv_permittivities[1], 0.0)
                    & jnp.allclose(inv_permittivities[2], 0.0)
                    & jnp.allclose(inv_permittivities[3], 0.0)
                    & jnp.allclose(inv_permittivities[5], 0.0)
                    & jnp.allclose(inv_permittivities[6], 0.0)
                    & jnp.allclose(inv_permittivities[7], 0.0)
                )

                is_isotropic = is_diagonal_isotropic | is_full_isotropic

                def _raise_if_anisotropic(is_iso):
                    if not is_iso:
                        raise NotImplementedError(
                            "Gaussian or planewave sources within anisotropic materials are not supported yet."
                        )

                jax.debug.callback(_raise_if_anisotropic, is_isotropic)

            # For now just return the isotropic result
            inv_perm_eff = inv_permittivities[0]
        else:
            inv_perm_eff = inv_permittivities

        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim == 4:
            # Remove when anisotropic case is verified
            if inv_permeabilities.shape[0] == 3 or inv_permeabilities.shape[0] == 9:
                # Check if diagonal isotropic (shape[0] == 3)
                is_diagonal_isotropic = (
                    (inv_permeabilities.shape[0] == 3)
                    & jnp.allclose(inv_permeabilities[0], inv_permeabilities[1])
                    & jnp.allclose(inv_permittivities[1], inv_permittivities[2])
                )

                # Check if full tensor isotropic (shape[0] == 9)
                is_full_isotropic = (
                    (inv_permeabilities.shape[0] == 9)
                    & jnp.allclose(inv_permeabilities[0], inv_permeabilities[4])
                    & jnp.allclose(inv_permeabilities[4], inv_permeabilities[8])
                    & jnp.allclose(inv_permeabilities[1], 0.0)
                    & jnp.allclose(inv_permeabilities[2], 0.0)
                    & jnp.allclose(inv_permeabilities[3], 0.0)
                    & jnp.allclose(inv_permeabilities[5], 0.0)
                    & jnp.allclose(inv_permeabilities[6], 0.0)
                    & jnp.allclose(inv_permeabilities[7], 0.0)
                )

                is_isotropic = is_diagonal_isotropic | is_full_isotropic

                def _raise_if_anisotropic(is_iso):
                    if not is_iso:
                        raise NotImplementedError(
                            "Gaussian or planewave sources within anisotropic materials are not supported yet."
                        )

                jax.debug.callback(_raise_if_anisotropic, is_isotropic)

            # For now just return the isotropic result
            inv_perm_eff_perm = inv_permeabilities[0]
        else:
            inv_perm_eff_perm = inv_permeabilities

        refractive_idx = 1 / jnp.sqrt(inv_perm_eff * inv_perm_eff_perm)

    velocity = (constants.c / refractive_idx)[None, ...]
    time_offset_E = travel_offset_E * distance_scale / (velocity * time_step_duration)
    time_offset_H = travel_offset_H * distance_scale / (velocity * time_step_duration)
    return time_offset_E, time_offset_H


def polygon_to_mask(
    boundary: tuple[float, float, float, float],
    resolution: float,
    polygon_vertices: np.ndarray,
) -> np.ndarray:
    """
    Generate a 2D binary mask from a polygon.

    Args:
        boundary (tuple[float, float, float, float]): tuple of (min_x, min_y, max_x, max_y)
            Rectangular boundary in metrical units (meter).
        resolution (float): float
            Grid resolution (spacing between grid points) in metrical units
        polygon_vertices (np.ndarray): list of (x, y) tuples
            Vertices of the polygon in metrical units. Last point should equal first point.
            Must have shape (N, 2).
    Returns:
        np.ndarray: 2D binary mask where 1 indicates inside polygon, 0 indicates outside
    """
    assert polygon_vertices.ndim == 2
    assert polygon_vertices.shape[1] == 2
    min_x, min_y, max_x, max_y = boundary

    x_coords = np.arange(min_x, max_x + 0.5 * resolution, resolution)
    y_coords = np.arange(min_y, max_y + 0.5 * resolution, resolution)
    return polygon_to_mask_at_points(x_coords, y_coords, polygon_vertices)


def polygon_to_mask_at_points(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    polygon_vertices: np.ndarray,
) -> np.ndarray:
    """Generate a 2D polygon mask at explicit sample coordinates.

    This is the rectilinear-grid counterpart to ``polygon_to_mask``.  The caller
    supplies the physical cell-center coordinates directly, so non-uniform grids
    do not need to be resampled onto a synthetic uniform lattice.
    """
    assert polygon_vertices.ndim == 2
    assert polygon_vertices.shape[1] == 2
    x_coords = np.asarray(x_coords)
    y_coords = np.asarray(y_coords)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
    points = np.column_stack((X.ravel(), Y.ravel()))
    polygon_path = Path(polygon_vertices)
    inside_polygon = polygon_path.contains_points(points)
    return inside_polygon.reshape(X.shape).astype(bool)


def multi_polygons_to_mask(
    boundary: tuple[float, float, float, float],
    resolution: float,
    polygon_list: list[np.ndarray],
) -> np.ndarray:
    """
    Generate a 2D binary mask from a list of polygons.

    Args:
        boundary (tuple[float, float, float, float]): tuple of (min_x, min_y, max_x, max_y)
            Rectangular boundary in metrical units (meter).
        resolution (float): float
            Grid resolution (spacing between grid points) in metrical units.
        polygon_list (list[np.ndarray]): list of polygon vertex arrays.
            Each array must have shape (N, 2) with (x, y) vertices in metrical units.
    Returns:
        np.ndarray: 2D boolean mask where True indicates inside at least one polygon.
    """
    if len(polygon_list) == 0:
        min_x, min_y, max_x, max_y = boundary
        x_coords = np.arange(min_x, max_x + 0.5 * resolution, resolution)
        y_coords = np.arange(min_y, max_y + 0.5 * resolution, resolution)
        return np.zeros((len(x_coords), len(y_coords)), dtype=bool)
    result = polygon_to_mask(boundary, resolution, polygon_list[0])
    for poly in polygon_list[1:]:
        result |= polygon_to_mask(boundary, resolution, poly)
    return result
