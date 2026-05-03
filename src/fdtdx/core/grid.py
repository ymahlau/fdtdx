import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.path import Path

from fdtdx import constants
from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field


@autoinit
class GridSpec(TreeClass):
    """Rectilinear simulation grid described by physical cell-edge coordinates.

    This is the canonical grid representation used by fdtdx internals.  A uniform
    grid is represented by equally spaced edge arrays, not by a separate scalar
    code path.  Keeping one representation is important for the non-uniform grid
    migration: placement, PML profiles, mode-solver coordinates, detector weights,
    and Yee update metrics should all ask the grid for physical distances instead
    of deriving them from a global ``resolution`` value.

    The arrays store cell *edges* in metres.  For a grid with ``nx`` cells along
    x, ``x_edges`` has shape ``(nx + 1,)`` and must be strictly increasing.  Cell
    widths, centers, face areas, and volumes are derived from these arrays.

    Notes:
        The first implementation focuses on rectilinear grids.  It intentionally
        does not encode automatic mesh generation policy.  A mesh generator should
        produce a ``GridSpec``; the solver should only consume ``GridSpec``.
    """

    x_edges: jax.Array = frozen_field()
    y_edges: jax.Array = frozen_field()
    z_edges: jax.Array = frozen_field()

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

    @classmethod
    def uniform(cls, shape: tuple[int, int, int], spacing: float, origin: tuple[float, float, float] = (0, 0, 0)):
        """Create a rectilinear ``GridSpec`` for a uniform grid.

        Args:
            shape: Number of cells in ``(x, y, z)``.
            spacing: Uniform cell width in metres.
            origin: Physical coordinate of the lower domain corner.

        Returns:
            A ``GridSpec`` whose edge arrays are equally spaced.
        """
        if spacing <= 0:
            raise ValueError(f"Uniform grid spacing must be positive, got {spacing}.")
        if any(n <= 0 for n in shape):
            raise ValueError(f"Uniform grid shape entries must be positive, got {shape}.")
        edge_arrays = tuple(origin[axis] + spacing * jnp.arange(shape[axis] + 1) for axis in range(3))
        return cls(x_edges=edge_arrays[0], y_edges=edge_arrays[1], z_edges=edge_arrays[2])

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
        return float(jnp.min(jnp.asarray([jnp.min(self.dx), jnp.min(self.dy), jnp.min(self.dz)])))

    @property
    def is_uniform(self) -> bool:
        """Whether all cell widths match a single spacing within numerical tolerance."""
        spacing = self.dx[0]
        return bool(jnp.allclose(self.dx, spacing) and jnp.allclose(self.dy, spacing) and jnp.allclose(self.dz, spacing))

    @property
    def uniform_spacing(self) -> float:
        """Return the scalar spacing for a uniform grid or raise for non-uniform grids.

        This compatibility escape hatch should only be used by code that has not
        yet been migrated to metric-aware helpers.  It deliberately raises for
        non-uniform grids so unsupported paths fail loudly.
        """
        if not self.is_uniform:
            raise ValueError("This operation still requires a uniform grid.")
        return float(self.dx[0])

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

    def slice_extent(self, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> tuple[float, float, float]:
        """Physical side lengths covered by a 3D grid slice."""
        return tuple(self.axis_extent(axis, slice_tuple[axis]) for axis in range(3))  # type: ignore[return-value]

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
        transverse_axes = [a for a in range(3) if a != axis]
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

    # phase variation
    x, y, z = jnp.meshgrid(
        jnp.arange(spatial_shape[0]),
        jnp.arange(spatial_shape[1]),
        jnp.arange(spatial_shape[2]),
        indexing="ij",
    )
    xyz = jnp.stack([x, y, z], axis=-1)
    center_list = [center[0], center[1]]
    propagation_axis = spatial_shape.index(1)
    center_list.insert(propagation_axis, jnp.array(0))
    center_3d = jnp.asarray(center_list, dtype=wave_vector.dtype)[None, None, None, :]
    xyz = xyz - center_3d

    # yee grid offsets
    xyz_E = jnp.stack(
        [
            xyz + jnp.asarray([0.5, 0, 0])[None, None, None, :],
            xyz + jnp.asarray([0, 0.5, 0])[None, None, None, :],
            xyz + jnp.asarray([0, 0, 0.5])[None, None, None, :],
        ]
    )
    xyz_H = jnp.stack(
        [
            xyz + jnp.asarray([0, 0.5, 0.5])[None, None, None, :],
            xyz + jnp.asarray([0.5, 0, 0.5])[None, None, None, :],
            xyz + jnp.asarray([0.5, 0.5, 0])[None, None, None, :],
        ]
    )

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
    time_offset_E = travel_offset_E * resolution / (velocity * time_step_duration)
    time_offset_H = travel_offset_H * resolution / (velocity * time_step_duration)
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

    # Create coordinate arrays
    x_coords = np.arange(min_x, max_x + 0.5 * resolution, resolution)
    y_coords = np.arange(min_y, max_y + 0.5 * resolution, resolution)

    # Create meshgrid
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

    # Flatten coordinates for point-in-polygon test
    points = np.column_stack((X.ravel(), Y.ravel()))

    # Create matplotlib Path object from polygon vertices
    polygon_path = Path(polygon_vertices)

    # Test which points are inside the polygon
    inside_polygon = polygon_path.contains_points(points)

    # Reshape back to 2D grid
    mask = inside_polygon.reshape(Y.shape).astype(bool)

    return mask
