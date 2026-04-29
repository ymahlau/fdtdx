import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.path import Path
from scipy.spatial import KDTree

from fdtdx import constants


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


def get_voxel_centers_numpy(grid_shape: tuple[int, int, int], resolution: float) -> np.ndarray:
    """
    Generates 3D coordinates for the centers of a voxel grid centered around the origin.

    Args:
        grid_shape (tuple[int, int, int]): The number of voxels along the X, Y, and Z axes (nx, ny, nz).
        resolution (float): The physical size of each cubic voxel edge.

    Returns:
        np.ndarray: A 2D array of shape (nx * ny * nz, 3) containing the (x, y, z) coordinates
        of all voxel centers.
    """
    nx, ny, nz = grid_shape
    x = (np.arange(nx) + 0.5) * resolution - nx * resolution / 2
    y = (np.arange(ny) + 0.5) * resolution - ny * resolution / 2
    z = (np.arange(nz) + 0.5) * resolution - nz * resolution / 2
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def compute_lookup_grid_dims(
    vertices_2d: np.ndarray,
    faces: np.ndarray,
    fixed_lookup_grid_size: tuple[int, int] | None,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """
    Calculates the dimensions and bounding box for a 2D spatial lookup grid.

    If a fixed size is not provided, the function uses a heuristic based on the
    average face area to target roughly 2-4 faces per grid cell.

    Args:
        vertices_2d (np.ndarray): A 2D array of shape (V, 2) containing the XY coordinates of mesh vertices.
        faces (np.ndarray): A 2D array of shape (F, 3) representing the mesh triangles.
        fixed_lookup_grid_size (tuple[int, int] | None): An optional tuple specifying the exact
            (GridX, GridY) dimensions. If None, dimensions are computed dynamically.

    Returns:
        tuple[int, int, np.ndarray, np.ndarray]: A tuple containing:
            - grid_x (int): Number of cells along the X-axis.
            - grid_y (int): Number of cells along the Y-axis.
            - min_bounds (np.ndarray): The (min_x, min_y) bounding box coordinates.
            - max_bounds (np.ndarray): The (max_x, max_y) bounding box coordinates.

    Raises:
        ValueError: If `fixed_lookup_grid_size` is neither a tuple nor None.
    """
    # 1. Compute global bounding box
    min_bounds = vertices_2d.min(axis=0)
    max_bounds = vertices_2d.max(axis=0)
    mesh_width, mesh_height = max_bounds - min_bounds

    if isinstance(fixed_lookup_grid_size, tuple):
        # Fixed size provided by user (GridX, GridY)
        return fixed_lookup_grid_size[0], fixed_lookup_grid_size[1], min_bounds, max_bounds

    elif fixed_lookup_grid_size is None:
        # Dynamic calculation based on heuristic
        num_faces = len(faces)
        # Rough heuristic: target ~2-4 faces per cell area
        avg_face_area = (mesh_width * mesh_height) / num_faces
        cell_size = np.sqrt(avg_face_area) * 2.0  # Tuning factor

        # Ensure at least a 1x1 grid
        grid_x = max(1, int(np.ceil(mesh_width / cell_size)))
        grid_y = max(1, int(np.ceil(mesh_height / cell_size)))

        return grid_x, grid_y, min_bounds, max_bounds

    raise ValueError(
        "fixed_lookup_grid_size must be a tuple (GridX, GridY) or None for automatic dimension computation"
    )


def build_spatial_lookup_grid_csr(
    vertices: np.ndarray,  # shape (V, 3)
    faces: np.ndarray,  # shape (F, 3)
    fixed_lookup_grid_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray, np.ndarray]:
    """
    Builds a Compressed Sparse Row (CSR) style 2D lookup grid for fast Z-axis ray casting.

    This function projects mesh faces onto the XY plane and maps them into a 2D grid.
    The CSR format allows for highly efficient memory access when looking up which
    faces overlap a specific (x, y) coordinate cell.

    Args:
        vertices (np.ndarray): A 2D array of shape (V, 3) representing mesh vertices.
        faces (np.ndarray): A 2D array of shape (F, 3) representing mesh triangles.
        fixed_lookup_grid_size (tuple[int, int] | None, optional): Explicit grid dimensions
            (GridX, GridY). Defaults to None for automatic calculation.

    Returns:
        tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray, np.ndarray]: A tuple containing:
            - offsets (np.ndarray): 1D int64 array of length (grid_x * grid_y + 1). Cell k's faces
            can be found in `indices[offsets[k]:offsets[k+1]]`.
            - indices (np.ndarray): 1D int64 array containing the face indices grouped by cell.
            - grid_dims (tuple[int, int]): The (grid_x, grid_y) dimensions of the grid.
            - min_b (np.ndarray): The 2D world-space origin (min_x, min_y) of the grid.
            - cell_size (np.ndarray): The 2D world-space dimensions (width, height) of a single cell.
    """
    vertices_2d = vertices[:, :2]
    grid_x, grid_y, min_b, max_b = compute_lookup_grid_dims(vertices_2d, faces, fixed_lookup_grid_size)

    mesh_size = max_b - min_b
    cell_size = np.where(mesh_size > 0, mesh_size / np.array([grid_x, grid_y]), 1.0)

    # 1. Per-face 2D AABB.
    face_verts = vertices_2d[faces]  # (F, 3, 2)
    face_min = face_verts.min(axis=1)  # (F, 2)
    face_max = face_verts.max(axis=1)  # (F, 2)

    # 2. AABB -> integer cell index range, clipped to the grid.
    hi = np.array([grid_x - 1, grid_y - 1], dtype=np.int64)
    cell_min = np.floor((face_min - min_b) / cell_size).astype(np.int64)
    cell_max = np.floor((face_max - min_b) / cell_size).astype(np.int64)
    np.clip(cell_min, 0, hi, out=cell_min)
    np.clip(cell_max, 0, hi, out=cell_max)

    span_x = cell_max[:, 0] - cell_min[:, 0] + 1
    span_y = cell_max[:, 1] - cell_min[:, 1] + 1
    per_face = span_x * span_y
    total = int(per_face.sum())
    F = len(faces)

    # 3. Expand to one entry per (face, cell-it-touches) pair.
    face_idx_flat = np.repeat(np.arange(F, dtype=np.int64), per_face)

    starts = np.empty(F, dtype=np.int64)
    starts[0] = 0
    np.cumsum(per_face[:-1], out=starts[1:])
    local = np.arange(total, dtype=np.int64) - np.repeat(starts, per_face)

    rep_min_x = np.repeat(cell_min[:, 0], per_face)
    rep_min_y = np.repeat(cell_min[:, 1], per_face)
    rep_span_x = np.repeat(span_x, per_face)

    cx = rep_min_x + (local % rep_span_x)
    cy = rep_min_y + (local // rep_span_x)
    cell_flat = cx * grid_y + cy  # row-major over (grid_x, grid_y)

    # 4. Group by cell.
    order = np.argsort(cell_flat, kind="stable")
    indices = face_idx_flat[order]  # CSR "indices"
    sorted_cells = cell_flat[order]

    # 5. counts -> offsets via cumsum with a leading zero.
    num_cells = grid_x * grid_y
    counts = np.bincount(sorted_cells, minlength=num_cells)
    offsets = np.empty(num_cells + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    return offsets, indices, (grid_x, grid_y), min_b, cell_size


def points_in_mesh_csr(
    vertices: np.ndarray,  # (V, 3)
    faces: np.ndarray,  # (F, 3)
    query_points: np.ndarray,  # (Q, 3)
    offsets: np.ndarray,
    indices: np.ndarray,
    grid_dims: tuple[int, int],
    min_b: np.ndarray,
    cell_size: np.ndarray,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """
    Inside/outside test for a watertight mesh via +Z parity ray casting.

    For each query point, this casts a vertical (+Z) ray and counts how many surface
    triangles it crosses, using the CSR XY grid to limit the candidate set of faces.
    According to the Jordan curve theorem, an odd number of crossings means the point
    is inside the mesh.

    Caveat: Query points lying exactly on a triangle edge or vertex of the XY projection
    can be misclassified due to floating-point ties. Perturb inputs by a tiny epsilon
    if this matters in practice.

    Args:
        vertices (np.ndarray): Array of shape (V, 3) containing mesh vertices.
        faces (np.ndarray): Array of shape (F, 3) containing mesh face indices.
        query_points (np.ndarray): Array of shape (Q, 3) containing points to test.
        offsets (np.ndarray): CSR offsets array from the lookup grid.
        indices (np.ndarray): CSR indices array from the lookup grid.
        grid_dims (tuple[int, int]): The (grid_x, grid_y) dimensions of the lookup grid.
        min_b (np.ndarray): The 2D world-space origin of the grid.
        cell_size (np.ndarray): The 2D dimensions of a single cell.
        chunk_size (int, optional): Number of query points to process per batch to limit memory
            usage (peaks at ~chunk_size * mean_candidates_per_cell). Defaults to 100_000.

    Returns:
        np.ndarray: A 1D boolean array of shape (Q,) where True indicates the query point
        is inside the watertight mesh.
    """
    Q = len(query_points)
    if Q == 0:
        return np.zeros(0, dtype=bool)

    grid_x, grid_y = grid_dims
    hi = np.array([grid_x - 1, grid_y - 1], dtype=np.int64)
    inside = np.zeros(Q, dtype=bool)

    # Promote to float64 for stable signed-area predicates.
    verts = vertices.astype(np.float64, copy=False)

    for c0 in range(0, Q, chunk_size):
        c1 = min(c0 + chunk_size, Q)
        qp = query_points[c0:c1].astype(np.float64, copy=False)
        Qc = c1 - c0

        # Cell index per query. Far-outside points are simply clipped to a
        # boundary cell; their 2D containment test will fail for all faces
        # there, yielding 0 crossings → "outside".
        xy = qp[:, :2]
        ij = np.floor((xy - min_b) / cell_size).astype(np.int64)
        np.clip(ij, 0, hi, out=ij)
        k = ij[:, 0] * grid_y + ij[:, 1]

        starts = offsets[k]
        ends = offsets[k + 1]
        counts = (ends - starts).astype(np.int64)

        total = int(counts.sum())
        if total == 0:
            continue

        # Expand to flat (query_local, face) pairs.
        q_flat = np.repeat(np.arange(Qc, dtype=np.int64), counts)
        pair_starts = np.empty(Qc, dtype=np.int64)
        pair_starts[0] = 0
        np.cumsum(counts[:-1], out=pair_starts[1:])
        local = np.arange(total, dtype=np.int64) - np.repeat(pair_starts, counts)
        f_flat = indices[np.repeat(starts, counts) + local]

        # Gather triangle vertices for each pair.
        tri = verts[faces[f_flat]]  # (T, 3, 3)
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        x0, y0, z0 = v0[:, 0], v0[:, 1], v0[:, 2]
        x1, y1, z1 = v1[:, 0], v1[:, 1], v1[:, 2]
        x2, y2, z2 = v2[:, 0], v2[:, 1], v2[:, 2]

        qf = qp[q_flat]
        px, py, pz = qf[:, 0], qf[:, 1], qf[:, 2]

        # Signed-area numerators (un-normalized barycentrics in 2D).
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        u_num = (y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)
        v_num = (y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)
        w_num = det - u_num - v_num

        # Strict 2D containment: all three numerators share det's sign,
        # det ≠ 0 (skip XY-degenerate / vertical triangles).
        s = np.sign(det)
        inside_2d = (s != 0) & (np.sign(u_num) == s) & (np.sign(v_num) == s) & (np.sign(w_num) == s)

        # z at intersection via barycentric interp; only valid where inside_2d.
        safe_det = np.where(det != 0, det, 1.0)
        z_hit = (u_num * z0 + v_num * z1 + w_num * z2) / safe_det

        # Surface crossings strictly above the query → contribute to parity.
        hit = inside_2d & (z_hit > pz)
        crossings = np.bincount(q_flat[hit], minlength=Qc)
        inside[c0:c1] = (crossings & 1) == 1

    return inside


def calculate_points_in_mesh(
    vertices: np.ndarray,  # (V, 3)
    faces: np.ndarray,  # (F, 3)
    query_points: np.ndarray,  # (Q, 3)
    fixed_lookup_grid_size: tuple[int, int] | None = None,
    chunk_size: int = 100_000,
):
    """
    High-level wrapper to determine which points lie inside a watertight mesh.

    This function automatically builds the necessary CSR spatial lookup grid and then
    performs the +Z parity ray casting test for all provided query points.

    Args:
        vertices (np.ndarray): Array of shape (V, 3) containing mesh vertices.
        faces (np.ndarray): Array of shape (F, 3) containing mesh face indices.
        query_points (np.ndarray): Array of shape (Q, 3) containing points to test.
        fixed_lookup_grid_size (tuple[int, int] | None, optional): Explicit dimensions for
            the internal CSR grid (GridX, GridY). Defaults to None for automatic sizing.
        chunk_size (int, optional): Number of query points to process per internal batch.
            Defaults to 100_000.

    Returns:
        np.ndarray: A 1D boolean array of shape (Q,) where True indicates the query point
        is inside the watertight mesh.
    """
    offsets, indices, grid_dims, min_b, cell_size = build_spatial_lookup_grid_csr(
        vertices=vertices,
        faces=faces,
    )
    binary_inside = points_in_mesh_csr(
        vertices=vertices,
        faces=faces,
        query_points=query_points,
        offsets=offsets,
        indices=indices,
        grid_dims=grid_dims,
        min_b=min_b,
        cell_size=cell_size,
        chunk_size=100_000,
    )
    return binary_inside


def exact_analytical_fractions(
    vertices: np.ndarray,
    faces: np.ndarray,
    query_points: np.ndarray,
    resolution: float,
    binary_inside: np.ndarray | None = None,
    fixed_lookup_grid_size: tuple[int, int] | None = None,
    chunk_size: int = 100_000,
    mc_samples_per_voxel: int = 5000,
    mc_batch_size: int = 1_000_000,
) -> np.ndarray:
    """
    Calculates subpixel volume fractions of a triangular mesh for a set of voxel-center query points.

    Each query point is treated as the center of a cubic voxel with side length `resolution`.
    For each voxel, the function returns the fractional volume occupied by the watertight
    mesh. Computation is routed by the number of intersecting faces (k) per voxel:

    - k = 1 (Single Face): Exact analytical 8-corner formula with topological fallbacks
      for 1D (axis-aligned face), 2D (extruded diagonal), and 3D (corner slicing) intersections.
    - k >= 2 (Multiple Faces): Vectorized Monte-Carlo multi-plane integration (capped at 3
      planes) over a localized convex corner.

    Args:
        vertices (np.ndarray): Array of shape (V, 3) containing the mesh vertex coordinates.
        faces (np.ndarray): Array of shape (F, 3) of triangle face indices.
        query_points (np.ndarray): Array of shape (Q, 3) of voxel-center coordinates.
            Each point represents the center of a cubic voxel with side `resolution`.
        resolution (float): Side length of a single cubic voxel.
        binary_inside (np.ndarray | None, optional): Boolean array of shape (Q,) used as
            the base fraction (0.0 or 1.0) before boundary corrections.

            IMPORTANT: "inside" here refers to the voxel CENTER being inside the mesh
            (a point-in-mesh test on `query_points`), NOT all 8 corners of the voxel
            being inside. Voxels straddling the surface are flagged either way and then
            overridden with a fractional value computed below.

            If None, it is computed internally via `calculate_points_in_mesh` using
            `fixed_lookup_grid_size` and `chunk_size`. Defaults to None.
        fixed_lookup_grid_size (tuple[int, int] | None, optional): Forwarded to
            `calculate_points_in_mesh` when `binary_inside` is computed internally.
            Ignored if `binary_inside` is supplied. Defaults to None (auto-sizing).
        chunk_size (int, optional): Forwarded to `calculate_points_in_mesh` when
            `binary_inside` is computed internally. Ignored if `binary_inside` is supplied.
            Defaults to 100_000.
        mc_samples_per_voxel (int, optional): Total Monte-Carlo samples drawn per
            k >= 2 voxel. Higher values reduce variance at linear cost. Defaults to 5000.
        mc_batch_size (int, optional): Approximate cap on the number of (voxel, sample)
            evaluations performed in parallel per batch. Controls peak memory; total work
            is unaffected. If `mc_batch_size >= N_complex_voxels`, all complex voxels are
            evaluated together with `mc_batch_size // N` samples per iteration; otherwise
            voxels are processed in chunks of `mc_batch_size` with 1 sample per iteration.
            Defaults to 1_000_000.

    Returns:
        np.ndarray: 1D float64 array of shape (Q,) with volume fractions in [0.0, 1.0].

    Notes:
        - A KDTree broad-phase narrows candidate (voxel, face) pairs.
        - For k >= 2 voxels, only the first 3 intersecting faces are used (convex-corner
          approximation).
    """
    pts = np.asarray(query_points, dtype=np.float64)
    Q = len(pts)

    # Compute binary_inside if not provided
    if binary_inside is None:
        binary_inside = calculate_points_in_mesh(
            vertices=vertices,
            faces=faces,
            query_points=pts,
            fixed_lookup_grid_size=fixed_lookup_grid_size,
            chunk_size=chunk_size,
        )

    fractions = np.asarray(binary_inside).astype(np.float64).flatten()

    # 1. Bounding Boxes & Normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    tri_min = np.minimum(np.minimum(v0, v1), v2) - resolution / 2.0
    tri_max = np.maximum(np.maximum(v0, v1), v2) + resolution / 2.0

    normals = np.cross(v1 - v0, v2 - v0)
    normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, normals_norm, out=np.zeros_like(normals), where=normals_norm > 1e-10)

    # 2. Broad-phase KD-Tree
    tri_centers = (v0 + v1 + v2) / 3.0
    tree = KDTree(tri_centers)
    neighbors_list = tree.query_ball_point(pts, r=resolution * 1.5)

    lens = np.array([len(n) for n in neighbors_list])
    if np.sum(lens) == 0:
        return fractions

    v_idx = np.repeat(np.arange(Q), lens)
    f_idx = np.concatenate([n for n in neighbors_list if n])

    candidate_pts = pts[v_idx]
    candidate_min = tri_min[f_idx]
    candidate_max = tri_max[f_idx]

    is_inside_aabb = np.all((candidate_pts >= candidate_min) & (candidate_pts <= candidate_max), axis=1)

    valid_v_idx = v_idx[is_inside_aabb]
    valid_f_idx = f_idx[is_inside_aabb]

    if len(valid_v_idx) == 0:
        return fractions

    # ---------------------------------------------------------
    # 3. Sorting & Routing by Intersection Count
    # ---------------------------------------------------------

    sort_args = np.argsort(valid_v_idx)
    sorted_v_idx = valid_v_idx[sort_args]
    sorted_f_idx = valid_f_idx[sort_args]

    unique_voxels, start_indices, counts = np.unique(sorted_v_idx, return_index=True, return_counts=True)

    # Cap counts at 3 (if a voxel has >3 faces, we just use the first 3 to define the corner)
    counts = np.clip(counts, 1, 3)

    # --- ROUTE K=1 (Exact Analytical) ---
    mask_k1 = counts == 1
    v_idx_k1 = unique_voxels[mask_k1]
    f_idx_k1 = sorted_f_idx[start_indices[mask_k1]]

    if len(v_idx_k1) > 0:
        pts_k1 = pts[v_idx_k1]
        n_k1 = normals[f_idx_k1]
        p0_k1 = v0[f_idx_k1]

        alpha = np.sum(n_k1 * (p0_k1 - pts_k1), axis=1) / resolution

        # Cube is symmetric: sort absolute normals descending to seamlessly handle 1D/2D collapses
        n_abs = np.abs(n_k1)
        n_sorted = np.sort(n_abs, axis=1)[:, ::-1]
        n0, n1, n2 = n_sorted[:, 0], n_sorted[:, 1], n_sorted[:, 2]

        vols_k1 = np.zeros_like(alpha)

        # Topological routing thresholds
        eps = 1e-6
        valid_n = n0 > 0  # skip degenerate (zero-area) faces whose normal is the zero vector
        mask_1d = (n1 < eps) & valid_n
        mask_2d = (n1 >= eps) & (n2 < eps) & valid_n
        mask_3d = (n2 >= eps) & valid_n

        # --- 1D Case (Axis-aligned face) ---
        if np.any(mask_1d):
            a1 = alpha[mask_1d]
            nx = n0[mask_1d]
            vols_k1[mask_1d] = a1 / nx + 0.5

        # --- 2D Case (Extruded diagonal) ---
        if np.any(mask_2d):
            a2 = alpha[mask_2d]
            nx = n0[mask_2d]
            ny = n1[mask_2d]

            ix = np.array([-1, 1, -1, 1])
            iy = np.array([-1, -1, 1, 1])
            signs = ix * iy

            term = a2[:, None] + 0.5 * (nx[:, None] * ix + ny[:, None] * iy)
            vol_terms = signs * (np.maximum(term, 0) ** 2)
            vols_k1[mask_2d] = np.sum(vol_terms, axis=1) / (2.0 * nx * ny)

        # --- 3D Case (Corner slicing) ---
        if np.any(mask_3d):
            a3 = alpha[mask_3d]
            nx = n0[mask_3d]
            ny = n1[mask_3d]
            nz = n2[mask_3d]

            ix = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
            iy = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
            iz = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
            signs = ix * iy * iz  # [FIX]: was 'is_really_zero' (NameError)

            term = a3[:, None] + 0.5 * (nx[:, None] * ix + ny[:, None] * iy + nz[:, None] * iz)
            vol_terms = signs * (np.maximum(term, 0) ** 3)
            vols_k1[mask_3d] = np.sum(vol_terms, axis=1) / (6.0 * nx * ny * nz)

        fractions[v_idx_k1] = np.clip(vols_k1, 0.0, 1.0)

    # --- ROUTE K >= 2 (Vectorized Monte-Carlo Multi-Plane Integration) ---
    mask_k_multi = counts > 1
    v_idx_multi = unique_voxels[mask_k_multi]

    if len(v_idx_multi) > 0:
        starts_multi = start_indices[mask_k_multi]
        c_multi = counts[mask_k_multi]

        N_multi = len(v_idx_multi)
        normals_multi = np.zeros((N_multi, 3, 3))
        p0_multi = np.zeros((N_multi, 3, 3))

        idx_p1 = sorted_f_idx[starts_multi]
        normals_multi[:, 0, :] = normals[idx_p1]
        p0_multi[:, 0, :] = v0[idx_p1]

        idx_p2 = sorted_f_idx[starts_multi + 1]
        normals_multi[:, 1, :] = normals[idx_p2]
        p0_multi[:, 1, :] = v0[idx_p2]

        has_3 = c_multi == 3
        if np.any(has_3):
            idx_p3 = sorted_f_idx[starts_multi[has_3] + 2]
            normals_multi[has_3, 2, :] = normals[idx_p3]
            p0_multi[has_3, 2, :] = v0[idx_p3]

        normals_multi[~has_3, 2, :] = normals_multi[~has_3, 1, :]
        p0_multi[~has_3, 2, :] = p0_multi[~has_3, 1, :]

        pts_multi = pts[v_idx_multi]
        total_inside_count = np.zeros(N_multi, dtype=np.int64)

        # Choose voxel-chunk size and samples-per-iteration so that each
        # batch evaluates ~mc_batch_size (voxel, sample) pairs in parallel.
        if N_multi >= mc_batch_size:
            voxels_per_chunk = mc_batch_size
            samples_per_iter = 1
        else:
            voxels_per_chunk = N_multi
            samples_per_iter = max(1, mc_batch_size // N_multi)

        for v_start in range(0, N_multi, voxels_per_chunk):
            v_end = min(v_start + voxels_per_chunk, N_multi)
            pts_chunk = pts_multi[v_start:v_end]
            normals_chunk = normals_multi[v_start:v_end]
            p0_chunk = p0_multi[v_start:v_end]

            samples_done = 0
            while samples_done < mc_samples_per_voxel:
                s = min(samples_per_iter, mc_samples_per_voxel - samples_done)
                mc_template = np.random.uniform(-0.5, 0.5, size=(s, 3)) * resolution
                voxel_mc_grids = pts_chunk[:, None, :] + mc_template[None, :, :]
                v_diff = voxel_mc_grids[:, :, None, :] - p0_chunk[:, None, :, :]
                distances = np.sum(normals_chunk[:, None, :, :] * v_diff, axis=3)
                is_inside = np.all(distances <= 0, axis=2)
                total_inside_count[v_start:v_end] += np.sum(is_inside, axis=1)
                samples_done += s

        vols_multi = total_inside_count / mc_samples_per_voxel
        fractions[v_idx_multi] = vols_multi

    return fractions
