import numpy as np
from scipy.spatial import KDTree


def extract_surface(
    cells_and_types: list[tuple[str, np.ndarray]] | tuple[str, np.ndarray],
    extract_outer_surface_of_tetra: bool = True,
) -> np.ndarray:
    """
    Extract triangular surface faces from a list of (cell_type, connectivity) pairs.

    For tetrahedral cells, the outer boundary is determined by finding faces that
    appear exactly once (shared by only one tetrahedron in the group).

    Parameters
    ----------
    cells_and_types : list[tuple[str, np.ndarray]] or tuple[str, np.ndarray]
        A single tuple or a list of tuples containing the cell type and its connectivity
        array. Supported cell types are `"triangle"`, `"quad"`, and `"tetra"`.

    Returns
    -------
    np.ndarray
        An array of triangular surface faces with shape `(N, 3)` containing the vertex
        indices, where `N` is the total number of triangular faces extracted.

    Notes
    -----
    - **"triangle"**: The connectivity array is passed through as-is.
    - **"quad"**: Each quadrilateral cell is split into two triangular faces: `[0, 1, 2]` and `[0, 2, 3]`.
    - **"tetra"**: Extracts all faces from the tetrahedra and retains only those that are not shared by another tetrahedron (i.e., the outer boundary faces with a count of 1).

    """
    if isinstance(cells_and_types, tuple):
        cells_and_types = [cells_and_types]

    all_faces: list[np.ndarray] = []
    for cell_type, connectivity in cells_and_types:
        if cell_type == "triangle":
            all_faces.append(connectivity)
        elif cell_type == "quad":
            q = connectivity
            all_faces.append(q[:, [0, 1, 2]])
            all_faces.append(q[:, [0, 2, 3]])
        elif cell_type == "tetra":
            t = connectivity
            faces = np.vstack(
                [
                    t[:, [0, 2, 1]],
                    t[:, [0, 1, 3]],
                    t[:, [0, 3, 2]],
                    t[:, [1, 2, 3]],
                ]
            )
            if extract_outer_surface_of_tetra:
                faces_sorted = np.sort(faces, axis=1)
                _, inv, counts = np.unique(faces_sorted, axis=0, return_inverse=True, return_counts=True)
                all_faces.append(faces[counts[inv] == 1])
            else:
                all_faces.append(faces)
        else:
            raise ValueError(f"Unsupported cell type '{cell_type}'. Supported types: 'triangle', 'quad', 'tetra'.")
    if not all_faces:
        raise ValueError("No supported cell types found; cannot extract surface.")
    return np.concatenate(all_faces, axis=0)


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
) -> np.ndarray:
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
        fixed_lookup_grid_size=fixed_lookup_grid_size,
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
        chunk_size=chunk_size,
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


def query_point_in_tetra_precomputed(
    queries: np.ndarray, tree: KDTree, inv_matrix: np.ndarray, v0: np.ndarray, k=10, epsilon=1e-8
) -> tuple[np.ndarray, np.ndarray]:
    queries = np.asarray(queries, dtype=np.float64)
    num_queries = len(queries)

    # --- Broad Phase ---
    # Query KD-Tree for the 'k' nearest centroids for all 'Q' points
    _, candidate_idx = tree.query(queries, k=k)

    if k == 1:
        candidate_idx = candidate_idx[:, np.newaxis]  # Ensure 2D shape (Q, 1)

    # --- Narrow Phase ---
    # Gather the pre-computed v0 and E_inv for the specific candidate tetrahedra
    matrix_inv_c = inv_matrix[candidate_idx]  # Shape: (Q, k, 3, 3)
    v0_c = v0[candidate_idx]  # Shape: (Q, k, 3)

    # Calculate vector from the tetrahedron's v0 to the query point
    diff = queries[:, np.newaxis, :] - v0_c  # Shape: (Q, k, 3)

    # Add an axis to treat diff as a column vector, then batch multiply
    lambdas_123 = matrix_inv_c @ diff[..., np.newaxis]  # Shape: (Q, k, 3, 1)
    lambdas_123 = lambdas_123.squeeze(-1)  # Shape: (Q, k, 3)

    # Calculate the 4th barycentric coordinate (lambda 0)
    lambda0 = 1.0 - np.sum(lambdas_123, axis=-1)  # Shape: (Q, k)

    # Combine all four barycentric coordinates
    all_lambdas = np.concatenate([lambda0[..., np.newaxis], lambdas_123], axis=-1)  # (Q, k, 4)

    # Point is inside if all coordinates are >= 0 (accounting for float tolerance)
    inside_mask = np.all(all_lambdas >= -epsilon, axis=-1)  # Shape: (Q, k), boolean

    # --- Resolve Final Output ---
    # Find points that matched at least one candidate tetrahedron
    is_inside = np.any(inside_mask, axis=-1)  # Shape: (Q,), boolean

    # Get the specific index of the tetrahedron it fell into
    tet_indices = np.full(num_queries, -1, dtype=np.int32)

    # np.argmax returns the index of the first True value along the k-axis
    first_match_idx = np.argmax(inside_mask, axis=-1)

    # Extract the actual mesh indices using the argmax results
    valid_tets = candidate_idx[np.arange(num_queries), first_match_idx]

    # Write the valid indices to our output array, only where a match was found
    tet_indices[is_inside] = valid_tets[is_inside]

    return is_inside, tet_indices


def query_point_in_tetra(
    vertices: np.ndarray,
    tetras: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    epsilon=1e-8,
):
    """
    Convenience wrapper to query if points are inside tetrahedra.
    Builds the necessary KDTree and inverse matrices if they are not provided.
    """
    # Build the KDTree if it wasn't pre-computed and passed in
    tree = build_tetra_kdtree(vertices, tetras)

    # Build the inverse matrices if they weren't pre-computed and passed in
    v0, inv_matrix = build_tetra_inverse_matrices(vertices, tetras)

    # 3. Call the core precomputed query function
    is_inside, tet_indices = query_point_in_tetra_precomputed(
        queries=queries, tree=tree, inv_matrix=inv_matrix, v0=v0, k=k, epsilon=epsilon
    )

    return is_inside, tet_indices


def build_tetra_kdtree(vertices: np.ndarray, tetras: np.ndarray) -> KDTree:
    centroids = vertices[tetras].mean(axis=1)
    return KDTree(centroids)


def build_tetra_inverse_matrices(vertices: np.ndarray, tetras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (v0, inv_matrix) with shapes (T, 3) and (T, 3, 3)."""
    T_verts = vertices[tetras]  # (T, 4, 3)
    v0 = T_verts[:, 0, :]  # (T, 3)
    edges = T_verts[:, 1:4, :] - v0[:, None, :]  # (T, 3, 3)
    matrix = np.transpose(edges, (0, 2, 1))  # (T, 3, 3)
    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"One or more tetrahedra are degenerate (flat). {e!s}")
    return v0, inv_matrix


def _narrow_phase(
    points: np.ndarray,  # (..., 3)
    candidate_idx: np.ndarray,  # (..., k)         broadcastable with points[..., 0]
    v0: np.ndarray,  # (T, 3)
    inv_matrix: np.ndarray,  # (T, 3, 3)
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Test whether each point lies inside any of its k candidate tetrahedra.

    `points` and `candidate_idx` must broadcast against each other on all leading
    axes; concretely we expect ``points.shape[:-1] == candidate_idx.shape[:-1]``
    after broadcasting. The function never materialises a (Q, n^3, k, ...) tensor
    larger than what the caller already implies via shapes.

    Returns
    -------
    inside_mask : bool array of shape ``points.shape[:-1] + (k,)``
        True where the point is inside that particular candidate.
    matched_tet : int array of shape ``points.shape[:-1]``
        Mesh index of the first candidate that contained the point, or -1.
    """
    # Gather candidate tetra data. Shapes after gather:
    #   v0_c  : leading_shape + (k, 3)
    #   inv_c : leading_shape + (k, 3, 3)
    v0_c = v0[candidate_idx]
    inv_c = inv_matrix[candidate_idx]

    # Vector from each candidate's v0 to the query point.
    # points[..., None, :] broadcasts a singleton k-axis.
    diff = points[..., None, :] - v0_c  # (..., k, 3)

    # Batched matrix-vector product: (..., k, 3, 3) @ (..., k, 3, 1) -> (..., k, 3, 1)
    lambdas_123 = (inv_c @ diff[..., None]).squeeze(-1)  # (..., k, 3)

    lambda0 = 1.0 - lambdas_123.sum(axis=-1)  # (..., k)
    all_lambdas = np.concatenate([lambda0[..., None], lambdas_123], axis=-1)  # (..., k, 4)

    inside_per_cand = np.all(all_lambdas >= -epsilon, axis=-1)  # (..., k)

    # Resolve "first match" along the k axis without a Python loop.
    any_inside = inside_per_cand.any(axis=-1)  # (...,)
    first_match = inside_per_cand.argmax(axis=-1)  # (...,) int

    matched_tet = np.where(
        any_inside,
        np.take_along_axis(candidate_idx, first_match[..., None], axis=-1).squeeze(-1),
        -1,
    )
    return inside_per_cand, matched_tet


def build_tetra_radii(vertices: np.ndarray, tetras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-tetra centroid and bounding radius.

    `radii[t]` is the max distance from centroid_t to any of its four vertices,
    so any point inside tetra t lies within `radii[t]` of `centroids[t]`. Used
    as a conservative containment test against voxel half-diagonals.
    """
    T_verts = vertices[tetras]  # (T, 4, 3)
    centroids = T_verts.mean(axis=1)  # (T, 3)
    radii = np.linalg.norm(T_verts - centroids[:, None, :], axis=-1).max(axis=1)  # (T,)
    return centroids, radii


def voxel_in_tetra_ratio(
    vertices: np.ndarray,  # (V, 3)
    tetras: np.ndarray,  # (T, 4)
    queries: np.ndarray,  # (Q, 3)  voxel centers
    resolution: float | tuple[float, float, float],
    n_samples_per_axis: int = 8,
    k: int = 32,
    chunk_size: int = 1024,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Fraction of each cuboid voxel that lies inside the tetrahedral mesh.

    Tiered evaluation:
      0. AABB prefilter — voxels outside the (expanded) mesh bbox → ratio 0.
      1. Distance prefilter — voxels whose nearest tetra-centroid is further
         than R_max + voxel_half_diag away → ratio 0.
      2. Corner short-circuit — if all 8 voxel corners fall inside the same
         tetra, the voxel is fully covered → ratio 1.
      3. Sub-voxel sampling — only the boundary voxels reach this stage.

    Both tier-2 and tier-3 are chunked along the voxel axis, so peak memory
    is bounded by `chunk_size * n^3 * k_eff` floats regardless of mesh or
    grid size.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    tetras = np.asarray(tetras, dtype=np.int64)
    queries = np.asarray(queries, dtype=np.float64)

    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError(f"queries must have shape (Q, 3), got {queries.shape}")

    Q = queries.shape[0]
    if Q == 0:
        return np.zeros((0,), dtype=np.float64)

    res = np.asarray(resolution, dtype=np.float64)
    if res.ndim == 0:
        res = np.broadcast_to(res, (3,)).copy()
    elif res.shape != (3,):
        raise ValueError(f"resolution must be scalar or shape (3,), got {res.shape}")
    if np.any(res <= 0):
        raise ValueError(f"resolution must be positive, got {res}")

    n = int(n_samples_per_axis)
    if n < 1:
        raise ValueError(f"n_samples_per_axis must be >= 1, got {n}")

    half = res * 0.5
    voxel_half_diag = float(np.linalg.norm(half))

    # Output: outside-mesh voxels stay at 0.
    ratios = np.zeros(Q, dtype=np.float64)

    # ------------------------------------------------------------------
    # Tier 0: mesh-AABB prefilter.
    # ------------------------------------------------------------------
    mesh_min = vertices.min(axis=0)
    mesh_max = vertices.max(axis=0)
    in_aabb = np.all(
        (queries >= mesh_min - half) & (queries <= mesh_max + half),
        axis=1,
    )
    if not in_aabb.any():
        return ratios

    aabb_local = np.flatnonzero(in_aabb)
    queries_aabb = queries[aabb_local]

    # ------------------------------------------------------------------
    # Mesh-side precomputation (done after AABB filter, but on the full
    # mesh — these are O(T) and we need the KD-tree for the survivors).
    # ------------------------------------------------------------------
    T = tetras.shape[0]
    k_eff = int(min(k, T))
    tree = build_tetra_kdtree(vertices, tetras)
    v0, inv_matrix = build_tetra_inverse_matrices(vertices, tetras)
    _, tetra_radii = build_tetra_radii(vertices, tetras)
    R_max = float(tetra_radii.max())

    dists, candidate_idx = tree.query(queries_aabb, k=k_eff)
    if k_eff == 1:
        candidate_idx = candidate_idx[:, None]
        dists = dists[:, None]
    candidate_idx = candidate_idx.astype(np.int64, copy=False)

    # ------------------------------------------------------------------
    # Tier 1: distance prefilter.
    # No part of voxel v can lie in any tetra if the nearest centroid is
    # farther than R_max + voxel_half_diag.
    # ------------------------------------------------------------------
    near_mesh = dists[:, 0] <= R_max + voxel_half_diag
    if not near_mesh.any():
        return ratios

    near_local = np.flatnonzero(near_mesh)
    queries_near = queries_aabb[near_local]
    candidate_idx = candidate_idx[near_local]
    cand_dists = dists[near_local]
    final_idx = aabb_local[near_local]  # → original `queries`

    # Per-(voxel, candidate) validity using each tetra's own radius. A tetra
    # t can contribute to voxel v only if dist(v, centroid_t) ≤ R_t + d/2.
    # Invalid slots get masked out before the argmax / any reductions, so
    # they neither produce false matches nor inflate the ratio.
    cand_radii = tetra_radii[candidate_idx]  # (n_near, k_eff)
    cand_valid = cand_dists <= cand_radii + voxel_half_diag  # (n_near, k_eff)

    n_near = queries_near.shape[0]

    # ------------------------------------------------------------------
    # Tier 2: chunked 8-corner short-circuit.
    # We never materialise (n_near, 8, k_eff, …) tensors — chunked along
    # the voxel axis, with candidate gathers happening per chunk.
    # ------------------------------------------------------------------
    sign = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    corner_offsets = sign * half  # (8, 3)

    fully_inside = np.zeros(n_near, dtype=bool)

    for start in range(0, n_near, chunk_size):
        stop = min(start + chunk_size, n_near)
        sl = slice(start, stop)
        B = stop - start

        cand_b = candidate_idx[sl]  # (B, k_eff)
        valid_b = cand_valid[sl]  # (B, k_eff)

        v0_b = v0[cand_b]  # (B, k_eff, 3)
        inv_b = inv_matrix[cand_b]  # (B, k_eff, 3, 3)

        # corners: (B, 8, 3); broadcast against (B, 1, k_eff, 3) v0_b in diff.
        corners_b = queries_near[sl, None, :] + corner_offsets[None, :, :]
        diff = corners_b[:, :, None, :] - v0_b[:, None, :, :]  # (B, 8, k_eff, 3)

        # Batched matvec: result[b, c, k, i] = Σⱼ inv_b[b, k, i, j] · diff[b, c, k, j]
        lambdas_123 = np.einsum("bkij,bckj->bcki", inv_b, diff)
        lambda0 = 1.0 - lambdas_123.sum(axis=-1)  # (B, 8, k_eff)
        all_lambdas = np.concatenate([lambda0[..., None], lambdas_123], axis=-1)  # (B, 8, k_eff, 4)

        inside_per_cand = np.all(all_lambdas >= -epsilon, axis=-1)
        inside_per_cand &= valid_b[:, None, :]  # mask far candidates

        any_inside = inside_per_cand.any(axis=-1)  # (B, 8)
        first_match = inside_per_cand.argmax(axis=-1)  # (B, 8)
        row = np.arange(B)[:, None]
        matched = np.where(any_inside, cand_b[row, first_match], -1)  # (B, 8)

        all_in = (matched >= 0).all(axis=1)
        same_tet = (matched == matched[:, :1]).all(axis=1)
        fully_inside[sl] = all_in & same_tet

    ratios[final_idx[fully_inside]] = 1.0

    # ------------------------------------------------------------------
    # Tier 3: chunked sub-voxel sampling on remaining boundary voxels.
    # ------------------------------------------------------------------
    todo_local = np.flatnonzero(~fully_inside)
    if todo_local.size == 0:
        return np.clip(ratios, 0.0, 1.0)

    queries_todo = queries_near[todo_local]
    cand_todo = candidate_idx[todo_local]
    valid_todo = cand_valid[todo_local]
    final_idx_todo = final_idx[todo_local]
    n_todo = todo_local.size

    # Midpoint-rule sample template in voxel-local coords.
    axis_lin = (np.arange(n, dtype=np.float64) + 0.5) / n - 0.5
    gx, gy, gz = np.meshgrid(axis_lin, axis_lin, axis_lin, indexing="ij")
    template_scaled = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3) * res  # (n^3, 3)

    chunk_ratios = np.empty(n_todo, dtype=np.float64)
    for start in range(0, n_todo, chunk_size):
        stop = min(start + chunk_size, n_todo)
        sl = slice(start, stop)

        cand_chunk = cand_todo[sl]  # (B, k_eff)
        valid_chunk = valid_todo[sl]  # (B, k_eff)

        # Per-chunk gather — bounds peak memory at O(B · k_eff · 9).
        v0_chunk = v0[cand_chunk]  # (B, k_eff, 3)
        inv_chunk = inv_matrix[cand_chunk]  # (B, k_eff, 3, 3)

        samples = queries_todo[sl, None, :] + template_scaled[None, :, :]  # (B, n3, 3)
        diff = samples[:, :, None, :] - v0_chunk[:, None, :, :]  # (B, n3, k_eff, 3)
        lambdas_123 = np.einsum("bkij,bskj->bski", inv_chunk, diff)
        lambda0 = 1.0 - lambdas_123.sum(axis=-1)
        all_lambdas = np.concatenate([lambda0[..., None], lambdas_123], axis=-1)
        inside_per_cand = np.all(all_lambdas >= -epsilon, axis=-1)  # (B, n3, k_eff)
        inside_per_cand &= valid_chunk[:, None, :]
        chunk_ratios[sl] = inside_per_cand.any(axis=-1).mean(axis=-1)  # (B,)

    ratios[final_idx_todo] = chunk_ratios
    return np.clip(ratios, 0.0, 1.0)
