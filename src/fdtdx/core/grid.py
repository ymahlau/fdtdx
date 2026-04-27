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
    center_3d = jnp.asarray(center_list, dtype=jnp.float32)[None, None, None, :]
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


def _voxel_centers_numpy(grid_shape: tuple[int, int, int], resolution: float) -> np.ndarray:
    # ... (Keep existing _voxel_centers implementation here) ...
    nx, ny, nz = grid_shape
    x = (np.arange(nx) + 0.5) * resolution - nx * resolution / 2
    y = (np.arange(ny) + 0.5) * resolution - ny * resolution / 2
    z = (np.arange(nz) + 0.5) * resolution - nz * resolution / 2
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def exact_analytical_fractions(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid_shape: tuple[int, int, int],
    resolution: float,
    binary_inside: np.ndarray,
    mc_batch_size: int = 1000,
    mc_iterations: int = 5,
) -> np.ndarray:
    """
    Calculates subpixel volume fractions.
    - k=1 faces: Exact analytical 8-corner formula.
    - k>=2 faces: Vectorized Monte-Carlo integration for complex convex shapes.
    """
    # Helper to generate voxel centers
    x = np.arange(grid_shape[0]) * resolution
    y = np.arange(grid_shape[1]) * resolution
    z = np.arange(grid_shape[2]) * resolution
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    pts = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

    fractions = binary_inside.astype(np.float64).flatten()

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

    v_idx = np.repeat(np.arange(len(pts)), lens)
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

    # Sort pairs by voxel index so all faces for a single voxel are contiguous
    sort_args = np.argsort(valid_v_idx)
    sorted_v_idx = valid_v_idx[sort_args]
    sorted_f_idx = valid_f_idx[sort_args]

    # Get unique voxels, where they start in the sorted array, and how many faces they intersect
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

        # Distance calculation
        d_k1 = np.sum(n_k1 * (pts_k1 - p0_k1), axis=1)
        alpha = d_k1 / resolution
        n_abs = np.abs(n_k1)
        n_x, n_y, n_z = n_abs[:, 0], n_abs[:, 1], n_abs[:, 2]

        valid_normals_mask = (n_x >= 1e-6) | (n_y >= 1e-6) | (n_z >= 1e-6)

        ix = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
        iy = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
        iz = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        signs = ix * iy * iz * -1.0

        term = alpha[:, None] + 0.5 * (n_x[:, None] * ix + n_y[:, None] * iy + n_z[:, None] * iz)
        vol_terms = signs * (np.maximum(term, 0) ** 3)

        sign_multiplier = 1.0 / (6.0 * np.maximum(n_x, 1e-8) * np.maximum(n_y, 1e-8) * np.maximum(n_z, 1e-8))
        vols_k1 = np.sum(vol_terms, axis=1) * sign_multiplier

        fractions[v_idx_k1] = np.where(valid_normals_mask, np.clip(vols_k1, 0.0, 1.0), 0.0)

    # --- ROUTE K >= 2 (Vectorized Monte-Carlo Multi-Plane Integration) ---
    mask_k_multi = counts > 1
    v_idx_multi = unique_voxels[mask_k_multi]

    if len(v_idx_multi) > 0:
        starts_multi = start_indices[mask_k_multi]
        c_multi = counts[mask_k_multi]

        # PADDING STRATEGY: Create fixed (N, 3, 3) tensors for normals and origins
        N_multi = len(v_idx_multi)
        normals_multi = np.zeros((N_multi, 3, 3))
        p0_multi = np.zeros((N_multi, 3, 3))

        # Fill plane 1 (Everyone has at least 2)
        idx_p1 = sorted_f_idx[starts_multi]
        normals_multi[:, 0, :] = normals[idx_p1]
        p0_multi[:, 0, :] = v0[idx_p1]

        # Fill plane 2 (Everyone has at least 2)
        idx_p2 = sorted_f_idx[starts_multi + 1]
        normals_multi[:, 1, :] = normals[idx_p2]
        p0_multi[:, 1, :] = v0[idx_p2]

        # Fill plane 3 (Only where count == 3)
        has_3 = c_multi == 3
        if np.any(has_3):
            idx_p3 = sorted_f_idx[starts_multi[has_3] + 2]
            normals_multi[has_3, 2, :] = normals[idx_p3]
            p0_multi[has_3, 2, :] = v0[idx_p3]

        # For those that ONLY had 2 planes, duplicate plane 2 into slot 3.
        # This makes the 3rd plane evaluation redundant but keeps the matrix perfectly dense!
        normals_multi[~has_3, 2, :] = normals_multi[~has_3, 1, :]
        p0_multi[~has_3, 2, :] = p0_multi[~has_3, 1, :]

        # -- Monte-Carlo Integration --
        pts_multi = pts[v_idx_multi]  # Shape: (N_multi, 3)
        total_inside_count = np.zeros(N_multi, dtype=np.int32)

        for _ in range(mc_iterations):
            # Generate random points within a voxel bounds [-0.5, 0.5]
            # Shape: (mc_batch_size, 3)
            mc_template = np.random.uniform(-0.5, 0.5, size=(mc_batch_size, 3)) * resolution

            # Broadcast template to all multi-plane voxels
            # Shape: (N_multi, mc_batch_size, 3)
            voxel_mc_grids = pts_multi[:, None, :] + mc_template[None, :, :]

            # Calculate distance to all 3 planes simultaneously
            # P0 shape: (N_multi, 1, 3, 3) -> broadcast against grids
            # Normals shape: (N_multi, 1, 3, 3)
            v_diff = voxel_mc_grids[:, :, None, :] - p0_multi[:, None, :, :]  # (N_multi, mc_batch_size, 3, 3)

            # Dot product across the coordinate axis
            # Result shape: (N_multi, mc_batch_size, 3 planes)
            distances = np.sum(normals_multi[:, None, :, :] * v_diff, axis=3)

            # A point is inside if distance <= 0 for ALL 3 planes
            is_inside = np.all(distances <= 0, axis=2)  # Shape: (N_multi, mc_batch_size)

            # Accumulate the number of points that fell inside
            total_inside_count += np.sum(is_inside, axis=1)

        # Average the bools over total sampled points to get the exact fraction
        total_samples = mc_batch_size * mc_iterations
        vols_multi = total_inside_count / total_samples

        fractions[v_idx_multi] = vols_multi

    return fractions
