import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.static_material.static import StaticMultiMaterialObject

_RAY_CHUNK = 512  # points processed per batch in _contains_numpy

# Non-axis-aligned ray direction based on the golden ratio (φ).
# Irrational components make accidental edge/vertex hits essentially impossible
# for any regularly-structured mesh (boxes, extruded polygons, …).
_PHI = (1.0 + np.sqrt(5.0)) / 2.0
_RAY_DIR: np.ndarray = np.array([1.0, _PHI, _PHI**2])
_RAY_DIR = _RAY_DIR / np.linalg.norm(_RAY_DIR)


def _contains_numpy(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Test whether points lie inside a watertight triangular mesh.

    Implements the Moller-Trumbore ray-triangle intersection algorithm in the
    +z direction.  The intersection count parity determines inside/outside.
    No external library beyond numpy is required.

    Coordinates are normalised internally to a unit bounding box so the
    algorithm works correctly regardless of the physical scale of the mesh
    (e.g. nanometre-scale FDTD objects).

    Args:
        vertices: (V, 3) float array of mesh vertices.
        faces:    (F, 3) int array of triangle vertex indices.
        points:   (N, 3) float array of query point coordinates.

    Returns:
        (N,) bool array - True if the corresponding point is inside the mesh.
    """
    # Normalise to a unit bounding box so that absolute epsilon comparisons
    # stay meaningful regardless of the physical scale of the coordinates.
    scale = max(float(np.abs(vertices).max()), float(np.abs(points).max()), 1e-300)
    verts = vertices.astype(np.float64) / scale
    pts_n = points.astype(np.float64) / scale

    v0 = verts[faces[:, 0]]  # (F, 3)
    e1 = verts[faces[:, 1]] - v0  # (F, 3)
    e2 = verts[faces[:, 2]] - v0  # (F, 3)

    # h = cross(ray, e2) is independent of the query point.
    ray = _RAY_DIR
    h = np.cross(ray[np.newaxis, :], e2)  # (F, 3)
    a = np.einsum("fi,fi->f", e1, h)  # (F,)
    valid = np.abs(a) > 1e-10
    inv_a = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)  # (F,)

    n_pts = len(points)
    inside = np.zeros(n_pts, dtype=bool)

    for start in range(0, n_pts, _RAY_CHUNK):
        end = min(start + _RAY_CHUNK, n_pts)
        pts = pts_n[start:end]  # (C, 3)

        s = pts[:, np.newaxis, :] - v0[np.newaxis, :, :]  # (C, F, 3)
        u = inv_a[np.newaxis, :] * np.einsum("cfi,fi->cf", s, h)
        ok_u = valid[np.newaxis, :] & (u >= 0.0) & (u <= 1.0)

        q = np.cross(s, e1[np.newaxis, :, :])  # (C, F, 3)
        v = inv_a[np.newaxis, :] * np.einsum("i,cfi->cf", ray, q)
        ok_v = ok_u & (v >= 0.0) & (u + v <= 1.0)

        t = inv_a[np.newaxis, :] * np.einsum("fi,cfi->cf", e2, q)
        hits = np.sum(ok_v & (t > 1e-8), axis=1)  # (C,)
        inside[start:end] = hits % 2 == 1

    return inside


def _extract_surface(cells_and_types: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Extract triangular surface faces from a list of (cell_type, connectivity) pairs.

    For tetrahedral cells the outer boundary is determined by finding faces that
    appear exactly once (shared by only one tetrahedron in the group).
    """
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
                    t[:, [0, 1, 2]],
                    t[:, [0, 1, 3]],
                    t[:, [0, 2, 3]],
                    t[:, [1, 2, 3]],
                ]
            )
            faces_sorted = np.sort(faces, axis=1)
            _, inv, counts = np.unique(faces_sorted, axis=0, return_inverse=True, return_counts=True)
            all_faces.append(faces[counts[inv] == 1])
        else:
            raise ValueError(f"Unsupported cell type '{cell_type}'. Supported types: 'triangle', 'quad', 'tetra'.")
    if not all_faces:
        raise ValueError("No supported cell types found; cannot extract surface.")
    return np.concatenate(all_faces, axis=0)


def _voxel_centers(grid_shape: tuple[int, int, int], resolution: float) -> np.ndarray:
    """Return voxel center positions in mesh-local coordinates (origin = bounding-box center).

    Returns:
        Array of shape (nx*ny*nz, 3).
    """
    nx, ny, nz = grid_shape
    x = (np.arange(nx) + 0.5) * resolution - nx * resolution / 2
    y = (np.arange(ny) + 0.5) * resolution - ny * resolution / 2
    z = (np.arange(nz) + 0.5) * resolution - nz * resolution / 2
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


@autoinit
class MeshedObject(StaticMultiMaterialObject):
    """A single-material 3D object defined by a triangular surface mesh.

    The mesh coordinate system must be centered at the origin — vertices must be
    given such that the bounding-box center is at (0, 0, 0). This matches the
    convention used by :class:`ExtrudedPolygon`.

    Supported cell types for file loading: ``"triangle"`` (used directly),
    ``"quad"`` (each quad split into two triangles), and ``"tetra"`` (volumetric
    mesh; the outer boundary surface is extracted automatically).
    """

    #: (V, 3) float array of mesh vertices centered at the origin.
    vertices: np.ndarray = frozen_field()

    #: (F, 3) int array of triangle vertex indices.
    faces: np.ndarray = frozen_field()

    #: Name of the material in the ``materials`` dict to assign to all voxels.
    material_name: str = frozen_field()

    # ------------------------------------------------------------------
    # StaticMultiMaterialObject interface
    # ------------------------------------------------------------------

    def get_voxel_mask_for_shape(self) -> jax.Array:
        pts = _voxel_centers(self.grid_shape, self._config.resolution)
        inside = _contains_numpy(self.vertices, self.faces, pts)
        return jnp.asarray(inside.reshape(self.grid_shape), dtype=jnp.bool_)

    def get_material_mapping(self) -> jax.Array:
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        result = np.full(int(np.prod(self.grid_shape)), idx, dtype=np.int32)
        return jnp.asarray(result.reshape(self.grid_shape), dtype=jnp.int32)


def meshed_object_from_file(
    path: str | pathlib.Path,
    **kwargs: Any,
) -> MeshedObject:
    """Load a mesh file via meshio, auto-center it, and return a :class:`MeshedObject`.

    The bounding-box center of the loaded mesh is shifted to the origin so the
    result satisfies the centered-coordinate convention expected by
    :class:`MeshedObject`.

    Args:
        path: Path to the mesh file.  Any format supported by meshio is accepted
              (e.g. ``.vtk``, ``.msh``, ``.obj``, ``.stl``).
        **kwargs: Forwarded to :class:`MeshedObject` (``materials``,
                  ``material_name``, …).

    Returns:
        :class:`MeshedObject` with vertices centred around the origin in metres.
    """
    import meshio

    mesh = meshio.read(str(path))
    centre = 0.5 * (mesh.points.min(axis=0) + mesh.points.max(axis=0))
    mesh.points = mesh.points - centre
    cells_and_types = [(b.type, b.data) for b in mesh.cells]
    faces = _extract_surface(cells_and_types)
    return MeshedObject(vertices=mesh.points, faces=faces, **kwargs)
