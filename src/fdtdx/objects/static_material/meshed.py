import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.grid import get_voxel_centers_numpy
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.mesh import calculate_points_in_mesh, extract_surface
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.static_material.static import StaticMultiMaterialObject


@autoinit
class TetraMeshedObject(StaticMultiMaterialObject):
    """A single-material 3D object defined by a tetraeder volume mesh.

    The mesh coordinate system must be centered at the origin — vertices must be
    given such that the bounding-box center is at (0, 0, 0). This matches the
    convention used by :class:`ExtrudedPolygon`.

    """

    #: (V, 3) float array of mesh vertices centered at the origin.
    vertices: np.ndarray = frozen_field()

    #: (T, 4) int array of tetrahedron vertex indices.
    tetra: np.ndarray = frozen_field()

    #: Name of the material in the ``materials`` dict to assign to all voxels.
    material_name: str = frozen_field()

    #: If True, calculates exact analytical subpixel volume fractions for boundary voxels
    subpixel_smoothing: bool = frozen_field(default=False)

    def get_voxel_mask_for_shape(self) -> jax.Array:
        pts = get_voxel_centers_numpy(self.grid_shape, self._config.resolution)
        inside = calculate_points_in_mesh(self.vertices, self.faces, pts)
        return jnp.asarray(inside.reshape(self.grid_shape), dtype=jnp.bool_)

    def get_material_mapping(self) -> jax.Array:
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        result = np.full(int(np.prod(self.grid_shape)), idx, dtype=np.int32)
        return jnp.asarray(result.reshape(self.grid_shape), dtype=jnp.int32)


@autoinit
class TriangleMeshedObject(StaticMultiMaterialObject):
    """A single-material 3D object defined by a triangular surface mesh.

    The mesh coordinate system must be centered at the origin — vertices must be
    given such that the bounding-box center is at (0, 0, 0). This matches the
    convention used by :class:`ExtrudedPolygon`.

    Supported cell types for file loading: ``"triangle"`` (used directly),
    ``"quad"`` (each quad split into two triangles).
    """

    #: (V, 3) float array of mesh vertices centered at the origin.
    vertices: np.ndarray = frozen_field()

    #: (F, 3) int array of triangle vertex indices.
    faces: np.ndarray = frozen_field()

    #: Name of the material in the ``materials`` dict to assign to all voxels.
    material_name: str = frozen_field()

    #: If True, calculates exact analytical subpixel volume fractions for boundary voxels
    subpixel_smoothing: bool = frozen_field(default=False)

    def get_voxel_mask_for_shape(self) -> jax.Array:
        pts = get_voxel_centers_numpy(self.grid_shape, self._config.resolution)
        inside = calculate_points_in_mesh(self.vertices, self.faces, pts)
        return jnp.asarray(inside.reshape(self.grid_shape), dtype=jnp.bool_)

    def get_material_mapping(self) -> jax.Array:
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        result = np.full(int(np.prod(self.grid_shape)), idx, dtype=np.int32)
        return jnp.asarray(result.reshape(self.grid_shape), dtype=jnp.int32)


def meshed_object_from_file(
    path: str | pathlib.Path,
    **kwargs: Any,
) -> TriangleMeshedObject:
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
    faces = extract_surface(cells_and_types)
    return TriangleMeshedObject(vertices=mesh.points, faces=faces, **kwargs)
