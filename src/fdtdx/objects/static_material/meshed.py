import pathlib
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx import SimulationConfig, frozen_private_field
from fdtdx.core.grid import get_voxel_centers_numpy
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.mesh import calculate_points_in_mesh, extract_surface, voxel_in_tetra_ratio
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.static_material.static import StaticMultiMaterialObject
from fdtdx.typing import SliceTuple3D


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
    
    _ratio_inside: np.ndarray = frozen_private_field()
    
    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        sizes = [x[1] - x[0] for x in grid_slice_tuple]
        voxel_centers = get_voxel_centers_numpy(
            grid_shape=(sizes[0], sizes[1], sizes[2]),
            resolution=config.resolution,
        )
        ratio_inside = voxel_in_tetra_ratio(
            vertices=self.vertices,
            tetras=self.tetra,
            queries=voxel_centers,
            resolution=config.resolution,
            k=32,
            n_samples_per_axis=8,
        )
        self = self.aset("_ratio_inside", ratio_inside)
        return self

    def get_voxel_mask_for_shape(self) -> jax.Array:
        inside = jnp.where(self._ratio_inside < 1e-6, 0.0, 1.0)
        return jnp.asarray(inside.reshape(self.grid_shape), dtype=jnp.bool_)

    def get_material_mapping(self) -> jax.Array:
        assert len(self.materials) == 2, "Currently only implemented for two materials"
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        jax_ratio_arr = jnp.asarray(self._ratio_inside, dtype=jnp.int32)
        result= idx * jax_ratio_arr + (1 - idx) * (1 - jax_ratio_arr)
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
