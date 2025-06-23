from typing import Self

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import polygon_to_mask
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.static_material.static import StaticMultiMaterialObject
from fdtdx.typing import SliceTuple3D


@autoinit
class ExtrudedPolygon(StaticMultiMaterialObject):
    """A polygon object specified by a list of vertices. The coordinate system has its origin at the lower left of the
    bounding box of the polygon.

    Attributes:
        material_name: Name of the material in the materials dictionary to be used for the object
        axis: The extrusion axis.
        vertices: numpy array of shape (N, 2) specifying the position of vertices in metrical units (meter).
    """
    material_name: str = frozen_field()
    axis: int = frozen_field()
    vertices: np.ndarray = frozen_field()

    @property
    def horizontal_axis(self) -> int:
        """Gets the horizontal axis perpendicular to the fiber axis.

        Returns:
            int: The index of the horizontal axis (0=x or 1=y).
        """
        return 1 if self.axis == 0 else 0

    @property
    def vertical_axis(self) -> int:
        """Gets the vertical axis perpendicular to the fiber axis.

        Returns:
            int: The index of the vertical axis (1=y or 2=z).
        """
        return 1 if self.axis == 2 else 2

    def get_voxel_mask_for_shape(self) -> jax.Array:
        width = self.grid_shape[self.horizontal_axis]
        height = self.grid_shape[self.vertical_axis]
        
        half_res = 0.5 * self._config.resolution
        maxx = (width - 0.5) * self._config.resolution
        maxy = (height - 0.5) * self._config.resolution
        mask_2d = polygon_to_mask(
            boundary=(half_res, half_res, maxx, maxy),
            resolution=self._config.resolution,
            polygon_vertices=self.vertices,
        )
        extrusion_height = self.grid_shape[self.axis]
        mask = jnp.repeat(
            jnp.expand_dims(jnp.asarray(mask_2d, dtype=jnp.bool), axis=self.axis),
            repeats=extrusion_height,
            axis=self.axis,
        )
        
        return mask

    def get_material_mapping(
        self,
    ) -> jax.Array:
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        arr = jnp.ones(self.grid_shape, dtype=jnp.int32) * idx
        return arr
