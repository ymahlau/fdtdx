import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.grid import polygon_to_mask
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.static_material.static import StaticMultiMaterialObject


@autoinit
class ExtrudedPolygon(StaticMultiMaterialObject):
    """A polygon object specified by a list of vertices. The coordinate system has its origin at the lower left of the
    bounding box of the polygon.

    Attributes:
        material_name (str): Name of the material in the materials dictionary to be used for the object
        axis (int): The extrusion axis.
        vertices (np.ndarray): numpy array of shape (N, 2) specifying the position of vertices in metrical units
            (meter).
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

    @property
    def centered_vertices(self) -> np.ndarray:
        vx = self.vertices[:, 0] + 0.5 * self.real_shape[self.horizontal_axis]
        vy = self.vertices[:, 1] + 0.5 * self.real_shape[self.vertical_axis]
        return np.stack((vx, vy), axis=-1)

    def get_voxel_mask_for_shape(self) -> jax.Array:
        n_horizontal = self.grid_shape[self.horizontal_axis]
        n_vertical = self.grid_shape[self.vertical_axis]

        half_res = 0.5 * self._config.resolution
        max_horizontal = (n_horizontal - 0.5) * self._config.resolution
        max_vertical = (n_vertical - 0.5) * self._config.resolution

        # move vertices to center

        mask_2d = polygon_to_mask(
            boundary=(half_res, half_res, max_horizontal, max_vertical),
            resolution=self._config.resolution,
            polygon_vertices=self.centered_vertices,
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
