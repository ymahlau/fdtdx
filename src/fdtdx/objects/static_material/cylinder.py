import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.static_material.static import StaticMultiMaterialObject


@autoinit
class Cylinder(StaticMultiMaterialObject):
    """A cylindrical optical fiber with configurable properties.

    This class represents a cylindrical fiber with customizable radius, material,
    and orientation. The fiber can be positioned along any of the three principal axes.

    Attributes:
        radius (float): The radius of the fiber in meter.
        axis (int): The principal axis along which the fiber extends (0=x, 1=y, 2=z).
        material_name (str): Name of the material in the materials dictionary to be used for the object.
    """

    radius: float = frozen_field()
    axis: int = frozen_field()
    material_name: str = frozen_field()

    @property
    def horizontal_axis(self) -> int:
        """Gets the horizontal axis perpendicular to the fiber axis.

        Returns:
            int: The index of the horizontal axis (0=x or 1=y).
        """
        if self.axis == 0:
            return 1
        return 0

    @property
    def vertical_axis(self) -> int:
        """Gets the vertical axis perpendicular to the fiber axis.

        Returns:
            int: The index of the vertical axis (1=y or 2=z).
        """
        if self.axis == 2:
            return 1
        return 2

    def get_voxel_mask_for_shape(self) -> jax.Array:
        width = self.grid_shape[self.vertical_axis]
        height = self.grid_shape[self.horizontal_axis]
        center = (height / 2, width / 2)
        grid_radius_exact = self.radius / self._config.resolution
        grid = (
            jnp.stack(jnp.meshgrid(*map(jnp.arange, (width, height)), indexing="xy"), axis=-1)
            - jnp.asarray(center)
            + 0.5
        ) / jnp.asarray(grid_radius_exact)

        mask = (grid**2).sum(axis=-1) < 1
        mask = jnp.expand_dims(mask, axis=self.axis)
        return mask

    def get_material_mapping(
        self,
    ) -> jax.Array:
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        arr = jnp.ones(self.grid_shape, dtype=jnp.int32) * idx
        return arr
