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

    """

    #: The radius of the fiber in meter.
    radius: float = frozen_field()

    #: The principal axis along which the fiber extends (0=x, 1=y, 2=z).
    axis: int = frozen_field()

    #: Name of the material in the materials dictionary to be used for the object.
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
        def local_centers(axis: int) -> jax.Array:
            """Return physical cell centers relative to this object's lower edge."""
            lower, upper = self.grid_slice_tuple[axis]
            grid = self._config.realized_grid
            if grid is None:
                spacing = self._config.require_uniform_grid()
                return (jnp.arange(self.grid_shape[axis]) + 0.5) * spacing
            edges = grid.edges(axis)
            return 0.5 * (edges[lower:upper] + edges[lower + 1 : upper + 1]) - edges[lower]

        horizontal = local_centers(self.horizontal_axis)
        vertical = local_centers(self.vertical_axis)
        horizontal_grid, vertical_grid = jnp.meshgrid(horizontal, vertical, indexing="ij")
        center_h = 0.5 * self.real_shape[self.horizontal_axis]
        center_v = 0.5 * self.real_shape[self.vertical_axis]
        grid = jnp.stack((horizontal_grid - center_h, vertical_grid - center_v), axis=-1) / self.radius

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
