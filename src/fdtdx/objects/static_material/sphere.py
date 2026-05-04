import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.static_material.static import StaticMultiMaterialObject


@autoinit
class Sphere(StaticMultiMaterialObject):
    """A sphere or ellipsoid object with configurable properties.

    This class represents a sphere or ellipsoid with customizable radius/radii and material.
    When all three radii are equal, the shape is a perfect sphere.

    """

    #: The default radius of the sphere in meter (used if specific axis radii are not provided).
    radius: float = frozen_field()

    #: Name of the sphere material in the materials dictionary to be used for the object.
    material_name: str = frozen_field()
    # Optional parameters for ellipsoid shape

    #: The radius along the x-axis in meter. If none, use radius. Defaults to None.
    radius_x: float | None = frozen_field(default=None)

    #: The radius along the y-axis in meter. If none, use radius. Defaults to None.
    radius_y: float | None = frozen_field(default=None)

    #: The radius along the z-axis in meter. If none, use radius. Defaults to None.
    radius_z: float | None = frozen_field(default=None)

    def get_voxel_mask_for_shape(self) -> jax.Array:
        """Generates a voxel mask for a sphere or ellipsoid shape.

        Returns:
            jax.Array: Boolean mask where True indicates voxels inside the sphere/ellipsoid.
        """
        # Determine the radii for each axis
        radius_x = self.radius_x if self.radius_x is not None else self.radius
        radius_y = self.radius_y if self.radius_y is not None else self.radius
        radius_z = self.radius_z if self.radius_z is not None else self.radius

        def local_centers(axis: int) -> jax.Array:
            """Return physical cell centers relative to this object's lower edge."""
            lower, upper = self.grid_slice_tuple[axis]
            grid = self._config.realized_grid
            if grid is None:
                spacing = self._config.require_uniform_grid()
                return (jnp.arange(self.grid_shape[axis]) + 0.5) * spacing
            edges = grid.edges(axis)
            return 0.5 * (edges[lower:upper] + edges[lower + 1 : upper + 1]) - edges[lower]

        # Create 3D grid
        x, y, z = jnp.meshgrid(local_centers(0), local_centers(1), local_centers(2), indexing="ij")
        center_x, center_y, center_z = (0.5 * axis_size for axis_size in self.real_shape)

        # Calculate normalized squared distances for each dimension using the ellipsoid equation
        x_term = ((x - center_x) / radius_x) ** 2
        y_term = ((y - center_y) / radius_y) ** 2
        z_term = ((z - center_z) / radius_z) ** 2

        # Create mask based on ellipsoid equation: points inside if x^2/a^2 + y^2/b^2 + z^2/c^2 < 1
        mask = (x_term + y_term + z_term) < 1

        return mask

    def get_material_mapping(
        self,
    ) -> jax.Array:
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        arr = jnp.ones(self.grid_shape, dtype=jnp.int32) * idx
        return arr
