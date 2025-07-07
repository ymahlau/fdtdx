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

    Attributes:
        radius (float): The default radius of the sphere in meter (used if specific axis radii are not provided).
        material_name (str): Name of the sphere material in the materials dictionary to be used for the object.
        radius_x (float | None, optional): The radius along the x-axis in meter. If none, use radius. Defaults to None.
        radius_y (float | None, optional): The radius along the y-axis in meter. If none, use radius. Defaults to None.
        radius_z (float | None, optional): The radius along the z-axis in meter. If none, use radius. Defaults to None.
    """

    radius: float = frozen_field()
    material_name: str = frozen_field()
    # Optional parameters for ellipsoid shape
    radius_x: float | None = frozen_field(default=None)
    radius_y: float | None = frozen_field(default=None)
    radius_z: float | None = frozen_field(default=None)

    def get_voxel_mask_for_shape(self) -> jax.Array:
        """Generates a voxel mask for a sphere or ellipsoid shape.

        Returns:
            jax.Array: Boolean mask where True indicates voxels inside the sphere/ellipsoid.
        """
        # Get dimensions
        x_dim, y_dim, z_dim = self.grid_shape

        # Define center of the ellipsoid
        center = (x_dim / 2, y_dim / 2, z_dim / 2)

        # Determine the radii for each axis
        radius_x = self.radius_x if self.radius_x is not None else self.radius
        radius_y = self.radius_y if self.radius_y is not None else self.radius
        radius_z = self.radius_z if self.radius_z is not None else self.radius

        # Convert radii to grid units
        grid_radius_x = radius_x / self._config.resolution
        grid_radius_y = radius_y / self._config.resolution
        grid_radius_z = radius_z / self._config.resolution

        # Create 3D grid
        x, y, z = jnp.meshgrid(jnp.arange(x_dim), jnp.arange(y_dim), jnp.arange(z_dim), indexing="ij")

        # Calculate normalized squared distances for each dimension using the ellipsoid equation
        x_term = ((x - center[0] + 0.5) / grid_radius_x) ** 2
        y_term = ((y - center[1] + 0.5) / grid_radius_y) ** 2
        z_term = ((z - center[2] + 0.5) / grid_radius_z) ** 2

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
