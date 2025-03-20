from typing import Self

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import extended_autoinit, field, frozen_private_field
from fdtdx.core.plotting.colors import LIGHT_GREY
from fdtdx.materials import Material
from fdtdx.objects.static_material.multi_material import StaticMultiMaterialObject
from fdtdx.typing import UNDEFINED_SHAPE_3D, PartialGridShape3D, PartialRealShape3D, SliceTuple3D


@extended_autoinit
class Cylinder(StaticMultiMaterialObject):
    """A cylindrical optical fiber with configurable properties.

    This class represents a cylindrical fiber with customizable radius, permittivity,
    and orientation. The fiber can be positioned along any of the three principal axes
    and supports both inner and outer permittivity configuration.

    Attributes:
        radius: The radius of the fiber in meter.
        fiber_permittivity: The relative permittivity (εᵣ) of the fiber material.
        axis: The principal axis along which the fiber extends (0=x, 1=y, 2=z).
        outer_permittivity: The relative permittivity of the surrounding medium (default=1).
        permittivity_config: Dictionary storing the permittivity values for different regions.
        partial_voxel_grid_shape: The shape of the voxel grid in grid units.
        partial_voxel_real_shape: The shape of the voxel grid in physical units.
        color: RGB color tuple for visualization (default=LIGHT_GREY).
    """

    radius: float = field(kind="KW_ONLY")
    material: Material = field(kind="KW_ONLY")
    axis: int = field(kind="KW_ONLY")
    permittivity_config: dict[str, Material] = frozen_private_field()
    partial_voxel_grid_shape: PartialGridShape3D = frozen_private_field(default=UNDEFINED_SHAPE_3D)
    partial_voxel_real_shape: PartialRealShape3D = frozen_private_field(default=UNDEFINED_SHAPE_3D)
    color: tuple[float, float, float] = LIGHT_GREY

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

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        """Places the fiber on the simulation grid.

        Configures the voxel grid shape and permittivity settings for the fiber
        based on the given grid slice and configuration.

        Args:
            grid_slice_tuple: 3D tuple of slices defining the grid region.
            config: Configuration object containing simulation parameters.
            key: JAX random key for stochastic operations.

        Returns:
            Self: Updated instance with grid placement configured.
        """
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        permittivity_config = {
            "material": self.material,
        }
        self = self.aset("permittivity_config", permittivity_config)
        return self

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
        return jnp.zeros(
            self.grid_shape,
            dtype=jnp.int32,
        )
