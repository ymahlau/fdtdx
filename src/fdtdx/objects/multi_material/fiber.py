from typing import Self

import jax
import jax.numpy as jnp
import pytreeclass as tc

from fdtdx.core.config import SimulationConfig
from fdtdx.core.plotting.colors import LIGHT_GREY
from fdtdx.core.typing import UNDEFINED_SHAPE_3D, PartialGridShape3D, PartialRealShape3D, SliceTuple3D
from fdtdx.objects.multi_material.multi_material import MultiMaterial


@tc.autoinit
class Fiber(MultiMaterial):
    """A cylindrical optical fiber with configurable properties.

    This class represents a cylindrical fiber with customizable radius, permittivity,
    and orientation. The fiber can be positioned along any of the three principal axes
    and supports both inner and outer permittivity configuration.

    Attributes:
        radius: The radius of the fiber in simulation units.
        fiber_permittivity: The relative permittivity (εᵣ) of the fiber material.
        axis: The principal axis along which the fiber extends (0=x, 1=y, 2=z).
        resolution: The spatial resolution of the simulation grid.
        outer_permittivity: The relative permittivity of the surrounding medium (default=1).
        permittivity_config: Dictionary storing the permittivity values for different regions.
        partial_voxel_grid_shape: The shape of the voxel grid in grid units.
        partial_voxel_real_shape: The shape of the voxel grid in physical units.
        color: RGB color tuple for visualization (default=LIGHT_GREY).
    """

    radius: float = tc.field(init=True, kind="KW_ONLY")  # type: ignore
    fiber_permittivity: float = tc.field(init=True, kind="KW_ONLY")  # type: ignore
    axis: int = tc.field(init=True, kind="KW_ONLY")  # type: ignore
    resolution: float = tc.field(init=True, kind="KW_ONLY")  # type: ignore
    outer_permittivity: float = 1
    permittivity_config: dict[str, float] = tc.field(  # type: ignore
        default=None, init=False, on_getattr=[tc.unfreeze], on_setattr=[tc.freeze]
    )
    partial_voxel_grid_shape: PartialGridShape3D = tc.field(default=UNDEFINED_SHAPE_3D, init=False)  # type: ignore
    partial_voxel_real_shape: PartialRealShape3D = tc.field(default=None, init=False)  # type: ignore
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
        voxel_grid_shape = (
            self.resolution,
            self.resolution,
            self.resolution,
        )
        self = self.aset("partial_voxel_real_shape", voxel_grid_shape)
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        permittivity_config = {
            "fiber": self.fiber_permittivity,
            "outside": self.outer_permittivity,
        }
        self = self.aset("permittivity_config", permittivity_config)
        return self

    def get_inv_permittivity(
        self,
        prev_inv_permittivity: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permittivity and info dict
        """Calculates the inverse permittivity distribution for the fiber.

        Computes a 3D array of inverse permittivity values, with the fiber region
        having 1/fiber_permittivity and the surrounding region having 1/outer_permittivity.
        The fiber cross-section is circular in the plane perpendicular to its axis.

        Args:
            prev_inv_permittivity: Previous inverse permittivity array (unused).
            params: Optional parameter dictionary (unused).

        Returns:
            tuple[jax.Array, dict]: Tuple containing:
                - 3D array of inverse permittivity values
                - Empty info dictionary
        """
        del params, prev_inv_permittivity
        inv_permittivity = jnp.ones(shape=self.grid_shape, dtype=jnp.float32)
        inv_permittivity = inv_permittivity / self.outer_permittivity

        width = self.grid_shape[self.vertical_axis]
        height = self.grid_shape[self.horizontal_axis]
        center = (height / 2, width / 2)
        grid_radius_exact = self.radius / self.resolution
        grid = (
            jnp.stack(jnp.meshgrid(*map(jnp.arange, (width, height)), indexing="xy"), axis=-1)
            - jnp.asarray(center)
            + 0.5
        ) / jnp.asarray(grid_radius_exact)

        mask = (grid**2).sum(axis=-1) < 1
        mask = jnp.expand_dims(mask, axis=self.axis)

        inv_permittivity = jnp.where(mask, 1 / self.fiber_permittivity, inv_permittivity)
        inv_permittivity = inv_permittivity
        return inv_permittivity, {}

    def get_inv_permeability(
        self,
        prev_inv_permeability: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permeability and info dict
        """Returns the inverse permeability distribution.

        The fiber does not modify the permeability, so this returns the input unchanged.

        Args:
            prev_inv_permeability: Previous inverse permeability array.
            params: Optional parameter dictionary (unused).

        Returns:
            tuple[jax.Array, dict]: Tuple containing:
                - Unmodified input permeability array
                - Empty info dictionary
        """
        del params
        return prev_inv_permeability, {}
