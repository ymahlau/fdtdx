from abc import ABC, abstractmethod

import jax

from fdtdx.core.jax.pytrees import extended_autoinit, field, frozen_field
from fdtdx.core.plotting.colors import LIGHT_BLUE, LIGHT_BROWN, LIGHT_GREY
from fdtdx.materials import Material
from fdtdx.objects.object import OrderableObject


@extended_autoinit
class StaticMaterialObject(OrderableObject):
    material: Material = field(kind="KW_ONLY")
    color: tuple[float, float, float] | None = LIGHT_GREY


@extended_autoinit
class StaticMultiMaterialObject(OrderableObject, ABC):
    materials: dict[str, Material] = frozen_field(kind="KW_ONLY")
    color: tuple[float, float, float] | None = LIGHT_GREY

    @abstractmethod
    def get_voxel_mask_for_shape(self) -> jax.Array:
        """Get a binary mask of the objects shape. Everything voxel not in the mask, will not be updated by
        this object. For example, can be used to approximate a round shape.
        The mask is calculated in device voxel size, not in simulation voxels.

        Returns:
            jax.Array: Binary mask representing the voxels occupied by the object
        """
        raise NotImplementedError()

    @abstractmethod
    def get_material_mapping(
        self,
    ) -> jax.Array:
        """Returns an array, which represents the material index at every voxel. Specifically, it returns the
        index of the ordered material list.

        Returns:
            jax.Array: Index array
        """
        raise NotImplementedError()


@extended_autoinit
class SimulationVolume(StaticMaterialObject):
    """Background material for the entire simulation volume.

    Defines the default material properties for the simulation background.
    Usually represents air/vacuum with εᵣ=1.0 and μᵣ=1.0.
    """

    placement_order = -1000
    material: Material = field(
        default=Material(
            permittivity=1.0,
            permeability=1.0,
        ),
        kind="KW_ONLY",
    )


@extended_autoinit
class Substrate(StaticMaterialObject):
    """Material representing a substrate layer.

    Used to model substrate materials like silicon dioxide.
    Visualized in light brown color by default.
    """

    color: tuple[float, float, float] | None = LIGHT_BROWN


@extended_autoinit
class Waveguide(StaticMaterialObject):
    """Material for optical waveguides.

    Used to model waveguide structures that can guide electromagnetic waves.
    Visualized in light blue color by default.

    Attributes:
        permittivity: Required relative permittivity of the waveguide material
        color: RGB tuple for visualization, defaults to light blue
    """

    color: tuple[float, float, float] | None = LIGHT_BLUE
