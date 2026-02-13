from abc import ABC, abstractmethod

import jax

from fdtdx.colors import Color, XKCD_LIGHT_GREY
from fdtdx.core.jax.pytrees import autoinit, field, frozen_field
from fdtdx.materials import Material
from fdtdx.objects.object import OrderableObject


@autoinit
class UniformMaterialObject(OrderableObject):
    #: the material object
    material: Material = field()

    #: the color object
    color: Color | None = frozen_field(default=XKCD_LIGHT_GREY)


@autoinit
class StaticMultiMaterialObject(OrderableObject, ABC):
    #: the static material
    materials: dict[str, Material] = field()

    #: the color of the material
    color: Color | None = frozen_field(default=XKCD_LIGHT_GREY)

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


@autoinit
class SimulationVolume(UniformMaterialObject):
    """Background material for the entire simulation volume.

    Defines the default material properties for the simulation background.
    Usually represents air/vacuum with εᵣ=1.0 and μᵣ=1.0.
    """

    #: an integer values of the placement order
    placement_order: int = frozen_field(default=-1000)

    #: the static material
    material: Material = field(
        default=Material(
            permittivity=(1.0, 1.0, 1.0),
            permeability=(1.0, 1.0, 1.0),
        ),
    )
