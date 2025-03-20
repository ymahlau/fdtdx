from abc import ABC, abstractmethod

import jax

from fdtdx.core.jax.pytrees import extended_autoinit, frozen_field
from fdtdx.materials import ContinuousMaterialRange, Material
from fdtdx.objects.object import OrderableObject


@extended_autoinit
class StaticMultiMaterialObject(OrderableObject, ABC):
    material: dict[str, Material] | ContinuousMaterialRange = frozen_field(kind="KW_ONLY")  # type: ignore

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
