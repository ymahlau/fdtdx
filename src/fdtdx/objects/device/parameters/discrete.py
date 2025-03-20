from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_field, frozen_private_field
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.core.misc import PaddingConfig, get_air_name
from fdtdx.materials import Material, compute_ordered_names
from fdtdx.objects.device.parameters.binary_transform import (
    binary_median_filter,
    connect_holes_and_structures,
    remove_floating_polymer,
)


@extended_autoinit
class DiscreteTransformation(ExtendedTreeClass, ABC):
    _material: dict[str, Material] = frozen_private_field()
    _config: SimulationConfig = frozen_private_field()

    def init_module(
        self: Self,
        config: SimulationConfig,
        material: dict[str, Material],
    ) -> Self:
        self = self.aset("_config", config)
        self = self.aset("_material", material)
        return self

    @abstractmethod
    def __call__(
        self,
        material_indices: jax.Array,
    ) -> jax.Array:
        raise NotImplementedError()


@extended_autoinit
class RemoveFloatingMaterial(DiscreteTransformation):
    """Finds all material that floats in the air and sets their permittivity to air.

    This constraint module identifies regions of material that are not connected to any
    substrate or boundary and converts them to air. This helps ensure physically
    realizable designs by eliminating floating/disconnected material regions.

    The module only works with binary material systems (2 permittivities) where one
    material represents air.
    """

    def __call__(
        self,
        material_indices: jax.Array,
    ) -> jax.Array:
        if len(self._material) != 2:
            raise NotImplementedError("Remove floating material currently only implemented for single material")
        air_name = get_air_name(self._material)
        ordered_name_list = compute_ordered_names(self._material)
        air_idx = ordered_name_list.index(air_name)

        is_material_matrix = material_indices != air_idx
        is_material_after_removal = remove_floating_polymer(is_material_matrix)
        result = (1 - air_idx) * is_material_after_removal + air_idx * ~is_material_after_removal
        result = straight_through_estimator(material_indices, result)
        return result


@extended_autoinit
class ConnectHolesAndStructures(DiscreteTransformation):
    """Connects floating polymer regions and ensures air holes connect to outside.

    This constraint module ensures physical realizability of designs by:
    1. Either connecting floating polymer regions to the substrate or removing them
    2. Ensuring all air holes are connected to the outside (no trapped air)

    The bottom (lower z) is treated as the substrate reference.

    Attributes:
        fill_material: Name of material to use for filling gaps when connecting regions.
            Required when working with more than 2 materials.
    """

    fill_material: str | None = frozen_field(default=None)

    def __call__(
        self,
        material_indices: jax.Array,
    ) -> jax.Array:
        if len(self._material) > 2 and self.fill_material is None:
            raise Exception(
                "ConnectHolesAndStructures: Need to specify fill material when working with more than a single material"
            )
        air_name = get_air_name(self._material)
        ordered_name_list = compute_ordered_names(self._material)
        air_idx = ordered_name_list.index(air_name)
        is_material_matrix = material_indices != air_idx
        feasible_material_matrix = connect_holes_and_structures(is_material_matrix)

        result = jnp.empty_like(material_indices)
        # set air
        result = jnp.where(
            feasible_material_matrix,
            -1,  # this is set below
            air_idx,
        )
        # material where previously was material
        result = jnp.where(feasible_material_matrix & is_material_matrix, material_indices, result)

        # material, where previously was air
        fill_name = self.fill_material
        if fill_name is None:
            fill_name = ordered_name_list[1 - air_idx]
        fill_idx = ordered_name_list.index(fill_name)
        result = jnp.where(
            feasible_material_matrix & ~is_material_matrix,
            fill_idx,
            result,
        )
        return result


BOTTOM_Z_PADDING_CONFIG_REPEAT = PaddingConfig(
    modes=("edge", "edge", "edge", "edge", "constant", "edge"), widths=(20,), values=(1,)
)

BOTTOM_Z_PADDING_CONFIG = PaddingConfig(
    modes=("constant", "constant", "constant", "constant", "constant", "constant"),
    widths=(10,),
    values=(1, 0, 1, 1, 1, 0),
)


@extended_autoinit
class BinaryMedianFilterModule(DiscreteTransformation):
    """Performs 3D binary median filtering on the design.

    Applies a 3D median filter to smooth and clean up binary material distributions.
    This helps remove small features and noise while preserving larger structures.

    Attributes:
        padding_cfg: Configuration for padding behavior at boundaries.
        kernel_sizes: 3-tuple of kernel sizes for each dimension.
        num_repeats: Number of times to apply the filter consecutively.
    """

    padding_cfg: PaddingConfig = frozen_field()
    kernel_sizes: tuple[int, int, int] = frozen_field()
    num_repeats: int = frozen_field(default=1)

    def __call__(
        self,
        material_indices: jax.Array,
    ) -> jax.Array:
        if len(self._material) != 2:
            raise Exception("BinaryMedianFilterModule only works for two materials!")
        cur_arr = material_indices
        for _ in range(self.num_repeats):
            cur_arr = binary_median_filter(
                arr_3d=cur_arr,
                kernel_sizes=self.kernel_sizes,
                padding_cfg=self.padding_cfg,
            )
        result = straight_through_estimator(material_indices, cur_arr)
        return result
