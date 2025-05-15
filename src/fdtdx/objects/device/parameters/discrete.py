from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_field, frozen_private_field
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.core.misc import PaddingConfig, get_background_material_name
from fdtdx.materials import Material, compute_ordered_names
from fdtdx.objects.device.parameters.binary_transform import (
    binary_median_filter,
    connect_holes_and_structures,
    remove_floating_polymer,
)
from fdtdx.objects.device.parameters.transform import (
    SameShapeBinaryParameterTransform, 
    SameShapeDiscreteParameterTransform
)


@extended_autoinit
class RemoveFloatingMaterial(SameShapeBinaryParameterTransform):
    """Finds all material that floats in the air and sets their permittivity to air.

    This constraint module identifies regions of material that are not connected to any
    substrate or boundary and converts them to air. This helps ensure physically
    realizable designs by eliminating floating/disconnected material regions.

    The module only works with binary material systems (2 permittivities) where one
    material represents air.
    """
    background_material: str | None = frozen_field(default=None)

    def __call__(
        self,
        params: dict[str, jax.Array] | jax.Array,
        **kwargs,
    ) -> dict[str, jax.Array] | jax.Array:
        del kwargs
        if isinstance(params, dict):
            raise Exception(
                f"RemoveFloatingMaterial only implemented for a single array as input. Please make sure that the "
                "previous transformation outputs a single array"
            )
        if self.background_material is None:
            background_name = get_background_material_name(self._materials)
        else:
            background_name = self.background_material
        ordered_name_list = compute_ordered_names(self._materials)
        background_idx = ordered_name_list.index(background_name)
        
        is_material_matrix = params != background_idx
        is_material_after_removal = remove_floating_polymer(is_material_matrix)
        result = (1 - background_idx) * is_material_after_removal + background_idx * ~is_material_after_removal
        result = straight_through_estimator(params, result)
        return result


@extended_autoinit
class ConnectHolesAndStructures(SameShapeDiscreteParameterTransform):
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
    background_material: str | None = frozen_field(default=None)

    def __call__(
        self,
        params: dict[str, jax.Array] | jax.Array,
        **kwargs,
    ) -> dict[str, jax.Array] | jax.Array:
        del kwargs
        if isinstance(params, dict):
            raise Exception(
                f"ConnectHolesAndStructures only implemented for a single array as input. Please make sure that the "
                "previous transformation outputs a single array"
            )
        if len(self._materials) > 2 and self.fill_material is None:
            raise Exception(
                "ConnectHolesAndStructures: Need to specify fill_material when working with more than two materials"
            )
        if self.background_material is None:
            background_name = get_background_material_name(self._materials)
        else:
            background_name = self.background_material
        ordered_name_list = compute_ordered_names(self._materials)
        air_idx = ordered_name_list.index(background_name)
        is_material_matrix = params != air_idx
        feasible_material_matrix = connect_holes_and_structures(is_material_matrix)

        result = jnp.empty_like(params)
        # set air
        result = jnp.where(
            feasible_material_matrix,
            -1,  # this is set below
            air_idx,
        )
        # material where previously was material
        result = jnp.where(feasible_material_matrix & is_material_matrix, params, result)

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
        result = straight_through_estimator(params, result)
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
class BinaryMedianFilterModule(SameShapeBinaryParameterTransform):
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
        params: dict[str, jax.Array] | jax.Array,
        **kwargs,
    ) -> dict[str, jax.Array] | jax.Array:
        del kwargs
        if isinstance(params, dict):
            raise Exception(
                f"BinaryMedianFilterModule only implemented for a single array as input. Please make sure that the "
                "previous transformation outputs a single array"
            )
        cur_arr = params
        for _ in range(self.num_repeats):
            cur_arr = binary_median_filter(
                arr_3d=cur_arr,
                kernel_sizes=self.kernel_sizes,
                padding_cfg=self.padding_cfg,
            )
        result = straight_through_estimator(params, cur_arr)
        return result
