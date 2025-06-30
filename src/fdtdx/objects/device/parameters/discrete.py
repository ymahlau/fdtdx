from typing import Sequence

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field, frozen_private_field
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.core.misc import PaddingConfig, get_background_material_name
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.device.parameters.binary_transform import (
    binary_median_filter,
    connect_holes_and_structures,
    remove_floating_polymer,
)
from fdtdx.objects.device.parameters.transform import (
    SameShapeTypeParameterTransform,
)
from fdtdx.typing import ParameterType


@autoinit
class RemoveFloatingMaterial(SameShapeTypeParameterTransform):
    """Finds all material that floats in the air and sets their permittivity to air.

    This constraint module identifies regions of material that are not connected to any
    substrate or boundary and converts them to air. This helps ensure physically
    realizable designs by eliminating floating/disconnected material regions.

    The module only works with binary material systems (2 permittivities) where one
    material represents air.

    Attributes:
        background_material (str | None, optional): Name of the background material. If none, the material with the
            lowest permittivity is used. Defaults to None.
    """

    background_material: str | None = frozen_field(default=None)

    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=(ParameterType.DISCRETE, ParameterType.BINARY),
    )
    _check_single_array: bool = frozen_private_field(default=True)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        if self.background_material is None:
            background_name = get_background_material_name(self._materials)
        else:
            background_name = self.background_material
        ordered_name_list = compute_ordered_names(self._materials)
        background_idx = ordered_name_list.index(background_name)

        single_key = list(params.keys())[0]
        param_arr = params[single_key]
        is_material_matrix = param_arr != background_idx
        is_material_after_removal = remove_floating_polymer(is_material_matrix)
        result = (1 - background_idx) * is_material_after_removal + background_idx * ~is_material_after_removal
        result = straight_through_estimator(param_arr, result)
        return {single_key: result}


@autoinit
class ConnectHolesAndStructures(SameShapeTypeParameterTransform):
    """Connects floating polymer regions and ensures air holes connect to outside.

    This constraint module ensures physical realizability of designs by:
    1. Either connecting floating polymer regions to the substrate or removing them
    2. Ensuring all air holes are connected to the outside (no trapped air)

    The bottom (lower z) is treated as the substrate reference.

    Attributes:
        fill_material (str | None, optional): Name of material to use for filling gaps when connecting regions.
            Required when working with more than 2 materials. Defaults to None.
        background_material (str | None, optional): Name of the background material. If none, the material with the
            lowest permittivity is used. Defaults to None.
    """

    fill_material: str | None = frozen_field(default=None)
    background_material: str | None = frozen_field(default=None)
    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=(ParameterType.DISCRETE, ParameterType.BINARY),
    )
    _check_single_array: bool = frozen_private_field(default=True)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        if len(self._materials) > 2 and self.fill_material is None:
            raise Exception(
                "ConnectHolesAndStructures: Need to specify fill_material when working with more than two materials"
            )
        if self.background_material is None:
            background_name = get_background_material_name(self._materials)
        else:
            background_name = self.background_material
        ordered_name_list = compute_ordered_names(self._materials)
        background_idx = ordered_name_list.index(background_name)

        single_key = list(params.keys())[0]
        param_arr = params[single_key]
        is_material_matrix = param_arr != background_idx
        feasible_material_matrix = connect_holes_and_structures(is_material_matrix)

        result = jnp.empty_like(param_arr)
        # set air
        result = jnp.where(
            feasible_material_matrix,
            -1,  # this is set below
            background_idx,
        )
        # material where previously was material
        result = jnp.where(feasible_material_matrix & is_material_matrix, param_arr, result)

        # material, where previously was background material (air)
        fill_name = self.fill_material
        if fill_name is None:
            fill_name = ordered_name_list[1 - background_idx]
        fill_idx = ordered_name_list.index(fill_name)
        result = jnp.where(
            feasible_material_matrix & ~is_material_matrix,
            fill_idx,
            result,
        )
        result = straight_through_estimator(param_arr, result)
        return {single_key: result}


BOTTOM_Z_PADDING_CONFIG_REPEAT = PaddingConfig(
    modes=("edge", "edge", "edge", "edge", "constant", "edge"), widths=(20,), values=(1,)
)

BOTTOM_Z_PADDING_CONFIG = PaddingConfig(
    modes=("constant", "constant", "constant", "constant", "constant", "constant"),
    widths=(10,),
    values=(1, 0, 1, 1, 1, 0),
)


@autoinit
class BinaryMedianFilterModule(SameShapeTypeParameterTransform):
    """Performs 3D binary median filtering on the design.

    Applies a 3D median filter to smooth and clean up binary material distributions.
    This helps remove small features and noise while preserving larger structures.

    Attributes:
        padding_cfg (PaddingConfig): Configuration for padding behavior at boundaries.
        kernel_sizes (tuple[int, int, int]): 3-tuple of kernel sizes for each dimension.
        num_repeats (int, optional): Number of times to apply the filter consecutively. Defaults to one.
    """

    padding_cfg: PaddingConfig = frozen_field()
    kernel_sizes: tuple[int, int, int] = frozen_field()
    num_repeats: int = frozen_field(default=1)

    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=ParameterType.BINARY,
    )
    _check_single_array: bool = frozen_private_field(default=True)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        single_key = list(params.keys())[0]
        param_arr = params[single_key]
        cur_arr = param_arr
        for _ in range(self.num_repeats):
            cur_arr = binary_median_filter(
                arr_3d=cur_arr,
                kernel_sizes=self.kernel_sizes,
                padding_cfg=self.padding_cfg,
            )
        result = straight_through_estimator(param_arr, cur_arr)
        return {single_key: result}
