from abc import ABC, abstractmethod
from typing import Self, Sequence
import jax
import jax.numpy as jnp
import pytreeclass as tc

from fdtdx.constraints.binary_transform import binary_median_filter, connect_holes_and_structures, remove_floating_polymer
from fdtdx.core.misc import PaddingConfig
from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit, frozen_field

@tc.autoinit
class DiscreteParameterMapping(ExtendedTreeClass, ABC):
    _allowed_permittivities: Sequence[float] = tc.field(
        default=None,
        init=False,
    )  # type: ignore
    _permittivity_names: Sequence[str] = tc.field(
        default=None,
        init=False,
        on_setattr=[tc.freeze],
        on_getattr=[tc.unfreeze],
    )  # type: ignore
    
    @abstractmethod
    def __call__(
        self,
        params: jax.Array,
    ) -> jax.Array:
        del params
        raise NotImplementedError()
    
    def init_module(
        self: Self,
        allowed_permittivities: Sequence[float],
        permittivity_names: Sequence[str],
    ) -> Self:
        self = self.aset("_allowed_permittivities", allowed_permittivities)
        self = self.aset("_permittivity_names", permittivity_names)
        return self


@tc.autoinit
class RemoveFloatingMaterial(DiscreteParameterMapping):
    """
    Finds all material that floats in the air and sets their permittivity to air
    """
    
    def __call__(
        self,
        params: jax.Array,
    ) -> jax.Array:
        if len(self._allowed_permittivities) != 2:
            raise NotImplementedError(
                f"Remove floating material currently only implemented for single material"
            )
        air_idx = self._allowed_permittivities.index(1)
        is_material_matrix = (params != air_idx)
        result = remove_floating_polymer(is_material_matrix)
        result = result.astype(params.dtype)
        return result


@tc.autoinit
class ConnectHolesAndStructures(DiscreteParameterMapping):
    """
    Connects all floating polymer or throws it away and connects all air holes to the outside.
    Bottom is lower z.
    """
    
    fill_material: str | None = tc.field(
        default=None,
        init=True,
        on_getattr=[tc.unfreeze],
        on_setattr=[tc.freeze],
    ) # type: ignore
    
    def __call__(
        self,
        params: jax.Array,
    ) -> jax.Array:
        if len(self._allowed_permittivities) > 2 and self.fill_material is None:
            raise Exception(
                f"Need to specify fill material when working with more than a single material"
            )
        air_idx = self._allowed_permittivities.index(1)
        is_material_matrix = (params != air_idx)
        feasible_material_matrix = connect_holes_and_structures(is_material_matrix)
        
        result = jnp.empty_like(params)
        # set air
        result = jnp.where(
            feasible_material_matrix,
            -1,  # this is set below
            air_idx,
        )
        # material where previously was material
        result = jnp.where(
            feasible_material_matrix & is_material_matrix,
            params,
            result
        )

        # material, where previously was air
        fill_idx = self._permittivity_names.index(self.fill_material)
        result = jnp.where(
            feasible_material_matrix & ~is_material_matrix,
            fill_idx,
            result,
        )
        return result


BOTTOM_Z_PADDING_CONFIG_REPEAT = PaddingConfig(
    modes=("edge", "edge", "edge", "edge", "constant", "edge"),
    widths=(20,),
    values=(1,)
)

BOTTOM_Z_PADDING_CONFIG = PaddingConfig(
    modes=("constant", "constant", "constant", "constant", "constant", "constant"),
    widths=(10,),
    values=(1, 0, 1, 1, 1, 0),
)

@extended_autoinit
class BinaryMedianFilterMapping(DiscreteParameterMapping):
    """
    Performs a 3D median filter over the design
    """
    padding_cfg: PaddingConfig = frozen_field()
    kernel_sizes: tuple[int, int, int] = frozen_field()
    num_repeats: int = frozen_field(default=1)
    
    def __call__(
        self,
        params: jax.Array,
    ) -> jax.Array:
        if params.ndim != 3:
            raise Exception(f"Invalid parameter shape: {params.shape}")
        if len(self._allowed_permittivities) != 2:
            raise Exception(f"BinaryMedianFilterMapping only works for two materials")
        
        for _ in range(self.num_repeats):
            params = binary_median_filter(
                arr_3d=params,
                kernel_sizes=self.kernel_sizes,
                padding_cfg=self.padding_cfg,
            )
        return params