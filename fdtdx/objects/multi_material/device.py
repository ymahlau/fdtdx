from typing import Self
import jax
import jax.numpy as jnp
import pytreeclass as tc

from fdtdx.constraints.mapping import ConstraintMapping
from fdtdx.core.config import SimulationConfig
from fdtdx.core.plotting.colors import PINK
from fdtdx.objects.multi_material.multi_material import MultiMaterial
from fdtdx.core.misc import expand_matrix
from fdtdx.core.jax.typing import SliceTuple3D

@tc.autoinit
class Device(MultiMaterial):
    """
    Object with an Array of permittivities that are optimizable
    """
    name: str = tc.field(  # type: ignore
        init=True, 
        kind="KW_ONLY",
        on_getattr=[tc.unfreeze],
        on_setattr=[tc.freeze],
    )
    constraint_mapping: ConstraintMapping = tc.field(init=True, kind="KW_ONLY")  # type: ignore
    dtype: jnp.dtype = tc.field( # type: ignore
        init=True,
        default=jnp.float32,
        kind="KW_ONLY",
        on_getattr=[tc.unfreeze],
        on_setattr=[tc.freeze],
    )
    color: tuple[float, float, float] = PINK
    permittivity_config: dict[str, float] = tc.field(default=None, init=False)   # type: ignore
    
    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        self = self.aset(
            "permittivity_config",
            self.constraint_mapping.permittivity_config
        )
        mapping = self.constraint_mapping.init_module(self.matrix_voxel_grid_shape)
        self = self.aset("constraint_mapping", mapping)
        
        return self
    
    def init_params(
        self,
        key: jax.Array,
    ) -> dict[str, jax.Array]:
        params = jax.random.uniform(
            key=key,
            shape=self.matrix_voxel_grid_shape,
            minval=0,  # parameter always live between 0 and 1, mapped later to inv_perm
            maxval=1,
            dtype=self.dtype,
        )
        return {
            'params': params
        }

    def get_inv_permittivity(
        self,
        prev_inv_permittivity: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permittivity and info dict
        del prev_inv_permittivity
        if params is None:
            raise Exception(f"Device params cannot be None")
        quantized_array = self.constraint_mapping(params['params'])
        extended_params = expand_matrix(
            matrix=quantized_array,
            grid_points_per_voxel=self.single_voxel_grid_shape,
            add_channels=False,
        )
        return extended_params, {}
    
    def get_inv_permeability(
        self,
        prev_inv_permeability: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permeability and info dict
        del params
        return prev_inv_permeability, {}
    
    def get_indices(
        self,
        params: dict[str, jax.Array],
    ) -> jax.Array:
        quantized_array = self.constraint_mapping(
            params['params'],
            return_indices=True,
        )
        return quantized_array
