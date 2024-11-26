from abc import ABC, abstractmethod
from typing import Literal, Self
import jax
import jax.numpy as jnp
import pytreeclass as tc
from loguru import logger

from fdtdx.constraints.utils import compute_allowed_indices, nearest_index
from fdtdx.core.misc import get_air_name
from fdtdx.core.jax.pytrees import ExtendedTreeClass
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.core.jax.typing import GridShape3D


@tc.autoinit
class ContinousParameterMapping(ExtendedTreeClass, ABC):
    _permittivity_config: dict[str, float] = tc.field(default=None, init=False)  # type: ignore
    _matrix_voxel_grid_shape: GridShape3D = tc.field(default=None, init=False)  # type: ignore
    
    def init_params(
        self,
        permittivity_config: dict[str, float],
        matrix_voxel_grid_shape: GridShape3D,
    ) -> Self:
        self = self.aset("_permittivity_config", permittivity_config)
        self = self.aset("_matrix_voxel_grid_shape", matrix_voxel_grid_shape)
        return self
    
    @abstractmethod
    def __call__(
        self,
        params: jax.Array,
    ) -> jax.Array:
        del params
        raise NotImplementedError()
    
    
@tc.autoinit
class PillarMapping(ContinousParameterMapping):
    axis: int = tc.field(init=True, kind="KW_ONLY") # type: ignore
    single_polymer_columns: bool = tc.field(init=True, kind="KW_ONLY") # type: ignore
    distance_metric: Literal[
        "euclidean",
        "permittivity_differences_plus_average_permittivity",
    ] = tc.field(
        default="permittivity_differences_plus_average_permittivity",
        init=True,
        on_setattr=[tc.freeze],
        on_getattr=[tc.unfreeze],
    )  # type: ignore
    _allowed_indices: jax.Array = tc.field(
        default=None, 
        init=False,
        on_setattr=[tc.freeze],
        on_getattr=[tc.unfreeze],
    )  # type: ignore
    _index_map: dict[str, int] = tc.field(
        default=None,
        init=False,
        on_setattr=[tc.freeze],
        on_getattr=[tc.unfreeze],
    )  # type: ignore
    _latent_vals: jax.Array = tc.field(
        default=None,
        init=False,
        on_setattr=[tc.freeze],
        on_getattr=[tc.unfreeze],
    )  # type: ignore
    
    def init_params(
        self,
        permittivity_config: dict[str, float],
        matrix_voxel_grid_shape: GridShape3D,
    ) -> Self:
        self = super().init_params(
            permittivity_config=permittivity_config,
            matrix_voxel_grid_shape=matrix_voxel_grid_shape,
        )
        # construct index map
        index_map, latent_vals = {}, []
        inv_perms = [1/x for x in permittivity_config.values()]
        max_inv_perm, min_inv_perm = max(inv_perms), min(inv_perms)
        for idx, (k, v) in enumerate(permittivity_config.items()):
            index_map[k] = idx
            cur_latent = (1/v - min_inv_perm) / (max_inv_perm - min_inv_perm)
            latent_vals.append(cur_latent)
        self = self.aset("_index_map", index_map)
        self = self.aset("_latent_vals", jnp.asarray(latent_vals, dtype=jnp.float32))
        
        air_name = get_air_name(self._permittivity_config)
        indices = list(index_map.values())
        allowed_columns = compute_allowed_indices(
            num_layers=matrix_voxel_grid_shape[self.axis],
            indices=indices,
            fill_holes_with_index=[index_map[air_name]],
            single_polymer_columns=self.single_polymer_columns,
        )
        self = self.aset("_allowed_indices", allowed_columns)
        logger.info(f"{allowed_columns=}")
        return self
    
    
    def __call__(
        self,
        params: jax.Array,
    ) -> jax.Array:
        
        nearest_allowed_index = nearest_index(
            values=params,
            allowed_values=self._latent_vals,
            axis=self.axis,
            distance_metric=self.distance_metric,
            allowed_indices=self._allowed_indices,
            return_distances=False,
        )
        latents = self._latent_vals[self._allowed_indices]
        quantized_latents = latents[nearest_allowed_index]
        if self.axis == 2:
            pass  # no transposition needed
        elif self.axis == 1:
            quantized_latents = jnp.transpose(quantized_latents, axes=(0, 2, 1))
        elif self.axis == 0:
            quantized_latents = jnp.transpose(quantized_latents, axes=(2, 0, 1))
        else:
            raise Exception(f"invalid axis: {self.axis}")
        
        quantized_latents = straight_through_estimator(params, quantized_latents)
        return quantized_latents

