from typing import Self, Sequence
import jax
import jax.numpy as jnp

from fdtdx.constraints.discrete import DiscreteParameterMapping
from fdtdx.constraints.continuous import ContinousParameterMapping
from fdtdx.constraints.projection import ParameterProjection
from fdtdx.core.jax.pytrees import ExtendedTreeClass
from fdtdx.core.jax.ste import straight_through_estimator
from fdtdx.core.jax.typing import INVALID_SHAPE_3D, GridShape3D


class ConstraintMapping(ExtendedTreeClass):
    
    def __init__(
        self,
        permittivity_config: dict[str, float],
        projection: ParameterProjection | None = None,
        continuous_modules: Sequence[ContinousParameterMapping] = [],
        discrete_modules: Sequence[DiscreteParameterMapping] = [],
        
    ):
        self.permittivity_config = permittivity_config
        self.projection = projection
        self.continuous_modules = continuous_modules
        self.discrete_modules = discrete_modules
        
        self.matrix_voxel_grid_shape: GridShape3D = INVALID_SHAPE_3D
    
    @property
    def ordered_permittivity_tuples(self) -> list[tuple[str, float]]:
        kv = list(self.permittivity_config.items())
        kv_sorted = sorted(kv, key=lambda x: x[1])
        return kv_sorted
    
    @property
    def allowed_permittivities(self):
        name_val_list = self.ordered_permittivity_tuples
        perms = jnp.asarray([v[1] for v in name_val_list], dtype=jnp.float32)
        return perms
    
    @property
    def allowed_inverse_permittivities(self):
        return 1.0 / self.allowed_permittivities
    
    def params_shape(self) -> tuple[int, ...]:
        if self.projection is not None:
            return self.projection.params_shape()
        return self.matrix_voxel_grid_shape
    
    def init_module(
        self: Self,
        matrix_voxel_grid_shape: GridShape3D,
    ) -> Self:
        # init attributes
        self = self.aset("matrix_voxel_grid_shape", matrix_voxel_grid_shape)
        
        # init continuous modules
        cont_modules = []
        for m in self.continuous_modules:
            m_init = m.init_params(
                permittivity_config=self.permittivity_config,
                matrix_voxel_grid_shape=matrix_voxel_grid_shape,
            )
            cont_modules.append(m_init)
        self = self.aset("continuous_modules", cont_modules)
        
        # init discrete modules
        discrete_module_list = []
        perm_tuples = self.ordered_permittivity_tuples
        for m in self.discrete_modules:
            m_new = m.init_module(
                permittivity_names=[v[0] for v in perm_tuples],
                allowed_permittivities=[v[1] for v in perm_tuples],
            )
            discrete_module_list.append(m_new)
        
        self = self.aset("discrete_modules", discrete_module_list)
        
        return self
    

    def __call__(
        self,
        params: jax.Array,
        return_indices: bool = False,
    ) -> jax.Array:
        if self.matrix_voxel_grid_shape == INVALID_SHAPE_3D:
            raise Exception(f"Mapping is not yet initialized")
        
        if self.projection is not None:
            params = self.projection(params)

        # continous mappings
        for m in self.continuous_modules:
            params = m(params)
        
        # scale 
        allowed_max = self.allowed_inverse_permittivities.max()
        allowed_min = self.allowed_inverse_permittivities.min()
        allowed_range = allowed_max - allowed_min
        scaled = (params * allowed_range + allowed_min)
        
        # convert to closest discrete value
        dist = jnp.abs(
            scaled[..., None] - self.allowed_inverse_permittivities[None, None, None, :]
        )
        discrete = jnp.argmin(dist, axis=-1).astype(jnp.int32)
        
        # discrete mappings
        for m in self.discrete_modules:
            discrete = m(discrete)
        
        if return_indices:
            return discrete
            
        # bool to permittivities
        output = self.allowed_inverse_permittivities[discrete.astype(jnp.int32)]
        result = straight_through_estimator(scaled, output)
        
        return result
