from abc import ABC, abstractmethod
import jax

from fdtdx.core.jax.pytrees import ExtendedTreeClass
from fdtdx.core.jax.typing import INVALID_SHAPE_3D, GridShape3D


class ParameterProjection(ExtendedTreeClass, ABC):
    
    def __init__(
        self,
    ):
        self.voxel_grid_shape: GridShape3D = INVALID_SHAPE_3D
    
    @abstractmethod
    def params_shape(self) -> tuple[int, ...]:
        raise NotImplementedError()
    
    def __call__(
        self,
        params: jax.Array,
    ) -> jax.Array:
        del params
        raise NotImplementedError()
