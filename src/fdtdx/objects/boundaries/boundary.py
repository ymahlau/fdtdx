from abc import ABC, abstractmethod

import jax

from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit
from fdtdx.objects.object import SimulationObject
from fdtdx.typing import GridShape3D


@extended_autoinit
class BaseBoundaryState(ExtendedTreeClass):
    pass


@extended_autoinit
class BaseBoundary(SimulationObject, ABC):
    @abstractmethod
    def init_state(
        self,
    ) -> BaseBoundaryState:
        raise NotImplementedError()

    @abstractmethod
    def reset_state(self, state: BaseBoundaryState) -> BaseBoundaryState:
        raise NotImplementedError()

    @abstractmethod
    def update_E_boundary_state(
        self,
        boundary_state: BaseBoundaryState,
        H: jax.Array,
    ) -> BaseBoundaryState:
        raise NotImplementedError()

    @abstractmethod
    def update_H_boundary_state(
        self,
        boundary_state: BaseBoundaryState,
        E: jax.Array,
    ) -> BaseBoundaryState:
        raise NotImplementedError()

    @abstractmethod
    def update_E(
        self,
        E: jax.Array,
        boundary_state: BaseBoundaryState,
        inverse_permittivity: jax.Array,
    ) -> jax.Array:
        raise NotImplementedError()

    @abstractmethod
    def update_H(
        self,
        H: jax.Array,
        boundary_state: BaseBoundaryState,
        inverse_permeability: jax.Array,
    ) -> jax.Array:
        raise NotImplementedError()

    @abstractmethod
    def boundary_interface_grid_shape(self) -> GridShape3D:
        raise NotImplementedError()
