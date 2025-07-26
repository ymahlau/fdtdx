from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import jax

from fdtdx.core.jax.pytrees import TreeClass, autoinit
from fdtdx.objects.object import SimulationObject
from fdtdx.typing import GridShape3D


@autoinit
class BaseBoundaryState(TreeClass):
    pass

T = TypeVar("T", bound=BaseBoundaryState)

@autoinit
class BaseBoundary(SimulationObject, ABC, Generic[T]):
    @property
    @abstractmethod
    def descriptive_name(self) -> str:
        """Gets a human-readable name describing this boundary's location."""
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def thickness(self) -> int:
        """Gets the thickness of the boundary in grid points."""
        raise NotImplementedError()

    @abstractmethod
    def init_state(
        self,
    ) -> T:
        raise NotImplementedError()

    @abstractmethod
    def reset_state(self, state: T) -> T:
        raise NotImplementedError()

    @abstractmethod
    def update_E_boundary_state(
        self,
        boundary_state: T,
        H: jax.Array,
    ) -> T:
        raise NotImplementedError()

    @abstractmethod
    def update_H_boundary_state(
        self,
        boundary_state: T,
        E: jax.Array,
    ) -> T:
        raise NotImplementedError()

    @abstractmethod
    def update_E(
        self,
        E: jax.Array,
        boundary_state: T,
        inverse_permittivity: jax.Array,
    ) -> jax.Array:
        raise NotImplementedError()

    @abstractmethod
    def update_H(
        self,
        H: jax.Array,
        boundary_state: T,
        inverse_permeability: jax.Array | float,
    ) -> jax.Array:
        raise NotImplementedError()

    @abstractmethod
    def boundary_interface_grid_shape(self) -> GridShape3D:
        raise NotImplementedError()
