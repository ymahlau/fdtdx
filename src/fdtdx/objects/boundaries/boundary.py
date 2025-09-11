from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

import jax

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field
from fdtdx.objects.object import SimulationObject
from fdtdx.typing import GridShape3D, Slice3D, SliceTuple3D


@autoinit
class BaseBoundaryState(TreeClass):
    pass


T = TypeVar("T", bound=BaseBoundaryState)


@autoinit
class BaseBoundary(SimulationObject, ABC, Generic[T]):
    """Base class for all boundary conditions in FDTD simulations.

    This class defines the interface for boundary conditions, including methods
    for initializing, resetting, and updating boundary states, as well as updating
    the electric and magnetic fields at the boundaries.
    """

    #: Principal axis for boundary (0=x, 1=y, 2=z)
    axis: int = frozen_field()

    #: Direction along axis ("+" or "-")
    direction: Literal["+", "-"] = frozen_field()

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

    def interface_grid_shape(self) -> GridShape3D:
        if self.axis == 0:
            return 1, self.grid_shape[1], self.grid_shape[2]
        elif self.axis == 1:
            return self.grid_shape[0], 1, self.grid_shape[2]
        elif self.axis == 2:
            return self.grid_shape[0], self.grid_shape[1], 1
        raise Exception(f"Invalid axis: {self.axis=}")

    def interface_slice_tuple(self) -> SliceTuple3D:
        slice_list = [*self._grid_slice_tuple]
        if self.direction == "+":
            slice_list[self.axis] = (self._grid_slice_tuple[self.axis][0], self._grid_slice_tuple[self.axis][0] + 1)
        elif self.direction == "-":
            slice_list[self.axis] = (self._grid_slice_tuple[self.axis][1] - 1, self._grid_slice_tuple[self.axis][1])
        return slice_list[0], slice_list[1], slice_list[2]

    def interface_slice(self) -> Slice3D:
        slice_list = [*self.grid_slice]
        if self.direction == "+":
            slice_list[self.axis] = slice(
                self._grid_slice_tuple[self.axis][0], self._grid_slice_tuple[self.axis][0] + 1
            )
        elif self.direction == "-":
            slice_list[self.axis] = slice(
                self._grid_slice_tuple[self.axis][1] - 1, self._grid_slice_tuple[self.axis][1]
            )
        return slice_list[0], slice_list[1], slice_list[2]
