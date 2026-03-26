from abc import ABC, abstractmethod
from typing import Literal

import jax

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.object import SimulationObject
from fdtdx.typing import GridShape3D, Slice3D, SliceTuple3D


@autoinit
class BaseBoundary(SimulationObject, ABC):
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

    @property
    def uses_wrap_padding(self) -> bool:
        """Whether this boundary's axis should use wrap (periodic) padding.

        Returns True for boundaries that connect opposite sides of the domain
        (periodic, Bloch). Returns False for terminating boundaries (PEC, PMC, PML).
        """
        return False

    def apply_pad_correction(
        self, padded_fields: jax.Array, volume_shape: tuple[int, int, int], resolution: float
    ) -> jax.Array:
        """Apply boundary-specific correction to padded fields.

        Called after basic wrap/constant padding. Default is a no-op.
        Subclasses like BlochBoundary override this to apply phase shifts
        to ghost cells.

        Args:
            padded_fields: Padded field array of shape (3, Nx+2, Ny+2, Nz+2)
            volume_shape: Full simulation volume shape (Nx, Ny, Nz)
            resolution: Grid resolution in meters

        Returns:
            Padded fields with boundary-specific corrections applied
        """
        return padded_fields

    def apply_post_E_update(self, E: jax.Array) -> jax.Array:
        """Apply boundary-specific enforcement after E field update.

        Called after each E field update (forward and reverse). Default is a no-op.
        Subclasses like PEC override this to zero tangential E components.

        Args:
            E: Electric field array of shape (3, Nx, Ny, Nz)

        Returns:
            E field with boundary conditions enforced
        """
        return E

    def apply_post_H_update(self, H: jax.Array) -> jax.Array:
        """Apply boundary-specific enforcement after H field update.

        Called after each H field update (forward and reverse). Default is a no-op.
        Subclasses like PMC override this to zero tangential H components.

        Args:
            H: Magnetic field array of shape (3, Nx, Ny, Nz)

        Returns:
            H field with boundary conditions enforced
        """
        return H

    def apply_field_reset(self, fields: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Apply boundary-specific field reset during backward propagation.

        Called during the backward pass to restore each boundary region to its
        correct state. Default is a no-op. Subclasses like PML override this to
        zero their region; BlochBoundary overrides to copy from the opposite face.

        Args:
            fields: Dict mapping field names (e.g. 'E', 'H') to their arrays

        Returns:
            Updated fields dict with this boundary's reset applied
        """
        return fields

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
