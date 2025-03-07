from typing import Literal

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import extended_autoinit, field, frozen_field
from fdtdx.core.jax.typing import GridShape3D, Slice3D, SliceTuple3D
from fdtdx.core.plotting.colors import LIGHT_BLUE
from fdtdx.objects.boundaries.boundary import BaseBoundary, BaseBoundaryState


@extended_autoinit
class PeriodicBoundaryState(BaseBoundaryState):
    """State container for periodic boundary conditions.

    Stores the field values at the opposite boundary for periodic wrapping.

    Attributes:
        E_opposite: Electric field values from opposite boundary
        H_opposite: Magnetic field values from opposite boundary
    """

    E_opposite: jax.Array
    H_opposite: jax.Array


@extended_autoinit
class PeriodicBoundary(BaseBoundary):
    """Implements periodic boundary conditions.

    The periodic boundary connects opposite sides of the simulation domain,
    making waves that exit one side reenter from the opposite side.

    Attributes:
        axis: Principal axis for periodicity (0=x, 1=y, 2=z)
        direction: Direction along axis ("+" or "-")
        color: RGB color tuple for visualization
    """

    axis: int = field(kind="KW_ONLY")
    direction: Literal["+", "-"] = frozen_field(kind="KW_ONLY")  # type: ignore
    color: tuple[float, float, float] = LIGHT_BLUE

    @property
    def descriptive_name(self) -> str:
        """Gets a human-readable name describing this periodic boundary's location.

        Returns:
            str: Description like "min_x" or "max_z" indicating position
        """
        axis_str = "x" if self.axis == 0 else "y" if self.axis == 1 else "z"
        direction_str = "min" if self.direction == "-" else "max"
        return f"{direction_str}_{axis_str}"

    @property
    def thickness(self) -> int:
        """Gets the thickness of the periodic boundary layer in grid points.

        Returns:
            int: Number of grid points in the boundary layer (always 1 for periodic)
        """
        return 1

    def init_state(
        self,
    ) -> PeriodicBoundaryState:
        """Initializes the periodic boundary state.

        Creates storage for field values from opposite boundary needed for
        periodic wrapping.

        Returns:
            PeriodicBoundaryState: Initialized state container
        """
        dtype = self._config.dtype
        ext_shape = (3,) + self.grid_shape

        boundary_state = PeriodicBoundaryState(
            E_opposite=jnp.zeros(shape=ext_shape, dtype=dtype),
            H_opposite=jnp.zeros(shape=ext_shape, dtype=dtype),
        )
        return boundary_state

    def reset_state(self, state: PeriodicBoundaryState) -> PeriodicBoundaryState:
        """Resets the periodic boundary state.

        Args:
            state: Current boundary state to reset

        Returns:
            PeriodicBoundaryState: New state with zeroed fields
        """
        new_state = PeriodicBoundaryState(
            E_opposite=state.E_opposite * 0,
            H_opposite=state.H_opposite * 0,
        )
        return new_state

    def boundary_interface_grid_shape(self) -> GridShape3D:
        """Gets the shape of the periodic boundary interface.

        Returns:
            GridShape3D: 3D shape tuple with 1 in the periodic axis dimension
        """
        if self.axis == 0:
            return 1, self.grid_shape[1], self.grid_shape[2]
        elif self.axis == 1:
            return self.grid_shape[0], 1, self.grid_shape[2]
        elif self.axis == 2:
            return self.grid_shape[0], self.grid_shape[1], 1
        raise Exception(f"Invalid axis: {self.axis=}")

    def boundary_interface_slice_tuple(self) -> SliceTuple3D:
        """Gets the slice tuple for accessing the periodic boundary interface.

        Returns:
            SliceTuple3D: Tuple of slices defining the interface boundary
        """
        slice_list = [*self._grid_slice_tuple]
        if self.direction == "+":
            slice_list[self.axis] = (self._grid_slice_tuple[self.axis][0], self._grid_slice_tuple[self.axis][0] + 1)
        elif self.direction == "-":
            slice_list[self.axis] = (self._grid_slice_tuple[self.axis][1] - 1, self._grid_slice_tuple[self.axis][1])
        return slice_list[0], slice_list[1], slice_list[2]

    def boundary_interface_slice(self) -> Slice3D:
        """Gets the slice object for accessing the periodic boundary interface.

        Returns:
            Slice3D: Slice object defining the interface boundary
        """
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

    def update_E_boundary_state(
        self,
        boundary_state: PeriodicBoundaryState,
        H: jax.Array,
    ) -> PeriodicBoundaryState:
        """Updates the periodic boundary state for electric field.

        Stores both E and H field values from the opposite boundary for periodic wrapping.

        Args:
            boundary_state: Current boundary state
            H: Magnetic field array

        Returns:
            PeriodicBoundaryState: Updated boundary state
        """
        # Get field values from opposite boundary
        opposite_slice = list(self.grid_slice)
        if self.direction == "+":
            opposite_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][1] - 1, self._grid_slice_tuple[self.axis][1]
            )
        else:
            opposite_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][0], self._grid_slice_tuple[self.axis][0] + 1
            )

        # Store H field values from opposite boundary
        H_opposite = jnp.array(H[..., opposite_slice[0], opposite_slice[1], opposite_slice[2]])

        return PeriodicBoundaryState(
            E_opposite=boundary_state.E_opposite,  # Keep existing E values
            H_opposite=H_opposite,  # Update H values
        )

    def update_H_boundary_state(
        self,
        boundary_state: PeriodicBoundaryState,
        E: jax.Array,
    ) -> PeriodicBoundaryState:
        """Updates the periodic boundary state for magnetic field.

        Stores both E and H field values from the opposite boundary for periodic wrapping.

        Args:
            boundary_state: Current boundary state
            E: Electric field array

        Returns:
            PeriodicBoundaryState: Updated boundary state
        """
        # Get field values from opposite boundary
        opposite_slice = list(self.grid_slice)
        if self.direction == "+":
            opposite_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][1] - 1, self._grid_slice_tuple[self.axis][1]
            )
        else:
            opposite_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][0], self._grid_slice_tuple[self.axis][0] + 1
            )

        # Store E field values from opposite boundary
        E_opposite = jnp.array(E[..., opposite_slice[0], opposite_slice[1], opposite_slice[2]])

        return PeriodicBoundaryState(
            E_opposite=E_opposite,  # Update E values
            H_opposite=boundary_state.H_opposite,  # Keep existing H values
        )

    def update_E(
        self,
        E: jax.Array,
        boundary_state: PeriodicBoundaryState,
        inverse_permittivity: jax.Array,
    ) -> jax.Array:
        """Updates the electric field at the periodic boundary.

        Applies periodic boundary conditions by copying field values from the
        opposite boundary, maintaining field continuity.

        Args:
            E: Electric field array to update
            boundary_state: Current boundary state
            inverse_permittivity: Inverse permittivity array

        Returns:
            jax.Array: Updated electric field array
        """
        # Get the boundary slice
        boundary_slice = list(self.grid_slice)
        if self.direction == "+":
            boundary_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][0], self._grid_slice_tuple[self.axis][0] + 1
            )
            # Copy from opposite boundary (last slice)
            opposite_slice = list(self.grid_slice)
            opposite_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][1] - 1, self._grid_slice_tuple[self.axis][1]
            )
        else:
            boundary_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][1] - 1, self._grid_slice_tuple[self.axis][1]
            )
            # Copy from opposite boundary (first slice)
            opposite_slice = list(self.grid_slice)
            opposite_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][0], self._grid_slice_tuple[self.axis][0] + 1
            )

        # Copy field values from opposite boundary
        E = E.at[..., boundary_slice[0], boundary_slice[1], boundary_slice[2]].set(
            E[..., opposite_slice[0], opposite_slice[1], opposite_slice[2]]
        )

        return E

    def update_H(
        self,
        H: jax.Array,
        boundary_state: PeriodicBoundaryState,
        inverse_permeability: jax.Array,
    ) -> jax.Array:
        """Updates the magnetic field at the periodic boundary.

        Applies periodic boundary conditions by copying field values from the
        opposite boundary, maintaining field continuity.

        Args:
            H: Magnetic field array to update
            boundary_state: Current boundary state
            inverse_permeability: Inverse permeability array

        Returns:
            jax.Array: Updated magnetic field array
        """
        # Get the boundary slice
        boundary_slice = list(self.grid_slice)
        if self.direction == "+":
            boundary_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][0], self._grid_slice_tuple[self.axis][0] + 1
            )
            # Copy from opposite boundary (last slice)
            opposite_slice = list(self.grid_slice)
            opposite_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][1] - 1, self._grid_slice_tuple[self.axis][1]
            )
        else:
            boundary_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][1] - 1, self._grid_slice_tuple[self.axis][1]
            )
            # Copy from opposite boundary (first slice)
            opposite_slice = list(self.grid_slice)
            opposite_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][0], self._grid_slice_tuple[self.axis][0] + 1
            )

        # Copy field values from opposite boundary
        H = H.at[..., boundary_slice[0], boundary_slice[1], boundary_slice[2]].set(
            H[..., opposite_slice[0], opposite_slice[1], opposite_slice[2]]
        )

        return H
