import functools

import jax
import jax.numpy as jnp
from typing_extensions import override

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.plotting.colors import LIGHT_BLUE
from fdtdx.objects.boundaries.boundary import BaseBoundary, BaseBoundaryState
from fdtdx.units.unitful import Unitful
from fdtdx.units import m, A, V


@autoinit
class PeriodicBoundaryState(BaseBoundaryState):
    """State container for periodic boundary conditions.

    Stores the field values at the opposite boundary for periodic wrapping.

    Attributes:
        E_opposite (jax.Array): Electric field values from opposite boundary
        H_opposite (jax.Array): Magnetic field values from opposite boundary
    """

    E_opposite: Unitful
    H_opposite: Unitful


@autoinit
class PeriodicBoundary(BaseBoundary[PeriodicBoundaryState]):
    """Implements periodic boundary conditions.

    The periodic boundary connects opposite sides of the simulation domain,
    making waves that exit one side reenter from the opposite side.

    Attributes:
        color (tuple[float, float, float] | None): RGB color tuple for visualization. Defaults to light blue.
    """

    color: tuple[float, float, float] | None = frozen_field(default=LIGHT_BLUE)

    @property
    @override
    def descriptive_name(self) -> str:
        """Gets a human-readable name describing this periodic boundary's location.

        Returns:
            str: Description like "min_x" or "max_z" indicating position
        """
        axis_str = "x" if self.axis == 0 else "y" if self.axis == 1 else "z"
        direction_str = "min" if self.direction == "-" else "max"
        return f"{direction_str}_{axis_str}"

    @property
    @override
    def thickness(self) -> int:
        """Gets the thickness of the periodic boundary layer in grid points.

        Returns:
            int: Number of grid points in the boundary layer (always 1 for periodic)
        """
        return 1

    @override
    def init_state(
        self,
    ) -> PeriodicBoundaryState:
        dtype = self._config.dtype
        ext_shape = (3,) + self.grid_shape

        boundary_state = PeriodicBoundaryState(
            E_opposite=(V / m) * jnp.zeros(shape=ext_shape, dtype=dtype),
            H_opposite=(A / m) * jnp.zeros(shape=ext_shape, dtype=dtype),
        )
        return boundary_state

    @override
    def reset_state(self, state: PeriodicBoundaryState) -> PeriodicBoundaryState:
        # Reset the state by zeroing out the fields
        return PeriodicBoundaryState(
            E_opposite=state.E_opposite * 0,
            H_opposite=state.H_opposite * 0,
        )

    @override
    def update_E_boundary_state(
        self,
        boundary_state: PeriodicBoundaryState,
        H: Unitful,
    ) -> PeriodicBoundaryState:
        # Store H field values from opposite boundary
        H_opposite = H[..., *self.opposite_slice]

        return PeriodicBoundaryState(
            E_opposite=boundary_state.E_opposite,  # Keep existing E values
            H_opposite=H_opposite,  # Update H values
        )

    @override
    def update_H_boundary_state(
        self,
        boundary_state: PeriodicBoundaryState,
        E: Unitful,
    ) -> PeriodicBoundaryState:
        # Store E field values from opposite boundary
        E_opposite = E[..., *self.opposite_slice]

        return PeriodicBoundaryState(
            E_opposite=E_opposite,  # Update E values
            H_opposite=boundary_state.H_opposite,  # Keep existing H values
        )

    @functools.cached_property
    def boundary_slice(self) -> tuple[slice, ...]:
        """Get the slice for the current boundary.

        Returns:
            tuple[slice, ...]: Slice for the boundary
        """
        boundary_slice = list(self.grid_slice)
        if self.direction == "+":
            boundary_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][0], self._grid_slice_tuple[self.axis][0] + 1
            )
        else:
            boundary_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][1] - 1, self._grid_slice_tuple[self.axis][1]
            )
        return tuple(boundary_slice)

    @functools.cached_property
    def opposite_slice(self) -> tuple[slice, ...]:
        """Get the slice for the opposite boundary.

        Returns:
            tuple[slice, ...]: Slice for the opposite boundary
        """
        opposite_slice = list(self.grid_slice)
        if self.direction == "+":
            opposite_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][1] - 1, self._grid_slice_tuple[self.axis][1]
            )
        else:
            opposite_slice[self.axis] = slice(
                self._grid_slice_tuple[self.axis][0], self._grid_slice_tuple[self.axis][0] + 1
            )
        return tuple(opposite_slice)

    @override
    def update_E(
        self,
        E: Unitful,
        boundary_state: PeriodicBoundaryState,
        inverse_permittivity: jax.Array,
    ) -> Unitful:
        del boundary_state, inverse_permittivity

        # Copy field values from opposite boundary
        E = E.at[..., *self.boundary_slice].set(E[..., *self.opposite_slice])

        return E

    @override
    def update_H(
        self,
        H: Unitful,
        boundary_state: PeriodicBoundaryState,
        inverse_permeability: jax.Array | float,
    ) -> Unitful:
        del boundary_state, inverse_permeability

        # Copy field values from opposite boundary
        H = H.at[..., *self.boundary_slice].set(H[..., *self.opposite_slice])

        return H
