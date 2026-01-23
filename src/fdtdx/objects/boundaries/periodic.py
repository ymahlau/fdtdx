import functools

from typing_extensions import override

from fdtdx.colors import XKCD_LIGHT_BLUE
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.boundaries.boundary import BaseBoundary


@autoinit
class PeriodicBoundary(BaseBoundary):
    """Implements periodic boundary conditions.

    The periodic boundary connects opposite sides of the simulation domain,
    making waves that exit one side reenter from the opposite side.
    """

    #: RGB color tuple for visualization. Defaults to light blue.
    color: tuple[float, float, float] | None = frozen_field(default=XKCD_LIGHT_BLUE)

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
