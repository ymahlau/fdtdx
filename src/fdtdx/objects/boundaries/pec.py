import jax
from typing_extensions import override

from fdtdx.colors import XKCD_RED, Color
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.boundaries.boundary import BaseBoundary


@autoinit
class PerfectElectricConductor(BaseBoundary):
    """Implements perfect electric conductor (PEC) boundary conditions.

    PEC enforces E_tangential = 0 at the boundary wall. Zero-padding provides
    the correct ghost cell values for H (and for E_normal), but the curl_H
    computation at the boundary produces nonzero updates for tangential E
    components. This class explicitly zeros them after each E update.

    Component zeroing per axis:
    - PEC on x-face: zero Ey, Ez (tangential to x)
    - PEC on y-face: zero Ex, Ez (tangential to y)
    - PEC on z-face: zero Ex, Ey (tangential to z)
    """

    #: RGB color tuple for visualization. Defaults to red.
    color: Color | None = frozen_field(default=XKCD_RED)

    @property
    @override
    def descriptive_name(self) -> str:
        """Gets a human-readable name describing this PEC boundary's location.

        Returns:
            str: Description like "min_x" or "max_z" indicating position
        """
        axis_str = "x" if self.axis == 0 else "y" if self.axis == 1 else "z"
        direction_str = "min" if self.direction == "-" else "max"
        return f"{direction_str}_{axis_str}"

    @property
    @override
    def thickness(self) -> int:
        """Gets the thickness of the PEC boundary layer in grid points.

        Returns:
            int: Number of grid points in the boundary layer (always 1 for PEC)
        """
        return 1

    @property
    def tangential_components(self) -> tuple[int, int]:
        """Gets the indices of E field components tangential to this boundary.

        Returns:
            tuple[int, int]: Indices of the two tangential components (0=Ex, 1=Ey, 2=Ez)
        """
        if self.axis == 0:
            return (1, 2)  # Ey, Ez tangential to x-face
        elif self.axis == 1:
            return (0, 2)  # Ex, Ez tangential to y-face
        else:
            return (0, 1)  # Ex, Ey tangential to z-face

    @override
    def apply_post_E_update(self, E: jax.Array) -> jax.Array:
        """Zeros tangential E components at this PEC boundary face."""
        comp1, comp2 = self.tangential_components
        sx, sy, sz = self.grid_slice
        E = E.at[comp1, sx, sy, sz].set(0)
        E = E.at[comp2, sx, sy, sz].set(0)
        return E
