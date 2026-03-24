from typing_extensions import override

from fdtdx.colors import XKCD_RED, Color
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.boundaries.boundary import BaseBoundary


@autoinit
class PerfectElectricConductor(BaseBoundary):
    """Implements perfect electric conductor (PEC) boundary conditions.

    PEC enforces E_tangential = 0 at the boundary wall. This is the default
    Yee grid termination behavior — zero-padding already provides PEC conditions.
    This class makes the boundary type explicit for configuration and user intent.
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
