from typing import Literal

import pytreeclass

from fdtdx.core.jax.pytrees import ExtendedTreeClass
from fdtdx.core.jax.typing import PartialGridShape3D
from fdtdx.objects.boundaries.boundary_utils import axis_direction_from_kind
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.object import PositionConstraint, SimulationObject


@pytreeclass.autoinit
class BoundaryConfig(ExtendedTreeClass):
    """Configuration class for PML boundary conditions.

    This class stores thickness and kappa parameters for Perfectly Matched Layer (PML)
    boundaries in all six directions (min/max x/y/z). The parameters control the absorption
    properties and physical size of the PML regions.

    Attributes:
        thickness_grid_minx (int): Number of grid cells for PML at minimum x boundary. Default 10.
        thickness_grid_maxx (int): Number of grid cells for PML at maximum x boundary. Default 10.
        thickness_grid_miny (int): Number of grid cells for PML at minimum y boundary. Default 10.
        thickness_grid_maxy (int): Number of grid cells for PML at maximum y boundary. Default 10.
        thickness_grid_minz (int): Number of grid cells for PML at minimum z boundary. Default 10.
        thickness_grid_maxz (int): Number of grid cells for PML at maximum z boundary. Default 10.
        kappa_start_minx (float): Initial kappa value at min x boundary. Default 1.0.
        kappa_end_minx (float): Final kappa value at min x boundary. Default 1.5.
        kappa_start_maxx (float): Initial kappa value at max x boundary. Default 1.0.
        kappa_end_maxx (float): Final kappa value at max x boundary. Default 1.5.
        kappa_start_miny (float): Initial kappa value at min y boundary. Default 1.0.
        kappa_end_miny (float): Final kappa value at min y boundary. Default 1.5.
        kappa_start_maxy (float): Initial kappa value at max y boundary. Default 1.0.
        kappa_end_maxy (float): Final kappa value at max y boundary. Default 1.5.
        kappa_start_minz (float): Initial kappa value at min z boundary. Default 1.0.
        kappa_end_minz (float): Final kappa value at min z boundary. Default 1.5.
        kappa_start_maxz (float): Initial kappa value at max z boundary. Default 1.0.
        kappa_end_maxz (float): Final kappa value at max z boundary. Default 1.5.
    """

    thickness_grid_minx: int = 10
    thickness_grid_maxx: int = 10
    thickness_grid_miny: int = 10
    thickness_grid_maxy: int = 10
    thickness_grid_minz: int = 10
    thickness_grid_maxz: int = 10
    kappa_start_minx: float = 1.0
    kappa_end_minx: float = 1.5
    kappa_start_maxx: float = 1.0
    kappa_end_maxx: float = 1.5
    kappa_start_miny: float = 1.0
    kappa_end_miny: float = 1.5
    kappa_start_maxy: float = 1.0
    kappa_end_maxy: float = 1.5
    kappa_start_minz: float = 1.0
    kappa_end_minz: float = 1.5
    kappa_start_maxz: float = 1.0
    kappa_end_maxz: float = 1.5

    def get_dict(self) -> dict[str, int]:
        """Gets a dictionary mapping boundary names to their grid thicknesses.

        Returns:
            dict[str, int]: Dictionary with keys 'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z'
                mapping to their respective grid thickness values.
        """
        return {
            "min_x": self.thickness_grid_minx,
            "max_x": self.thickness_grid_maxx,
            "min_y": self.thickness_grid_miny,
            "max_y": self.thickness_grid_maxy,
            "min_z": self.thickness_grid_minz,
            "max_z": self.thickness_grid_maxz,
        }

    def get_kappa_dict(
        self,
        prop: Literal["kappa_start", "kappa_end"],
    ) -> dict[str, float]:
        """Gets a dictionary mapping boundary names to their kappa values.

        Args:
            prop: Which kappa property to get, either "kappa_start" or "kappa_end"

        Returns:
            dict[str, float]: Dictionary with keys 'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z'
                mapping to their respective kappa values.

        Raises:
            Exception: If prop is not "kappa_start" or "kappa_end"
        """
        if prop == "kappa_start":
            return {
                "min_x": self.kappa_start_minx,
                "max_x": self.kappa_start_maxx,
                "min_y": self.kappa_start_miny,
                "max_y": self.kappa_start_maxy,
                "min_z": self.kappa_start_minz,
                "max_z": self.kappa_start_maxz,
            }
        elif prop == "kappa_end":
            return {
                "min_x": self.kappa_end_minx,
                "max_x": self.kappa_end_maxx,
                "min_y": self.kappa_end_miny,
                "max_y": self.kappa_end_maxy,
                "min_z": self.kappa_end_minz,
                "max_z": self.kappa_end_maxz,
            }
        else:
            raise Exception(f"Unknown: {prop=}")

    def get_inside_boundary_slice(self) -> tuple[slice, slice, slice]:
        """Gets slice objects for the non-PML interior region of the simulation volume.

        Returns:
            tuple[slice, slice, slice]: Three slice objects for indexing the x, y, z dimensions
                respectively, excluding the PML boundary regions.
        """
        return (
            slice(
                self.thickness_grid_minx + 1,
                -self.thickness_grid_maxx - 1,
            ),
            slice(
                self.thickness_grid_miny + 1,
                -self.thickness_grid_maxy - 1,
            ),
            slice(
                self.thickness_grid_minz + 1,
                -self.thickness_grid_maxz - 1,
            ),
        )

    @classmethod
    def from_uniform_bound(cls, thickness: int, kappa_start: float = 1, kappa_end: float = 1.5):
        """Creates a BoundaryConfig with uniform parameters for all boundaries.

        Args:
            thickness: Grid thickness to use for all PML boundaries
            kappa_start: Initial kappa value for all boundaries. Defaults to 1.0.
            kappa_end: Final kappa value for all boundaries. Defaults to 1.5.

        Returns:
            BoundaryConfig: New config object with uniform parameters
        """
        return cls(
            thickness_grid_minx=thickness,
            thickness_grid_maxx=thickness,
            thickness_grid_miny=thickness,
            thickness_grid_maxy=thickness,
            thickness_grid_minz=thickness,
            thickness_grid_maxz=thickness,
            kappa_start_minx=kappa_start,
            kappa_end_minx=kappa_end,
            kappa_start_maxx=kappa_start,
            kappa_end_maxx=kappa_end,
            kappa_start_miny=kappa_start,
            kappa_end_miny=kappa_end,
            kappa_start_maxy=kappa_start,
            kappa_end_maxy=kappa_end,
            kappa_start_minz=kappa_start,
            kappa_end_minz=kappa_end,
            kappa_start_maxz=kappa_start,
            kappa_end_maxz=kappa_end,
        )


def pml_objects_from_config(
    config: BoundaryConfig,
    volume: SimulationObject,
) -> tuple[dict[str, PerfectlyMatchedLayer], list[PositionConstraint]]:
    """Creates PML boundary objects from a boundary configuration.

    Creates PerfectlyMatchedLayer objects for all six boundaries (min/max x/y/z)
    based on the provided configuration. Also generates position constraints to
    properly place the PML objects relative to the simulation volume.

    Args:
        config: Configuration object containing PML parameters
        volume: The main simulation volume object that the PMLs will surround

    Returns:
        tuple containing:
            - dict mapping boundary names ('min_x', 'max_x', etc) to PML objects
            - list of PositionConstraint objects for placing the PMLs
    """
    boundaries, constraints = {}, []
    thickness_dict = config.get_dict()
    kappa_start_dict = config.get_kappa_dict("kappa_start")
    kappa_end_dict = config.get_kappa_dict("kappa_start")

    for kind, thickness in thickness_dict.items():
        axis, direction = axis_direction_from_kind(kind)
        kappa_start, kappa_end = kappa_start_dict[kind], kappa_end_dict[kind]

        grid_shape_list: list[int | None] = [None, None, None]
        grid_shape_list[axis] = thickness
        grid_shape: PartialGridShape3D = tuple(grid_shape_list)  # type: ignore

        other_axes = [0, 1, 2]
        del other_axes[axis]

        cur_pml = PerfectlyMatchedLayer(
            axis=axis, partial_grid_shape=grid_shape, kappa_start=kappa_start, kappa_end=kappa_end, direction=direction
        )

        direction_int = -1 if direction == "-" else 1
        pos_constraint = cur_pml.place_relative_to(
            volume,
            axes=(axis, other_axes[0], other_axes[1]),
            own_positions=(direction_int, 0, 0),
            other_positions=(direction_int, 0, 0),
        )

        boundaries[kind] = cur_pml
        constraints.append(pos_constraint)

    return boundaries, constraints
