from typing import Literal, Union

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.boundaries.periodic import PeriodicBoundary
from fdtdx.objects.boundaries.utils import axis_direction_from_kind
from fdtdx.objects.object import PositionConstraint
from fdtdx.objects.static_material.static import SimulationVolume
from fdtdx.typing import PartialGridShape3D


@autoinit
class BoundaryConfig(TreeClass):
    """Configuration class for boundary conditions.

    This class stores parameters for boundary conditions in all six directions (min/max x/y/z).
    Supports both PML and periodic boundaries. For PML, the parameters control the absorption
    properties and physical size of the PML regions.

    Attributes:
        boundary_type_minx (str): Boundary type at minimum x ("pml" or "periodic"). Default "pml".
        boundary_type_maxx (str): Boundary type at maximum x ("pml" or "periodic"). Default "pml".
        boundary_type_miny (str): Boundary type at minimum y ("pml" or "periodic"). Default "pml".
        boundary_type_maxy (str): Boundary type at maximum y ("pml" or "periodic"). Default "pml".
        boundary_type_minz (str): Boundary type at minimum z ("pml" or "periodic"). Default "pml".
        boundary_type_maxz (str): Boundary type at maximum z ("pml" or "periodic"). Default "pml".
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

    boundary_type_minx: str = frozen_field(default="pml")
    boundary_type_maxx: str = frozen_field(default="pml")
    boundary_type_miny: str = frozen_field(default="pml")
    boundary_type_maxy: str = frozen_field(default="pml")
    boundary_type_minz: str = frozen_field(default="pml")
    boundary_type_maxz: str = frozen_field(default="pml")
    thickness_grid_minx: int = frozen_field(default=10)
    thickness_grid_maxx: int = frozen_field(default=10)
    thickness_grid_miny: int = frozen_field(default=10)
    thickness_grid_maxy: int = frozen_field(default=10)
    thickness_grid_minz: int = frozen_field(default=10)
    thickness_grid_maxz: int = frozen_field(default=10)
    kappa_start_minx: float = frozen_field(default=1.0)
    kappa_end_minx: float = frozen_field(default=1.5)
    kappa_start_maxx: float = frozen_field(default=1.0)
    kappa_end_maxx: float = frozen_field(default=1.5)
    kappa_start_miny: float = frozen_field(default=1.0)
    kappa_end_miny: float = frozen_field(default=1.5)
    kappa_start_maxy: float = frozen_field(default=1.0)
    kappa_end_maxy: float = frozen_field(default=1.5)
    kappa_start_minz: float = frozen_field(default=1.0)
    kappa_end_minz: float = frozen_field(default=1.5)
    kappa_start_maxz: float = frozen_field(default=1.0)
    kappa_end_maxz: float = frozen_field(default=1.5)

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

    def get_type_dict(self) -> dict[str, str]:
        """Gets a dictionary mapping boundary names to their boundary types.

        Returns:
            dict[str, str]: Dictionary with keys 'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z'
                mapping to their respective boundary types ("pml" or "periodic").
        """
        return {
            "min_x": self.boundary_type_minx,
            "max_x": self.boundary_type_maxx,
            "min_y": self.boundary_type_miny,
            "max_y": self.boundary_type_maxy,
            "min_z": self.boundary_type_minz,
            "max_z": self.boundary_type_maxz,
        }

    def get_kappa_dict(
        self,
        prop: Literal["kappa_start", "kappa_end"],
    ) -> dict[str, float]:
        """Gets a dictionary mapping boundary names to their kappa values.

        Args:
            prop (Literal["kappa_start", "kappa_end"]): Which kappa property to get,
                either "kappa_start" or "kappa_end".

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
                self.thickness_grid_minx + 1 if self.boundary_type_minx == "pml" else 0,
                -self.thickness_grid_maxx - 1 if self.boundary_type_maxx == "pml" else None,
            ),
            slice(
                self.thickness_grid_miny + 1 if self.boundary_type_miny == "pml" else 0,
                -self.thickness_grid_maxy - 1 if self.boundary_type_maxy == "pml" else None,
            ),
            slice(
                self.thickness_grid_minz + 1 if self.boundary_type_minz == "pml" else 0,
                -self.thickness_grid_maxz - 1 if self.boundary_type_maxz == "pml" else None,
            ),
        )

    @classmethod
    def from_uniform_bound(
        cls,
        thickness: int = 10,
        boundary_type: str = "pml",
        kappa_start: float = 1,
        kappa_end: float = 1.5,
    ) -> "BoundaryConfig":
        """Creates a BoundaryConfig with uniform parameters for all boundaries.

        Args:
            thickness (int, optional): Grid thickness to use for all PML boundaries. Defaults to 10.
            boundary_type (str, optional): Type of boundary to use ("pml" or "periodic"). Defaults to "pml".
            kappa_start (float, optional): Initial kappa value for all boundaries. Defaults to 1.0.
            kappa_end (float, optional): Final kappa value for all boundaries. Defaults to 1.5.

        Returns:
            BoundaryConfig: New config object with uniform parameters
        """
        return cls(
            boundary_type_minx=boundary_type,
            boundary_type_maxx=boundary_type,
            boundary_type_miny=boundary_type,
            boundary_type_maxy=boundary_type,
            boundary_type_minz=boundary_type,
            boundary_type_maxz=boundary_type,
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


def boundary_objects_from_config(
    config: BoundaryConfig,
    volume: SimulationVolume,
) -> tuple[dict[str, Union[PerfectlyMatchedLayer, PeriodicBoundary]], list[PositionConstraint]]:
    """Creates boundary objects from a boundary configuration.

    Creates PerfectlyMatchedLayer or PeriodicBoundary objects for all six boundaries
    (min/max x/y/z) based on the provided configuration. Also generates position
    constraints to properly place the boundary objects relative to the simulation volume.

    Args:
        config (BoundaryConfig): Configuration object containing boundary parameters
        volume (SimulationVolume): The main simulation volume object that the boundaries will surround

    Returns:
        tuple[dict[str, Union[PerfectlyMatchedLayer, PeriodicBoundary]], list[PositionConstraint]]: tuple containing:
            - dict mapping boundary names ('min_x', 'max_x', etc) to boundary objects
            - list of PositionConstraint objects for placing the boundaries
    """
    boundaries, constraints = {}, []
    thickness_dict = config.get_dict()
    type_dict = config.get_type_dict()
    kappa_start_dict = config.get_kappa_dict("kappa_start")
    kappa_end_dict = config.get_kappa_dict("kappa_end")

    for kind, thickness in thickness_dict.items():
        axis, direction = axis_direction_from_kind(kind)
        boundary_type = type_dict[kind]
        kappa_start, kappa_end = kappa_start_dict[kind], kappa_end_dict[kind]

        grid_shape_list: list[int | None] = [None, None, None]
        grid_shape_list[axis] = thickness if boundary_type == "pml" else 1
        grid_shape: PartialGridShape3D = tuple(grid_shape_list)  # type: ignore

        other_axes = [0, 1, 2]
        del other_axes[axis]

        if boundary_type == "pml":
            cur_boundary = PerfectlyMatchedLayer(
                axis=axis,
                partial_grid_shape=grid_shape,
                kappa_start=kappa_start,
                kappa_end=kappa_end,
                direction=direction,
            )
        else:  # periodic
            cur_boundary = PeriodicBoundary(
                axis=axis,
                partial_grid_shape=grid_shape,
                direction=direction,
            )

        direction_int = -1 if direction == "-" else 1
        pos_constraint = cur_boundary.place_relative_to(
            volume,
            axes=(axis, other_axes[0], other_axes[1]),
            own_positions=(direction_int, 0, 0),
            other_positions=(direction_int, 0, 0),
        )

        boundaries[kind] = cur_boundary
        constraints.append(pos_constraint)

    return boundaries, constraints
