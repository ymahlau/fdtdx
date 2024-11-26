from typing import Literal
import pytreeclass

from fdtdx.core.jax.pytrees import ExtendedTreeClass
from fdtdx.objects.boundaries.boundary_utils import axis_direction_from_kind
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.core.jax.typing import PartialGridShape3D
from fdtdx.objects.object import PositionConstraint, SimulationObject


@pytreeclass.autoinit
class BoundaryConfig(ExtendedTreeClass):
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

    def get_dict(
        self
    ) -> dict[str, int]:
        return {
            "min_x": self.thickness_grid_minx,
            "max_x": self.thickness_grid_maxx,
            "min_y": self.thickness_grid_miny,
            "max_y": self.thickness_grid_maxy,
            "min_z": self.thickness_grid_minz,
            "max_z": self.thickness_grid_maxz,
        }
        
    
    def get_kappa_dict(
        self, prop: Literal["kappa_start", "kappa_end"],
    ) -> dict[str, float]:
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
    
    def get_inside_boundary_slice(self):
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
    def from_uniform_bound(
        cls, thickness: int, kappa_start: float = 1, kappa_end: float = 1.5
    ):
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
            axis=axis,
            partial_grid_shape=grid_shape,
            kappa_start=kappa_start,
            kappa_end=kappa_end,
            direction=direction
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

