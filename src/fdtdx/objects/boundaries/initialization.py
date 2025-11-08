from typing import Literal, Union

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field
from fdtdx.config import SimulationConfig
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
    """

    #: Boundary type at minimum x ("pml" or "periodic"). Default "pml".
    boundary_type_minx: str = frozen_field(default="pml")

    #: Boundary type at maximum x ("pml" or "periodic"). Default "pml".
    boundary_type_maxx: str = frozen_field(default="pml")

    #: Boundary type at minimum y ("pml" or "periodic"). Default "pml".
    boundary_type_miny: str = frozen_field(default="pml")

    #: Boundary type at maximum y ("pml" or "periodic"). Default "pml".
    boundary_type_maxy: str = frozen_field(default="pml")

    #: Boundary type at minimum z ("pml" or "periodic"). Default "pml".
    boundary_type_minz: str = frozen_field(default="pml")

    #: Number of grid cells for PML at maximum z boundary. Default 10.
    boundary_type_maxz: str = frozen_field(default="pml")

    #: Number of grid cells for PML at minimum x boundary. Default 10.
    thickness_grid_minx: int = frozen_field(default=10)

    #: Number of grid cells for PML at maximum x boundary. Default 10.
    thickness_grid_maxx: int = frozen_field(default=10)

    #: Boundary type at minimum y ("pml" or "periodic"). Default "pml".
    thickness_grid_miny: int = frozen_field(default=10)

    #: Number of grid cells for PML at maximum y boundary. Default 10.
    thickness_grid_maxy: int = frozen_field(default=10)

    #: Number of grid cells for PML at minimum z boundary. Default 10.
    thickness_grid_minz: int = frozen_field(default=10)

    #: Number of grid cells for PML at maximum z boundary. Default 10.
    thickness_grid_maxz: int = frozen_field(default=10)

    #: Initial kappa value at min x boundary. Default 1.0.
    kappa_start_minx: float = frozen_field(default=1.0)

    #: Final kappa value at min x boundary. Default 1.5.
    kappa_end_minx: float = frozen_field(default=1.5)

    #: Initial kappa value at max x boundary. Default 1.0.
    kappa_start_maxx: float = frozen_field(default=1.0)

    #: Final kappa value at max x boundary. Default 1.5.
    kappa_end_maxx: float = frozen_field(default=1.5)

    #: Initial kappa value at min y boundary. Default 1.0.
    kappa_start_miny: float = frozen_field(default=1.0)

    #: Final kappa value at min y boundary. Default 1.5.
    kappa_end_miny: float = frozen_field(default=1.5)

    #: Initial kappa value at max y boundary. Default 1.0.
    kappa_start_maxy: float = frozen_field(default=1.0)

    #: Final kappa value at max y boundary. Default 1.5.
    kappa_end_maxy: float = frozen_field(default=1.5)

    #: Initial kappa value at min z boundary. Default 1.0.
    kappa_start_minz: float = frozen_field(default=1.0)

    #: Final kappa value at min z boundary. Default 1.5.
    kappa_end_minz: float = frozen_field(default=1.5)

    #: Initial kappa value at max z boundary. Default 1.0.
    kappa_start_maxz: float = frozen_field(default=1.0)

    #: Final kappa value at max z boundary. Default 1.5.
    kappa_end_maxz: float = frozen_field(default=1.5)

    #: Initial alpha value at min x boundary. Default 1e-8.
    alpha_start_minx: float = frozen_field(default=1e-8)

    #: Final alpha value at min x boundary. Default 1e-8.
    alpha_end_minx: float = frozen_field(default=1e-8)

    #: Initial alpha value at max x boundary. Default 1e-8.
    alpha_start_maxx: float = frozen_field(default=1e-8)

    #: Final alpha value at max x boundary. Default 1e-8.
    alpha_end_maxx: float = frozen_field(default=1e-8)

    #: Initial alpha value at min y boundary. Default 1e-8.
    alpha_start_miny: float = frozen_field(default=1e-8)

    #: Final alpha value at min y boundary. Default 1e-8.
    alpha_end_miny: float = frozen_field(default=1e-8)

    #: Initial alpha value at max y boundary. Default 1e-8.
    alpha_start_maxy: float = frozen_field(default=1e-8)

    #: Final alpha value at max y boundary. Default 1e-8.
    alpha_end_maxy: float = frozen_field(default=1e-8)

    #: Initial alpha value at min z boundary. Default 1e-8.
    alpha_start_minz: float = frozen_field(default=1e-8)

    #: Final alpha value at min z boundary. Default 1e-8.
    alpha_end_minz: float = frozen_field(default=1e-8)

    #: Initial alpha value at max z boundary. Default 1e-8.
    alpha_start_maxz: float = frozen_field(default=1e-8)

    #: Final alpha value at max z boundary. Default 1e-8.
    alpha_end_maxz: float = frozen_field(default=1e-8)

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

    def get_alpha_dict(
        self,
        prop: Literal["alpha_start", "alpha_end"],
    ) -> dict[str, float]:
        """Gets a dictionary mapping boundary names to their alpha values.

        Args:
            prop (Literal["alpha_start", "alpha_end"]): Which alpha property to get,
                either "alpha_start" or "alpha_end".

        Returns:
            dict[str, float]: Dictionary with keys 'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z'
                mapping to their respective alpha values.

        Raises:
            Exception: If prop is not "alpha_start" or "alpha_end"
        """
        if prop == "alpha_start":
            return {
                "min_x": self.alpha_start_minx,
                "max_x": self.alpha_start_maxx,
                "min_y": self.alpha_start_miny,
                "max_y": self.alpha_start_maxy,
                "min_z": self.alpha_start_minz,
                "max_z": self.alpha_start_maxz,
            }
        elif prop == "alpha_end":
            return {
                "min_x": self.alpha_end_minx,
                "max_x": self.alpha_end_maxx,
                "min_y": self.alpha_end_miny,
                "max_y": self.alpha_end_maxy,
                "min_z": self.alpha_end_minz,
                "max_z": self.alpha_end_maxz,
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
        alpha_start: float = 1e-8,
        alpha_end: float = 1e-8,
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
            alpha_start_minx=alpha_start,
            alpha_end_minx=alpha_end,
            alpha_start_maxx=alpha_start,
            alpha_end_maxx=alpha_end,
            alpha_start_miny=alpha_start,
            alpha_end_miny=alpha_end,
            alpha_start_maxy=alpha_start,
            alpha_end_maxy=alpha_end,
            alpha_start_minz=alpha_start,
            alpha_end_minz=alpha_end,
            alpha_start_maxz=alpha_start,
            alpha_end_maxz=alpha_end,
        )


def boundary_objects_from_config(
    config: BoundaryConfig,
    volume: SimulationVolume,
    sim_config: SimulationConfig,
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
    boundaries: dict[str, Union[PerfectlyMatchedLayer, PeriodicBoundary]] = {}
    constraints: list[PositionConstraint] = []

    # Face settings
    thickness_dict = config.get_dict()
    type_dict = config.get_type_dict()
    kappa_start_dict = config.get_kappa_dict("kappa_start")
    kappa_end_dict = config.get_kappa_dict("kappa_end")
    alpha_start_dict = config.get_alpha_dict("alpha_start")
    alpha_end_dict = config.get_alpha_dict("alpha_end")

    # Helper: per-axis index and face keys
    axes = ("x", "y", "z")
    face_min_keys = ("min_x", "min_y", "min_z")
    face_max_keys = ("max_x", "max_y", "max_z")

    # Periodic faces are still added as-is (they don't overlap).
    for key in (*face_min_keys, *face_max_keys):
        if type_dict[key] == "periodic":
            ax, direction = axis_direction_from_kind(key)
            direction_int = -1 if direction == "-" else 1
            grid_shape_list: list[int | None] = [None, None, None]
            grid_shape_list[ax] = 1  # zero-thickness placeholder for periodic
            grid_shape: PartialGridShape3D = tuple(grid_shape_list)  # type: ignore
            other_axes = [0, 1, 2]
            del other_axes[ax]

            pb = PeriodicBoundary(
                axis=ax,
                partial_grid_shape=grid_shape,
                direction=direction,
            )
            pos = pb.place_relative_to(
                volume,
                axes=(ax, other_axes[0], other_axes[1]),
                own_positions=(direction_int, 0, 0),
                other_positions=(direction_int, 0, 0),
            )
            boundaries[key] = pb
            constraints.append(pos)

    # If no PMLs configured, we are done
    if all(type_dict[k] != "pml" for k in (*face_min_keys, *face_max_keys)):
        return boundaries, constraints

    vol_shape: PartialGridShape3D = volume.partial_grid_shape
    if any(s is None for s in volume.partial_grid_shape):
        if any(s is None for s in volume.partial_real_shape):
            raise Exception("Either SimulationVolume.partial_grid_shape or SimulationVolume.partial_real_shape must be fully specified before boundary creation")
        vol_shape = (
            int(round(volume.partial_real_shape[0] / sim_config.resolution)),
            int(round(volume.partial_real_shape[1] / sim_config.resolution)),
            int(round(volume.partial_real_shape[2] / sim_config.resolution)),
        )
    print(vol_shape)
    full_x, full_y, full_z = int(vol_shape[0]), int(vol_shape[1]), int(vol_shape[2])

    # Thicknesses per face (0 if not PML)
    tmin = [
        thickness_dict["min_x"] if type_dict["min_x"] == "pml" else 0,
        thickness_dict["min_y"] if type_dict["min_y"] == "pml" else 0,
        thickness_dict["min_z"] if type_dict["min_z"] == "pml" else 0,
    ]
    tmax = [
        thickness_dict["max_x"] if type_dict["max_x"] == "pml" else 0,
        thickness_dict["max_y"] if type_dict["max_y"] == "pml" else 0,
        thickness_dict["max_z"] if type_dict["max_z"] == "pml" else 0,
    ]

    # Compute interior lengths (disjoint middle segments)
    Lx = full_x - tmin[0] - tmax[0]
    Ly = full_y - tmin[1] - tmax[1]
    Lz = full_z - tmin[2] - tmax[2]
    if Lx < 0 or Ly < 0 or Lz < 0:
        raise ValueError(
            f"PML thickness exceeds volume size: "
            f"L=({full_x},{full_y},{full_z}), tmin={tuple(tmin)}, tmax={tuple(tmax)}"
        )

    # Alpha/kappa per-face helpers (fallbacks even if one side is not PML won't be used)
    def get_alpha_kappa_for_sign(ax: int, s: int) -> tuple[float, float, float, float]:
        if s < 0:
            face = f"min_{axes[ax]}"
        elif s > 0:
            face = f"max_{axes[ax]}"
        else:
            face = f"min_{axes[ax]}"  # unused for s==0
        return (
            alpha_start_dict[face],
            alpha_end_dict[face],
            kappa_start_dict[face],
            kappa_end_dict[face],
        )

    # Build segments per axis: (label, sign, length)
    segs = []
    for ax, L in enumerate((Lx, Ly, Lz)):
        seg_ax: list[tuple[str, int, int]] = []
        if tmin[ax] > 0:
            seg_ax.append(("min", -1, tmin[ax]))
        seg_ax.append(("mid", 0, L))
        if tmax[ax] > 0:
            seg_ax.append(("max", 1, tmax[ax]))
        segs.append(seg_ax)

    # Helper to pick primary axis for a block
    def pick_primary_axis(signs: tuple[int, int, int]) -> int:
        for i in range(3):
            if signs[i] != 0:
                return i
        return 0  # shouldn't happen because we skip all-zero blocks

    def block_key(sx: tuple[str, int], sy: tuple[str, int], sz: tuple[str, int]) -> str:
        lx, sxv = sx
        ly, syv = sy
        lz, szv = sz
        parts = []
        if sxv != 0:
            parts.append(f"{'min' if sxv < 0 else 'max'}_x")
        if syv != 0:
            parts.append(f"{'min' if syv < 0 else 'max'}_y")
        if szv != 0:
            parts.append(f"{'min' if szv < 0 else 'max'}_z")

        # Face-center convenience aliases for non-overlap stripes
        if sxv != 0 and syv == 0 and szv == 0:
            return "min_x" if sxv < 0 else "max_x"
        if syv != 0 and sxv == 0 and szv == 0:
            return "min_y" if syv < 0 else "max_y"
        if szv != 0 and sxv == 0 and syv == 0:
            return "min_z" if szv < 0 else "max_z"

        return "pml_" + ("_".join(parts) if parts else "interior")

    for lx, sx, nx in segs[0]:
        for ly, sy, ny in segs[1]:
            for lz, sz, nz in segs[2]:
                if sx == 0 and sy == 0 and sz == 0:
                    continue

                if nx == 0 or ny == 0 or nz == 0:
                    continue

                direction_params = (sx, sy, sz)
                primary = pick_primary_axis(direction_params)
                direction = "-" if direction_params[primary] < 0 else "+"
                sizes = (nx, ny, nz)

                a_s, a_e, k_s, k_e = get_alpha_kappa_for_sign(primary, direction_params[primary])

                grid_shape: PartialGridShape3D = (sizes[0], sizes[1], sizes[2])  # type: ignore

                pml = PerfectlyMatchedLayer(
                    axis=primary,
                    partial_grid_shape=grid_shape,
                    kappa_start=k_s,
                    kappa_end=k_e,
                    alpha_start=a_s,
                    alpha_end=a_e,
                    direction=direction,
                    direction_params=direction_params,
                )

                order = (primary, (primary + 1) % 3, (primary + 2) % 3)
                pos_map = {0: sx, 1: sy, 2: sz}
                own_positions = (pos_map[order[0]], pos_map[order[1]], pos_map[order[2]])
                other_positions = own_positions

                constraint = pml.place_relative_to(
                    volume,
                    axes=order,
                    own_positions=own_positions,
                    other_positions=other_positions,
                )

                key = block_key((lx, sx), (ly, sy), (lz, sz))
                base_key = key
                suffix = 1
                while key in boundaries:
                    suffix += 1
                    key = f"{base_key}__{suffix}"

                boundaries[key] = pml
                constraints.append(constraint)

    return boundaries, constraints
