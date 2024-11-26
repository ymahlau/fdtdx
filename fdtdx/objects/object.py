from abc import ABC
import abc
from dataclasses import dataclass
from typing import Self

import jax
import pytreeclass as tc

from fdtdx.core.config import DUMMY_SIMULATION_CONFIG, SimulationConfig
from fdtdx.core.misc import ensure_slice_tuple
from fdtdx.core.jax.pytrees import ExtendedTreeClass
from fdtdx.core.jax.typing import (
    INVALID_SLICE_TUPLE_3D,
    UNDEFINED_SHAPE_3D,
    GridShape3D,
    PartialGridShape3D,
    PartialRealShape3D,
    RealShape3D,
    Slice3D,
    SliceTuple3D,
)

_GLOBAL_COUNTER = 0

@tc.autoinit
class UniqueName(ExtendedTreeClass):
    def __call__(self, x):
        global _GLOBAL_COUNTER
        if x is None:
            name = f"Object_{_GLOBAL_COUNTER}"
            _GLOBAL_COUNTER += 1
            return name
        return x

@dataclass(kw_only=True, frozen=True)
class PositionConstraint:
    owner: "SimulationObject"  # "parent" object
    target: "SimulationObject"  # "child" object
    axes: tuple[int, ...]
    owner_positions: tuple[float, ...]
    target_positions: tuple[float, ...]
    margins: tuple[float, ...]
    grid_margins: tuple[int, ...]


@dataclass(kw_only=True, frozen=True)
class SizeConstraint:
    owner: "SimulationObject"  # "parent" object
    target: "SimulationObject"  # "child" object
    axes: tuple[int, ...]
    other_axes: tuple[int, ...]
    proportions: tuple[float, ...]
    offsets: tuple[float, ...]
    grid_offsets: tuple[int, ...]


@tc.autoinit
class SimulationObject(ExtendedTreeClass, ABC):
    partial_real_shape: PartialRealShape3D = UNDEFINED_SHAPE_3D
    partial_grid_shape: PartialGridShape3D = UNDEFINED_SHAPE_3D
    placement_order: int = 0
    color: tuple[float, float, float] | None = None  # RGB, interval[0, 1]
    _grid_slice_tuple: SliceTuple3D = tc.field(  # type: ignore
        default=INVALID_SLICE_TUPLE_3D,
        init=False,
        on_getattr=[tc.unfreeze],
        on_setattr=[tc.freeze],
    )
    _config: SimulationConfig = DUMMY_SIMULATION_CONFIG
    name: str = tc.field(
        default=None,
        init=True,
        on_setattr=[UniqueName(), tc.freeze],
        on_getattr=[tc.unfreeze],
    )  # type: ignore
    
    @property
    def grid_slice_tuple(self) -> SliceTuple3D:
        if self._grid_slice_tuple == INVALID_SLICE_TUPLE_3D:
            raise Exception(f"Object is not yet initialized: {self}")
        return self._grid_slice_tuple
    
    @property
    def grid_slice(self) -> Slice3D:
        tpl = ensure_slice_tuple(self._grid_slice_tuple)
        if len(tpl) != 3:
            raise Exception(f"Invalid slice tuple, this should never happen: {tpl}")
        return tpl[0], tpl[1], tpl[2]
    
    @property
    def real_shape(self) -> RealShape3D:
        grid_shape = self.grid_shape
        real_shape = (
            grid_shape[0] * self._config.resolution,
            grid_shape[1] * self._config.resolution,
            grid_shape[2] * self._config.resolution,
        )
        return real_shape
    
    @property
    def grid_shape(self) -> GridShape3D:
        if self._grid_slice_tuple == INVALID_SLICE_TUPLE_3D:
            raise Exception("Cannot compute shape on non-initialized object")
        return (
            self._grid_slice_tuple[0][1] - self._grid_slice_tuple[0][0],
            self._grid_slice_tuple[1][1] - self._grid_slice_tuple[1][0],
            self._grid_slice_tuple[2][1] - self._grid_slice_tuple[2][0],
        )
    
    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        del key
        if self._grid_slice_tuple != INVALID_SLICE_TUPLE_3D:
            raise Exception(f"Object is already compiled to grid: {self}")
        for axis in range(3):
            s1, s2 = grid_slice_tuple[axis]
            if s1 < 0 or s2 < 0 or s2 <= s1:
                raise Exception(f"Invalid placement of object {self} at {grid_slice_tuple}")
        self = self.aset("_grid_slice_tuple", grid_slice_tuple)
        self = self.aset("_config", config)

        return self

    @abc.abstractmethod
    def get_inv_permittivity(
        self,
        prev_inv_permittivity: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permittivity and info dict
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_inv_permeability(
        self,
        prev_inv_permeability: jax.Array,
        params: dict[str, jax.Array] | None,
    ) -> tuple[jax.Array, dict]:  # permeability and info dict
        raise NotImplementedError()
    

    def place_relative_to(
        self,
        other: "SimulationObject",
        axes: tuple[int, ...],
        own_positions: tuple[float, ...],
        other_positions: tuple[float, ...],  # positions are relative to size
        margins: tuple[float, ...] | None = None,  # absolute coordinates
        grid_margins: tuple[int, ...] | None = None,
    ) -> PositionConstraint:
        if margins is None:
            margins = tuple([0 for _ in axes])
        if grid_margins is None:
            grid_margins = tuple([0 for _ in axes])
        if (
            len(axes) != len(own_positions)
            or len(axes) != len(other_positions)
            or len(axes) != len(margins)
            or len(axes) != len(grid_margins)
        ):
            raise Exception("All inputs should have same lengths")
        constraint = PositionConstraint(
            axes=axes,
            owner=other,
            target=self,
            owner_positions=other_positions,
            target_positions=own_positions,
            margins=margins,
            grid_margins=grid_margins,
        )
        return constraint

    def size_relative_to(
        self,
        other: "SimulationObject",
        axes: tuple[int, ...],
        other_axes: tuple[int, ...] | None = None,
        proportions: tuple[float, ...] | None = None,
        offsets: tuple[float, ...] | None = None,
        grid_offsets: tuple[int, ...] | None = None,
    ) -> SizeConstraint:
        if offsets is None:
            offsets = tuple([0 for _ in axes])
        if grid_offsets is None:
            grid_offsets = tuple([0 for _ in axes])
        if proportions is None:
            proportions = tuple([1.0 for _ in axes])
        if other_axes is None:
            other_axes = tuple([a for a in axes])
        if (
            len(axes) != len(proportions)
            or len(axes) != len(offsets)
            or len(axes) != len(grid_offsets)
        ):
            raise Exception("All inputs should have same lengths")
        constraint = SizeConstraint(
            owner=other,
            target=self,
            axes=axes,
            other_axes=other_axes,
            proportions=proportions,
            offsets=offsets,
            grid_offsets=grid_offsets,
        )
        return constraint

    def same_size(
        self,
        other: "SimulationObject",
        axes: tuple[int, ...] = (0, 1, 2),
        proportions: tuple[float, ...] = (1, 1, 1),
        offsets: tuple[float, ...] = (0, 0, 0),
        grid_offsets: tuple[int, ...] = (0, 0, 0),
    ) -> SizeConstraint:
        num_axis = len(axes)
        if num_axis != 3:
            proportions = proportions[:num_axis]
            offsets = offsets[:num_axis]
            grid_offsets = grid_offsets[:num_axis]
        constraint = self.size_relative_to(
            other=other,
            axes=axes,
            proportions=proportions,
            offsets=offsets,
            grid_offsets=grid_offsets,
        )
        return constraint

    def place_at_center(
        self,
        other: "SimulationObject",
        axes: tuple[int, ...] = (0, 1, 2),
        own_positions: tuple[float, ...] = (0, 0, 0),
        other_positions: tuple[float, ...] = (
            0,
            0,
            0,
        ),  # positions are relative to size
        margins: tuple[float, ...] | None = None,  # absolute coordinates
        grid_margins: tuple[int, ...] | None = None,
    ) -> PositionConstraint:
        constraint = self.place_relative_to(
            other=other,
            axes=axes,
            own_positions=own_positions,
            other_positions=other_positions,
            margins=margins,
            grid_margins=grid_margins,
        )
        return constraint

    def same_position_and_size(
        self,
        other: "SimulationObject",
    ) -> tuple[PositionConstraint, SizeConstraint]:
        size_constraint = self.size_relative_to(
            other=other,
            axes=(0, 1, 2),
            proportions=(1, 1, 1),
            offsets=(0, 0, 0),
            grid_offsets=(0, 0, 0),
        )
        pos_constraint = self.place_at_center(
            other=other,
            axes=(0, 1, 2),
        )
        return pos_constraint, size_constraint

    def place_face_to_face_positive_direction(
        self,
        other: "SimulationObject",
        axes: tuple[int],
        own_positions: tuple[float] = (-1,),
        other_positions: tuple[float] = (1,),  # positions are relative to size
        margins: tuple[float] = (0,),  # absolute coordinates
        grid_margins: tuple[int] = (0,),
    ) -> PositionConstraint:
        constraint = self.place_relative_to(
            other=other,
            axes=axes,
            own_positions=own_positions,
            other_positions=other_positions,
            margins=margins,
            grid_margins=grid_margins,
        )
        return constraint

    def place_face_to_face_negative_direction(
        self,
        other: "SimulationObject",
        axes: tuple[int],
        own_positions: tuple[float] = (1,),
        other_positions: tuple[float] = (-1,),  # positions are relative to size
        margins: tuple[float] = (0,),  # absolute coordinates
        grid_margins: tuple[int] = (0,),
    ) -> PositionConstraint:
        constraint = self.place_relative_to(
            other=other,
            axes=axes,
            own_positions=own_positions,
            other_positions=other_positions,
            margins=margins,
            grid_margins=grid_margins,
        )
        return constraint
    
    def place_above(
        self,
        other: "SimulationObject",
        own_positions: tuple[float] = (-1,),
        other_positions: tuple[float] = (1,),  # positions are relative to size
        margins: tuple[float] = (0,),  # absolute coordinates
        grid_margins: tuple[int] = (0,),
    ) -> PositionConstraint:
        constraint = self.place_face_to_face_positive_direction(
            other=other,
            axes=(2,),
            own_positions=own_positions,
            other_positions=other_positions,
            margins=margins,
            grid_margins=grid_margins,
        )
        return constraint

    def place_below(
        self,
        other: "SimulationObject",
        own_positions: tuple[float] = (1,),
        other_positions: tuple[float] = (-1,),  # positions are relative to size
        margins: tuple[float] = (0,),  # absolute coordinates
        grid_margins: tuple[int] = (0,),
    ) -> PositionConstraint:
        constraint = self.place_face_to_face_negative_direction(
            other=other,
            axes=(2,),
            own_positions=own_positions,
            other_positions=other_positions,
            margins=margins,
            grid_margins=grid_margins,
        )
        return constraint

    def check_overlap(
        self,
        other: "SimulationObject",
    ) -> bool:
        for axis in range(3):
            s_start, s_end = self._grid_slice_tuple[axis]
            o_start, o_end = other._grid_slice_tuple[axis]
            if o_start <= s_start <= o_end:
                return True
            if o_start <= s_end <= o_end:
                return True
        return False
    
    def __eq__(
        self: Self,
        other: "SimulationObject",
    ) -> bool:
        return self.name == other.name
    
    def __hash__(
        self
    ) -> int:
        return hash(self.name)




