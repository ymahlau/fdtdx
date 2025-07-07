from abc import ABC
from dataclasses import dataclass
from typing import Literal, Optional, Self

import jax

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import (
    TreeClass,
    autoinit,
    frozen_field,
    frozen_private_field,
    private_field,
)
from fdtdx.core.misc import ensure_slice_tuple
from fdtdx.typing import (
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


@autoinit
class UniqueName(TreeClass):
    """Generates unique names for simulation objects.

    A utility class that ensures each simulation object gets a unique name by
    maintaining a global counter. If no name is provided, generates names in
    the format "Object_N" where N is an incrementing counter.
    """

    def __call__(self, x: str | None) -> str:
        """Generate a unique name if none is provided.

        Args:
            x (str | None): The proposed name or None

        Returns:
            str: Either the input name if provided, or a new unique name
        """
        global _GLOBAL_COUNTER
        if x is None:
            name = f"Object_{_GLOBAL_COUNTER}"
            _GLOBAL_COUNTER += 1
            return name
        return x


@dataclass(kw_only=True, frozen=True)
class PositionConstraint:
    """Defines a positional relationship between two simulation objects.

    A constraint that positions one object relative to another, with optional
    margins and offsets. Used to specify how objects should be placed in the
    simulation volume relative to each other.

    Attributes:
        object (SimulationObject): The "child" object whose position is being adjusted
        other_object (SimulationObject): The "parent" object that serves as reference
        axes (tuple[int, ...]): Which axes (x,y,z) this constraint applies to
        object_positions (tuple[float, ...]): Relative positions on child object (-1 to 1)
        other_object_positions (tuple[float, ...]): Relative positions on parent object (-1 to 1)
        margins (tuple[float, ...]): Optional real-space margins between objects
        grid_margins (tuple[int, ...]): Optional grid-space margins between objects
    """

    object: "SimulationObject"  # "child" object, whose pos is adjusted
    other_object: "SimulationObject"  # "parent" object
    axes: tuple[int, ...]
    object_positions: tuple[float, ...]
    other_object_positions: tuple[float, ...]
    margins: tuple[float, ...]
    grid_margins: tuple[int, ...]


@dataclass(kw_only=True, frozen=True)
class SizeConstraint:
    """Defines a size relationship between two simulation objects.

    A constraint that sets the size of one object relative to another, with
    optional proportions and offsets. Used to specify how objects should be
    sized relative to each other in the simulation.

    Attributes:
        object (SimulationObject): The "child" object whose size is being adjusted
        other_object (SimulationObject): The "parent" object that serves as reference
        axes (tuple[int, ...]): Which axes of the child to constrain
        other_axes (tuple[int, ...]): Which axes of the parent to reference
        proportions (tuple[float, ...]): Size multipliers relative to parent
        offsets (tuple[float, ...]): Additional real-space size offsets
        grid_offsets (tuple[int, ...]): Additional grid-space size offsets
    """

    object: "SimulationObject"  # "child" object, whose size is adjusted
    other_object: "SimulationObject"  # "parent" object
    axes: tuple[int, ...]
    other_axes: tuple[int, ...]
    proportions: tuple[float, ...]
    offsets: tuple[float, ...]
    grid_offsets: tuple[int, ...]


@dataclass(kw_only=True, frozen=True)
class SizeExtensionConstraint:
    """Defines how an object extends toward another object or boundary.

    A constraint that extends one object's size until it reaches another object
    or the simulation boundary. Can extend in positive or negative direction
    along an axis.

    Attributes:
        object (SimulationObject): The object being extended
        other_object (Optional["SimulationObject"]): Optional target object to extend to
        axis (int): Which axis to extend along
        direction (Literal["+", "-"]): Direction to extend ('+' or '-')
        other_position (float): Relative position on target (-1 to 1)
        offset (float): Additional real-space offset
        grid_offset (int): Additional grid-space offset
    """

    object: "SimulationObject"  # "child" object, whose size is adjusted
    other_object: Optional["SimulationObject"]  # "parent" object
    axis: int
    direction: Literal["+", "-"]
    other_position: float
    offset: float
    grid_offset: int


@dataclass(kw_only=True, frozen=True)
class GridCoordinateConstraint:
    """Constrains an object's position to specific grid coordinates.

    Forces specific sides of an object to align with given grid coordinates.
    Used for precise positioning in the discretized simulation space.

    Attributes:
        object (SimulationObject): The object to position
        axes (tuple[int, ...]): Which axes to constrain
        sides (tuple[Literal["+", "-"], ...]): Which side of each axis ('+' or '-')
        coordinates (tuple[int, ...]): Grid coordinates to align with
    """

    object: "SimulationObject"
    axes: tuple[int, ...]
    sides: tuple[Literal["+", "-"], ...]
    coordinates: tuple[int, ...]


@dataclass(kw_only=True, frozen=True)
class RealCoordinateConstraint:
    """Constrains an object's position to specific real-space coordinates.

    Forces specific sides of an object to align with given real-space coordinates.
    Used for precise positioning in physical units.

    Attributes:
        object (SimulationObject): The object to position
        axes (tuple[int, ...]): Which axes to constrain
        sides (tuple[Literal["+", "-"], ...]): Which side of each axis ('+' or '-')
        coordinates (tuple[float, ...]): Real-space coordinates to align with
    """

    object: "SimulationObject"
    axes: tuple[int, ...]
    sides: tuple[Literal["+", "-"], ...]
    coordinates: tuple[float, ...]


@autoinit
class SimulationObject(TreeClass, ABC):
    """Abstract base class for objects in a 3D simulation environment.

    This class provides the foundation for simulation objects with spatial properties and positioning capabilities
    in both real and grid coordinate systems. It supports random positioning offsets.

    Attributes:
        partial_real_shape (PartialRealShape3D, optional): The object's shape in real-world
            coordinates. Defaults to UNDEFINED_SHAPE_3D if not specified.
        partial_grid_shape (PartialGridShape3D, optional): The object's shape in grid coordinates.
            Defaults to UNDEFINED_SHAPE_3D if not specified.
        color (tuple[float, float, float] | None, optional): RGB color values for the object,
            where each component is in the interval [0, 1]. None indicates no color
            is specified. Defaults to None.
        name (str, optional): Unique identifier for the object. Automatically enforced to be
            unique through the UniqueName validator. The user can also set a name manually.
        max_random_real_offsets (tuple[float, float, float], optional): Maximum random offset
            values that can be applied to the object's position in real coordinates
            for each axis (x, y, z). Defaults to (0, 0, 0) for no random offset.
        max_random_grid_offsets (tuple[int, int, int], optional): Maximum random offset values
            that can be applied to the object's position in grid coordinates for each
            axis (x, y, z). Defaults to (0, 0, 0) for no random offset.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    partial_real_shape: PartialRealShape3D = frozen_field(default=UNDEFINED_SHAPE_3D)
    partial_grid_shape: PartialGridShape3D = frozen_field(default=UNDEFINED_SHAPE_3D)
    color: tuple[float, float, float] | None = frozen_field(default=None)  # RGB, interval[0, 1]
    name: str = frozen_field(  # type: ignore
        default=None,
        on_setattr=[UniqueName()],
    )
    max_random_real_offsets: tuple[float, float, float] = frozen_field(default=(0, 0, 0))
    max_random_grid_offsets: tuple[int, int, int] = frozen_field(default=(0, 0, 0))

    _grid_slice_tuple: SliceTuple3D = frozen_private_field(
        default=INVALID_SLICE_TUPLE_3D,
    )
    _config: SimulationConfig = private_field()

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
        self = self.aset("_config", config, create_new_ok=True)
        return self

    def apply(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
    ) -> Self:
        del key, inv_permittivities, inv_permeabilities
        return self

    def place_relative_to(
        self,
        other: "SimulationObject",
        axes: tuple[int, ...] | int,
        own_positions: tuple[float, ...] | float,
        other_positions: tuple[float, ...] | float,
        margins: tuple[float, ...] | float | None = None,
        grid_margins: tuple[int, ...] | int | None = None,
    ) -> PositionConstraint:
        """Creates a PositionalConstraint between two objects. The constraint is defined by anchor points on
        both objects, which are constrainted to be at the same position. Anchors are defined in relative coordinates,
        i.e. a position of -1 is the left object boundary in the repective axis and a position of +1 the right boundary.

        Args:
            other (SimulationObject): Another object in the simulation scene
            axes (tuple[int, ...] | int): Eiter a single integer or a tuple describing the axes of the constraints
            own_positions (tuple[float, ...] | float): The positions of the own anchor in the axes. Must have the same lengths as axes
            other_positions (tuple[float, ...] | float): The positions of the other objects' anchor in the axes. Must have the same lengths as axes
            margins (tuple[float, ...] | float | None, optional): The margins between the anchors of both objects in
                meters. Must have the same lengths as axes. If None, no margin is used. Defaults to None.
            grid_margins (tuple[int, ...] | int | None, optional): The margins between the anchors of both objects
                in Yee-grid voxels. Must have the same lengths as axes. If none, no margin is used. Defaults to None.

        Returns:
            PositionConstraint: Positional constraint between this object and the other
        """
        if isinstance(axes, int):
            axes = (axes,)
        if isinstance(own_positions, int | float):
            own_positions = (float(own_positions),)
        if isinstance(other_positions, int | float):
            other_positions = (float(other_positions),)
        if isinstance(margins, int | float):
            margins = (float(margins),)
        if isinstance(grid_margins, int):
            grid_margins = (grid_margins,)
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
            other_object=other,
            object=self,
            other_object_positions=other_positions,
            object_positions=own_positions,
            margins=margins,
            grid_margins=grid_margins,
        )
        return constraint

    def size_relative_to(
        self,
        other: "SimulationObject",
        axes: tuple[int, ...] | int,
        other_axes: tuple[int, ...] | int | None = None,
        proportions: tuple[float, ...] | float | None = None,
        offsets: tuple[float, ...] | float | None = None,
        grid_offsets: tuple[int, ...] | int | None = None,
    ) -> SizeConstraint:
        """Creates a SizeConstraint between two objects. The constraint defines the size of this object relative
        to another object, allowing for proportional scaling and offsets in specified axes.

        Args:
            other (SimulationObject): Another object in the simulation scene
            axes (tuple[int, ...] | int): Either a single integer or a tuple describing which axes of this object to
                constrain.
            other_axes (tuple[int, ...] | int | None, optional): Either a single integer or a tuple describing which
                axes of the other object to reference. If None, uses the same axes as specified in 'axes'. Defaults
                to None.
            proportions (tuple[float, ...] | float | None, optional): Scale factors to apply to the other object's
                dimensions. Must have same length as axes. If None, uses 1.0 (same size). Defaults to None.
            offsets (tuple[float, ...] | float | None, optional): Additional size offsets in meters to apply after
                scaling. Must have same length as axes. If None, no offset is used. Defaults to None.
            grid_offsets (tuple[int, ...] | int | None, optional): Additional size offsets in Yee-grid voxels to
                apply after scaling. Must have same length as axes. If None, no offset is used. Defaults to None.

        Returns:
            SizeConstraint: Size constraint between this object and the other
        """
        if isinstance(axes, int):
            axes = (axes,)
        if isinstance(other_axes, int):
            other_axes = (other_axes,)
        if isinstance(proportions, int | float):
            proportions = (float(proportions),)
        if isinstance(offsets, int | float):
            offsets = (offsets,)
        if isinstance(grid_offsets, int):
            grid_offsets = (grid_offsets,)
        if offsets is None:
            offsets = tuple([0 for _ in axes])
        if grid_offsets is None:
            grid_offsets = tuple([0 for _ in axes])
        if proportions is None:
            proportions = tuple([1.0 for _ in axes])
        if other_axes is None:
            other_axes = tuple([a for a in axes])
        if len(axes) != len(proportions) or len(axes) != len(offsets) or len(axes) != len(grid_offsets):
            raise Exception("All inputs should have same lengths")
        constraint = SizeConstraint(
            other_object=other,
            object=self,
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
        axes: tuple[int, ...] | int = (0, 1, 2),
        offsets: tuple[float, ...] | float | None = None,
        grid_offsets: tuple[int, ...] | int | None = None,
    ) -> SizeConstraint:
        """Creates a SizeConstraint that makes this object the same size as another object along specified axes.
        This is a convenience wrapper around size_relative_to() with proportions set to 1.0.

        Args:
            other (SimulationObject): Another object in the simulation scene
            axes (tuple[int, ...] | int, optional): Either a single integer or a tuple describing which axes should
                have the same size. Defaults to all axes (0, 1, 2).
            offsets (tuple[float, ...] | float | None, optional): Additional size offsets in meters to apply.
                Must have same length as axes. If None, no offset is used. Defaults to None.
            grid_offsets (tuple[int, ...] | int | None, optional): Additional size offsets in Yee-grid voxels to
                apply. Must have same length as axes. If None, no offset is used. Defaults to None.

        Returns:
            SizeConstraint: Size constraint ensuring equal sizes between objects
        """
        if isinstance(axes, int):
            axes = (axes,)
        proportions = tuple([1 for _ in axes])
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
        axes: tuple[int, ...] | int = (0, 1, 2),
        own_positions: tuple[float, ...] | float | None = None,
        other_positions: tuple[float, ...] | float | None = None,
        margins: tuple[float, ...] | float | None = None,
        grid_margins: tuple[int, ...] | int | None = None,
    ) -> PositionConstraint:
        """Creates a PositionConstraint that centers this object relative to another object along specified axes.
        This is a convenience wrapper around place_relative_to() with default positions at the center (0).

        Args:
            other (SimulationObject): Another object in the simulation scene
            axes (tuple[int, ...] | int, optional): Either a single integer or a tuple describing which axes to center
                on. Defaults to all axes (0, 1, 2).
            own_positions (tuple[float, ...] | float | None, optional): Relative positions on this object (-1 to 1).
                If None, uses center (0). Defaults to None.
            other_positions (tuple[float, ...] | float | None, optional): Relative positions on other object (-1 to 1).
                If None, uses center (0). Defaults to None.
            margins (tuple[float, ...] | float | None, optional): Additional margins in meters between objects.
                Must have same length as axes. If None, no margin is used. Defaults to None.
            grid_margins ( tuple[int, ...] | int | None, optional): Additional margins in Yee-grid voxels between
                objects. Must have same length as axes. If None, no margin is used. Defaults to None.

        Returns:
            PositionConstraint: Position constraint centering objects relative to each other
        """
        if isinstance(axes, int):
            axes = (axes,)
        if own_positions is None:
            own_positions = tuple([0 for _ in axes])
        if other_positions is None:
            other_positions = tuple([0 for _ in axes])
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
        axes: tuple[int, ...] | int = (0, 1, 2),
    ) -> tuple[PositionConstraint, SizeConstraint]:
        """Creates both position and size constraints to make this object match another object's position and size.
        This is a convenience wrapper combining place_at_center() and same_size().

        Args:
            other (SimulationObject): Another object in the simulation scene
            axes (tuple[int, ...] | int, optional): Either a single integer or a tuple describing which axes to match.
                Defaults to all axes (0, 1, 2).

        Returns:
            tuple[PositionConstraint, SizeConstraint]: Position and size constraints for matching objects
        """
        size_constraint = self.same_size(
            other=other,
            axes=axes,
        )
        pos_constraint = self.place_at_center(
            other=other,
            axes=axes,
        )
        return pos_constraint, size_constraint

    def face_to_face_positive_direction(
        self,
        other: "SimulationObject",
        axes: tuple[int, ...] | int,
        margins: tuple[float, ...] | float | None = None,
        grid_margins: tuple[int, ...] | int | None = None,
    ) -> PositionConstraint:
        """Creates a PositionConstraint that places this object facing another object in the positive direction
        of specified axes. The objects will touch at their facing boundaries unless margins are specified.

        Args:
            other (SimulationObject): Another object in the simulation scene
            axes (tuple[int, ...] | int): Either a single integer or a tuple describing which axes to align on
            margins (tuple[float, ...] | float | None, optional): Additional margins in meters between the facing
                surfaces. Must have same length as axes. If None, no margin is used. Defaults to None.
            grid_margins (tuple[int, ...] | int | None, optional): Additional margins in Yee-grid voxels between the
                facing surfaces. Must have same length as axes. If None, no margin is used. Defaults to None

        Returns:
            PositionConstraint: Position constraint aligning objects face-to-face in positive direction
        """
        if isinstance(axes, int):
            axes = (axes,)
        own_positions = tuple([-1 for _ in axes])
        other_positions = tuple([1 for _ in axes])
        constraint = self.place_relative_to(
            other=other,
            axes=axes,
            own_positions=own_positions,
            other_positions=other_positions,
            margins=margins,
            grid_margins=grid_margins,
        )
        return constraint

    def face_to_face_negative_direction(
        self,
        other: "SimulationObject",
        axes: tuple[int, ...] | int,
        margins: tuple[float, ...] | float | None = None,
        grid_margins: tuple[int, ...] | int | None = None,
    ) -> PositionConstraint:
        """Creates a PositionConstraint that places this object facing another object in the negative direction
        of specified axes. The objects will touch at their facing boundaries unless margins are specified.

        Args:
            other (SimulationObject): Another object in the simulation scene
            axes (tuple[int, ...] | int): Either a single integer or a tuple describing which axes to align on
            margins (tuple[float, ...] | float | None, optional): Additional margins in meters between the facing
                surfaces. Must have same length as axes. If None, no margin is used. Defaults to None.
            grid_margins (tuple[int, ...] | int | None, optional): Additional margins in Yee-grid voxels between the
                facing surfaces. Must have same length as axes. If None, no margin is used. Defaults to None.

        Returns:
            PositionConstraint: Position constraint aligning objects face-to-face in negative direction
        """
        if isinstance(axes, int):
            axes = (axes,)
        own_positions = tuple([1 for _ in axes])
        other_positions = tuple([-1 for _ in axes])
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
        margins: tuple[float, ...] | float | None = None,
        grid_margins: tuple[int, ...] | int | None = None,
    ) -> PositionConstraint:
        """Creates a PositionConstraint that places this object above another object along the z-axis.
        This is a convenience wrapper around face_to_face_positive_direction() for axis 2 (z-axis).

        Args:
            other (SimulationObject): Another object in the simulation scene
            margins (tuple[float, ...] | float | None, optional): Additional vertical margins in meters between objects.
                If None, no margin is used. Defaults to None.
            grid_margins (tuple[int, ...] | int | None, optional): Additional vertical margins in Yee-grid voxels
                between objects. If None, no margin is used. Defaults to None.

        Returns:
            PositionConstraint: Position constraint placing this object above the other
        """
        constraint = self.face_to_face_positive_direction(
            other=other,
            axes=(2,),
            margins=margins,
            grid_margins=grid_margins,
        )
        return constraint

    def place_below(
        self,
        other: "SimulationObject",
        margins: tuple[float, ...] | float | None = None,
        grid_margins: tuple[int, ...] | int | None = None,
    ) -> PositionConstraint:
        """Creates a PositionConstraint that places this object below another object along the z-axis.
        This is a convenience wrapper around face_to_face_negative_direction() for axis 2 (z-axis).

        Args:
            other (SimulationObject): Another object in the simulation scene
            margins (tuple[float, ...] | float | None, optional): Additional vertical margins in meters between objects.
                If None, no margin is used. Defaults to None.
            grid_margins (tuple[int, ...] | int | None, optional): Additional vertical margins in Yee-grid voxels
                between objects. If None, no margin is used. Defaults to None.

        Returns:
            PositionConstraint: Position constraint placing this object below the other
        """
        constraint = self.face_to_face_negative_direction(
            other=other,
            axes=(2,),
            margins=margins,
            grid_margins=grid_margins,
        )
        return constraint

    def set_grid_coordinates(
        self,
        axes: tuple[int, ...] | int,
        sides: tuple[Literal["+", "-"], ...] | Literal["+", "-"],
        coordinates: tuple[int, ...] | int,
    ) -> GridCoordinateConstraint:
        """Creates a GridCoordinateConstraint that forces specific sides of this object to align with
        given grid coordinates. Used for precise positioning in the discretized simulation space.

        Args:
            axes (tuple[int, ...] | int): Either a single integer or a tuple describing which axes to constrain
            sides (tuple[Literal["+", "-"], ...] | Literal["+", "-"]): Either a single string or a tuple of strings
                ('+' or '-') indicating which side of each axis to constrain. Must have same length as axes.
            coordinates (tuple[int, ...] | int): Either a single integer or a tuple of integers specifying the
                grid coordinates to align with. Must have same length as axes.

        Returns:
            GridCoordinateConstraint: Constraint forcing alignment with specific grid coordinates
        """
        if isinstance(axes, int):
            axes = (axes,)
        if isinstance(sides, str):
            sides = (sides,)
        if isinstance(coordinates, int):
            coordinates = (coordinates,)
        if len(axes) != len(sides) or len(axes) != len(coordinates):
            raise Exception("All inputs need to have the same lengths!")
        return GridCoordinateConstraint(
            object=self,
            axes=axes,
            sides=sides,
            coordinates=coordinates,
        )

    def set_real_coordinates(
        self,
        axes: tuple[int, ...] | int,
        sides: tuple[Literal["+", "-"], ...] | Literal["+", "-"],
        coordinates: tuple[float, ...] | float,
    ) -> RealCoordinateConstraint:
        """Creates a RealCoordinateConstraint that forces specific sides of this object to align with
        given real-space coordinates. Used for precise positioning in physical units.

        Args:
            axes (tuple[int, ...] | int): Either a single integer or a tuple describing which axes to constrain
            sides (tuple[Literal["+", "-"], ...] | Literal["+", "-"]): Either a single string or a tuple of
                strings ('+' or '-') indicating which side of each axis to constrain. Must have same length as axes.
            coordinates (tuple[float, ...] | float): Either a single float or a tuple of floats specifying the
                real-space coordinates in meters to align with. Must have same length as axes.

        Returns:
            RealCoordinateConstraint: Constraint forcing alignment with specific real-space coordinates
        """
        if isinstance(axes, int):
            axes = (axes,)
        if isinstance(sides, str):
            sides = (sides,)
        if isinstance(coordinates, int | float):
            coordinates = (float(coordinates),)
        if len(axes) != len(sides) or len(axes) != len(coordinates):
            raise Exception("All inputs need to have the same lengths!")
        return RealCoordinateConstraint(
            object=self,
            axes=axes,
            sides=sides,
            coordinates=coordinates,
        )

    def extend_to(
        self,
        other: "SimulationObject | None",
        axis: int,
        direction: Literal["+", "-"],
        other_position: float | None = None,
        offset: float = 0,
        grid_offset: int = 0,
    ) -> SizeExtensionConstraint:
        """Creates a SizeExtensionConstraint that extends this object along a specified axis until it
        reaches another object or the simulation boundary. The extension can be in either positive or
        negative direction.

        Args:
            other (SimulationObject | None): Target object to extend to, or None to extend to simulation boundary
            axis (int): Which axis to extend along (0, 1, or 2)
            direction (Literal["+", "-"]): Direction to extend in ('+' or '-')
            other_position (float | None, optional): Relative position on target object (-1 to 1) to extend to.
                If None, defaults to the corresponding side (-1 for '+' direction, 1 for '-' direction). Defaults to
                None.
            offset (float, optional): Additional offset in meters to apply after extension. Ignored when extending to
                simulation boundary. Defaults to zero.
            grid_offset (int, optional): Additional offset in Yee-grid voxels to apply after extension. Ignored when
                extending to simulation boundary. Defaults to zero.

        Returns:
            SizeExtensionConstraint: Constraint defining how the object extends
        """
        # default: extend to corresponding side
        if other_position is None:
            other_position = -1 if direction == "+" else 1
        if other is None:
            if offset != 0 or grid_offset != 0:
                raise Exception("Cannot use offset when extending object to infinity")
        return SizeExtensionConstraint(
            object=self,
            other_object=other,
            axis=axis,
            direction=direction,
            other_position=other_position,
            offset=offset,
            grid_offset=grid_offset,
        )

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
        other,
    ) -> bool:
        if not isinstance(other, SimulationObject):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


@autoinit
class OrderableObject(SimulationObject):
    placement_order: int = frozen_field(default=0)
